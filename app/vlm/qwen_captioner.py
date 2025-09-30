"""
VRAM-conserving Qwen2.5-VL captioner (stable):
- bf16 precision
- 32k tokenizer context
- Accelerate device_map="auto" with explicit max_memory (GPU cap + CPU offload)
- Attention implementation set via model config (sdpa / flash_attention_2 / eager)
- OOM / load guards with clean CPU fallback
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from PIL import Image, ImageOps
import os

# allocator hint before torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from ..core.settings import settings
from .captioner import caption_image_stub

_QWEN_SINGLETON = None  # cached instance

@dataclass
class QwenConfig:
    model_id: str
    device: str          # "auto" | "cpu" | "cuda"
    dtype: str           # ignored; we lock bf16 per user request
    max_new_tokens: int
    temperature: float
    top_p: float
    # memory / offload
    offload_folder: str
    gpu_max_gb: float
    cpu_max_gb: float
    attn_impl: str       # "sdpa", "flash_attention_2", "eager"
    context_tokens: int

def _normalize_model_id(mid: str) -> str:
    mid = (mid or "").strip()
    if not mid:
        return "Qwen/Qwen2.5-VL-3B-Instruct"
    if "/" not in mid:
        return "Qwen/" + mid
    return mid

def _bf16():
    return torch.bfloat16

def _max_memory_map(gpu_gb: float, cpu_gb: float) -> dict:
    mm = {"cpu": f"{int(cpu_gb)}GiB"}
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        mm[0] = f"{int(gpu_gb)}GiB"   # integer GPU key
    return mm

class QwenCaptioner:
    def __init__(self, cfg: QwenConfig):
        """
        Load Qwen with offload and stable attention config.
        """
        from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig

        self.cfg = cfg
        model_id = _normalize_model_id(cfg.model_id)
        dtype = _bf16()

        # processor / tokenizer
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, use_fast=True
        )
        try:
            tok = getattr(self.processor, "tokenizer", None)
            if tok is not None:
                tok.model_max_length = int(cfg.context_tokens)  # 32k
        except Exception:
            pass

        # prepare config with attention implementation
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        try:
            # not all models expose this; safe to set when present
            setattr(config, "attn_implementation", cfg.attn_impl)
        except Exception:
            pass

        offload_dir = str(cfg.offload_folder)
        os.makedirs(offload_dir, exist_ok=True)

        force_cpu = (cfg.device == "cpu") or (not torch.cuda.is_available())
        if force_cpu:
            # Pure CPU load
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                config=config,
                dtype=dtype,
                device_map={"": "cpu"},
                offload_folder=offload_dir,
                trust_remote_code=True,
            ).eval()
            self._device = torch.device("cpu")
        else:
            # Auto GPU+CPU offload with integer GPU keys
            try:
                max_mem = _max_memory_map(cfg.gpu_max_gb, cfg.cpu_max_gb)
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    config=config,
                    torch_dtype=dtype,
                    device_map="auto",
                    max_memory=max_mem,
                    offload_folder=offload_dir,
                    trust_remote_code=True,
                ).eval()
                self._device = next(self.model.parameters()).device
            except RuntimeError:
                # OOM -> fallback to CPU
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    config=config,
                    torch_dtype=dtype,
                    device_map={"": "cpu"},
                    offload_folder=offload_dir,
                    trust_remote_code=True,
                ).eval()
                self._device = torch.device("cpu")

    def _apply_orientation(self, img: Image.Image) -> Image.Image:
        return ImageOps.exif_transpose(img).convert("RGB")

    def caption(self, img: Image.Image) -> str:
        """
        One-sentence, parts/materials-focused caption with bf16 + offload.
        """
        img = self._apply_orientation(img)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text",
                     "text": ("You are a concise home-repair assistant. "
                              "Describe the photo in one sentence, focusing on materials, components, and condition.")}
                ]
            }
        ]

        chat_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            images=[img],
            text=chat_text,
            return_tensors="pt",
            padding=True
        )

        # Move inputs to the model device
        dev = self._device
        for k, v in inputs.items():
            if hasattr(v, "to"):
                inputs[k] = v.to(dev)

        # Prefer SDPA mem-efficient kernels when on CUDA and requested
        if dev.type == "cuda" and getattr(torch.backends, "cuda", None):
            if getattr(torch.backends.cuda, "sdp_kernel", None) and self.cfg.attn_impl == "sdpa":
                torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)

        # Generate
        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
        )

        # Decode only continuation
        in_ids = inputs.get("input_ids")
        gen_ids = gen[:, in_ids.shape[1]:] if in_ids is not None else gen
        if hasattr(self.processor, "batch_decode"):
            out = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
            text = (out[0] if out else "").strip()
        else:
            tok = getattr(self.processor, "tokenizer", None)
            text = tok.decode(gen_ids[0], skip_special_tokens=True).strip() if tok else ""

        return text or caption_image_stub(img)

def get_captioner():
    """
    Return a cached captioner configured for bf16 + long context + offload.
    """
    global _QWEN_SINGLETON
    if not settings.use_qwen_captioner:
        return type("StubWrapper", (), {"caption": caption_image_stub})()

    if _QWEN_SINGLETON is not None:
        return _QWEN_SINGLETON

    cfg = QwenConfig(
        model_id=settings.qwen_model_id,
        device=settings.qwen_device,
        dtype="bfloat16",
        max_new_tokens=settings.qwen_max_new_tokens,
        temperature=settings.qwen_temperature,
        top_p=settings.qwen_top_p,
        offload_folder=str(settings.qwen_offload_folder),
        gpu_max_gb=float(settings.qwen_gpu_max_gb),
        cpu_max_gb=float(settings.qwen_cpu_max_gb),
        attn_impl=settings.qwen_attn_impl,
        context_tokens=int(settings.qwen_context_tokens),
    )

    try:
        _QWEN_SINGLETON = QwenCaptioner(cfg)
    except Exception as e:
        _QWEN_SINGLETON = type("StubWithReason", (), {
            "_reason": str(e),
            "caption": caption_image_stub
        })()

    return _QWEN_SINGLETON

