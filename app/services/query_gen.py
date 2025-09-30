"""
Purpose:
- Turn (image caption + optional user text) into a succinct, actionable "how do I ..." query.
- Uses local Qwen2.5-VL (language-only) via transformers text generation.
- Optional fallback: OpenAI API if OPENAI_API_KEY is set in env/.env.

Design:
- Short system prompt for safety + DIY scope.
- If user text is present, specialize the query with materials/type when possible.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import os

from ..core.settings import settings

# --- Optional OpenAI fallback (kept minimal; only if key is present) ---
def _openai_fallback(prompt: str, max_tokens: int = 48) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
    if not api_key:
        return None
    try:
        # lightweight runtime import to avoid hard dependency if unused
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        # Use the smallest capable instruct model you have access to; replace if needed
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"You generate one concise, practical home-repair search query."},
                {"role":"user","content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or None
    except Exception:
        return None

# --- Qwen (text-only) generator via transformers ---
def _qwen_text_only(prompt: str, max_new_tokens: int = 48, temperature: float = 0.2, top_p: float = 0.9) -> Optional[str]:
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_id = settings.qwen_model_id if "/" in settings.qwen_model_id else f"Qwen/{settings.qwen_model_id}"
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if settings.qwen_device == "auto" else None,
            trust_remote_code=True,
        ).eval()
        if settings.qwen_device in ("cpu","cuda"):
            mdl.to(settings.qwen_device)
        ids = tok.encode(prompt, return_tensors="pt")
        dev = next(mdl.parameters()).device
        ids = ids.to(dev)
        out = mdl.generate(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        # Decode only the newly generated continuation
        cont = out[:, ids.shape[1]:]
        txt = tok.decode(cont[0], skip_special_tokens=True).strip()
        return txt or None
    except Exception:
        return None

@dataclass
class QueryInputs:
    caption: str
    user_text: Optional[str] = None

SYSTEM_HINT = (
    "You generate ONE concise, specific DIY home-repair search query. "
    "Prefer material/species (oak vs pine), fixture/tool names, and symptoms. "
    "Avoid brand names, fluff, and pronouns. Return just the query."
)

def synthesize_query(inp: QueryInputs) -> str:
    """
    Compose a small prompt and try Qwen, then OpenAI fallback.
    """
    if inp.user_text and inp.user_text.strip():
        core = f'User asked: "{inp.user_text.strip()}". Image hint: "{inp.caption}".'
    else:
        core = f'No user text. From image hint: "{inp.caption}". Assume user intent: "how do I handle this situation?"'

    prompt = f"{SYSTEM_HINT}\n\n{core}\n\nReturn one query starting with a verb (e.g., 'refinish...', 'replace...', 'diagnose...')."
    # Try Qwen text-only first
    q = _qwen_text_only(prompt, max_new_tokens=48, temperature=0.2, top_p=0.9)
    if q:
        return q
    # Fallback to OpenAI if available
    q2 = _openai_fallback(prompt, max_tokens=48)
    if q2:
        return q2
    # Last resort: simple heuristic
    return f"how to handle: {inp.caption}"

