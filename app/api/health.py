# Common language: Environment/ops probe that surfaces version pins, paths, and index status.
# Use this before/after upgrades to confirm no silent drift.

from fastapi import APIRouter
from ..core.settings import settings
from ..rag.indexer import get_index_status
from pydantic import BaseModel, Field
from typing import List
from ..services.safety import analyze_snippets
from ..vlm.qwen_captioner import get_captioner
from pathlib import Path
import sys, importlib

router = APIRouter(tags=["health"])

class SafetyTestIn(BaseModel):
    texts: List[str] = Field(default_factory=list, description="Snippets to scan for hazards")

@router.post("/safety/test")
def safety_test(payload: SafetyTestIn):
    """
    Quick diagnostic endpoint to see how the safety analyzer flags text.
    """
    return analyze_snippets(payload.texts)

def _ver(modname: str) -> str:
    try:
        m = importlib.import_module(modname)
        return getattr(m, "__version__", "unknown")
    except Exception:
        return "not-installed"

def _file_info(p: Path):
    try:
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        return {"path": str(p), "exists": exists, "size": size}
    except Exception:
        return {"path": str(p), "exists": False, "size": 0}

@router.get("/healthz")
def healthz():
    # Load whitelist domains count without reading the whole file here; the search layer merges them at call-time.
    whitelist_file = settings.search_whitelist_file
    qwen = get_captioner()
    qcfg = getattr(qwen, "cfg", None)
    qwen_status = {
        "using_qwen": (getattr(qwen, "__class__", None).__name__ not in ("StubWrapper", "StubWithReason")),
        "model_id": getattr(qcfg, "model_id", settings.qwen_model_id),
        "device": getattr(qcfg, "device", settings.qwen_device),
        "dtype": getattr(qcfg, "dtype", settings.qwen_dtype),
        "load_warning": getattr(qwen, "_reason", None),
    }
    return {
        "status": "ok",
        "python": sys.version.split()[0],
        "versions": {
            "numpy": _ver("numpy"),
            "pypdf": _ver("pypdf"),
            "faiss": _ver("faiss"),
            "fastapi": _ver("fastapi"),
            "uvicorn": _ver("uvicorn"),
            "pydantic_settings": _ver("pydantic_settings"),
            "sentence_transformers": _ver("sentence_transformers"),
            "httpx": _ver("httpx"),
            "selectolax": _ver("selectolax"),
        },
        "config": {
            "pdf_dir": str(settings.pdf_dir),
            "data_dir": str(settings.data_dir),
            "vlm_model": settings.vlm_model_name,
            "embed_model": settings.embed_model_name,
        },
        "search_config": {
            "whitelist_file": _file_info(whitelist_file),
            "env_keys_present": {
                "GOOGLE_API_KEY": bool(settings.google_api_key),
                "GOOGLE_TEXT_CSE_ID": bool(settings.google_text_cse_id),
                "GOOGLE_IMAGE_CSE_ID": bool(settings.google_image_cse_id),
            },
        },
        "rag_index": get_index_status(settings.data_dir),
        "qwen": qwen_status,
    }

