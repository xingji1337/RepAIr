"""
Purpose:
- /caption now uses local Qwen if available; falls back to stub automatically.
"""

from fastapi import APIRouter, UploadFile, File
from PIL import Image
from io import BytesIO
from ..vlm.qwen_captioner import get_captioner

router = APIRouter(prefix="/api/v1/vlm", tags=["vlm"])

@router.post("/caption")
async def caption(image: UploadFile = File(...)):
    try:
        raw = await image.read()
        img = Image.open(BytesIO(raw)).convert("RGB")
        cap = get_captioner().caption(img)
        return {"ok": True, "caption": cap, "filename": image.filename}
    except Exception as e:
        return {"ok": False, "error": f"caption-failed: {e!r}", "filename": image.filename}

