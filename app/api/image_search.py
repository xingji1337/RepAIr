"""
Purpose:
- Build CLIP image index over page images, and search by uploaded image.
- Adds safety flags based on generated caption text (simple pass-through).
"""

from fastapi import APIRouter, UploadFile, File, Query
from pathlib import Path
from io import BytesIO
from PIL import Image

from ..core.settings import settings
from ..rag.image_indexer import build_image_index, search_by_image
from ..vlm.captioner import caption_image_stub
from ..services.safety import analyze_snippets

router = APIRouter(prefix="/api/v1/rag/image", tags=["image-rag"])

@router.post("/index/build")
def build_image_faiss():
    return build_image_index(
        page_root=Path(settings.page_images_dir),
        index_dir=Path(settings.image_index_dir),
        model_name=settings.image_embed_model_name,
        batch_size=32,
    )

@router.post("/search")
async def image_search(k: int = Query(default=5, ge=1, le=20),
                       image: UploadFile = File(...)):
    # Decode image
    raw = await image.read()
    img = Image.open(BytesIO(raw)).convert("RGB")

    # Optional: get a stub caption for safety labels (later swap with real VLM)
    cap = caption_image_stub(img)
    safety = analyze_snippets([cap])

    # Search neighbors
    res = search_by_image(
        img=img,
        index_dir=Path(settings.image_index_dir),
        model_name=settings.image_embed_model_name,
        k=k,
    )
    if not res.get("ok"):
        return res

    # Attach caption + flags to each result (helps the app render)
    return {
        **res,
        "query_caption": cap,
        "warnings": safety["warnings"],
    }

