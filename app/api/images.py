"""
Purpose:
- Expose endpoints to render PDFs to page images and list what's available.
"""

from fastapi import APIRouter, Query
from pathlib import Path
from ..core.settings import settings
from ..rag.images import render_pdf_pages, list_page_images

router = APIRouter(prefix="/api/v1/rag/pages", tags=["pages"])

@router.post("/build")
def build_pages(dpi: int = 150, limit: int | None = Query(default=None, ge=1, description="Limit # of PDFs")):
    """
    Render PDFs under settings.pdf_dir to JPEGs under settings.page_images_dir.
    """
    return render_pdf_pages(
        pdf_dir=Path(settings.pdf_dir),
        out_dir=Path(settings.page_images_dir),
        dpi=dpi,
        limit=limit
    )

@router.get("/list")
def list_pages():
    """
    List discovered page images (capped for payload safety).
    """
    pages = list_page_images(Path(settings.page_images_dir))
    return {"ok": True, "count": len(pages), "pages": pages}

