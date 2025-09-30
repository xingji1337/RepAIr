"""
Purpose:
- Render each PDF into per-page JPEGs for later captioning and image-based search.
- Deterministic storage under settings.page_images_dir/<doc_hash>/page_XXXX.jpg
- Returns a registry (list of dicts) with file, page, and image path.

System requirements:
- poppler-utils installed (pdftoppm callable)
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from pathlib import Path
import hashlib
from pdf2image import convert_from_path  # uses poppler's pdftoppm under the hood

@dataclass
class PageImage:
    pdf_path: str
    doc_hash: str
    page_index: int     # 0-based
    image_path: str     # absolute path to the saved jpg

def _hash_path(p: Path) -> str:
    # Stable hash for directory naming
    return hashlib.sha1(str(p.resolve()).encode("utf-8")).hexdigest()[:16]

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def render_pdf_pages(pdf_dir: Path, out_dir: Path, dpi: int = 150, limit: int | None = None) -> Dict[str, Any]:
    """
    Walk pdf_dir, render pages to JPEGs into out_dir, and write a registry mapping:
      data/page_images/registry.jsonl with lines:
      {"doc_hash": "...", "pdf_path": "..."}
    """
    pdfs = sorted(pdf_dir.glob("**/*.pdf"))
    if limit is not None:
        pdfs = pdfs[:limit]

    _ensure_dir(out_dir)
    pages: List[PageImage] = []
    skipped: List[str] = []
    errors: List[str] = []

    registry_path = out_dir / "registry.jsonl"
    seen_hashes: set[str] = set()
    with registry_path.open("a", encoding="utf-8") as regf:
        for pdf in pdfs:
            try:
                doc_id = _hash_path(pdf)
                # write registry line once per doc_hash
                if doc_id not in seen_hashes:
                    regf.write(json.dumps({"doc_hash": doc_id, "pdf_path": str(pdf.resolve())}) + "\n")
                    seen_hashes.add(doc_id)

                target_dir = out_dir / doc_id
                _ensure_dir(target_dir)
                imgs = convert_from_path(str(pdf), dpi=dpi)
                for i, img in enumerate(imgs):
                    name = f"page_{i+1:04d}.jpg"
                    outp = target_dir / name
                    img.save(str(outp), format="JPEG", quality=90)
                    pages.append(PageImage(
                        pdf_path=str(pdf.resolve()),
                        doc_hash=doc_id,
                        page_index=i,
                        image_path=str(outp.resolve()),
                    ))
            except Exception as e:
                skipped.append(str(pdf.resolve()))
                errors.append(f"{pdf.name}: {e!r}")
                continue

    return {
        "ok": True,
        "count": len(pages),
        "pages": [asdict(p) for p in pages[:200]],
        "skipped": skipped,
        "errors": errors,
        "note": "Preview capped to 200 pages; full set is on disk.",
    }


def list_page_images(out_dir: Path) -> List[Dict[str, Any]]:
    """
    Return a lightweight list of discovered page images with doc_hash and page index.
    """
    if not out_dir.exists():
        return []
    out: List[Dict[str, Any]] = []
    for doc_dir in sorted(out_dir.glob("*")):
        if not doc_dir.is_dir():
            continue
        for jpg in sorted(doc_dir.glob("page_*.jpg")):
            try:
                page_idx = int(jpg.stem.split("_")[1]) - 1
            except Exception:
                page_idx = -1
            out.append({
                "doc_hash": doc_dir.name,
                "page_index": page_idx,
                "image_path": str(jpg.resolve()),
            })
    # Cap to keep API light; CLI or filesystem can inspect the rest
    return out[:1000]

