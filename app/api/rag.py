# Common language: Expose build/search/status endpoints for the PDF-backed RAG.
# - /build  : walk PDFs, chunk, embed, persist FAISS
# - /search : semantic nearest-neighbor over the built index
# - /status : quick view of index presence, chunk count, and timestamps

# Purpose: wrap build in try/except so the API always returns JSON.

from fastapi import APIRouter, Query
from pathlib import Path
from ..core.settings import settings
from ..rag.indexer import build_index, search as rag_search, get_index_status
from ..services.safety import analyze_snippets

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])

@router.post("/build")
def build(force: bool = False, max_pdfs: int | None = None):
    """
    Build (or rebuild) the FAISS index from PDFs at settings.pdf_dir.
    - force: reserved for future freshness checks
    - max_pdfs: optional limit to debug a smaller subset first
    """
    try:
        summary = build_index(
            pdf_dir=Path(settings.pdf_dir),
            data_dir=Path(settings.data_dir),
            embed_model_name=settings.embed_model_name,
            max_pdfs=max_pdfs,
        )
        return summary
    except Exception as e:
        return {"ok": False, "error": f"build-failed: {e!r}"}


@router.get("/search")
def search(q: str = Query(..., min_length=2, description="query text"), k: int = 5):
    """
    Semantic search over the built index plus safety advisories.
    Returns:
      - results: list with provenance and per-result 'flags' (if any)
      - warnings: unique advisories (GAS/APPLIANCE/ELECTRICAL)
    """
    base = rag_search(query=q, data_dir=Path(settings.data_dir), top_k=k)
    if not base.get("ok"):
        return base

    # Extract snippets, analyze safety, then attach flags/warnings
    snippets = [r.get("snippet", "") for r in base.get("results", [])]
    safety = analyze_snippets(snippets)
    flags_by_idx = safety["per_item_flags"]

    results_with_flags = []
    for i, r in enumerate(base["results"]):
        r2 = dict(r)
        r2["flags"] = flags_by_idx[i]
        results_with_flags.append(r2)

    return {
        **base,
        "results": results_with_flags,
        "warnings": safety["warnings"],
    }


@router.get("/status")
def status():
    """
    Lightweight index status probe for dashboards/ops.
    """
    return get_index_status(data_dir=Path(settings.data_dir))

