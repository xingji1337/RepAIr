"""
Purpose:
- Expose /api/v1/search/* endpoints backing the whitelist + service orchestrator.
"""

from fastapi import APIRouter
from ..core.settings import settings
from ..search.schema import SearchQuery, SearchResponse
from ..search.whitelist import load_whitelists
from ..search.service import search_service

router = APIRouter(prefix="/api/v1/search", tags=["search"])

@router.get("/status", response_model=SearchResponse)
def search_status():
    domains, repos = load_whitelists(settings.search_whitelist_file, settings.allowed_domains, settings.allowed_repos)
    # Return as a "status-like" SearchResponse for consistency (no results)
    return SearchResponse(
        ok=True,
        query="",
        results=[],
        allowed_domains=domains,
        notes=f"{len(domains)} domains, {len(repos)} repos whitelisted."
    )

@router.post("/query", response_model=SearchResponse)
def search_query(payload: SearchQuery):
    return search_service(payload)

