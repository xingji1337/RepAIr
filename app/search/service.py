"""
Purpose:
- The "service" orchestrates query -> (future) fetch -> parse -> rank -> response.
- For now it's a stub that validates whitelist and echoes intent.
"""

from __future__ import annotations
from typing import List
from .schema import SearchQuery, SearchResponse, SearchResult
from .whitelist import load_whitelists, is_url_allowed
from ..core.settings import settings
from .fetcher import fetch_many
from .google_cse import google_search_text

def search_service(payload: SearchQuery) -> SearchResponse:
    # Load whitelist (merges file + defaults)
    allowed_domains, _repos = load_whitelists(
        settings.search_whitelist_file, settings.allowed_domains, settings.allowed_repos
    )

    urls = [str(u) for u in (payload.urls or [])]
    bad_urls = [u for u in urls if not is_url_allowed(u, allowed_domains)]
    if bad_urls:
        return SearchResponse(
            ok=False,
            query=payload.query,
            results=[],
            allowed_domains=allowed_domains,
            notes=f"Blocked URLs outside whitelist: {', '.join(bad_urls)}",
        )

    results: List[SearchResult] = []
    notes = []

    if urls:
        fetched = fetch_many(urls, max_results=payload.max_results)
        results.extend(fetched)
        notes.append("Fetched direct URLs.")
    else:
        # Whitelist-first Google search: one request per allowed domain
        g = google_search_text(payload.query, allowed_domains, max_results=payload.max_results)
        results.extend(g)
        if g:
            notes.append("Google CSE (domain-loop) + whitelist.")
        else:
            notes.append("No whitelisted hits from Google CSE.")

    return SearchResponse(
        ok=True,
        query=payload.query,
        results=results,
        allowed_domains=allowed_domains,
        notes=" ".join(notes),
    )




