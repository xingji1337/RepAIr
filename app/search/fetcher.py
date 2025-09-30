"""
Purpose:
- Fetch and parse allowed web pages into structured results.
- Enforces whitelist from settings.
"""

from __future__ import annotations
import httpx
from selectolax.parser import HTMLParser
from typing import List, Optional
from ..core.settings import settings
from .whitelist import load_whitelists, is_url_allowed
from .schema import SearchResult

def fetch_url(url: str) -> Optional[SearchResult]:
    """
    Fetch a single URL (must be whitelisted) and parse basic info.
    Returns SearchResult or None if disallowed/failure.
    """
    # Load whitelist fresh (merges file + defaults)
    domains, _repos = load_whitelists(
        settings.search_whitelist_file, settings.allowed_domains, settings.allowed_repos
    )
    if not is_url_allowed(url, domains):
        return None

    try:
        resp = httpx.get(url, timeout=10.0, follow_redirects=True)
        resp.raise_for_status()
    except Exception:
        return None

    parser = HTMLParser(resp.text)

    title = (parser.css_first("title").text(strip=True) if parser.css_first("title") else url)
    snippet_node = parser.css_first("p") or parser.css_first("div")
    snippet = snippet_node.text(strip=True)[:300] if snippet_node else ""

    return SearchResult(title=title, url=url, snippet=snippet, score=1.0)

def fetch_many(urls: List[str], max_results: int = 5) -> List[SearchResult]:
    out: List[SearchResult] = []
    for u in urls[:max_results]:
        res = fetch_url(u)
        if res:
            out.append(res)
    return out

