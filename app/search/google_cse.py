"""
Purpose:
- Query Google Custom Search JSON API for web results.
- STRICT whitelist: query each allowed domain with siteSearch=domain.
- Merge, deduplicate, and cap results.

Notes:
- Requires: settings.google_api_key, settings.google_text_cse_id (from .env or env)
- We keep requests small (num<=10) and short timeouts to be polite.
"""

from __future__ import annotations
import os
from typing import List, Dict, Set
import httpx
from urllib.parse import urlparse
from .schema import SearchResult
from ..core.settings import settings

GOOGLE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"

def _domain_of(url: str) -> str:
    try:
        d = urlparse(url).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return ""

def _api_params(query: str, api_key: str, cx: str, domain: str, num: int) -> Dict[str, str]:
    # siteSearch targets a single domain; we call it once per allowed domain.
    return {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": str(min(num, 10)),
        "safe": "active",
        "siteSearch": domain,
        "siteSearchFilter": "i",  # include only this site
    }

def google_search_text(query: str, allowed_domains: List[str], max_results: int = 5) -> List[SearchResult]:
    api_key = settings.google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("google_api_key")
    cx = settings.google_text_cse_id or os.getenv("GOOGLE_TEXT_CSE_ID") or os.getenv("google_text_cse_id")
    if not api_key or not cx or not allowed_domains:
        return []

    results: List[SearchResult] = []
    seen_urls: Set[str] = set()

    # Try each allowed domain until we collect enough results
    for dom in allowed_domains:
        try:
            params = _api_params(query, api_key, cx, dom, num=max_results)
            r = httpx.get(GOOGLE_ENDPOINT, params=params, timeout=12.0)
            r.raise_for_status()
            data = r.json()
        except Exception:
            continue

        for it in (data.get("items") or []):
            link = it.get("link")
            if not link:
                continue
            # hard check in case CSE returns a subdomain off-target
            if _domain_of(link) not in {d.lower() for d in allowed_domains}:
                continue
            if link in seen_urls:
                continue
            seen_urls.add(link)

            title = it.get("title") or link
            snippet = (it.get("snippet") or "")[:300]
            results.append(SearchResult(title=title, url=link, snippet=snippet, score=1.0))

            if len(results) >= max_results:
                break

        if len(results) >= max_results:
            break

    return results[:max_results]

