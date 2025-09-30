"""
Purpose:
- Pydantic models for search in/out so the API is self-documenting and stable.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional

class SearchQuery(BaseModel):
    query: str = Field(..., min_length=2, description="User's search text")
    max_results: int = Field(5, ge=1, le=25, description="Max results to return")
    # Optional direct URLs to fetch (must be whitelisted) in addition to query-based results
    urls: Optional[List[HttpUrl]] = None

class SearchResult(BaseModel):
    title: str
    url: HttpUrl
    snippet: str = ""
    score: float = 0.0

class SearchResponse(BaseModel):
    ok: bool = True
    query: str
    results: List[SearchResult] = []
    allowed_domains: List[str] = []
    notes: Optional[str] = None

