"""
Purpose:
- Load and normalize a whitelist of domains and repos from config + a file.
- Provide helpers to check if a URL is allowed.
"""

from __future__ import annotations
from typing import Tuple, List
from urllib.parse import urlparse
from pathlib import Path

def _read_lines(path: Path) -> List[str]:
    try:
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [ln.strip() for ln in text.splitlines() if ln.strip()]
    except Exception:
        return []

def _extract_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        # strip leading "www."
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""

def load_whitelists(file_path: Path, defaults_domains: List[str], defaults_repos: List[str]) -> Tuple[List[str], List[str]]:
    """
    Load whitelist from a text file (one URL per line), merge with defaults.
    Returns (domains, repos). De-duplicates and normalizes domains.
    """
    lines = _read_lines(file_path)
    file_domains: List[str] = []
    file_repos: List[str] = []

    for ln in lines:
        # quick heuristic: if contains 'github.com' or ends with .git, treat as repo
        if "github.com" in ln or ln.endswith(".git"):
            file_repos.append(ln)
        else:
            dom = _extract_domain(ln)
            if dom:
                file_domains.append(dom)

    # Build final unique lists
    domains = list(dict.fromkeys(
        [d.lower() for d in defaults_domains] + [d.lower() for d in file_domains]
    ))
    repos = list(dict.fromkeys(defaults_repos + file_repos))
    return domains, repos

def is_url_allowed(url: str, allowed_domains: List[str]) -> bool:
    """Return True if URL's domain (sans www.) is in allowed_domains."""
    dom = _extract_domain(url)
    return dom in set([d.lower() for d in allowed_domains])

