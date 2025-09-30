"""
Purpose:
- Apply SearchLVLM-style filters (room/material/component/tool) to a mixed list of results.
- Simple, explainable matching: case-insensitive keyword sets with synonyms.
- AND semantics across provided filters: if multiple filters are given, all must match.

What it scans in each result:
- "image" items: join their snippets list into one text blob
- "text" items: the "snippet" field
- "web" items: title + snippet
"""

from __future__ import annotations
from typing import List, Dict, Any
import re

def _compile_terms(syn_map: dict[str, list[str]], target_key: str | None) -> list[re.Pattern]:
    """
    Given a synonyms map and a requested key (e.g., "kitchen"), return compiled regex patterns.
    If key not present or empty, return [] (no constraint).

    Implementation detail:
    - We escape each synonym with re.escape()
    - Then we replace literal escaped spaces (`\ `) with a character class `[\s-]`
      so "breaker panel" matches "breaker-panel" or "breaker panel".
    - We build the final pattern with word boundaries.
    """
    if not target_key:
        return []

    key = target_key.strip().lower()
    variants = syn_map.get(key, [key])

    patterns: list[re.Pattern] = []
    for v in variants:
        esc = re.escape(v)
        esc = esc.replace(r"\ ", r"[\s-]")  # allow space or hyphen between words
        pat = rf"\b{esc}\b"
        patterns.append(re.compile(pat, re.IGNORECASE))
    return patterns


def _text_of(item: dict) -> str:
    t = []
    if item.get("type") == "image":
        for s in item.get("snippets", []):
            t.append(str(s))
    elif item.get("type") == "text":
        t.append(str(item.get("snippet","")))
    elif item.get("type") == "web":
        t.append(str(item.get("title","")))
        t.append(str(item.get("snippet","")))
    return " ".join(t)

def _match_any(text: str, patterns: list[re.Pattern]) -> bool:
    if not patterns:  # no constraint
        return True
    return any(p.search(text) for p in patterns)

def apply_filters(
    items: List[Dict[str, Any]],
    *,
    room: str | None,
    material: str | None,
    component: str | None,
    tool: str | None,
    cfg: Dict[str, dict[str, list[str]]],
) -> Dict[str, Any]:
    """
    Return filtered list + notes.
    cfg keys expected: room, material, component, tool -> synonyms mapping dicts
    """
    # Compile constraint patterns (empty = no constraint)
    pats_room = _compile_terms(cfg["room"], room)
    pats_mat = _compile_terms(cfg["material"], material)
    pats_comp = _compile_terms(cfg["component"], component)
    pats_tool = _compile_terms(cfg["tool"], tool)

    kept, dropped = [], 0
    for it in items:
        text = _text_of(it)
        if not (_match_any(text, pats_room) and
                _match_any(text, pats_mat) and
                _match_any(text, pats_comp) and
                _match_any(text, pats_tool)):
            dropped += 1
            continue
        kept.append(it)

    notes = {
        "kept": len(kept),
        "dropped": dropped,
        "criteria": {
            "room": room,
            "material": material,
            "component": component,
            "tool": tool,
        }
    }
    return {"items": kept, "notes": notes}

