"""
Purpose:
- Given image neighbors (doc_hash, page_index), fetch nearby text chunks from RAG meta
  based on the matching PDF path and page window, then fuse into a single result row.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
from .indexer import load_index_and_meta  # reuse to read rag.index_meta.json
from ..services.safety import analyze_snippets
from ..core.settings import settings
from pathlib import Path
import hashlib

def _build_page_map(meta_d: Dict[str, Any]) -> Dict[str, List[Tuple[int, Dict[str, Any]]]]:
    out: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for c in meta_d.get("chunks", []):
        pdf = c.get("file_path","")
        page = int(c.get("page_from", 0))
        out.setdefault(pdf, []).append((page, c))
    for pdf, lst in out.items():
        lst.sort(key=lambda x: x[0])
    return out

def _nearby_chunks(page_map: Dict[str, List[Tuple[int, Dict[str, Any]]]],
                   pdf_path: str, page_index: int, window: int = 1, max_return: int = 3) -> List[str]:
    lst = page_map.get(pdf_path, [])
    if not lst:
        return []
    targets = set(range(max(0, page_index - window), page_index + window + 1))
    hits: List[str] = []
    for p, c in lst:
        if p in targets:
            snip = (c.get("text","") or "").replace("\n"," ").strip()
            if snip:
                hits.append(snip[:400])
            if len(hits) >= max_return:
                break
    return hits

def _load_registry(page_root: Path) -> dict[str, str]:
    reg = {}
    rp = page_root / "registry.jsonl"
    if rp.exists():
        for ln in rp.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not ln.strip():
                continue
            try:
                obj = json.loads(ln)
                reg[str(obj.get("doc_hash",""))] = str(obj.get("pdf_path",""))
            except Exception:
                continue
    return reg

def _hash_path(p: Path) -> str:
    return hashlib.sha1(str(p.resolve()).encode("utf-8")).hexdigest()[:16]

def _build_registry_from_pdfs(pdf_dir: Path) -> dict[str, str]:
    reg = {}
    for pdf in sorted(Path(pdf_dir).glob("**/*.pdf")):
        reg[_hash_path(pdf)] = str(pdf.resolve())
    return reg

def fuse_image_hits_with_text(image_hits: List[Dict[str, Any]], data_dir: Path) -> Dict[str, Any]:
    """
    For each image neighbor -> ensure pdf_path is filled, attach nearby text snippet(s), and safety advisories.
    Never throws; always returns a dict with ok:True/False.
    """
    try:
        # Load text meta
        _index, meta, _norm = load_index_and_meta(data_dir)
        if not meta:
            return {"ok": False, "reason": "RAG text index not built."}

        meta_d = {"chunks": [vars(c) for c in meta.chunks]}
        page_map = _build_page_map(meta_d)

        # Registry (doc_hash -> pdf_path)
        page_root = Path(settings.page_images_dir)
        pdf_dir = Path(settings.pdf_dir)
        reg = _load_registry(page_root) or _build_registry_from_pdfs(pdf_dir)

        fused = []
        all_snips: List[str] = []

        for e in image_hits:
            pdf = e.get("pdf_path") or reg.get(str(e.get("doc_hash","")), "")
            page = int(e.get("page_index", 0))
            snips = _nearby_chunks(page_map, pdf, page, window=1, max_return=2) if pdf else []
            all_snips.extend(snips)
            fused.append({
                **e,
                "pdf_path": pdf,
                "snippets": snips,
            })

        safety = analyze_snippets(all_snips)
        return {"ok": True, "results": fused, "warnings": safety["warnings"]}

    except Exception as e:
        return {"ok": False, "reason": f"fusion-failed: {e!r}"}


