"""
Purpose:
- Build and query a FAISS index over rendered page images.
- Uses a CLIP image encoder via sentence-transformers to get embeddings.
- Maps neighbors back to PDF page image + source PDF path.

Artifacts under settings.image_index_dir:
  - img.index.faiss       : FAISS index
  - img.index_meta.json   : metadata with mapping entries
  - img.index_norm.npy    : bool flag indicating L2-normalization
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import numpy as np
from PIL import Image

from sentence_transformers import SentenceTransformer
import faiss  # type: ignore

@dataclass
class ImgEntry:
    doc_hash: str
    page_index: int          # 0-based
    pdf_path: str            # absolute path to the PDF
    image_path: str          # absolute path to the page JPEG

@dataclass
class ImgMeta:
    embed_model: str
    dim: int
    entries: List[ImgEntry]

def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms
    
def _load_registry(page_root: Path) -> Dict[str, str]:
    """
    Read data/page_images/registry.jsonl -> {doc_hash: pdf_path}
    """
    reg = {}
    rp = page_root / "registry.jsonl"
    if not rp.exists():
        return reg
    for ln in rp.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
            reg[str(obj.get("doc_hash",""))] = str(obj.get("pdf_path",""))
        except Exception:
            continue
    return reg

def _find_page_images(page_root: Path) -> List[Path]:
    """
    Scan the page images directory: data/page_images/<doc_hash>/page_XXXX.jpg
    """
    if not page_root.exists():
        return []
    out: List[Path] = []
    for docdir in sorted(page_root.glob("*")):
        if not docdir.is_dir():
            continue
        for jpg in sorted(docdir.glob("page_*.jpg")):
            out.append(jpg)
    return out

def build_image_index(page_root: Path, index_dir: Path, model_name: str, batch_size: int = 32) -> Dict[str, Any]:
    """
    Build FAISS index over all page JPGs under page_root.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "img.index.faiss"
    meta_path = index_dir / "img.index_meta.json"
    norm_path = index_dir / "img.index_norm.npy"

    # Discover images
    imgs = _find_page_images(page_root)
    if not imgs:
        # Clean up stale index if any
        if index_path.exists(): index_path.unlink()
        meta_path.write_text(json.dumps({"entries": [], "embed_model": model_name, "dim": 0}, indent=2))
        np.save(norm_path, np.array([True], dtype=bool))
        return {"ok": True, "images": 0, "note": "No page images found."}

    registry = _load_registry(page_root)

    # Prepare entries map and PIL images
    entries: List[ImgEntry] = []
    for p in imgs:
        doc_hash = p.parent.name
        try:
            page_idx = int(p.stem.split("_")[1]) - 1
        except Exception:
            page_idx = -1
        entries.append(ImgEntry(
            doc_hash=doc_hash,
            page_index=page_idx,
            pdf_path=registry.get(doc_hash, ""),  
            image_path=str(p.resolve())
        ))

    # Load encoder
    encoder = SentenceTransformer(model_name)
    dim = encoder.get_sentence_embedding_dimension()

    # Encode in batches to avoid RAM spikes
    embs_all: List[np.ndarray] = []
    for i in range(0, len(entries), batch_size):
        batch_entries = entries[i:i+batch_size]
        pil_batch = []
        for e in batch_entries:
            try:
                pil_batch.append(Image.open(e.image_path).convert("RGB"))
            except Exception:
                pil_batch.append(Image.new("RGB", (224,224), color=(200,200,200)))
        # sentence-transformers v3 supports image encoding via encode() with a list of PIL Images
        vecs = encoder.encode(pil_batch, batch_size=len(pil_batch), convert_to_numpy=True, show_progress_bar=False)
        embs_all.append(vecs)

    embs = np.vstack(embs_all).astype(np.float32)
    embs = _l2_normalize(embs)  # cosine via inner product

    # FAISS index
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, str(index_path))

    meta = ImgMeta(embed_model=model_name, dim=int(embs.shape[1]), entries=entries)
    meta_path.write_text(json.dumps({
        "embed_model": meta.embed_model,
        "dim": meta.dim,
        "entries": [asdict(e) for e in meta.entries],
    }, ensure_ascii=False, indent=2))
    np.save(norm_path, np.array([True], dtype=bool))

    return {"ok": True, "images": len(entries), "dim": meta.dim}

def _load_index(index_dir: Path) -> Tuple[faiss.Index | None, Dict[str, Any] | None, bool]:
    index_path = index_dir / "img.index.faiss"
    meta_path = index_dir / "img.index_meta.json"
    norm_path = index_dir / "img.index_norm.npy"
    if not (index_path.exists() and meta_path.exists()):
        return None, None, False
    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text())
    norm = bool(np.load(norm_path)[0]) if norm_path.exists() else False
    return index, meta, norm

def search_by_image(img: Image.Image, index_dir: Path, model_name: str, k: int = 5) -> Dict[str, Any]:
    index, meta, norm = _load_index(index_dir)
    if index is None or not meta:
        return {"ok": False, "reason": "Image index not built."}

    encoder = SentenceTransformer(model_name)
    q = encoder.encode([img.convert("RGB")], convert_to_numpy=True)
    if norm:
        q = _l2_normalize(q.astype(np.float32))

    D, I = index.search(q.astype(np.float32), k)
    scores = D[0].tolist()
    idxs = I[0].tolist()

    entries = meta.get("entries", [])
    out = []
    for s, i in zip(scores, idxs):
        if i < 0 or i >= len(entries):
            continue
        e = entries[i]
        out.append({
            "score": float(s),
            "doc_hash": e["doc_hash"],
            "page_index": e["page_index"],
            "image_path": e["image_path"],
            "pdf_path": e.get("pdf_path", ""),
        })
    return {"ok": True, "results": out}

