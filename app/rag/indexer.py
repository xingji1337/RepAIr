"""
Purpose:
- Build and query a lightweight semantic index over local PDFs (text-first RAG).
- Hardened to avoid import-time failures: heavy deps are imported inside functions.
- Produces artifacts in data_dir:
    rag.index.faiss       - FAISS index
    rag.index_meta.json   - metadata (model, dim, chunk provenance, built_at)
    rag.index_norm.npy    - bool flag for L2-normalization

Notes:
- We do NOT import faiss / sentence_transformers / pypdf at module import time.
  That prevents reload/import failures that hide function names (your error).
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
import json
import numpy as np

# ---------------------------
# Data structures
# ---------------------------

@dataclass
class RagPaths:
    base_dir: Path
    index_path: Path
    meta_path: Path
    norm_path: Path

    @classmethod
    def from_data_dir(cls, data_dir: Path) -> "RagPaths":
        data_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            base_dir=data_dir,
            index_path=data_dir / "rag.index.faiss",
            meta_path=data_dir / "rag.index_meta.json",
            norm_path=data_dir / "rag.index_norm.npy",
        )

@dataclass
class ChunkRecord:
    doc_id: int
    file_path: str
    page_from: int
    page_to: int
    text: str

@dataclass
class RagMeta:
    embed_model: str
    dim: int
    chunks: List[ChunkRecord]
    built_at: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "embed_model": self.embed_model,
            "dim": self.dim,
            "built_at": self.built_at,
            "chunks": [asdict(c) for c in self.chunks],
        }

# ---------------------------
# Small helpers (no heavy deps here)
# ---------------------------

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize for cosine via inner product."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

# ---------------------------
# Import-on-demand utilities
# ---------------------------

def _import_pypdf():
    try:
        from pypdf import PdfReader  # type: ignore
        return PdfReader
    except Exception as e:
        raise ImportError(f"pypdf import failed: {e!r}") from e

def _import_st():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer
    except Exception as e:
        raise ImportError(f"sentence-transformers import failed: {e!r}") from e

def _import_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except Exception as e:
        raise ImportError(f"faiss import failed: {e!r}") from e

# ---------------------------
# Text extraction & chunking
# ---------------------------

def extract_pdf_texts(pdf_path: Path) -> List[Tuple[int, str]]:
    """Return list of (page_index, page_text)."""
    PdfReader = _import_pypdf()
    reader = PdfReader(str(pdf_path))
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append((i, txt))
    return pages

def chunk_page_text(text: str, max_chars: int = 1000, overlap: int = 150) -> List[str]:
    """Greedy char-based chunking with overlap."""
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

# ---------------------------
# Embedding & FAISS (lazy imports)
# ---------------------------

def get_embedder(model_name: str):
    SentenceTransformer = _import_st()
    return SentenceTransformer(model_name)

def build_faiss_index(embeddings: np.ndarray, use_ip: bool = True):
    faiss = _import_faiss()
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) if use_ip else faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    return index

# ---------------------------
# Build pipeline
# ---------------------------

def build_index(pdf_dir: Path, data_dir: Path, embed_model_name: str, max_pdfs: int | None = None) -> Dict[str, Any]:
    """
    Walk pdf_dir, extract+chunk text, embed, build FAISS, persist artifacts to data_dir.
    Robustness features:
      - per-PDF try/except (skip bad files, record errors)
      - optional max_pdfs limit for debugging
      - batch embedding to avoid memory spikes
    Returns a structured summary including skipped files and errors.
    """
    paths = RagPaths.from_data_dir(data_dir)
    embedder = get_embedder(embed_model_name)
    dim = embedder.get_sentence_embedding_dimension()

    pdfs = sorted(pdf_dir.glob("**/*.pdf"))
    if max_pdfs is not None:
        pdfs = pdfs[:max_pdfs]

    chunk_records: List[ChunkRecord] = []
    skipped: List[str] = []
    errors: List[str] = []

    for doc_id, pdf in enumerate(pdfs):
        try:
            page_texts = extract_pdf_texts(pdf)  # may raise for malformed PDFs
            for (page_i, page_text) in page_texts:
                for piece in chunk_page_text(page_text):
                    if piece.strip():
                        chunk_records.append(
                            ChunkRecord(
                                doc_id=doc_id,
                                file_path=str(pdf.resolve()),
                                page_from=page_i,
                                page_to=page_i,
                                text=piece,
                            )
                        )
        except Exception as e:
            skipped.append(str(pdf.resolve()))
            errors.append(f"{pdf.name}: {e!r}")
            continue

    # Handle empty corpus gracefully
    if not chunk_records:
        meta = RagMeta(
            embed_model=embed_model_name,
            dim=dim,
            chunks=[],
            built_at=datetime.utcnow().isoformat(timespec="seconds"),
        )
        paths.meta_path.write_text(json.dumps(meta.to_json(), ensure_ascii=False, indent=2))
        # remove any stale index
        if paths.index_path.exists():
            paths.index_path.unlink()
        np.save(paths.norm_path, np.array([True], dtype=bool))
        return {
            "ok": True,
            "pdfs_scanned": len(pdfs),
            "chunks": 0,
            "skipped": skipped,
            "errors": errors,
            "note": "No text extracted.",
            "built_at": meta.built_at,
        }

    # Embed in batches to avoid large memory spikes
    texts = [c.text for c in chunk_records]
    embs_all: List[np.ndarray] = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            embs = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        except Exception as e:
            # if an embedding batch fails, record and skip that batch
            errors.append(f"embed-batch {i}-{i+len(batch)}: {e!r}")
            continue
        embs_all.append(embs)

    if not embs_all:
        # nothing embedded successfully
        meta = RagMeta(
            embed_model=embed_model_name,
            dim=dim,
            chunks=[],
            built_at=datetime.utcnow().isoformat(timespec="seconds"),
        )
        paths.meta_path.write_text(json.dumps(meta.to_json(), ensure_ascii=False, indent=2))
        if paths.index_path.exists():
            paths.index_path.unlink()
        np.save(paths.norm_path, np.array([True], dtype=bool))
        return {
            "ok": False,
            "reason": "all-embeddings-failed",
            "pdfs_scanned": len(pdfs),
            "skipped": skipped,
            "errors": errors,
            "built_at": meta.built_at,
        }

    embs = np.vstack(embs_all)
    embs = l2_normalize(embs)  # cosine via IP

    # Build FAISS and persist
    index = build_faiss_index(embs, use_ip=True)
    _import_faiss().write_index(index, str(paths.index_path))

    meta = RagMeta(
        embed_model=embed_model_name,
        dim=int(embs.shape[1]),
        chunks=chunk_records,
        built_at=datetime.utcnow().isoformat(timespec="seconds"),
    )
    paths.meta_path.write_text(json.dumps(meta.to_json(), ensure_ascii=False, indent=2))
    np.save(paths.norm_path, np.array([True], dtype=bool))

    return {
        "ok": True,
        "pdfs_scanned": len(pdfs),
        "chunks": len(chunk_records),
        "dim": meta.dim,
        "skipped": skipped,
        "errors": errors,
        "built_at": meta.built_at,
    }


# ---------------------------
# Load & search
# ---------------------------

def load_index_and_meta(data_dir: Path):
    """Return (faiss_index|None, RagMeta|None, norm_flag: bool)."""
    paths = RagPaths.from_data_dir(data_dir)
    if not (paths.index_path.exists() and paths.meta_path.exists()):
        return None, None, False

    faiss = _import_faiss()
    index = faiss.read_index(str(paths.index_path))

    meta_d = json.loads(paths.meta_path.read_text())
    chunks = [ChunkRecord(**c) for c in meta_d.get("chunks", [])]
    meta = RagMeta(
        embed_model=meta_d["embed_model"],
        dim=meta_d["dim"],
        chunks=chunks,
        built_at=meta_d.get("built_at", ""),
    )
    use_norm = False
    if paths.norm_path.exists():
        arr = np.load(paths.norm_path)
        use_norm = bool(arr[0])
    return index, meta, use_norm

def search(query: str, data_dir: Path, top_k: int = 5) -> Dict[str, Any]:
    """Embed query and return top_k matches with provenance and snippets."""
    index, meta, norm = load_index_and_meta(data_dir)
    if index is None or meta is None or not meta.chunks:
        return {"ok": False, "reason": "Index not built."}

    embedder = get_embedder(meta.embed_model)
    q_vec = embedder.encode([query], convert_to_numpy=True)
    if norm:
        q_vec = l2_normalize(q_vec)

    D, I = index.search(q_vec.astype(np.float32), top_k)
    scores = D[0].tolist()
    idxs = I[0].tolist()

    results = []
    for score, idx in zip(scores, idxs):
        if idx < 0 or idx >= len(meta.chunks):
            continue
        c = meta.chunks[idx]
        snippet = c.text[:400].replace("\n", " ").strip()
        results.append({
            "score": float(score),
            "file_path": c.file_path,
            "page_from": c.page_from,
            "page_to": c.page_to,
            "snippet": snippet,
        })

    return {"ok": True, "query": query, "k": top_k, "results": results}

# ---------------------------
# Status helper
# ---------------------------

def get_index_status(data_dir: Path) -> Dict[str, Any]:
    """Return small status dict about the current index."""
    paths = RagPaths.from_data_dir(data_dir)
    exists = paths.index_path.exists() and paths.meta_path.exists()
    status = {
        "exists": exists,
        "chunks": 0,
        "built_at": None,
        "index_path": str(paths.index_path),
        "meta_path": str(paths.meta_path),
    }
    if not exists:
        return status
    try:
        meta_d = json.loads(paths.meta_path.read_text())
        status["chunks"] = len(meta_d.get("chunks", []))
        status["built_at"] = meta_d.get("built_at")
        if not status["built_at"]:
            ts = max(paths.index_path.stat().st_mtime, paths.meta_path.stat().st_mtime)
            status["built_at"] = datetime.fromtimestamp(ts).isoformat(timespec="seconds")
    except Exception as e:
        status["error"] = f"status-read-failed: {e!r}"
    return status

__all__ = [
    "build_index",
    "search",
    "get_index_status",
]

