"""
Adds:
- prompt->query generation
- text RAG + whitelist Google CSE
- simple rank fusion + safety tagging is already handled in RAG + fusion; we combine lists here.
"""

from fastapi import APIRouter, UploadFile, File, Form
from pathlib import Path
from io import BytesIO
from PIL import Image

from ..core.settings import settings
from ..vlm.qwen_captioner import get_captioner
from ..rag.image_indexer import search_by_image
from ..rag.fusion import fuse_image_hits_with_text
from ..rag.indexer import search as rag_search
from ..search.schema import SearchQuery
from ..search.service import search_service
from ..services.query_gen import synthesize_query, QueryInputs
from ..services.filters import apply_filters

router = APIRouter(prefix="/api/v1/rag/fusion", tags=["fusion"])

def _rank_key(r: dict) -> float:
    # Visual score present? use it; else 0.7 for text/web with normalized pseudo-score
    return float(r.get("score", 0.7))

def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return (s[: n - 1] + "…") if len(s) > n else s

def _coerce_results_for_merge(image_res: dict, rag_res: dict, web_res: dict) -> list[dict]:
    out: list[dict] = []

    # image (already fused with local snippets)
    img_cap = settings.fusion_per_source_cap
    for it in (image_res.get("results", [])[:img_cap]):
        out.append({
            "type": "image",
            "score": float(it.get("score", 0.0)),
            "image_path": it.get("image_path"),
            "pdf_path": it.get("pdf_path"),
            "page_index": it.get("page_index"),
            "snippets": [
                _truncate(sn, settings.snippet_char_limit)
                for sn in it.get("snippets", [])
            ],
        })
    # web
    if web_res.get("ok"):
        for hit in web_res.get("results", []):
            out.append({
                "type": "web",
                "score": float(hit.get("score", 0.5)),
                "title": hit.get("title",""),
                "url": str(hit.get("url","")),
                "snippet": hit.get("snippet",""),
            })
    return out

def _dedup_keep_best(items: list[dict]) -> list[dict]:
    """
    Deduplicate by (pdf_path,page_index) for local docs and by url for web.
    Keep the highest score item.
    """
    seen = {}
    for x in items:
        if x["type"] == "web":
            key = ("web", x.get("url",""))
        elif x["type"] in ("image","text"):
            key = ("pdf", x.get("pdf_path",""), x.get("page_index"))
        else:
            key = ("other", repr(x))
        if key not in seen or _rank_key(x) > _rank_key(seen[key]):
            seen[key] = x
    return sorted(seen.values(), key=_rank_key, reverse=True)

@router.post("/search")
async def fusion_search(
    image: UploadFile = File(...),
    text: str | None = Form(default=None),
    k: int = 5, 
    # --- SearchLVLM-style filters (all optional; AND semantics if provided) ---
    room: str | None = Form(default=None),
    material: str | None = Form(default=None),
    component: str | None = Form(default=None),
    tool: str | None = Form(default=None),
):
    try:
        # 1) Decode + caption
        raw = await image.read()
        img = Image.open(BytesIO(raw)).convert("RGB")
        caption = get_captioner().caption(img)

        # 2) Generate a succinct query
        query = synthesize_query(QueryInputs(caption=caption, user_text=text))

        # 3) Visual neighbors -> fused with local text snippets
        vis = search_by_image(
            img=img,
            index_dir=Path(settings.image_index_dir),
            model_name=settings.image_embed_model_name,
            k=k
        )
        if not vis.get("ok"):
            return vis
        fused_local = fuse_image_hits_with_text(vis["results"], data_dir=Path(settings.data_dir))
        if not fused_local.get("ok"):
            fused_local = {"ok": True, "results": [], "warnings": []}

        # 4) Text RAG over local PDFs (already returns a dict)
        rag = rag_search(query=query, data_dir=Path(settings.data_dir), top_k=k)

        # 5) Whitelist web search via Google CSE (returns Pydantic model)
        web_resp = search_service(SearchQuery(query=query, max_results=k))
        
        # Normalize to dict so downstream code can .get(...)
        if hasattr(web_resp, "model_dump"):
            web = web_resp.model_dump()
        elif isinstance(web_resp, dict):
            web = web_resp
        else:
            web = {"ok": False, "results": [], "notes": "unexpected web response type"}

        # 6) Merge + dedupe + rank
        merged = _coerce_results_for_merge(fused_local, rag, web)
        merged = _dedup_keep_best(merged)

        # ---- APPLY FILTERS (if any provided) --------------------------------
        # Build cfg from settings’ synonym maps
        cfg = {
            "room": settings.filter_synonyms_room,
            "material": settings.filter_synonyms_material,
            "component": settings.filter_synonyms_component,
            "tool": settings.filter_synonyms_tool,
        }
        filtered = apply_filters(
            merged,
            room=room, material=material, component=component, tool=tool,
            cfg=cfg
        )
        final_items = filtered["items"][: settings.fusion_max_total]
        filter_notes = filtered["notes"]

        return {
            "ok": True,
            "query_caption": caption,
            "query_text": query,
            "results": final_items,
            "warnings": fused_local.get("warnings", []),
            "notes": {
                "image_hits": len(fused_local.get("results", [])),
                "rag_hits": len(rag.get("results", [])) if rag.get("ok") else 0,
                "web_hits": len(web.get("results", [])) if web.get("ok") else 0,
                "filters": filter_notes,
            }
        }
    except Exception as e:
        return {"ok": False, "error": f"fusion-search-failed: {e!r}"}


