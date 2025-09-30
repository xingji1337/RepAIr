"""
Purpose:
- FastAPI application factory and router mounts.
- Adds CORS for local dev + future domain.
- Uvicorn will serve this on 0.0.0.0:8000 by default.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.settings import settings
from .api.health import router as health_router
from .api.rag import router as rag_router
from .api.vlm import router as vlm_router
from .api.search import router as search_router
from .api.images import router as pages_router
from .api.image_search import router as image_rag_router
from .api.fusion import router as fusion_router

def create_app() -> FastAPI:
    app = FastAPI(title="Vision-RAG API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Lightweight routes ready
    app.include_router(health_router)
    app.include_router(rag_router)
    app.include_router(vlm_router)
    app.include_router(pages_router)
    app.include_router(search_router)
    app.include_router(image_rag_router)
    app.include_router(fusion_router)
    return app

    

app = create_app()


