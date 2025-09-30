"""
Purpose:
- Centralized configuration using pydantic-settings.
- Reads from environment variables and optional .env file.
- Keeps paths and host/port tunable without code changes.
"""

# --- Purpose: robust settings with env-file support and safe handling of extra keys.
from typing import List, Optional
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Pydantic v2 config (env file + ignore unexpected env vars)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",          # <-- prevents crashes if extra env vars exist
    )
    
    # API host/port
    host: str = Field(default="0.0.0.0", description="Bind address for FastAPI/Uvicorn")
    port: int = Field(default=8000, description="Port for FastAPI/Uvicorn")

    # CORS
    cors_allow_origins: list[str] = Field(
        default=["http://localhost:3000", "http://192.168.1.180", "https://manbpro.com", "https://www.manbpro.com"],
        description="Allowed origins for browser apps"
    )

    # Data paths (defaults to your current PDF location)
    pdf_dir: Path = Field(
        default=Path("/home/manny-buff/projects/capstone/hw-rag/data/"),
        description="Directory containing PDFs or media for RAG"
    )
    data_dir: Path = Field(
        default=Path("./data"),
        description="Local data directory for caches, indices, uploads"
    )
    
    # Where the image FAISS + meta live
    image_index_dir: Path = Field(default=Path("./data/image_index"))
    
    # where rendered page images are stored
    page_images_dir: Path = Field(default=Path("./data/page_images"))

    # Model knobs (placeholders; we’ll wire up later)
    vlm_model_name: str = Field(default="Qwen2.5-VL-3B-Instruct", description="Default VLM")
    embed_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="RAG embedding model")
    image_embed_model_name: str = Field(default="sentence-transformers/clip-ViT-B-32")
    
    # --- Search config ---
    # Path to a simple text file with one URL per line (repos or sites you trust).
    search_whitelist_file: Path = Field(
        default=Path("/home/manny-buff/projects/vision-rag-api/whitelist_urls.txt"),
        description="File with allowed base URLs/domains for search/fetch."
    )
    search_max_results: int = Field(default=5, description="Default cap on search results")

    # Keep defaults in code; they merge with file contents at runtime
    allowed_domains: List[str] = Field(
        default=[
            # sane seeds; real list comes from whitelist_urls.txt
            "familyhandyman.com",
            "thisoldhouse.com",
            "bobvila.com",
            "thespruce.com",
            "epa.gov",
            "nahb.org",
        ],
        description="Domain whitelist for outbound search/fetch."
    )
    # we’re not using repos now, but keep an empty list so callers don’t break
    allowed_repos: List[str] = Field(default=[], description="(Unused now) Repo allowlist.")

    # Google Custom Search (text)
    # These can come from env (.env or shell): GOOGLE_API_KEY, GOOGLE_TEXT_CSE_ID
    google_api_key: Optional[str] = None
    google_text_cse_id: Optional[str] = None
    google_image_cse_id: Optional[str] = None
    
    # ---- Qwen captioner config ----
    qwen_model_id: str = Field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    qwen_device: str = Field(default="auto")       # "auto" | "cuda" | "cpu"
    qwen_dtype: str = Field(default="auto")        # "auto" | "bfloat16" | "float16" | "float32"
    qwen_max_new_tokens: int = Field(default=64)
    qwen_temperature: float = Field(default=0.2)
    qwen_top_p: float = Field(default=0.9)
    
    # ---- Qwen memory/offload controls ----
    qwen_offload_folder: Path = Field(default=Path("./data/qwen_offload"))
    qwen_gpu_max_gb: float = Field(default=15.0)   # cap GPU usage; leave headroom
    qwen_cpu_max_gb: float = Field(default=80.0)   # how much RAM you're ok to use
    qwen_attn_impl: str = Field(default="sdpa")    # "sdpa" (mem-efficient) | "flash_attention_2" (if installed) | "eager"

    # Context window target (tokenizer side). Qwen 2.5 supports long contexts; we set tokenizer max len.
    qwen_context_tokens: int = Field(default=32768)
    
    # toggle: if false, we keep using the stub
    use_qwen_captioner: bool = Field(default=True)

    # ---- Response sizing knobs ----
    fusion_max_total: int = Field(default=10)         # total max items returned after dedupe
    fusion_per_source_cap: int = Field(default=5)     # cap per source (image/text/web) before merge
    snippet_char_limit: int = Field(default=280)      # truncate long snippets for UI
        
    # ---- SearchLVLM-style filter config ----
    # Each key has a list of synonyms/variants we’ll match case-insensitively.
    filter_synonyms_room: dict[str, list[str]] = {
        "kitchen": ["kitchen", "galley", "cooktop", "range", "sink (kitchen)"],
        "bathroom": ["bathroom", "bath", "toilet", "vanity", "shower", "tub"],
        "garage": ["garage"],
        "basement": ["basement", "crawlspace"],
        "living room": ["living room", "family room", "den", "lounge"],
        "bedroom": ["bedroom"],
        "laundry": ["laundry", "washer", "dryer"],
        "outdoor": ["outdoor", "exterior", "porch", "deck", "patio"],
    }

    filter_synonyms_material: dict[str, list[str]] = {
        "wood": ["wood", "oak", "pine", "maple", "plywood", "lumber"],
        "metal": ["metal", "steel", "aluminum", "copper", "iron", "brass"],
        "pvc": ["pvc", "cpvc", "plastic pipe"],
        "pex": ["pex"],
        "copper": ["copper pipe", "copper tubing", "copper"],
        "tile": ["tile", "ceramic", "porcelain"],
        "drywall": ["drywall", "sheetrock", "gypsum board"],
    }

    filter_synonyms_component: dict[str, list[str]] = {
        "breaker panel": ["breaker panel", "service panel", "breaker box"],
        "gfci": ["gfci", "gfi"],
        "outlet": ["outlet", "receptacle"],
        "faucet": ["faucet", "tap", "spout", "cartridge"],
        "thermocouple": ["thermocouple"],
        "water heater": ["water heater", "tank", "anode", "pilot", "igniter"],
        "hinge": ["hinge"],
        "door": ["door", "door jamb", "strike plate"],
    }

    filter_synonyms_tool: dict[str, list[str]] = {
        "multimeter": ["multimeter", "voltmeter", "continuity tester"],
        "stud finder": ["stud finder"],
        "pipe wrench": ["pipe wrench"],
        "screwdriver": ["screwdriver", "driver bit"],
        "pliers": ["pliers", "needle-nose", "channel lock", "tongue-and-groove"],
    }

settings = Settings()

