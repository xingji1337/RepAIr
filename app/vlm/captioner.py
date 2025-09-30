"""
Purpose:
- Small interface for image captioning.
- Stub now; later swap in Qwen2.5-VL without changing the API surface.

Notes:
- We keep a simple function signature that takes a PIL image and returns a short caption.
"""

from __future__ import annotations
from typing import Optional
from PIL import Image

def caption_image_stub(img: Image.Image) -> str:
    """
    Very simple placeholder captioner. Replace with real VLM later.
    """
    w, h = img.size
    return f"Photo ({w}x{h}); captioning model not wired yet."

