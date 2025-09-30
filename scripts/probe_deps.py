"""
Purpose:
- Sanity-check critical library versions after upgrades.
- Import the exact modules we use and print versions so we can spot drift immediately.
"""

import sys
import numpy
import pypdf
import faiss
import fastapi
import uvicorn
from pydantic_settings import BaseSettings
from sentence_transformers import SentenceTransformer

print("python", sys.version)
print("numpy", numpy.__version__)
print("pypdf", pypdf.__version__)
print("faiss", faiss.__version__)
print("fastapi", fastapi.__version__)
print("uvicorn", uvicorn.__version__)
print("pydantic-settings", BaseSettings.__module__.split(".")[0])  # presence check
# lightweight embedder load to ensure ST works post-upgrade
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("st_dim", m.get_sentence_embedding_dimension())
print("OK")

