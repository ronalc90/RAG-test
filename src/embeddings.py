# src/embeddings.py
import os
from pathlib import Path
from typing import List
import numpy as np
from dotenv import load_dotenv

# Cargar .env desde la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip().strip('"')
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def _cheap_embed(s: str, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(s)) % (2**32))
    v = rng.standard_normal(dim).astype(np.float32)
    n = np.linalg.norm(v) + 1e-10
    return (v / n).astype(np.float32)

def embed_texts(texts: List[str]) -> List[np.ndarray]:
    if not texts:
        return []
    if _client:
        resp = _client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [np.array(item.embedding, dtype=np.float32) for item in resp.data]
    return [_cheap_embed(t) for t in texts]

def embed_text(text: str) -> np.ndarray:
    if not text:
        return _cheap_embed("")
    if _client:
        resp = _client.embeddings.create(model=EMBED_MODEL, input=[text])
        return np.array(resp.data[0].embedding, dtype=np.float32)
    return _cheap_embed(text)

