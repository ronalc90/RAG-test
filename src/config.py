# src/config.py
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# Carga .env desde la raíz del proyecto si existe
# (api.py ya usa BASE_DIR/.env, pero este config también sirve si se importa por separado)
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH if ENV_PATH.exists() else None)

def _getenv(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name)
    if val is None or (isinstance(val, str) and not val.strip()):
        return default
    return val.strip()

def _getbool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1","true","yes","y","on"}

def _getint(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip())
    except Exception:
        return default

# === Backends de BD / LLM ===
DB_BACKEND        = (_getenv("DB_BACKEND", "sqlite") or "sqlite").lower()  # sqlite | iris | postgres
OPENAI_API_KEY    = _getenv("OPENAI_API_KEY", "")
MISTRAL_API_KEY   = _getenv("MISTRAL_API_KEY", "")

# === SQLite (ruta por defecto en /data) ===
DATA_DIR          = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SQLITE_DB_PATH    = str(DATA_DIR / "app.sqlite3")   # Útil si algún módulo necesita ruta explícita

# === IRIS ===
IRIS_DSN          = _getenv("IRIS_DSN", "")         # p.ej. "localhost:1972/USER"
IRIS_USER         = _getenv("IRIS_USER", "")
IRIS_PASSWORD     = _getenv("IRIS_PASSWORD", "")

# === Postgres (si lo habilitas) ===
PG_HOST           = _getenv("PG_HOST", "localhost")
PG_PORT           = _getint("PG_PORT", 5432)
PG_DB             = _getenv("PG_DB", "appdb")
PG_USER           = _getenv("PG_USER", "postgres")
PG_PASSWORD       = _getenv("PG_PASSWORD", "")

# === Otros toggles útiles (opcional) ===
EMBEDDINGS_DIM    = _getint("EMBEDDINGS_DIM", 384)  # si tus embeddings requieren dim fija
DEBUG             = _getbool("DEBUG", False)
