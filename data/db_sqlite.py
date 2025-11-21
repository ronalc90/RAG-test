# src/db_sqlite.py
import os
import json
import sqlite3
from pathlib import Path

# Ruta de la BD: ./data/secop.db
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "secop.db"

# Conexión global (thread-safe básico para desarrollo)
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.row_factory = sqlite3.Row

def init_db():
    cur = _conn.cursor()
    # Tabla de documentos (proyectos SECOP)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS secop_documents (
            doc_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            titulo      TEXT NOT NULL,
            entidad     TEXT,
            source_path TEXT,
            metadata    TEXT
        )
    """)
    _conn.commit()

def insert_document(titulo: str, entidad: str = None, source_path: str = None, metadata: dict = None) -> int:
    cur = _conn.cursor()
    meta = json.dumps(metadata or {}, ensure_ascii=False)
    cur.execute("""
        INSERT INTO secop_documents (titulo, entidad, source_path, metadata)
        VALUES (?, ?, ?, ?)
    """, (titulo, entidad, source_path, meta))
    _conn.commit()
    return cur.lastrowid

def list_documents(limit: int = 50):
    cur = _conn.cursor()
    cur.execute("SELECT doc_id, titulo, entidad, source_path, metadata FROM secop_documents ORDER BY doc_id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    res = []
    for r in rows:
        item = dict(r)
        # metadata de JSON -> dict
        try:
            item["metadata"] = json.loads(item.get("metadata") or "{}")
        except Exception:
            pass
        res.append(item)
    return res
