# src/db_sqlite.py
from __future__ import annotations
import sqlite3, json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# Ruta a /data/app.sqlite3 (carpeta hermana de src/)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "app.sqlite3"

def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON;")
    return con

def init_db() -> None:
    """Crea tablas si no existen."""
    with _conn() as con:
        cur = con.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            titulo     TEXT NOT NULL,
            entidad    TEXT,
            archivo    TEXT,
            metadata   TEXT
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id     INTEGER NOT NULL,
            ord        INTEGER NOT NULL,
            text       TEXT NOT NULL,
            emb_json   TEXT NOT NULL,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_docid ON chunks(doc_id);")
        con.commit()

def insert_document(titulo: str, entidad: Optional[str], archivo: Optional[str], metadata: Optional[Dict[str, Any]]) -> int:
    meta = json.dumps(metadata or {}, ensure_ascii=False)
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            "INSERT INTO documents (titulo, entidad, archivo, metadata) VALUES (?, ?, ?, ?)",
            (titulo, entidad, archivo, meta)
        )
        con.commit()
        return int(cur.lastrowid)

def list_documents() -> List[Dict[str, Any]]:
    with _conn() as con:
        cur = con.cursor()
        rows = cur.execute("""
            SELECT doc_id, titulo, entidad, archivo, metadata
            FROM documents
            ORDER BY doc_id DESC
        """).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            meta_raw = r["metadata"]
            try:
                meta = json.loads(meta_raw or "{}")
            except Exception:
                meta = meta_raw
            out.append({
                "doc_id": int(r["doc_id"]),
                "titulo": r["titulo"],
                "entidad": r["entidad"],
                "archivo": r["archivo"],
                "metadata": meta
            })
        return out

def insert_chunks(doc_id: int, chunks: List[str], embs) -> int:
    try:
        embs_list = embs.tolist() if hasattr(embs, "tolist") else embs
    except Exception:
        embs_list = embs
    rows = []
    for i, (text, emb) in enumerate(zip(chunks, embs_list)):
        rows.append((doc_id, i, text, json.dumps(emb, separators=(",", ":"), ensure_ascii=False)))
    with _conn() as con:
        cur = con.cursor()
        cur.executemany(
            "INSERT INTO chunks (doc_id, ord, text, emb_json) VALUES (?, ?, ?, ?)",
            rows
        )
        con.commit()
        return len(rows)

def fetch_all_vectors() -> List[Tuple[int, int, int, str, List[float], str]]:
    with _conn() as con:
        cur = con.cursor()
        rows = cur.execute("""
            SELECT c.chunk_id, c.doc_id, c.ord, c.text, c.emb_json, d.titulo
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            ORDER BY c.doc_id, c.ord
        """).fetchall()
    out: List[Tuple[int, int, int, str, List[float], str]] = []
    for r in rows:
        try:
            emb = json.loads(r["emb_json"]) if r["emb_json"] else []
        except Exception:
            emb = []
        out.append((
            int(r["chunk_id"]),
            int(r["doc_id"]),
            int(r["ord"]),
            r["text"],
            emb,
            r["titulo"]
        ))
    return out

def get_document(doc_id: int) -> Optional[Dict[str, Any]]:
    with _conn() as con:
        cur = con.cursor()
        r = cur.execute("""
            SELECT doc_id, titulo, entidad, archivo, metadata
            FROM documents
            WHERE doc_id = ?
        """, (doc_id,)).fetchone()
        if not r:
            return None
        meta_raw = r["metadata"]
        try:
            meta = json.loads(meta_raw or "{}")
        except Exception:
            meta = meta_raw
        return {
            "doc_id": int(r["doc_id"]),
            "titulo": r["titulo"],
            "entidad": r["entidad"],
            "archivo": r["archivo"],
            "metadata": meta
        }

def fetch_doc_text(doc_id: int) -> str:
    with _conn() as con:
        cur = con.cursor()
        rows = cur.execute(
            "SELECT text FROM chunks WHERE doc_id = ? ORDER BY ord ASC",
            (doc_id,)
        ).fetchall()
        return "\n".join([r["text"] for r in rows])
