# src/db_postgres.py
from __future__ import annotations
import os, json
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import psycopg
from pgvector.psycopg import register_vector  # pip install pgvector

# Nota: Usa una sola variable de entorno para la dimensión
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/secop")

def _conn():
    # Registramos pgvector en cada conexión
    con = psycopg.connect(POSTGRES_DSN, autocommit=True)
    register_vector(con)
    return con

SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS projects (
  project_id  SERIAL PRIMARY KEY,
  name        TEXT NOT NULL,
  description TEXT,
  created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
  doc_id      SERIAL PRIMARY KEY,
  project_id  INT REFERENCES projects(project_id),
  titulo      TEXT NOT NULL,
  entidad     TEXT,
  source_path TEXT,
  metadata    JSONB DEFAULT '{{}}'
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id    SERIAL PRIMARY KEY,
  doc_id      INT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  ord         INT NOT NULL,
  text        TEXT NOT NULL,
  embedding   vector({EMBEDDING_DIM}) NOT NULL
);

DO $$
BEGIN
  IF to_regclass('public.chunks_embedding_hnsw') IS NULL THEN
    BEGIN
      CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_l2_ops);
    EXCEPTION WHEN undefined_object THEN
      BEGIN
        CREATE INDEX chunks_embedding_ivf ON chunks USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
      EXCEPTION WHEN OTHERS THEN
        NULL;
      END;
    END;
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_chunks_docid ON chunks(doc_id);
"""

# =========================
# Requeridas por api.py
# =========================
def init_db():
    with _conn() as con, con.cursor() as cur:
        cur.execute(SCHEMA_SQL)

def insert_document(titulo: str, entidad: str | None, source_path: str | None, metadata: Dict[str,Any] | None = None) -> int:
    with _conn() as con, con.cursor() as cur:
        cur.execute("""
          INSERT INTO documents(titulo, entidad, source_path, metadata)
          VALUES (%s, %s, %s, %s)
          RETURNING doc_id
        """, (titulo, entidad, source_path, json.dumps(metadata or {})))
        return int(cur.fetchone()[0])

def list_documents(limit: int = 50) -> List[Dict[str, Any]]:
    with _conn() as con, con.cursor() as cur:
        cur.execute("""
          SELECT doc_id, titulo, entidad, source_path, metadata
          FROM documents
          ORDER BY doc_id DESC
          LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({
            "doc_id": int(r[0]),
            "titulo": r[1],
            "entidad": r[2],
            "archivo": r[3],              # mapeo: source_path -> "archivo" para compatibilidad con api.py
            "metadata": r[4] or {}
        })
    return out

def insert_chunks(doc_id: int, chunks: List[str], embeddings) -> int:
    """
    Inserta N chunks. `embeddings` puede ser np.ndarray shape (N, D) o lista de listas.
    pgvector acepta listas de floats directamente con el adaptador registrado.
    """
    try:
        embs = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
    except Exception:
        embs = embeddings

    rows = [(doc_id, i, chunks[i], embs[i]) for i in range(len(chunks))]
    with _conn() as con, con.cursor() as cur:
        cur.executemany("INSERT INTO chunks(doc_id, ord, text, embedding) VALUES (%s, %s, %s, %s)", rows)
    return len(rows)

def fetch_all_vectors() -> List[Tuple[int, int, int, str, List[float], str]]:
    """
    Devuelve tuplas: (chunk_id, doc_id, ord, text, emb, titulo)
    `emb` como lista[float] (api.py lo convierte a np.asarray(...))
    """
    with _conn() as con, con.cursor() as cur:
        cur.execute("""
          SELECT c.chunk_id, c.doc_id, c.ord, c.text, c.embedding, d.titulo
          FROM chunks c
          JOIN documents d ON d.doc_id = c.doc_id
          ORDER BY c.doc_id, c.ord
        """)
        rows = cur.fetchall()

    out: List[Tuple[int, int, int, str, List[float], str]] = []
    for r in rows:
        # pgvector -> python list (gracias a register_vector)
        emb_obj = r[4]
        if isinstance(emb_obj, (list, tuple, np.ndarray)):
            emb_list = [float(x) for x in emb_obj]
        else:
            # Fallback defensivo: intentar parsear si viniera como string "[0.1, 0.2, ...]"
            s = str(emb_obj).strip().strip("[]")
            emb_list = [float(x) for x in s.split(",") if x.strip()] if s else []
        out.append((int(r[0]), int(r[1]), int(r[2]), r[3], emb_list, r[5]))
    return out

def get_document(doc_id: int) -> Optional[Dict[str, Any]]:
    with _conn() as con, con.cursor() as cur:
        cur.execute("SELECT doc_id, titulo, entidad, source_path, metadata FROM documents WHERE doc_id=%s", (doc_id,))
        r = cur.fetchone()
    if not r:
        return None
    return {
        "doc_id": int(r[0]),
        "titulo": r[1],
        "entidad": r[2],
        "archivo": r[3],                 # mapeo a "archivo"
        "metadata": r[4] or {}
    }

def fetch_doc_text(doc_id: int) -> str:
    with _conn() as con, con.cursor() as cur:
        cur.execute("SELECT text FROM chunks WHERE doc_id=%s ORDER BY ord ASC", (doc_id,))
        parts = [row[0] for row in cur.fetchall()]
    return "\n".join(parts)
# =========================

# (Opcionales que ya tenías)
def insert_project(name: str, description: str | None = None) -> int:
    with _conn() as con, con.cursor() as cur:
        cur.execute("INSERT INTO projects(name, description) VALUES (%s, %s) RETURNING project_id", (name, description))
        return int(cur.fetchone()[0])

def update_source_path(doc_id: int, source_path: str):
    with _conn() as con, con.cursor() as cur:
        cur.execute("UPDATE documents SET source_path=%s WHERE doc_id=%s", (source_path, doc_id))

def similarity_search(q_emb: np.ndarray, top_k: int = 5) -> List[Tuple[float, int, int, str, str]]:
    """
    Búsqueda vectorial nativa (no usada por api.py, pero útil si quieres acelerar /ask aquí).
    Devuelve (score, doc_id, ord, text, titulo) con score ~ cosino si ajustas operador.
    """
    q = q_emb.tolist()
    with _conn() as con, con.cursor() as cur:
        cur.execute("""
          SELECT
            1 - (c.embedding <=> %s::vector) AS score,   -- 1 - L2 distancia normalizada
            c.doc_id, c.ord, c.text, d.titulo
          FROM chunks c
          JOIN documents d ON d.doc_id = c.doc_id
          ORDER BY c.embedding <=> %s::vector
          LIMIT %s
        """, (q, q, top_k))
        rows = cur.fetchall()
    return [(float(r[0]), int(r[1]), int(r[2]), r[3], r[4]) for r in rows]
