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

        # Nueva tabla para contratos SECOP con estructura RAG
        cur.execute("""
        CREATE TABLE IF NOT EXISTS contratos (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            codigo_unico    TEXT UNIQUE NOT NULL,
            texto_total     TEXT NOT NULL,
            texto_indexar   TEXT NOT NULL,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_contratos_codigo ON contratos(codigo_unico);")

        # Tabla para embeddings de contratos
        cur.execute("""
        CREATE TABLE IF NOT EXISTS contrato_embeddings (
            emb_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            codigo_unico    TEXT NOT NULL,
            chunk_ord       INTEGER NOT NULL,
            chunk_text      TEXT NOT NULL,
            emb_json        TEXT NOT NULL,
            FOREIGN KEY (codigo_unico) REFERENCES contratos(codigo_unico) ON DELETE CASCADE
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_emb_codigo ON contrato_embeddings(codigo_unico);")
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


# ============== FUNCIONES PARA CONTRATOS SECOP ==============

def generar_codigo_unico(registro: Dict[str, Any], indice: int) -> str:
    """
    Genera un código único para un registro.
    Usa el código del proceso si existe, sino genera uno con prefijo SEC-{indice}.
    """
    # Intentar usar campos existentes como identificador
    for campo in ["codigo_de_secop", "numero_del_proceso", "referencia_del_contrato", "id_contrato"]:
        if campo in registro and registro[campo]:
            return str(registro[campo]).strip()
    # Si no hay código, generar uno
    return f"SEC-{indice:06d}"


def extraer_texto_indexar(registro: Dict[str, Any]) -> str:
    """
    Extrae los campos específicos para indexar/embeddings:
    - Departamento
    - Descripción del proceso
    - Objeto del contrato
    - Nombre de la entidad
    """
    campos = []

    # Departamento
    for key in ["departamento", "departamento_entidad", "departamento_ejecucion"]:
        if key in registro and registro[key]:
            campos.append(f"Departamento: {registro[key]}")
            break

    # Descripción del proceso
    if registro.get("descripcion_del_proceso"):
        campos.append(f"Descripción: {registro['descripcion_del_proceso']}")

    # Objeto del contrato
    for key in ["objeto_del_contrato", "objeto_a_contratar", "detalle_del_objeto_a_contratar"]:
        if key in registro and registro[key]:
            campos.append(f"Objeto: {registro[key]}")
            break

    # Nombre de la entidad
    if registro.get("nombre_entidad"):
        campos.append(f"Entidad: {registro['nombre_entidad']}")

    return "\n".join(campos)


def insert_contrato(registro: Dict[str, Any], indice: int) -> str:
    """
    Inserta un contrato en la base de datos.

    Args:
        registro: Diccionario JSON del contrato
        indice: Índice para generar código único si es necesario

    Returns:
        codigo_unico del contrato insertado
    """
    codigo_unico = generar_codigo_unico(registro, indice)
    texto_total = json.dumps(registro, ensure_ascii=False, indent=2)
    texto_indexar = extraer_texto_indexar(registro)

    with _conn() as con:
        cur = con.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO contratos (codigo_unico, texto_total, texto_indexar)
            VALUES (?, ?, ?)
        """, (codigo_unico, texto_total, texto_indexar))
        con.commit()

    return codigo_unico


def insert_contrato_embeddings(codigo_unico: str, chunks: List[str], embeddings) -> int:
    """
    Inserta embeddings de un contrato.

    Args:
        codigo_unico: Identificador del contrato
        chunks: Lista de textos (fragmentos del texto_indexar)
        embeddings: Lista de vectores de embeddings

    Returns:
        Número de embeddings insertados
    """
    try:
        embs_list = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
    except Exception:
        embs_list = embeddings

    rows = []
    for i, (chunk, emb) in enumerate(zip(chunks, embs_list)):
        rows.append((codigo_unico, i, chunk, json.dumps(emb, separators=(",", ":"), ensure_ascii=False)))

    with _conn() as con:
        cur = con.cursor()
        # Eliminar embeddings anteriores del mismo contrato
        cur.execute("DELETE FROM contrato_embeddings WHERE codigo_unico = ?", (codigo_unico,))
        cur.executemany(
            "INSERT INTO contrato_embeddings (codigo_unico, chunk_ord, chunk_text, emb_json) VALUES (?, ?, ?, ?)",
            rows
        )
        con.commit()
    return len(rows)


def get_contrato_by_codigo(codigo_unico: str) -> Optional[Dict[str, Any]]:
    """Obtiene un contrato por su código único."""
    with _conn() as con:
        cur = con.cursor()
        r = cur.execute("""
            SELECT id, codigo_unico, texto_total, texto_indexar, created_at
            FROM contratos WHERE codigo_unico = ?
        """, (codigo_unico,)).fetchone()
        if not r:
            return None
        return {
            "id": r["id"],
            "codigo_unico": r["codigo_unico"],
            "texto_total": json.loads(r["texto_total"]),
            "texto_indexar": r["texto_indexar"],
            "created_at": r["created_at"]
        }


def fetch_all_contrato_embeddings() -> List[Tuple[str, int, str, List[float]]]:
    """
    Obtiene todos los embeddings de contratos para búsqueda vectorial.

    Returns:
        Lista de (codigo_unico, chunk_ord, chunk_text, embedding)
    """
    with _conn() as con:
        cur = con.cursor()
        rows = cur.execute("""
            SELECT codigo_unico, chunk_ord, chunk_text, emb_json
            FROM contrato_embeddings
            ORDER BY codigo_unico, chunk_ord
        """).fetchall()

    out = []
    for r in rows:
        try:
            emb = json.loads(r["emb_json"]) if r["emb_json"] else []
        except Exception:
            emb = []
        out.append((r["codigo_unico"], r["chunk_ord"], r["chunk_text"], emb))
    return out


def count_contratos() -> int:
    """Cuenta el número de contratos en la base de datos."""
    with _conn() as con:
        cur = con.cursor()
        return cur.execute("SELECT COUNT(*) FROM contratos").fetchone()[0]


def list_contratos(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """Lista contratos con paginación."""
    with _conn() as con:
        cur = con.cursor()
        rows = cur.execute("""
            SELECT id, codigo_unico, texto_indexar, created_at
            FROM contratos
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """, (limit, offset)).fetchall()
        return [dict(r) for r in rows]
