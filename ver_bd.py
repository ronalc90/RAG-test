#!/usr/bin/env python3
"""
Script para visualizar el contenido de la base de datos
"""
import sqlite3
import json
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "app.sqlite3"

def mostrar_bd():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("=" * 80)
    print("📊 ESTADO DE LA BASE DE DATOS")
    print("=" * 80)
    print()

    # Mostrar documentos
    print("📄 DOCUMENTOS ALMACENADOS:")
    print("-" * 80)
    cursor.execute("SELECT doc_id, titulo, entidad, metadata FROM documents")
    docs = cursor.fetchall()

    for doc_id, titulo, entidad, metadata in docs:
        print(f"\n🔹 ID: {doc_id}")
        print(f"   Título: {titulo}")
        print(f"   Entidad: {entidad}")

        try:
            meta = json.loads(metadata) if metadata else {}
            if meta.get("url"):
                print(f"   URL: {meta['url'][:70]}...")
        except:
            pass

        # Contar chunks de este documento
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,))
        num_chunks = cursor.fetchone()[0]
        print(f"   📦 Fragmentos (chunks): {num_chunks}")

    print("\n" + "=" * 80)

    # Estadísticas generales
    cursor.execute("SELECT COUNT(*) FROM documents")
    total_docs = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM chunks")
    total_chunks = cursor.fetchone()[0]

    print("\n📈 ESTADÍSTICAS:")
    print("-" * 80)
    print(f"Total de documentos: {total_docs}")
    print(f"Total de fragmentos (chunks): {total_chunks}")
    print(f"Promedio de chunks por documento: {total_chunks / total_docs if total_docs > 0 else 0:.1f}")
    print()

    # Mostrar esquema de las tablas
    print("=" * 80)
    print("🗂️  ESTRUCTURA DE LA BASE DE DATOS:")
    print("-" * 80)

    print("\n📋 Tabla: documents")
    cursor.execute("PRAGMA table_info(documents)")
    for col in cursor.fetchall():
        print(f"   • {col[1]} ({col[2]})")

    print("\n📋 Tabla: chunks")
    cursor.execute("PRAGMA table_info(chunks)")
    for col in cursor.fetchall():
        col_name = col[1]
        col_type = col[2]
        if col_name == "embedding":
            print(f"   • {col_name} ({col_type}) - Vector de embeddings")
        else:
            print(f"   • {col_name} ({col_type})")

    print("\n" + "=" * 80)

    # Mostrar ejemplo de chunk
    print("\n📝 EJEMPLO DE FRAGMENTO (CHUNK):")
    print("-" * 80)
    cursor.execute("""
        SELECT c.chunk_id, c.doc_id, c.ord, c.text, d.titulo
        FROM chunks c
        JOIN documents d ON c.doc_id = d.doc_id
        LIMIT 1
    """)

    chunk = cursor.fetchone()
    if chunk:
        chunk_id, doc_id, order, text, doc_titulo = chunk
        print(f"Chunk ID: {chunk_id}")
        print(f"Documento: {doc_titulo}")
        print(f"Orden: {order}")
        print(f"Texto (primeros 200 caracteres):")
        print(f"   {text[:200]}...")

    print("\n" + "=" * 80)

    conn.close()

if __name__ == "__main__":
    mostrar_bd()
