#!/usr/bin/env python3
"""
Script para cargar contratos SECOP II en la base de datos RAG.
Genera códigos únicos, extrae texto para indexar, y crea embeddings.
"""
import json
from typing import List, Dict, Any
from src.db_sqlite import (
    init_db,
    insert_contrato,
    insert_contrato_embeddings,
    count_contratos,
    get_contrato_by_codigo
)
from src.secop_api import buscar_contratos
from src.embeddings import embed_texts
from src.chunking import chunk_text


def cargar_contratos_desde_api(
    entidad: str = None,
    objeto: str = None,
    limite: int = 1000
) -> int:
    """
    Carga contratos desde la API de SECOP II y los almacena en la BD.

    Returns:
        Número de contratos cargados
    """
    print(f"Buscando contratos (limite={limite})...")
    contratos = buscar_contratos(
        entidad=entidad,
        objeto_contratar=objeto,
        limite=limite
    )

    if not contratos:
        print("No se encontraron contratos")
        return 0

    print(f"Encontrados {len(contratos)} contratos. Procesando...")

    cargados = 0
    for i, contrato in enumerate(contratos, 1):
        try:
            codigo = insert_contrato(contrato, i)
            cargados += 1
            if i % 100 == 0:
                print(f"  Procesados {i}/{len(contratos)} contratos...")
        except Exception as e:
            print(f"  Error en contrato {i}: {e}")

    print(f"Cargados {cargados} contratos exitosamente")
    return cargados


def cargar_contratos_desde_json(filepath: str) -> int:
    """
    Carga contratos desde un archivo JSON local.

    Args:
        filepath: Ruta al archivo JSON con lista de contratos

    Returns:
        Número de contratos cargados
    """
    print(f"Leyendo {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        contratos = json.load(f)

    if isinstance(contratos, dict):
        # Si es un solo contrato, convertir a lista
        contratos = [contratos]

    print(f"Encontrados {len(contratos)} registros. Procesando...")

    cargados = 0
    for i, contrato in enumerate(contratos, 1):
        try:
            codigo = insert_contrato(contrato, i)
            cargados += 1
            if i % 100 == 0:
                print(f"  Procesados {i}/{len(contratos)} contratos...")
        except Exception as e:
            print(f"  Error en contrato {i}: {e}")

    print(f"Cargados {cargados} contratos exitosamente")
    return cargados


def generar_embeddings_contratos(batch_size: int = 50) -> int:
    """
    Genera embeddings para todos los contratos en la BD.

    Returns:
        Total de embeddings generados
    """
    from src.db_sqlite import _conn

    print("Obteniendo contratos sin embeddings...")
    with _conn() as con:
        cur = con.cursor()
        rows = cur.execute("""
            SELECT c.codigo_unico, c.texto_indexar
            FROM contratos c
            LEFT JOIN contrato_embeddings e ON c.codigo_unico = e.codigo_unico
            WHERE e.codigo_unico IS NULL
        """).fetchall()

    if not rows:
        print("Todos los contratos ya tienen embeddings")
        return 0

    print(f"Generando embeddings para {len(rows)} contratos...")
    total_embs = 0

    for i, row in enumerate(rows, 1):
        codigo = row["codigo_unico"]
        texto = row["texto_indexar"]

        if not texto.strip():
            continue

        try:
            # Dividir en chunks si es necesario
            chunks = chunk_text(texto, max_chars=500, overlap=50)
            if not chunks:
                chunks = [texto]

            # Generar embeddings
            embs = embed_texts(chunks)

            # Guardar
            n = insert_contrato_embeddings(codigo, chunks, embs)
            total_embs += n

            if i % 50 == 0:
                print(f"  Procesados {i}/{len(rows)} contratos ({total_embs} embeddings)")

        except Exception as e:
            print(f"  Error en {codigo}: {e}")

    print(f"Generados {total_embs} embeddings en total")
    return total_embs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cargar contratos SECOP II")
    parser.add_argument("--json", help="Cargar desde archivo JSON")
    parser.add_argument("--api", action="store_true", help="Cargar desde API SECOP")
    parser.add_argument("--entidad", help="Filtrar por entidad (solo con --api)")
    parser.add_argument("--objeto", help="Filtrar por objeto (solo con --api)")
    parser.add_argument("--limite", type=int, default=1000, help="Limite de registros")
    parser.add_argument("--embeddings", action="store_true", help="Generar embeddings")
    parser.add_argument("--stats", action="store_true", help="Mostrar estadisticas")

    args = parser.parse_args()

    # Inicializar BD
    init_db()

    if args.stats:
        total = count_contratos()
        print(f"Total contratos en BD: {total}")

    elif args.json:
        cargar_contratos_desde_json(args.json)
        if args.embeddings:
            generar_embeddings_contratos()

    elif args.api:
        cargar_contratos_desde_api(
            entidad=args.entidad,
            objeto=args.objeto,
            limite=args.limite
        )
        if args.embeddings:
            generar_embeddings_contratos()

    elif args.embeddings:
        generar_embeddings_contratos()

    else:
        parser.print_help()
