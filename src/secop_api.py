"""
Módulo para integración con la API de SECOP II
Documentación: https://www.colombiacompra.gov.co/secop/secop-ii
"""
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime

SECOP_API_BASE = "https://www.datos.gov.co/resource/jbjy-vk9h.json"

def buscar_contratos(
    entidad: Optional[str] = None,
    objeto_contratar: Optional[str] = None,
    fecha_desde: Optional[str] = None,
    fecha_hasta: Optional[str] = None,
    limite: int = 100
) -> List[Dict[str, Any]]:
    """
    Busca contratos en SECOP II usando la API de datos abiertos

    Args:
        entidad: Nombre de la entidad contratante
        objeto_contratar: Palabras clave del objeto a contratar
        fecha_desde: Fecha inicio (YYYY-MM-DD)
        fecha_hasta: Fecha fin (YYYY-MM-DD)
        limite: Número máximo de resultados

    Returns:
        Lista de contratos encontrados
    """
    params = {"$limit": limite}

    # Construir filtros
    where_clauses = []

    if entidad:
        where_clauses.append(f"nombre_entidad like '%{entidad}%'")

    if objeto_contratar:
        where_clauses.append(f"descripcion_del_proceso like '%{objeto_contratar}%'")

    if fecha_desde:
        where_clauses.append(f"fecha_de_firma >= '{fecha_desde}'")

    if fecha_hasta:
        where_clauses.append(f"fecha_de_firma <= '{fecha_hasta}'")

    if where_clauses:
        params["$where"] = " AND ".join(where_clauses)

    try:
        response = requests.get(SECOP_API_BASE, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error consultando SECOP II: {e}")
        return []


def obtener_estadisticas_entidad(entidad: str) -> Dict[str, Any]:
    """
    Obtiene estadísticas de contratación de una entidad
    """
    contratos = buscar_contratos(entidad=entidad, limite=1000)

    if not contratos:
        return {"error": "No se encontraron contratos"}

    total_contratos = len(contratos)

    # Calcular monto total (si está disponible)
    montos = []
    for c in contratos:
        try:
            monto = float(c.get("valor_del_contrato", 0))
            montos.append(monto)
        except:
            pass

    monto_total = sum(montos)
    monto_promedio = monto_total / len(montos) if montos else 0

    # Modalidades más usadas
    modalidades = {}
    for c in contratos:
        mod = c.get("modalidad_de_contratacion", "Desconocida")
        modalidades[mod] = modalidades.get(mod, 0) + 1

    return {
        "entidad": entidad,
        "total_contratos": total_contratos,
        "monto_total": monto_total,
        "monto_promedio": monto_promedio,
        "modalidades": modalidades,
        "contratos_muestra": contratos[:5]  # Primeros 5 contratos
    }


def buscar_proveedores_por_sector(sector: str, limite: int = 50) -> List[Dict[str, Any]]:
    """
    Busca proveedores que han trabajado en un sector específico
    """
    contratos = buscar_contratos(objeto_contratar=sector, limite=limite)

    proveedores = {}
    for c in contratos:
        proveedor = c.get("proveedor_adjudicado", "Desconocido")
        if proveedor not in proveedores:
            proveedores[proveedor] = {
                "nombre": proveedor,
                "num_contratos": 0,
                "contratos": []
            }
        proveedores[proveedor]["num_contratos"] += 1
        proveedores[proveedor]["contratos"].append({
            "entidad": c.get("nombre_entidad"),
            "objeto": c.get("descripcion_del_proceso", "")[:100],
            "valor": c.get("valor_del_contrato")
        })

    return sorted(proveedores.values(), key=lambda x: x["num_contratos"], reverse=True)


if __name__ == "__main__":
    # Ejemplo de uso
    print("🔍 Buscando contratos de tecnología...")
    contratos = buscar_contratos(objeto_contratar="tecnología", limite=5)

    for i, c in enumerate(contratos, 1):
        print(f"\n{i}. {c.get('descripcion_del_proceso', 'N/A')[:80]}")
        print(f"   Entidad: {c.get('nombre_entidad', 'N/A')}")
        print(f"   Valor: ${c.get('valor_del_contrato', 'N/A')}")
