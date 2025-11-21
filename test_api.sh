#!/bin/bash
# Script de prueba de todos los endpoints

echo "=== TEST 1: Health Check ==="
curl -s http://127.0.0.1:8001/ping | python3 -m json.tool

echo -e "\n\n=== TEST 2: Consulta de Contratos SECOP II ==="
curl -s "http://127.0.0.1:8001/secop/contratos?entidad=SENA&limite=3" | python3 -m json.tool | head -50

echo -e "\n\n=== TEST 3: Proveedores de Software ==="
curl -s "http://127.0.0.1:8001/secop/proveedores?sector=software" | python3 -m json.tool | head -30

echo -e "\n\n=== TEST 4: Consulta RAG Tradicional ==="
curl -s -X POST http://127.0.0.1:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"¿Cuáles son los requisitos habilitantes?","top_k":1}' | python3 -m json.tool

echo -e "\n\n=== TEST 5: Estado de la Base de Datos ==="
curl -s http://127.0.0.1:8001/database | python3 -m json.tool

echo -e "\n\n✅ Tests completados"
