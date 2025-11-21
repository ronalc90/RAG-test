#!/usr/bin/env python3
"""
Suite de Pruebas Sistemáticas - Sistema LLM SECOP II
Evalúa cumplimiento de requerimientos funcionales y no funcionales
"""
import sys
import time
import json
import requests
from datetime import datetime
from typing import Dict, List, Tuple

# Configuración
BASE_URL = "http://127.0.0.1:8001"
TIMEOUT = 30

# Colores para terminal
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Imprime encabezado de sección"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.END}\n")

def print_test(name: str, passed: bool, details: str = ""):
    """Imprime resultado de prueba"""
    status = f"{Colors.GREEN}✓ PASS{Colors.END}" if passed else f"{Colors.RED}✗ FAIL{Colors.END}"
    print(f"{status} {name}")
    if details:
        print(f"     {Colors.YELLOW}{details}{Colors.END}")

class TestResults:
    """Almacena resultados de pruebas"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.results = []
        self.start_time = time.time()

    def add(self, name: str, passed: bool, details: str = "", metric: float = None):
        self.total += 1
        if passed:
            self.passed += 1
        else:
            self.failed += 1

        self.results.append({
            "test": name,
            "passed": passed,
            "details": details,
            "metric": metric
        })

        print_test(name, passed, details)

    def summary(self):
        """Imprime resumen de resultados"""
        elapsed = time.time() - self.start_time

        print_header("RESUMEN DE PRUEBAS")
        print(f"Total de pruebas: {self.total}")
        print(f"{Colors.GREEN}Exitosas: {self.passed}{Colors.END}")
        print(f"{Colors.RED}Fallidas: {self.failed}{Colors.END}")
        print(f"Tasa de éxito: {(self.passed/self.total*100):.1f}%")
        print(f"Tiempo total: {elapsed:.2f}s\n")

        if self.failed > 0:
            print(f"{Colors.RED}⚠ Algunas pruebas fallaron{Colors.END}")
            return False
        else:
            print(f"{Colors.GREEN}✓ Todas las pruebas pasaron{Colors.END}")
            return True

# ============================================================================
# PRUEBAS FUNCIONALES
# ============================================================================

def test_rf1_consulta_lenguaje_natural(results: TestResults):
    """RF1: Sistema permite consultas en lenguaje natural"""
    print_header("RF1: Consulta en Lenguaje Natural")

    # Test 1: Pregunta simple
    try:
        response = requests.post(f"{BASE_URL}/ask", json={
            "query": "¿Cuáles son los requisitos habilitantes?",
            "top_k": 1
        }, timeout=TIMEOUT)

        data = response.json()
        passed = response.status_code == 200 and data.get("ok") and data.get("answer")
        details = f"Respuesta recibida: {len(data.get('answer', ''))} caracteres"
        results.add("RF1.1: Consulta simple", passed, details)
    except Exception as e:
        results.add("RF1.1: Consulta simple", False, f"Error: {str(e)}")

    # Test 2: Pregunta con datos SECOP II
    try:
        response = requests.post(f"{BASE_URL}/ask", json={
            "query": "¿Cuántos contratos de tecnología tiene el SENA?",
            "top_k": 1
        }, timeout=TIMEOUT)

        data = response.json()
        passed = response.status_code == 200 and data.get("ok")
        secop_included = data.get("secop_data_included", False)
        details = f"Datos SECOP II incluidos: {secop_included}"
        results.add("RF1.2: Consulta con datos históricos", passed, details)
    except Exception as e:
        results.add("RF1.2: Consulta con datos históricos", False, f"Error: {str(e)}")

def test_rf2_acceso_secop(results: TestResults):
    """RF2: Acceso a datos de SECOP II"""
    print_header("RF2: Acceso a Datos SECOP II")

    # Test 1: Consulta de contratos
    try:
        response = requests.get(f"{BASE_URL}/secop/contratos", params={
            "entidad": "SENA",
            "limite": 5
        }, timeout=TIMEOUT)

        data = response.json()
        passed = response.status_code == 200 and len(data.get("contratos", [])) > 0
        details = f"Contratos encontrados: {len(data.get('contratos', []))}"
        results.add("RF2.1: Consulta de contratos", passed, details)
    except Exception as e:
        results.add("RF2.1: Consulta de contratos", False, f"Error: {str(e)}")

    # Test 2: Estadísticas
    try:
        response = requests.get(f"{BASE_URL}/secop/estadisticas/SENA", timeout=TIMEOUT)

        data = response.json()
        passed = response.status_code == 200 and "total_contratos" in data
        details = f"Total contratos: {data.get('total_contratos', 0)}"
        results.add("RF2.2: Estadísticas por entidad", passed, details)
    except Exception as e:
        results.add("RF2.2: Estadísticas por entidad", False, f"Error: {str(e)}")

    # Test 3: Proveedores
    try:
        response = requests.get(f"{BASE_URL}/secop/proveedores", params={
            "sector": "software"
        }, timeout=TIMEOUT)

        data = response.json()
        passed = response.status_code == 200 and len(data.get("proveedores", [])) > 0
        details = f"Proveedores encontrados: {len(data.get('proveedores', []))}"
        results.add("RF2.3: Búsqueda de proveedores", passed, details)
    except Exception as e:
        results.add("RF2.3: Búsqueda de proveedores", False, f"Error: {str(e)}")

def test_rf3_busqueda_semantica(results: TestResults):
    """RF3: Búsqueda semántica (RAG)"""
    print_header("RF3: Búsqueda Semántica (RAG)")

    # Test 1: Similitud de embeddings
    try:
        response = requests.post(f"{BASE_URL}/ask", json={
            "query": "capacidad financiera requisitos",
            "top_k": 1
        }, timeout=TIMEOUT)

        data = response.json()
        match = data.get("matches", [{}])[0]
        score = match.get("score", 0)

        passed = score > 0.3  # Umbral mínimo de similitud
        details = f"Score de similitud: {score:.3f}"
        results.add("RF3.1: Similitud semántica", passed, details, score)
    except Exception as e:
        results.add("RF3.1: Similitud semántica", False, f"Error: {str(e)}")

def test_rf4_generacion_respuestas(results: TestResults):
    """RF4: Generación de respuestas contextualizadas"""
    print_header("RF4: Generación de Respuestas LLM")

    # Test 1: Respuesta coherente
    try:
        response = requests.post(f"{BASE_URL}/ask", json={
            "query": "¿Qué es la capacidad jurídica?",
            "top_k": 1
        }, timeout=TIMEOUT)

        data = response.json()
        answer = data.get("answer", "")

        # Verificar que la respuesta tiene contenido relevante
        passed = len(answer) > 50 and "capacidad" in answer.lower()
        details = f"Longitud respuesta: {len(answer)} caracteres"
        results.add("RF4.1: Respuesta coherente", passed, details)
    except Exception as e:
        results.add("RF4.1: Respuesta coherente", False, f"Error: {str(e)}")

def test_rf6_ingesta_documentos(results: TestResults):
    """RF6: Ingesta automática de documentos"""
    print_header("RF6: Ingesta Automática de Documentos")

    # Test 1: Base de datos con documentos
    try:
        response = requests.get(f"{BASE_URL}/database", timeout=TIMEOUT)

        data = response.json()
        total_docs = data.get("total_documentos", 0)
        total_chunks = data.get("estadisticas", {}).get("total_chunks", 0)

        passed = total_docs > 0 and total_chunks > 0
        details = f"Docs: {total_docs}, Chunks: {total_chunks}"
        results.add("RF6.1: Documentos procesados", passed, details)
    except Exception as e:
        results.add("RF6.1: Documentos procesados", False, f"Error: {str(e)}")

def test_rf8_api_rest(results: TestResults):
    """RF8: API REST completa"""
    print_header("RF8: API REST")

    endpoints = [
        ("GET", "/ping", "Health check"),
        ("GET", "/database", "Estado de BD"),
        ("POST", "/ask", "Consulta principal"),
        ("GET", "/secop/contratos", "Contratos SECOP II"),
        ("GET", "/secop/estadisticas/SENA", "Estadísticas"),
        ("GET", "/secop/proveedores?sector=software", "Proveedores"),
    ]

    for method, endpoint, name in endpoints:
        try:
            if method == "GET":
                response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
            else:
                response = requests.post(f"{BASE_URL}{endpoint}",
                                       json={"query": "test", "top_k": 1},
                                       timeout=10)

            passed = response.status_code == 200
            details = f"Status: {response.status_code}"
            results.add(f"RF8: {name}", passed, details)
        except Exception as e:
            results.add(f"RF8: {name}", False, f"Error: {str(e)}")

# ============================================================================
# PRUEBAS NO FUNCIONALES
# ============================================================================

def test_rnf1_rendimiento(results: TestResults):
    """RNF1: Rendimiento"""
    print_header("RNF1: Rendimiento")

    # Test 1: Tiempo de respuesta < 15s
    try:
        start = time.time()
        response = requests.post(f"{BASE_URL}/ask", json={
            "query": "¿Cuáles son los requisitos habilitantes?",
            "top_k": 1
        }, timeout=TIMEOUT)
        elapsed = time.time() - start

        passed = elapsed < 15.0 and response.status_code == 200
        details = f"Tiempo: {elapsed:.2f}s (límite: 15s)"
        results.add("RNF1.1: Tiempo de respuesta", passed, details, elapsed)
    except Exception as e:
        results.add("RNF1.1: Tiempo de respuesta", False, f"Error: {str(e)}")

    # Test 2: Consulta simple < 5s
    try:
        start = time.time()
        response = requests.get(f"{BASE_URL}/ping", timeout=5)
        elapsed = time.time() - start

        passed = elapsed < 5.0 and response.status_code == 200
        details = f"Tiempo: {elapsed:.3f}s (límite: 5s)"
        results.add("RNF1.2: Consulta simple rápida", passed, details, elapsed)
    except Exception as e:
        results.add("RNF1.2: Consulta simple rápida", False, f"Error: {str(e)}")

def test_rnf3_disponibilidad(results: TestResults):
    """RNF3: Disponibilidad"""
    print_header("RNF3: Disponibilidad")

    # Test 1: Sistema responde
    try:
        response = requests.get(f"{BASE_URL}/ping", timeout=5)
        data = response.json()

        passed = response.status_code == 200 and data.get("message") == "pong"
        details = f"Sistema: {data.get('db')}, LLM: {data.get('llm')}"
        results.add("RNF3.1: Sistema disponible", passed, details)
    except Exception as e:
        results.add("RNF3.1: Sistema disponible", False, f"Error: {str(e)}")

def test_rnf5_precision(results: TestResults):
    """RNF5: Precisión"""
    print_header("RNF5: Precisión (Dataset de Validación)")

    # Dataset de preguntas con respuestas esperadas
    test_cases = [
        {
            "query": "¿Cuáles son los requisitos habilitantes?",
            "keywords": ["capacidad", "jurídica", "financiera", "experiencia"],
            "min_score": 0.5
        },
        {
            "query": "¿Qué garantías se necesitan en obra pública?",
            "keywords": ["garantía", "cumplimiento", "anticipo"],
            "min_score": 0.4
        },
        {
            "query": "¿Cómo se evalúan las propuestas?",
            "keywords": ["criterio", "evaluación", "ponderación", "precio"],
            "min_score": 0.4
        }
    ]

    correct = 0
    for i, case in enumerate(test_cases, 1):
        try:
            response = requests.post(f"{BASE_URL}/ask", json={
                "query": case["query"],
                "top_k": 1
            }, timeout=TIMEOUT)

            data = response.json()
            answer = data.get("answer", "").lower()
            match = data.get("matches", [{}])[0]
            score = match.get("score", 0)

            # Verificar presencia de keywords
            keywords_found = sum(1 for kw in case["keywords"] if kw in answer)
            keyword_ratio = keywords_found / len(case["keywords"])

            passed = (score >= case["min_score"] and keyword_ratio >= 0.3) or keywords_found >= 2

            if passed:
                correct += 1

            details = f"Score: {score:.2f}, Keywords: {keywords_found}/{len(case['keywords'])}"
            results.add(f"RNF5.{i}: Precisión pregunta {i}", passed, details, score)
        except Exception as e:
            results.add(f"RNF5.{i}: Precisión pregunta {i}", False, f"Error: {str(e)}")

    # Precisión general
    precision = (correct / len(test_cases)) * 100 if test_cases else 0
    passed = precision >= 66.0  # Al menos 2/3
    details = f"Precisión: {precision:.1f}% (objetivo: ≥66%)"
    results.add("RNF5: Precisión general", passed, details, precision)

def test_rnf6_usabilidad(results: TestResults):
    """RNF6: Usabilidad"""
    print_header("RNF6: Usabilidad")

    # Test 1: Interfaz web disponible
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)

        passed = response.status_code == 200 and "html" in response.text.lower()
        details = f"Interfaz web accesible"
        results.add("RNF6.1: Interfaz web disponible", passed, details)
    except Exception as e:
        results.add("RNF6.1: Interfaz web disponible", False, f"Error: {str(e)}")

    # Test 2: Respuestas en español
    try:
        response = requests.post(f"{BASE_URL}/ask", json={
            "query": "¿Qué es un contrato?",
            "top_k": 1
        }, timeout=TIMEOUT)

        data = response.json()
        answer = data.get("answer", "")

        # Verificar caracteres en español
        spanish_chars = any(c in answer for c in ['á', 'é', 'í', 'ó', 'ú', 'ñ', 'ü'])
        passed = len(answer) > 0 and (spanish_chars or "contrato" in answer.lower())
        details = f"Respuesta en español: {'Sí' if spanish_chars else 'Probable'}"
        results.add("RNF6.2: Respuestas en español", passed, details)
    except Exception as e:
        results.add("RNF6.2: Respuestas en español", False, f"Error: {str(e)}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Ejecuta suite completa de pruebas"""
    print(f"\n{Colors.BOLD}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}SUITE DE PRUEBAS SISTEMÁTICAS - SISTEMA LLM SECOP II{Colors.END}")
    print(f"{Colors.BOLD}Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 80}{Colors.END}")

    results = TestResults()

    # Verificar que el servidor esté corriendo
    try:
        requests.get(f"{BASE_URL}/ping", timeout=5)
    except:
        print(f"\n{Colors.RED}ERROR: El servidor no está corriendo en {BASE_URL}{Colors.END}")
        print(f"{Colors.YELLOW}Por favor inicia el servidor: ./venv/bin/python api.py{Colors.END}\n")
        sys.exit(1)

    # Ejecutar pruebas funcionales
    test_rf1_consulta_lenguaje_natural(results)
    test_rf2_acceso_secop(results)
    test_rf3_busqueda_semantica(results)
    test_rf4_generacion_respuestas(results)
    test_rf6_ingesta_documentos(results)
    test_rf8_api_rest(results)

    # Ejecutar pruebas no funcionales
    test_rnf1_rendimiento(results)
    test_rnf3_disponibilidad(results)
    test_rnf5_precision(results)
    test_rnf6_usabilidad(results)

    # Mostrar resumen
    success = results.summary()

    # Guardar reporte
    report = {
        "timestamp": datetime.now().isoformat(),
        "total": results.total,
        "passed": results.passed,
        "failed": results.failed,
        "success_rate": (results.passed / results.total * 100) if results.total > 0 else 0,
        "tests": results.results
    }

    with open("test_results.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{Colors.CYAN}📄 Reporte guardado en: test_results.json{Colors.END}\n")

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
