# 🎯 GUIÓN DE SUSTENTACIÓN DEL PROYECTO
## **Sistema LLM de Consulta Inteligente para SECOP II**

---

## **FASE 1: INTRODUCCIÓN (2-3 minutos)**

### **Qué decir:**
"Buenos días/tardes. El día de hoy voy a presentar un **Sistema de Consulta Inteligente basado en LLM** para procesos de contratación pública en Colombia, específicamente integrado con **SECOP II**.

Este sistema permite a funcionarios públicos y ciudadanos:
1. Consultar información de contratos en tiempo real
2. Hacer preguntas en lenguaje natural sobre documentos de contratación
3. Obtener estadísticas y análisis de entidades públicas

El proyecto está construido con Python, FastAPI, y utiliza técnicas de RAG (Retrieval-Augmented Generation) con embeddings vectoriales."

### **Qué mostrar:**
- Abrir el navegador en `http://localhost:8000/ui`
- Mostrar la interfaz principal brevemente

---

## **FASE 2: ARQUITECTURA DEL SISTEMA (4-5 minutos)**

### **Qué decir:**
"Voy a explicar la arquitectura del sistema mostrando el código principal. Abramos el archivo `api.py`."

### **📍 Mostrar líneas 1-20** - Importaciones y dependencias
**Qué explicar:**
```
"Aquí vemos las importaciones principales:
- FastAPI: framework web moderno y de alto rendimiento
- ReportLab: generación de PDFs
- NumPy: cálculos vectoriales para embeddings
- python-dotenv: gestión de configuración sensible"
```

### **📍 Mostrar líneas 24-35** - Configuración y variables de entorno
**Qué explicar:**
```
"El sistema usa un patrón de configuración flexible:
- En la línea 24-29 creamos las carpetas necesarias automáticamente
- En la línea 31 cargamos variables del archivo .env
- En la línea 32 definimos el backend de base de datos (SQLite, PostgreSQL o InterSystems IRIS)
- En las líneas 33-35 detectamos automáticamente qué proveedor de LLM está disponible (OpenAI o Mistral)

Esto hace que el sistema funcione incluso SIN API keys, usando embeddings heurísticos."
```

### **📍 Mostrar líneas 40-56** - Importación dinámica de backends
**Qué explicar:**
```
"Aquí implementamos un patrón de diseño Strategy:
- Dependiendo de la variable DB_BACKEND, importamos diferentes módulos
- Esto permite cambiar de SQLite a PostgreSQL o IRIS sin modificar el código
- El sistema tiene fallback a SQLite si hay errores
- Es un ejemplo de Principio de Inversión de Dependencias (SOLID)"
```

---

## **FASE 3: MÓDULOS CORE (5-6 minutos)**

### **Qué decir:**
"Ahora veamos los módulos fundamentales que dan inteligencia al sistema."

### **📍 Abrir `src/embeddings.py` - líneas 1-44**
**Qué explicar:**
```
"Este módulo maneja la vectorización de texto:

LÍNEA 21: Inicializamos el cliente de OpenAI solo si hay API key
LÍNEA 23-27: Función _cheap_embed - embeddings heurísticos usando hash
  - Esto permite que el sistema funcione sin API keys
  - Usa el hash del texto como semilla para un vector normalizado
  - No es perfecto pero es funcional para demos

LÍNEA 29-35: embed_texts - vectorización por lotes
  - Si tenemos OpenAI, usa el modelo text-embedding-3-small
  - Si no, usa embeddings heurísticos
  - Retorna arrays de NumPy para cálculos eficientes

Esta flexibilidad es clave: el sistema funciona con o sin servicios externos."
```

### **📍 Abrir `src/chunking.py` - líneas 1-23**
**Qué explicar:**
```
"Este módulo implementa la segmentación de documentos:

LÍNEA 5: Función split_text con overlap
  - max_chars=1000: tamaño máximo de cada fragmento
  - overlap=150: solapamiento entre fragmentos para no perder contexto

LÍNEA 8-9: Limpieza de espacios en blanco y saltos de línea

LÍNEA 12-20: Algoritmo de ventana deslizante
  - Divide el texto en chunks
  - Mantiene overlap para preservar contexto entre fragmentos

Esto es fundamental para RAG: documentos grandes se dividen en piezas manejables."
```

### **📍 Abrir `src/db_sqlite.py`**

#### Mostrar líneas 19-43 - Inicialización de base de datos
**Qué explicar:**
```
"Aquí definimos el esquema de base de datos:

LÍNEA 24-30: Tabla 'documents'
  - Almacena metadata de cada documento subido
  - doc_id es autoincremental

LÍNEA 33-40: Tabla 'chunks'
  - Almacena fragmentos del texto con sus embeddings
  - emb_json guarda el vector como JSON
  - Relación uno-a-muchos con documents
  - ON DELETE CASCADE: si borro un doc, se borran sus chunks

LÍNEA 42: Índice para optimizar búsquedas por doc_id

Este diseño permite búsqueda vectorial eficiente."
```

#### Mostrar líneas 97-120 - Recuperación de vectores
**Qué explicar:**
```
"LÍNEA 97-120: fetch_all_vectors

Esta función es crucial para la búsqueda semántica:
- JOIN entre chunks y documents
- Retorna tuplas con chunk_id, doc_id, texto, embedding y título
- Deserializa los embeddings de JSON a listas de Python

Esto alimenta el algoritmo de similitud coseno para encontrar los chunks más relevantes."
```

---

## **FASE 4: INTEGRACIÓN CON SECOP II (3-4 minutos)**

### **📍 Abrir `src/secop_api.py`**

#### Mostrar líneas 9-57
**Qué explicar:**
```
"Este módulo integra con la API pública de datos abiertos de Colombia:

LÍNEA 9: URL base de SECOP II en datos.gov.co

LÍNEA 11-30: Función buscar_contratos
  - Parámetros opcionales: entidad, objeto, fechas, límite
  - Construye query dinámico

LÍNEA 36-46: Construcción de filtros WHERE
  - SQL injection safe usando la API de Socrata
  - Permite búsquedas combinadas

LÍNEA 51-54: Manejo de errores robusto
  - Timeout de 30 segundos
  - Retorna lista vacía en caso de error

Esta integración permite consultar contratos reales en tiempo real."
```

#### Mostrar líneas 60-96 - Estadísticas
**Qué explicar:**
```
"LÍNEA 60: obtener_estadisticas_entidad

Esta función calcula métricas agregadas:
- Total de contratos
- Monto total y promedio
- Distribución por modalidad de contratación

LÍNEA 72-78: Manejo de errores en montos
  - Try-except porque algunos contratos no tienen valor

Es un ejemplo de análisis de datos sobre APIs públicas."
```

---

## **FASE 5: ENDPOINTS DE LA API (6-7 minutos)**

### **Qué decir:**
"Ahora veamos los endpoints principales que exponen toda esta funcionalidad."

### **📍 Volver a `api.py` - líneas 1337-1396**

#### Mostrar líneas 1337-1370 - Endpoint de contratos
**Qué explicar:**
```
"LÍNEA 1337: @app.get('/secop/contratos')

Este endpoint permite búsqueda flexible:
- Query parameters opcionales
- Ejemplos en la documentación (línea 1348-1351)
- Retorna JSON con total, filtros y contratos

DEMOSTRACIÓN EN VIVO:
Ir al navegador: http://localhost:8000/secop/contratos?entidad=SENA&limite=5

Explicar el JSON retornado."
```

#### Mostrar líneas 1373-1380 - Endpoint de estadísticas
**Qué explicar:**
```
"LÍNEA 1373: @app.get('/secop/estadisticas/{entidad}')

Path parameter para la entidad.

DEMOSTRACIÓN:
http://localhost:8000/secop/estadisticas/SENA

Mostrar las estadísticas calculadas."
```

### **📍 Líneas 1490-1602** - Endpoint de preguntas con LLM

**Qué explicar:**
```
"LÍNEA 1490: @app.post('/ask')

Este es el corazón del sistema de RAG:

LÍNEA 1493-1510: Búsqueda semántica
  - Vectoriza la pregunta del usuario
  - Calcula similitud coseno con todos los chunks
  - Ordena por relevancia
  - Toma los top_k más similares

LÍNEA 1515-1521: Búsqueda de PDFs confiables
  - Si la pregunta coincide con keywords de PDFs específicos
  - Descarga y procesa automáticamente

LÍNEA 1536-1570: Generación de respuesta con LLM
  - Prompt engineering cuidadoso
  - Contexto de chunks relevantes
  - Instrucciones claras para respuestas precisas

LÍNEA 1575-1598: Modo heurístico (sin LLM)
  - Si no hay API key, usa reglas simples
  - Retorna contexto relevante sin generar texto

Este endpoint combina búsqueda vectorial + generación de lenguaje = RAG completo."
```

---

## **FASE 6: DEMOSTRACIÓN EN VIVO (5-6 minutos)**

### **Qué decir:**
"Ahora voy a demostrar el sistema funcionando en tiempo real."

### **Demo 1: Interfaz Web**
1. Abrir `http://localhost:8000/ui`
2. **Hacer una pregunta:**
   ```
   "¿Cuáles son los requisitos habilitantes para contratar con el Estado?"
   ```
3. **Explicar mientras carga:**
   - "El sistema está vectorizando mi pregunta"
   - "Buscando en los chunks más similares"
   - "Generando respuesta contextualizada"

4. **Mostrar la respuesta** y explicar las fuentes citadas

### **Demo 2: Consulta SECOP II**
1. En la interfaz, ir a la sección de SECOP
2. Buscar: `Entidad: SENA, Objeto: tecnología`
3. **Explicar:**
   - "Consulta en tiempo real a la API de datos abiertos"
   - Mostrar los contratos retornados

### **Demo 3: Documentación Interactiva**
1. Ir a `http://localhost:8000/docs`
2. Mostrar la documentación autogenerada por FastAPI
3. Probar un endpoint directamente (ej: `/ping`)

---

## **FASE 7: CARACTERÍSTICAS TÉCNICAS DESTACABLES (3-4 minutos)**

### **Qué decir y mostrar:**

**1. Arquitectura Multi-Backend**
- Volver a `api.py` líneas 40-56
- "Soporte para SQLite, PostgreSQL e InterSystems IRIS"

**2. Sistema Flexible sin Dependencias Externas**
- Volver a `src/embeddings.py` líneas 23-27
- "Funciona incluso sin OpenAI usando embeddings heurísticos"

**3. Patrón RAG (Retrieval-Augmented Generation)**
- `api.py` líneas 1493-1570
- "Combina búsqueda semántica con generación de lenguaje"

**4. Procesamiento Automático de PDFs**
- `api.py` líneas 72-117 (TRUSTED_PDFS)
- "PDFs confiables que se descargan y procesan automáticamente"

**5. API RESTful Moderna**
- "FastAPI con validación automática, documentación y type hints"

**6. Manejo de Errores Robusto**
- Try-except en múltiples niveles
- Fallbacks inteligentes

---

## **FASE 8: PRUEBAS Y VALIDACIÓN (2-3 minutos)**

### **Qué mostrar:**
```bash
# En terminal
./test_suite.py
```

### **Qué explicar:**
```
"El proyecto incluye una suite de pruebas automatizadas que valida:
- Requisitos funcionales (RF)
- Requisitos no funcionales (RNF)
- Integración con SECOP II
- Tiempos de respuesta

Aquí vemos los resultados de las pruebas..."
```

**Abrir** `http://localhost:8000/ui` y mostrar la sección de resultados de tests.

---

## **FASE 9: CONCLUSIONES (2 minutos)**

### **Qué decir:**

"Para concluir, este proyecto demuestra:

**1. Integración de tecnologías modernas:**
   - FastAPI, LLMs, Embeddings vectoriales, APIs públicas

**2. Diseño flexible y escalable:**
   - Múltiples backends de datos
   - Funciona con o sin servicios externos
   - Arquitectura modular

**3. Aplicación práctica:**
   - Resuelve un problema real de acceso a información pública
   - Simplifica consultas complejas con lenguaje natural

**4. Buenas prácticas:**
   - Código limpio y documentado
   - Manejo de errores robusto
   - Pruebas automatizadas
   - Type hints y validación

**Tecnologías utilizadas:**
- Python 3.11+
- FastAPI
- OpenAI API / Mistral AI
- SQLite/PostgreSQL/InterSystems IRIS
- ReportLab, NumPy, pypdf

**Posibles mejoras futuras:**
- Caché de embeddings
- Soporte para más formatos de documentos
- Dashboard de analytics
- Autenticación de usuarios"

---

## **FASE 10: PREGUNTAS Y RESPUESTAS**

### **Preguntas comunes y dónde mostrar el código:**

**P: "¿Cómo funciona la búsqueda semántica?"**
- R: Mostrar `api.py` líneas 1493-1510 y explicar similitud coseno

**P: "¿Qué pasa si no hay API key de OpenAI?"**
- R: Mostrar `src/embeddings.py` líneas 23-27 y 1575-1598 de `api.py`

**P: "¿Cómo se conecta con SECOP II?"**
- R: Mostrar `src/secop_api.py` completo

**P: "¿Cómo se almacenan los embeddings?"**
- R: Mostrar `src/db_sqlite.py` líneas 33-40 y 80-95

**P: "¿Qué es RAG?"**
- R: "Retrieval-Augmented Generation: combinar búsqueda de información relevante con generación de texto. Explicar flujo completo."

**P: "¿Por qué usar FastAPI en lugar de Flask o Django?"**
- R: "FastAPI ofrece:
  - Validación automática con Pydantic
  - Documentación interactiva automática (Swagger/OpenAPI)
  - Mejor rendimiento (async/await nativo)
  - Type hints completos"

**P: "¿Cómo escalaría este sistema para miles de usuarios?"**
- R: "Estrategias:
  - Usar PostgreSQL en lugar de SQLite
  - Implementar caché con Redis
  - Separar el procesamiento de PDFs en workers
  - Usar bases de datos vectoriales especializadas (Pinecone, Weaviate)
  - Load balancing con múltiples instancias"

---

## **📋 CHECKLIST ANTES DE LA PRESENTACIÓN**

- [ ] Servidor corriendo: `uvicorn api:app --reload --port 8000`
- [ ] Navegador abierto en `http://localhost:8000/ui`
- [ ] Documentos cargados en la base de datos (o listos para cargar)
- [ ] VSCode abierto con los archivos clave marcados
- [ ] Test suite ejecutado al menos una vez
- [ ] Conexión a internet (para demostrar SECOP II en vivo)
- [ ] Tener ejemplos de preguntas preparadas
- [ ] Revisar que las variables de entorno estén configuradas
- [ ] Tener backup de capturas de pantalla por si falla internet

---

## **⏱️ TIMING RECOMENDADO (Total: 25-30 minutos)**

1. Introducción: 2-3 min
2. Arquitectura: 4-5 min
3. Módulos Core: 5-6 min
4. SECOP II: 3-4 min
5. Endpoints API: 6-7 min
6. Demo en vivo: 5-6 min
7. Características técnicas: 3-4 min
8. Pruebas: 2-3 min
9. Conclusiones: 2 min
10. Preguntas: 5-10 min

---

## **💡 TIPS PARA UNA BUENA PRESENTACIÓN**

1. **Practica el flujo completo** al menos 2 veces antes
2. **Habla con confianza** - conoces tu código mejor que nadie
3. **Mantén contacto visual** con la audiencia, no solo la pantalla
4. **Explica el "por qué"** no solo el "qué" - demuestra pensamiento crítico
5. **Ten preparadas respuestas** a preguntas difíciles
6. **Si algo falla en vivo**, mantén la calma y explica qué debería pasar
7. **Usa terminología técnica apropiada** pero asegúrate de explicarla
8. **Muestra pasión** por tu proyecto - el entusiasmo es contagioso

---

## **📝 EJEMPLOS DE PREGUNTAS PARA DEMOSTRAR**

### Preguntas que funcionan bien con el sistema:

1. **Requisitos habilitantes:**
   - "¿Cuáles son los requisitos habilitantes para contratar con el Estado?"
   - "¿Qué documentos necesito para participar en una licitación?"

2. **Criterios de evaluación:**
   - "¿Cómo se evalúan las propuestas en una licitación pública?"
   - "¿Qué ponderación tienen los criterios económicos?"

3. **Plazos y garantías:**
   - "¿Cuáles son los plazos típicos en un proceso de contratación?"
   - "¿Qué tipos de garantías debo presentar?"

4. **SECOP II específico:**
   - "¿Cómo funciona la gestión contractual en SECOP II?"
   - "¿Qué es la validación de facturas en SECOP II?"

### Consultas SECOP que impresionan:

1. `Entidad: SENA` - Muestra gran volumen de contratos
2. `Objeto: tecnología` - Sector relevante y actual
3. `Entidad: Alcaldía de Bogotá, Objeto: infraestructura` - Búsqueda combinada

---

## **🎓 CONCEPTOS CLAVE PARA EXPLICAR SI PREGUNTAN**

### RAG (Retrieval-Augmented Generation)
"Es una técnica que combina búsqueda de información (retrieval) con generación de texto (generation). En lugar de que el LLM responda solo con su conocimiento entrenado, primero buscamos información relevante en nuestra base de datos y se la damos como contexto. Esto reduce alucinaciones y permite respuestas basadas en documentos específicos."

### Embeddings Vectoriales
"Son representaciones numéricas de texto en un espacio de alta dimensionalidad. Textos con significados similares tienen vectores cercanos. Esto nos permite hacer búsqueda semántica: encontrar información por significado, no solo por palabras exactas."

### Similitud Coseno
"Medida de similitud entre dos vectores basada en el ángulo entre ellos. Va de -1 a 1, donde 1 significa vectores idénticos. Es ideal para comparar embeddings porque normaliza por magnitud."

### API RESTful
"Interfaz de programación basada en HTTP que sigue principios REST. Usa verbos HTTP (GET, POST) de manera semántica, es stateless, y retorna datos estructurados (JSON). Permite que otros sistemas se integren fácilmente."

### FastAPI vs Flask
"FastAPI es más moderno: validación automática con Pydantic, documentación automática, soporte async nativo, y mejor rendimiento. Flask es más simple pero requiere más configuración manual."

---

**¡Éxito en tu presentación!** 🚀
