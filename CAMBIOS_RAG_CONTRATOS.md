# Documentación de Cambios - Sistema RAG para Contratos SECOP II

## Resumen de Requisitos Implementados

| Requisito | Estado | Ubicación |
|-----------|--------|-----------|
| Código único para cada registro | ✅ Cumplido | `src/db_sqlite.py:182-192` |
| Texto Total (JSON completo) | ✅ Cumplido | `src/db_sqlite.py:240` |
| Texto a Indexar (campos específicos) | ✅ Cumplido | `src/db_sqlite.py:195-225` |
| Embeddings del texto a indexar | ✅ Cumplido | `src/db_sqlite.py:254-284` |
| Mismo identificador para todos los vectores | ✅ Cumplido | `src/db_sqlite.py:272` |
| Recuperar texto total desde vector | ✅ Cumplido | `src/db_sqlite.py:287-303` |

---

## 1. Estructura de Base de Datos

### Archivo: `src/db_sqlite.py`
### Líneas: 44-67

```sql
-- Tabla principal de contratos (líneas 45-53)
CREATE TABLE IF NOT EXISTS contratos (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    codigo_unico    TEXT UNIQUE NOT NULL,  -- Clave única
    texto_total     TEXT NOT NULL,          -- JSON completo
    texto_indexar   TEXT NOT NULL,          -- Campos para embeddings
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla de embeddings (líneas 57-65)
CREATE TABLE IF NOT EXISTS contrato_embeddings (
    emb_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    codigo_unico    TEXT NOT NULL,          -- Vincula al contrato
    chunk_ord       INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    emb_json        TEXT NOT NULL,
    FOREIGN KEY (codigo_unico) REFERENCES contratos(codigo_unico)
);
```

**Por qué:** Se crearon dos tablas separadas para:
- `contratos`: Almacena el registro completo y el texto para indexar
- `contrato_embeddings`: Almacena los vectores, vinculados por `codigo_unico`

---

## 2. Generación de Código Único

### Archivo: `src/db_sqlite.py`
### Líneas: 182-192

```python
def generar_codigo_unico(registro: Dict[str, Any], indice: int) -> str:
    # Intentar usar campos existentes como identificador
    for campo in ["codigo_de_secop", "numero_del_proceso",
                  "referencia_del_contrato", "id_contrato"]:
        if campo in registro and registro[campo]:
            return str(registro[campo]).strip()
    # Si no hay código, generar uno
    return f"SEC-{indice:06d}"
```

**Lógica:**
1. Primero busca si el registro ya tiene un código en campos conocidos
2. Si no existe ninguno, genera `SEC-000001`, `SEC-000002`, etc.

**Cómo verificar:**
```python
# Registro con código existente
registro1 = {"numero_del_proceso": "ABC-123"}
generar_codigo_unico(registro1, 1)  # Retorna: "ABC-123"

# Registro sin código
registro2 = {"nombre_entidad": "SENA"}
generar_codigo_unico(registro2, 5)  # Retorna: "SEC-000005"
```

---

## 3. Texto Total (JSON Completo)

### Archivo: `src/db_sqlite.py`
### Línea: 240

```python
texto_total = json.dumps(registro, ensure_ascii=False, indent=2)
```

**Por qué:**
- `json.dumps()` convierte todo el diccionario a string JSON
- `ensure_ascii=False` preserva caracteres especiales (tildes, ñ)
- `indent=2` formatea legiblemente

**Resultado:** Se guarda el JSON completo desde `{` hasta `}` con todos los campos.

---

## 4. Texto a Indexar (Campos Específicos)

### Archivo: `src/db_sqlite.py`
### Líneas: 195-225

```python
def extraer_texto_indexar(registro: Dict[str, Any]) -> str:
    campos = []

    # 1. Departamento (líneas 205-209)
    for key in ["departamento", "departamento_entidad", "departamento_ejecucion"]:
        if key in registro and registro[key]:
            campos.append(f"Departamento: {registro[key]}")
            break

    # 2. Descripción del proceso (líneas 211-213)
    if registro.get("descripcion_del_proceso"):
        campos.append(f"Descripción: {registro['descripcion_del_proceso']}")

    # 3. Objeto del contrato (líneas 215-219)
    for key in ["objeto_del_contrato", "objeto_a_contratar",
                "detalle_del_objeto_a_contratar"]:
        if key in registro and registro[key]:
            campos.append(f"Objeto: {registro[key]}")
            break

    # 4. Nombre de la entidad (líneas 221-223)
    if registro.get("nombre_entidad"):
        campos.append(f"Entidad: {registro['nombre_entidad']}")

    return "\n".join(campos)
```

**Campos extraídos:**
| Campo Requerido | Campos JSON Buscados |
|-----------------|---------------------|
| Departamento | `departamento`, `departamento_entidad`, `departamento_ejecucion` |
| Descripción del proceso | `descripcion_del_proceso` |
| Objeto del contrato | `objeto_del_contrato`, `objeto_a_contratar`, `detalle_del_objeto_a_contratar` |
| Nombre de la entidad | `nombre_entidad` |

**Ejemplo de salida:**
```
Departamento: Cundinamarca
Descripción: Contratación de servicios de software
Objeto: Desarrollo de aplicación web
Entidad: SENA
```

---

## 5. Proceso de Embeddings

### Archivo: `src/db_sqlite.py`
### Líneas: 254-284

```python
def insert_contrato_embeddings(codigo_unico: str, chunks: List[str], embeddings) -> int:
    rows = []
    for i, (chunk, emb) in enumerate(zip(chunks, embs_list)):
        rows.append((
            codigo_unico,  # Mismo ID para todos los chunks
            i,             # Orden del chunk
            chunk,         # Texto del chunk
            json.dumps(emb)  # Vector embedding
        ))

    # Insertar en BD
    cur.executemany(
        "INSERT INTO contrato_embeddings (codigo_unico, chunk_ord, chunk_text, emb_json) VALUES (?, ?, ?, ?)",
        rows
    )
```

**Clave:** Todos los embeddings de un mismo contrato comparten el mismo `codigo_unico`.

---

## 6. Recuperación de Texto Total desde Vector

### Archivo: `src/db_sqlite.py`
### Líneas: 287-303 y 262-278

```python
# Obtener embeddings para búsqueda vectorial
def fetch_all_contrato_embeddings():
    # Retorna: (codigo_unico, chunk_ord, chunk_text, embedding)

# Obtener contrato completo por código
def get_contrato_by_codigo(codigo_unico: str):
    # Retorna: {
    #   "codigo_unico": "...",
    #   "texto_total": {...},  # JSON completo parseado
    #   "texto_indexar": "..."
    # }
```

**Flujo RAG:**
1. Usuario hace pregunta
2. Se genera embedding de la pregunta
3. Se busca en `contrato_embeddings` el vector más similar
4. Se obtiene el `codigo_unico` del match
5. Se usa `get_contrato_by_codigo()` para obtener `texto_total` completo
6. Se responde con toda la información estructurada

---

## 7. Archivos Modificados

| Archivo | Líneas Modificadas | Descripción |
|---------|-------------------|-------------|
| `src/db_sqlite.py` | 44-67 | Nuevas tablas `contratos` y `contrato_embeddings` |
| `src/db_sqlite.py` | 180-323 | Funciones SECOP (generar código, extraer texto, insertar, consultar) |
| `api.py` | 67-70 | Imports de nuevas funciones |
| `api.py` | 587-622 | UI: Pestaña "Base RAG" |
| `api.py` | 915-999 | JavaScript: Funciones RAG |
| `api.py` | 1399-1472 | Endpoints: `/rag/contratos`, `/rag/cargar`, `/rag/stats` |
| `cargar_contratos.py` | (nuevo) | Script CLI para carga masiva |

---

## 8. Cómo Verificar la Implementación

### Verificar estructura de BD:
```bash
sqlite3 data/app.sqlite3 ".schema contratos"
sqlite3 data/app.sqlite3 ".schema contrato_embeddings"
```

### Verificar carga de contratos:
```bash
# Cargar desde API
python cargar_contratos.py --api --limite 10 --stats

# Ver contratos cargados
sqlite3 data/app.sqlite3 "SELECT codigo_unico, substr(texto_indexar,1,100) FROM contratos LIMIT 5"
```

### Verificar desde UI:
1. Ejecutar: `python api.py`
2. Abrir: `http://127.0.0.1:8001`
3. Ir a pestaña "Base RAG"
4. Cargar contratos y verificar lista

### Verificar código único:
```python
from src.db_sqlite import generar_codigo_unico

# Test con código existente
r1 = {"numero_del_proceso": "CO1.PCCNTR.123456"}
print(generar_codigo_unico(r1, 1))  # CO1.PCCNTR.123456

# Test sin código
r2 = {"nombre_entidad": "Test"}
print(generar_codigo_unico(r2, 99))  # SEC-000099
```

---

## 9. Endpoints API Disponibles

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/rag/contratos` | GET | Lista contratos cargados |
| `/rag/contratos/{codigo}` | GET | Obtiene contrato por código único |
| `/rag/cargar` | POST | Carga contratos desde SECOP II |
| `/rag/stats` | GET | Estadísticas del sistema RAG |

---

## 10. Diagrama de Flujo

```
┌─────────────────┐
│  JSON SECOP II  │
│  (registro)     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  generar_codigo_unico()                 │
│  - Busca campo existente                │
│  - O genera SEC-XXXXXX                  │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  insert_contrato()                      │
│  - codigo_unico                         │
│  - texto_total = JSON completo          │
│  - texto_indexar = 4 campos específicos │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Tabla: contratos                       │
│  ┌─────────────┬─────────┬────────────┐ │
│  │codigo_unico │texto_tot│texto_index │ │
│  └─────────────┴─────────┴────────────┘ │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  insert_contrato_embeddings()           │
│  - Genera embeddings de texto_indexar   │
│  - Guarda con mismo codigo_unico        │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Tabla: contrato_embeddings             │
│  ┌─────────────┬─────────┬────────────┐ │
│  │codigo_unico │chunk_ord│emb_json    │ │
│  └─────────────┴─────────┴────────────┘ │
└─────────────────────────────────────────┘
```

---

*Documento generado: 2025-11-21*
