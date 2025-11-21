# api.py
from __future__ import annotations
from pathlib import Path
from io import BytesIO
from datetime import datetime
import os, json, zipfile
from typing import Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import numpy as np
import requests

# PDF
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics

# =========================
# Config y rutas
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ORIG_DIR = DATA_DIR / "originals"
UPLOADS_DIR = DATA_DIR / "uploads"
ORIG_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(dotenv_path=BASE_DIR / ".env")
DB_BACKEND = os.getenv("DB_BACKEND", "sqlite").lower()
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
MISTRAL_API_KEY = (os.getenv("MISTRAL_API_KEY") or "").strip()
LLM_PROVIDER = "openai" if OPENAI_API_KEY else ("mistral" if MISTRAL_API_KEY else "")

# =========================
# Backend de datos (asumimos que estos archivos src/* existen)
# =========================
try:
    if DB_BACKEND == "iris":
        from src.db_iris import init_db, insert_document, list_documents, insert_chunks, fetch_all_vectors, \
            get_document, fetch_doc_text
    elif DB_BACKEND == "postgres":
        from src.db_postgres import init_db, insert_document, list_documents, insert_chunks, fetch_all_vectors, \
            get_document, fetch_doc_text
    else:
        from src.db_sqlite import init_db, insert_document, list_documents, insert_chunks, fetch_all_vectors, \
            get_document, fetch_doc_text

        DB_BACKEND = "sqlite"
except Exception:
    from src.db_sqlite import init_db, insert_document, list_documents, insert_chunks, fetch_all_vectors, get_document, \
        fetch_doc_text

    DB_BACKEND = "sqlite"

try:
    init_db()
except Exception:
    pass

from src.chunking import split_text
from src.embeddings import embed_texts, embed_text
from src.secop_api import buscar_contratos, obtener_estadisticas_entidad, buscar_proveedores_por_sector
from src.db_sqlite import (
    insert_contrato, get_contrato_by_codigo, list_contratos, count_contratos,
    insert_contrato_embeddings, fetch_all_contrato_embeddings
)
from pypdf import PdfReader
from pydantic import BaseModel

# =========================
# PDFs Confiables para Búsqueda Automática
# =========================
TRUSTED_PDFS = [
    {
        "titulo": "Manual para determinar y verificar los requisitos habilitantes",
        "url": "https://operaciones.colombiacompra.gov.co/sites/cce_public/files/cce_documents/cce-eicp-ma-04._manual_requisitos_habilitantes_v3_29-09-2023.pdf",
        "entidad": "Colombia Compra Eficiente",
        "keywords": ["requisitos", "habilitantes", "capacidad", "juridica", "jurídica", "financiera", "organizacional",
                     "experiencia", "inhabilidades", "contrato"]
    },
    {
        "titulo": "Guía de criterios de evaluación",
        "url": "https://www.colombiacompra.gov.co/wp-content/uploads/2024/10/cce-sec-gi-18guiasecopii_eepclicitacionpublica20-04-2022.pdf",
        "entidad": "Colombia Compra Eficiente",
        "keywords": ["criterios", "evaluacion", "evaluación", "ponderacion", "ponderación", "metodologia",
                     "metodología", "precio", "experiencia"]
    },
    {
        "titulo": "Pliego de Condiciones Tipo – Obra Pública v2.0",
        "url": "https://www.colombiacompra.gov.co/wp-content/uploads/2024/08/20151115_pliego_de_condiciones_para_contrato_de_obra_publica_v2_0.pdf",
        "entidad": "Colombia Compra Eficiente",
        "keywords": ["pliego", "obra", "requisitos", "plazo", "garantias", "garantías", "ejecucion", "ejecución",
                     "condiciones", "pagos"]
    },
    {
        "titulo": "Guía – Gestión Contractual (SECOP II)",
        "url": "https://formacionvirtual.colombiacompra.gov.co/pluginfile.php/9193/mod_folder/content/0/M%C3%B3dulo%20VI/Gu%C3%ADa%20-%20Gesti%C3%B3n%20Contractual.pdf",
        "entidad": "Colombia Compra Eficiente",
        "keywords": ["gestion", "gestión", "contractual", "pagos", "plazos", "validación", "facturas", "parafiscales",
                     "cronograma", "secop", "secop ii"]
    },
]


# =========================
# Utilidades de Ingesta
# =========================
def _existing_urls_set() -> set:
    urls = set()
    try:
        docs = list_documents()
        for d in docs:
            meta_str = d.get("metadata", "{}")
            if isinstance(meta_str, str):
                try:
                    meta = json.loads(meta_str)
                except json.JSONDecodeError:
                    meta = {}
            else:
                meta = meta_str or {}

            if meta.get("url"):
                urls.add(meta["url"].strip())
    except Exception:
        pass
    return urls


def _pick_web_candidates(question: str, need: int) -> List[Dict[str, str]]:
    q = (question or "").lower()
    scored = []
    for item in TRUSTED_PDFS:
        score = sum(1 for kw in item.get("keywords", []) if kw.lower() in q)
        scored.append((score, item))
    scored.sort(reverse=True, key=lambda x: x[0])
    out = [it for sc, it in scored if sc > 0]
    return out[:need]


def _auto_ingest_from_web(question: str, min_docs: int = 1) -> int:
    existing_urls = _existing_urls_set()
    candidates = _pick_web_candidates(question, min_docs)
    count = 0
    for cand in candidates:
        url = cand.get("url")
        if not url or url in existing_urls:
            continue
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and "pdf" in r.headers.get("Content-Type", "").lower():
                doc_id = insert_document(cand["titulo"], cand.get("entidad"), None,
                                         json.dumps({"tipo": "pdf", "url": url}))
                ORIG_DIR.joinpath(f"doc_{doc_id}.pdf").write_bytes(r.content)
                reader = PdfReader(BytesIO(r.content))
                txt = "".join([(p.extract_text() or "") + "\n" for p in reader.pages])
                chunks = split_text(txt)
                if not chunks: continue
                embs = np.asarray(embed_texts(chunks), dtype=np.float32)
                insert_chunks(doc_id, chunks, embs)
                existing_urls.add(url)
                count += 1
                if count >= min_docs: break
        except Exception:
            continue
    return count


def create_synthetic_doc(question: str, answer_text: str) -> int:
    titulo = "Nota sin fuentes (autogenerada)"
    doc_id = insert_document(titulo, "Sistema", None, json.dumps({"tipo": "nota", "autogenerado": True, "q": question}))
    chunks = split_text(answer_text or "Respuesta generada sin fuentes.")
    if not chunks: chunks = [answer_text or "Respuesta generada sin fuentes."]
    embs = np.asarray(embed_texts(chunks), dtype=np.float32)
    insert_chunks(doc_id, chunks, embs)
    return doc_id


# =========================
# FastAPI App
# =========================
app = FastAPI(title="Dynamic RAG Assistant", version="1.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])

EMBED_UI = r"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Sistema LLM - Consulta SECOP II | Colombia Compra Eficiente</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    :root {
      --primary: #2563eb;
      --primary-dark: #1d4ed8;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --bg: #f8fafc;
      --bg-dark: #0f172a;
      --card: #ffffff;
      --text: #1e293b;
      --text-muted: #64748b;
      --border: #e2e8f0;
      --shadow: rgba(0,0,0,0.1);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      padding: 20px;
    }

    .container {
      max-width: 1200px;
      margin: 0 auto;
    }

    .header {
      background: white;
      border-radius: 16px;
      padding: 24px 32px;
      margin-bottom: 24px;
      box-shadow: 0 4px 6px var(--shadow);
    }

    .header h1 {
      font-size: 28px;
      color: var(--primary);
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .header p {
      color: var(--text-muted);
      font-size: 14px;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 12px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: 600;
      background: #e0e7ff;
      color: var(--primary);
    }

    .badge.success {
      background: #d1fae5;
      color: var(--success);
    }

    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin-bottom: 24px;
    }

    @media (max-width: 768px) {
      .grid { grid-template-columns: 1fr; }
    }

    .card {
      background: white;
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 4px 6px var(--shadow);
    }

    .card h2 {
      font-size: 18px;
      color: var(--text);
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .tabs {
      display: flex;
      gap: 8px;
      margin-bottom: 20px;
      border-bottom: 2px solid var(--border);
      padding-bottom: 2px;
    }

    .tab {
      padding: 10px 20px;
      border: none;
      background: none;
      color: var(--text-muted);
      cursor: pointer;
      font-weight: 500;
      border-bottom: 2px solid transparent;
      margin-bottom: -2px;
      transition: all 0.2s;
    }

    .tab.active {
      color: var(--primary);
      border-bottom-color: var(--primary);
    }

    .tab:hover {
      color: var(--primary);
    }

    .tab-content {
      display: none;
    }

    .tab-content.active {
      display: block;
    }

    label {
      display: block;
      font-size: 13px;
      font-weight: 600;
      color: var(--text);
      margin-bottom: 8px;
    }

    textarea, input, select {
      width: 100%;
      padding: 12px 16px;
      border: 1px solid var(--border);
      border-radius: 8px;
      font-size: 14px;
      font-family: inherit;
      transition: border-color 0.2s;
    }

    textarea:focus, input:focus, select:focus {
      outline: none;
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }

    textarea {
      min-height: 120px;
      resize: vertical;
    }

    textarea[readonly] {
      background: #f8fafc;
      cursor: default;
    }

    button {
      background: var(--primary);
      color: white;
      border: none;
      border-radius: 8px;
      padding: 12px 24px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }

    button:hover {
      background: var(--primary-dark);
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }

    button:active {
      transform: translateY(0);
    }

    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .btn-secondary {
      background: white;
      color: var(--primary);
      border: 1px solid var(--primary);
    }

    .btn-secondary:hover {
      background: var(--primary);
      color: white;
    }

    .result-card {
      background: #f8fafc;
      border-left: 4px solid var(--primary);
      border-radius: 8px;
      padding: 16px;
      margin-top: 16px;
    }

    .result-card h4 {
      font-size: 14px;
      color: var(--text);
      margin-bottom: 8px;
    }

    .result-card p {
      font-size: 13px;
      color: var(--text-muted);
      line-height: 1.6;
    }

    .stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-top: 16px;
    }

    .stat-box {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 16px;
      border-radius: 8px;
      text-align: center;
    }

    .stat-box h3 {
      font-size: 28px;
      margin-bottom: 4px;
    }

    .stat-box p {
      font-size: 12px;
      opacity: 0.9;
    }

    .examples {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }

    .example-chip {
      background: #f1f5f9;
      color: var(--text);
      padding: 6px 12px;
      border-radius: 16px;
      font-size: 12px;
      cursor: pointer;
      transition: all 0.2s;
      border: 1px solid transparent;
    }

    .example-chip:hover {
      background: var(--primary);
      color: white;
      transform: translateY(-2px);
    }

    .loading {
      display: inline-block;
      width: 16px;
      height: 16px;
      border: 2px solid rgba(255,255,255,0.3);
      border-top-color: white;
      border-radius: 50%;
      animation: spin 0.6s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .info-badge {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 11px;
      margin-top: 8px;
    }

    .info-badge.secop {
      background: #fef3c7;
      color: #92400e;
    }

    .info-badge.rag {
      background: #dbeafe;
      color: #1e40af;
    }

    a {
      color: var(--primary);
      text-decoration: none;
      font-weight: 500;
    }

    a:hover {
      text-decoration: underline;
    }

    .footer {
      text-align: center;
      color: white;
      margin-top: 32px;
      font-size: 13px;
      opacity: 0.9;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>
        🤖 Sistema LLM - Consulta SECOP II
        <span class="badge success">● En línea</span>
      </h1>
      <p>Asistente inteligente para consultas sobre contratación pública en Colombia</p>
    </div>

    <div class="grid">
      <!-- Panel de Consulta -->
      <div class="card" style="grid-column: 1 / -1;">
        <h2>💬 Consulta Inteligente (RAG + SECOP II)</h2>

        <div class="tabs">
          <button class="tab active" data-tab="ask">Preguntar</button>
          <button class="tab" data-tab="contracts">Contratos SECOP II</button>
          <button class="tab" data-tab="rag">Base RAG</button>
          <button class="tab" data-tab="stats">Estadísticas</button>
          <button class="tab" data-tab="tests">Pruebas</button>
        </div>

        <!-- Tab: Preguntar -->
        <div class="tab-content active" id="ask-content">
          <label>Escribe tu pregunta en lenguaje natural</label>
          <textarea id="q" placeholder="Ejemplo: ¿Cuáles son los requisitos habilitantes para licitar?"></textarea>

          <div class="examples">
            <span class="example-chip" onclick="setExample('¿Cuáles son los requisitos habilitantes?')">📋 Requisitos habilitantes</span>
            <span class="example-chip" onclick="setExample('¿Cuántos contratos de tecnología tiene el SENA?')">💻 Contratos SENA</span>
            <span class="example-chip" onclick="setExample('¿Qué garantías se necesitan en obra pública?')">🏗️ Garantías obra</span>
            <span class="example-chip" onclick="setExample('¿Cómo se evalúan las propuestas?')">⚖️ Evaluación</span>
          </div>

          <div style="margin-top:16px">
            <button id="askBtn" onclick="ask()">
              <span id="askIcon">🔍</span> Consultar
            </button>
          </div>

          <div id="answerBox" style="display:none; margin-top:20px">
            <label>Respuesta</label>
            <textarea id="answer" readonly></textarea>

            <div id="sourceInfo"></div>
          </div>
        </div>

        <!-- Tab: Contratos -->
        <div class="tab-content" id="contracts-content">
          <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px; margin-bottom:16px">
            <div>
              <label>Entidad</label>
              <input type="text" id="entity" placeholder="Ej: SENA, Ministerio de Educación">
            </div>
            <div>
              <label>Objeto del contrato</label>
              <input type="text" id="objeto" placeholder="Ej: tecnología, obra, software">
            </div>
          </div>

          <button onclick="searchContracts()">🔎 Buscar Contratos</button>

          <div id="contractsResult"></div>
        </div>

        <!-- Tab: Base RAG -->
        <div class="tab-content" id="rag-content">
          <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:12px; margin-bottom:20px">
            <div class="stat-box" id="ragStatContratos" style="text-align:center; padding:16px">
              <h3 style="margin:0; font-size:28px">-</h3>
              <p style="margin:4px 0 0 0; font-size:12px">Contratos</p>
            </div>
            <div class="stat-box" id="ragStatEmb" style="text-align:center; padding:16px">
              <h3 style="margin:0; font-size:28px">-</h3>
              <p style="margin:4px 0 0 0; font-size:12px">Con Embeddings</p>
            </div>
            <div class="stat-box" id="ragStatTotal" style="text-align:center; padding:16px">
              <h3 style="margin:0; font-size:28px">-</h3>
              <p style="margin:4px 0 0 0; font-size:12px">Total Embeddings</p>
            </div>
          </div>

          <div style="background:#f8fafc; padding:16px; border-radius:8px; margin-bottom:20px">
            <h4 style="margin:0 0 12px 0">Cargar Contratos desde SECOP II</h4>
            <div style="display:grid; grid-template-columns: 1fr 1fr 100px; gap:12px">
              <input type="text" id="ragEntidad" placeholder="Entidad (opcional)">
              <input type="text" id="ragObjeto" placeholder="Objeto (opcional)">
              <input type="number" id="ragLimite" value="100" min="1" max="1000" style="width:100%">
            </div>
            <button onclick="cargarContratosRAG()" style="margin-top:12px">📥 Cargar Contratos</button>
            <span id="ragCargaStatus" style="margin-left:12px; color:var(--text-muted)"></span>
          </div>

          <h4 style="margin-bottom:12px">Contratos en Base de Datos</h4>
          <div id="ragContratosList" style="max-height:400px; overflow-y:auto"></div>
        </div>

        <!-- Tab: Estadísticas -->
        <div class="tab-content" id="stats-content">
          <label>Entidad</label>
          <input type="text" id="statsEntity" placeholder="Ej: SENA">

          <div style="margin-top:12px">
            <button onclick="getStats()">📊 Obtener Estadísticas</button>
          </div>

          <div id="statsResult"></div>
        </div>

        <!-- Tab: Pruebas -->
        <div class="tab-content" id="tests-content">
          <div style="margin-bottom:20px">
            <p style="color: var(--text-secondary); margin-bottom:12px">
              Resultados de las pruebas sistemáticas del sistema
            </p>
            <button onclick="loadTests()">🔄 Cargar Resultados de Pruebas</button>
          </div>

          <div id="testsResult"></div>
        </div>
      </div>
    </div>

    <div class="footer">
      <p>🤖 Powered by GPT-4o-mini | 🔗 SECOP II Data | 📚 Colombia Compra Eficiente</p>
      <p style="margin-top:8px; opacity:0.7">Documentación: <a href="/database" style="color:white">Ver Base de Datos</a></p>
    </div>
  </div>

  <script>
    const $ = s => document.querySelector(s);
    const $$ = s => document.querySelectorAll(s);

    // Tab switching
    $$('.tab').forEach(tab => {
      tab.addEventListener('click', () => {
        const tabName = tab.dataset.tab;
        $$('.tab').forEach(t => t.classList.remove('active'));
        $$('.tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        $(`#${tabName}-content`).classList.add('active');
      });
    });

    function setExample(text) {
      $('#q').value = text;
    }

    async function call(path, opt = {}) {
      const r = await fetch(path, {headers: {"Content-Type": "application/json"}, ...opt});
      const text = await r.text();
      let data;
      try { data = JSON.parse(text); } catch { data = text; }
      return {ok: r.ok, data};
    }

    async function ask() {
      const btn = $('#askBtn');
      const icon = $('#askIcon');
      const answerBox = $('#answerBox');
      const answerEl = $('#answer');
      const sourceInfo = $('#sourceInfo');

      if (!$('#q').value.trim()) {
        alert('Por favor escribe una pregunta');
        return;
      }

      btn.disabled = true;
      icon.innerHTML = '<span class="loading"></span>';
      answerBox.style.display = 'block';
      answerEl.value = '🔍 Buscando en documentos y SECOP II...';
      sourceInfo.innerHTML = '';

      const payload = {query: $('#q').value, top_k: 1};
      const r = await call('/ask', {method: 'POST', body: JSON.stringify(payload)});

      btn.disabled = false;
      icon.innerHTML = '🔍';

      if (!r.ok || !r.data.ok) {
        answerEl.value = '❌ Error al procesar la consulta';
        return;
      }

      answerEl.value = r.data.answer || 'Sin respuesta';

      // Mostrar fuentes
      let sources = '';
      const match = r.data.matches?.[0];
      if (match) {
        sources += `<div class="result-card">
          <h4>📄 ${match.titulo}</h4>
          <p>${match.text_preview.substring(0, 200)}...</p>
          <div style="margin-top:8px">
            <a href="/download?doc_id=${match.doc_id}&q=${encodeURIComponent($('#q').value)}&a=${encodeURIComponent(answerEl.value)}" target="_blank">📥 Descargar PDF</a>
            <span class="info-badge rag">🎯 Similitud: ${(match.score * 100).toFixed(1)}%</span>
          </div>
        </div>`;
      }

      if (r.data.secop_data_included) {
        sources += `<span class="info-badge secop">✅ Datos de SECOP II incluidos (${r.data.secop_contracts_count} contratos)</span>`;
      }

      sourceInfo.innerHTML = sources;
    }

    async function searchContracts() {
      const entity = $('#entity').value;
      const objeto = $('#objeto').value;
      const result = $('#contractsResult');

      result.innerHTML = '<p style="margin-top:16px">🔄 Buscando...</p>';

      const params = new URLSearchParams();
      if (entity) params.append('entidad', entity);
      if (objeto) params.append('objeto', objeto);
      params.append('limite', '10');

      const r = await call(`/secop/contratos?${params}`);

      if (!r.ok || !r.data.contratos) {
        result.innerHTML = '<p style="color:red; margin-top:16px">❌ Error al buscar contratos</p>';
        return;
      }

      const contracts = r.data.contratos;
      if (contracts.length === 0) {
        result.innerHTML = '<p style="margin-top:16px">📭 No se encontraron contratos</p>';
        return;
      }

      let html = `<p style="margin-top:16px; font-weight:600">Encontrados: ${r.data.total} contratos</p>`;
      contracts.slice(0, 5).forEach((c, i) => {
        html += `<div class="result-card" style="margin-top:12px">
          <h4>${i + 1}. ${c.nombre_entidad}</h4>
          <p><strong>Objeto:</strong> ${c.descripcion_del_proceso?.substring(0, 150) || 'N/A'}...</p>
          <p><strong>Valor:</strong> $${parseInt(c.valor_del_contrato || 0).toLocaleString('es-CO')}</p>
          <p><strong>Modalidad:</strong> ${c.modalidad_de_contratacion || 'N/A'}</p>
        </div>`;
      });

      result.innerHTML = html;
    }

    async function getStats() {
      const entity = $('#statsEntity').value;
      const result = $('#statsResult');

      if (!entity.trim()) {
        alert('Por favor ingresa una entidad');
        return;
      }

      result.innerHTML = '<p style="margin-top:16px">📊 Calculando estadísticas...</p>';

      const r = await call(`/secop/estadisticas/${encodeURIComponent(entity)}`);

      if (!r.ok || r.data.error) {
        result.innerHTML = '<p style="color:red; margin-top:16px">❌ No se encontraron datos</p>';
        return;
      }

      const stats = r.data;
      let html = `<div class="stats">
        <div class="stat-box">
          <h3>${stats.total_contratos}</h3>
          <p>Contratos</p>
        </div>
        <div class="stat-box">
          <h3>$${(stats.monto_total / 1000000).toFixed(1)}M</h3>
          <p>Monto Total</p>
        </div>
        <div class="stat-box">
          <h3>$${(stats.monto_promedio / 1000000).toFixed(1)}M</h3>
          <p>Promedio</p>
        </div>
      </div>`;

      result.innerHTML = html;
    }

    async function loadTests() {
      const result = $('#testsResult');
      result.innerHTML = '<p style="margin-top:16px">⏳ Cargando resultados...</p>';

      const r = await call('/test/results');

      if (!r.ok || !r.data.ok) {
        result.innerHTML = `<p style="color:red; margin-top:16px">❌ ${r.data.error || 'Error al cargar resultados'}</p>`;
        return;
      }

      const data = r.data;
      const passed = data.passed || 0;
      const failed = data.failed || 0;
      const total = data.total || 0;
      const successRate = data.success_rate || 0;

      // Categorizar tests
      const functionalTests = data.tests.filter(t => t.test.startsWith('RF'));
      const nonFunctionalTests = data.tests.filter(t => t.test.startsWith('RNF'));
      const passedFunctional = functionalTests.filter(t => t.passed === true || (typeof t.passed === 'string' && t.passed.length > 0)).length;
      const passedNonFunctional = nonFunctionalTests.filter(t => t.passed === true || (typeof t.passed === 'string' && t.passed.length > 0)).length;

      let html = `
        <div style="margin-top:20px">
          <!-- Resumen Principal -->
          <div class="stats" style="margin-bottom:30px">
            <div class="stat-box" style="background: linear-gradient(135deg, #059669 0%, #10b981 100%)">
              <h3 style="color:white">${passed}/${total}</h3>
              <p style="color:rgba(255,255,255,0.9)">Pruebas Aprobadas</p>
            </div>
            <div class="stat-box" style="background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%)">
              <h3 style="color:white">${successRate.toFixed(1)}%</h3>
              <p style="color:rgba(255,255,255,0.9)">Tasa de Éxito</p>
            </div>
            <div class="stat-box" style="background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%)">
              <h3 style="color:white">${failed}</h3>
              <p style="color:rgba(255,255,255,0.9)">Pruebas Fallidas</p>
            </div>
          </div>

          <!-- Barra de Progreso -->
          <div style="margin-bottom:30px">
            <div style="display:flex; justify-content:space-between; margin-bottom:8px">
              <span style="font-weight:600; color:var(--text)">Progreso General</span>
              <span style="color:var(--text-secondary)">${passed} de ${total}</span>
            </div>
            <div style="background:#e5e7eb; border-radius:8px; height:24px; overflow:hidden">
              <div style="background:linear-gradient(90deg, #059669, #10b981); height:100%; width:${successRate}%; transition:width 0.5s; display:flex; align-items:center; justify-content:center; color:white; font-weight:600; font-size:12px">
                ${successRate.toFixed(1)}%
              </div>
            </div>
          </div>

          <!-- Gráfico de Categorías -->
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:30px">
            <div style="background:#f8fafc; border-radius:12px; padding:20px; border:1px solid #e5e7eb">
              <h4 style="margin:0 0 16px 0; color:var(--text); font-size:16px">📋 Funcionales (RF)</h4>
              <div style="text-align:center">
                <div style="position:relative; width:120px; height:120px; margin:0 auto">
                  <svg viewBox="0 0 36 36" style="transform:rotate(-90deg)">
                    <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      fill="none" stroke="#e5e7eb" stroke-width="3"/>
                    <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      fill="none" stroke="#10b981" stroke-width="3"
                      stroke-dasharray="${(passedFunctional/functionalTests.length*100).toFixed(1)}, 100"/>
                  </svg>
                  <div style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); font-size:20px; font-weight:700; color:#10b981">
                    ${passedFunctional}/${functionalTests.length}
                  </div>
                </div>
                <p style="margin-top:12px; color:var(--text-secondary); font-size:14px">
                  ${(passedFunctional/functionalTests.length*100).toFixed(1)}% aprobadas
                </p>
              </div>
            </div>

            <div style="background:#f8fafc; border-radius:12px; padding:20px; border:1px solid #e5e7eb">
              <h4 style="margin:0 0 16px 0; color:var(--text); font-size:16px">⚡ No Funcionales (RNF)</h4>
              <div style="text-align:center">
                <div style="position:relative; width:120px; height:120px; margin:0 auto">
                  <svg viewBox="0 0 36 36" style="transform:rotate(-90deg)">
                    <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      fill="none" stroke="#e5e7eb" stroke-width="3"/>
                    <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      fill="none" stroke="#3b82f6" stroke-width="3"
                      stroke-dasharray="${(passedNonFunctional/nonFunctionalTests.length*100).toFixed(1)}, 100"/>
                  </svg>
                  <div style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); font-size:20px; font-weight:700; color:#3b82f6">
                    ${passedNonFunctional}/${nonFunctionalTests.length}
                  </div>
                </div>
                <p style="margin-top:12px; color:var(--text-secondary); font-size:14px">
                  ${(passedNonFunctional/nonFunctionalTests.length*100).toFixed(1)}% aprobadas
                </p>
              </div>
            </div>
          </div>

          <!-- Detalles de Pruebas -->
          <div style="background:#f8fafc; border-radius:12px; padding:20px; border:1px solid #e5e7eb">
            <h4 style="margin:0 0 16px 0; color:var(--text); font-size:16px">📊 Detalle de Pruebas</h4>
            <div style="max-height:400px; overflow-y:auto">
              ${data.tests.map(test => {
                const isPassed = test.passed === true || (typeof test.passed === 'string' && test.passed.length > 0);
                const icon = isPassed ? '✅' : '❌';
                const color = isPassed ? '#10b981' : '#ef4444';
                const bgColor = isPassed ? '#f0fdf4' : '#fef2f2';

                return `
                  <div style="background:${bgColor}; border-left:4px solid ${color}; padding:12px; margin-bottom:8px; border-radius:6px">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px">
                      <span style="font-size:16px">${icon}</span>
                      <strong style="color:var(--text); font-size:14px">${test.test}</strong>
                    </div>
                    <p style="color:var(--text-secondary); font-size:13px; margin:0; padding-left:24px">
                      ${test.details || 'Sin detalles'}
                      ${test.metric ? ` (Métrica: ${test.metric.toFixed(3)})` : ''}
                    </p>
                  </div>
                `;
              }).join('')}
            </div>
          </div>

          <!-- Footer con timestamp -->
          <div style="margin-top:20px; text-align:center; color:var(--text-secondary); font-size:13px">
            <p>Última ejecución: ${new Date(data.timestamp).toLocaleString('es-CO')}</p>
          </div>
        </div>
      `;

      result.innerHTML = html;
    }

    // Permitir Enter para enviar
    $('#q').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && e.ctrlKey) {
        ask();
      }
    });

    // ========== Funciones RAG ==========
    async function loadRAGStats() {
      const r = await call('/rag/stats');
      if (r.ok && r.data.ok) {
        $('#ragStatContratos h3').textContent = r.data.total_contratos;
        $('#ragStatEmb h3').textContent = r.data.contratos_con_embeddings;
        $('#ragStatTotal h3').textContent = r.data.total_embeddings;
      }
    }

    async function loadRAGContratos() {
      const list = $('#ragContratosList');
      list.innerHTML = '<p>Cargando...</p>';

      const r = await call('/rag/contratos?limit=50');
      if (!r.ok || !r.data.ok) {
        list.innerHTML = '<p style="color:red">Error al cargar contratos</p>';
        return;
      }

      if (r.data.contratos.length === 0) {
        list.innerHTML = '<p style="color:var(--text-muted)">No hay contratos cargados. Use el formulario para cargar desde SECOP II.</p>';
        return;
      }

      let html = '';
      r.data.contratos.forEach(c => {
        const textoPreview = (c.texto_indexar || '').substring(0, 150);
        html += `
          <div class="result-card" style="margin-bottom:8px; cursor:pointer" onclick="verContratoRAG('${c.codigo_unico}')">
            <div style="display:flex; justify-content:space-between; align-items:center">
              <strong style="color:var(--primary)">${c.codigo_unico}</strong>
              <span style="font-size:11px; color:var(--text-muted)">${c.created_at || ''}</span>
            </div>
            <p style="margin-top:6px; font-size:13px">${textoPreview}...</p>
          </div>
        `;
      });

      list.innerHTML = html;
    }

    async function cargarContratosRAG() {
      const status = $('#ragCargaStatus');
      status.textContent = 'Cargando...';

      const entidad = $('#ragEntidad').value;
      const objeto = $('#ragObjeto').value;
      const limite = $('#ragLimite').value || 100;

      const params = new URLSearchParams();
      if (entidad) params.append('entidad', entidad);
      if (objeto) params.append('objeto', objeto);
      params.append('limite', limite);

      const r = await call(`/rag/cargar?${params}`, {method: 'POST'});

      if (r.ok && r.data.ok) {
        status.textContent = `Cargados ${r.data.cargados} contratos. Total: ${r.data.total_en_bd}`;
        status.style.color = 'var(--success)';
        loadRAGStats();
        loadRAGContratos();
      } else {
        status.textContent = r.data.error || 'Error al cargar';
        status.style.color = 'var(--danger)';
      }
    }

    async function verContratoRAG(codigo) {
      const r = await call(`/rag/contratos/${encodeURIComponent(codigo)}`);
      if (r.ok && r.data.ok) {
        const c = r.data.contrato;
        alert(`Código: ${c.codigo_unico}\n\nTexto a Indexar:\n${c.texto_indexar}\n\nJSON completo guardado en texto_total`);
      }
    }

    // Cargar datos RAG al hacer clic en la pestaña
    $$('.tab').forEach(tab => {
      tab.addEventListener('click', () => {
        if (tab.dataset.tab === 'rag') {
          loadRAGStats();
          loadRAGContratos();
        }
      });
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root(): return HTMLResponse(EMBED_UI)


@app.get("/ui", response_class=HTMLResponse)
def ui(): return HTMLResponse(EMBED_UI)


@app.get("/ping")
def ping(): return {"message": "pong", "db": DB_BACKEND, "llm": LLM_PROVIDER or "none"}


@app.get("/database", response_class=HTMLResponse)
def show_database():
    """Endpoint para mostrar el estado de la base de datos de forma visual"""
    docs = list_documents()

    total_chunks = 0
    docs_data = []

    for doc in docs:
        doc_id = doc.get("doc_id")
        # Contar chunks de este documento
        chunks_count = len([v for v in fetch_all_vectors() if v[1] == doc_id])
        total_chunks += chunks_count

        metadata = doc.get("metadata", "{}")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}

        docs_data.append({
            "doc_id": doc_id,
            "titulo": doc.get("titulo"),
            "entidad": doc.get("entidad"),
            "num_chunks": chunks_count,
            "tipo": metadata.get("tipo", "pdf"),
            "url": metadata.get("url", "N/A")
        })

    promedio = round(total_chunks / len(docs), 1) if docs else 0

    # Generar filas de la tabla
    table_rows = ""
    for doc in docs_data:
        titulo_display = doc['titulo'][:60] + "..." if len(doc['titulo']) > 60 else doc['titulo']
        entidad_display = doc['entidad'] or 'Sin entidad'
        url_cell = f'<a href="{doc["url"]}" target="_blank" class="url-link">🔗 Ver fuente</a>' if doc['url'] != 'N/A' else '<span style="color:#9ca3af">Sin URL</span>'

        table_rows += f"""
              <tr>
                <td><strong>ID {doc['doc_id']}</strong></td>
                <td>
                  <div class="doc-title">{titulo_display}</div>
                  <div class="doc-entity">{entidad_display}</div>
                </td>
                <td>
                  <span class="type-badge type-{doc['tipo']}">{doc['tipo']}</span>
                </td>
                <td>
                  <span class="chunk-badge">{doc['num_chunks']} chunks</span>
                </td>
                <td>
                  {url_cell}
                </td>
              </tr>
        """

    # Generar HTML visual
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Base de Datos - RAG System</title>
  <style>
    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }}

    :root {{
      --primary: #2563eb;
      --primary-dark: #1e40af;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --text: #1f2937;
      --text-secondary: #6b7280;
      --bg-light: #f8fafc;
      --border: #e5e7eb;
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      padding: 40px 20px;
    }}

    .container {{
      max-width: 1200px;
      margin: 0 auto;
    }}

    .header {{
      background: white;
      border-radius: 16px;
      padding: 32px;
      box-shadow: 0 10px 40px rgba(0,0,0,0.1);
      margin-bottom: 30px;
      text-align: center;
    }}

    .header h1 {{
      font-size: 32px;
      color: var(--text);
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 12px;
    }}

    .header p {{
      color: var(--text-secondary);
      font-size: 16px;
    }}

    .badge {{
      display: inline-block;
      padding: 4px 12px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: 600;
    }}

    .badge.success {{
      background: #d1fae5;
      color: #065f46;
    }}

    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 20px;
      margin-bottom: 30px;
    }}

    .stat-box {{
      background: white;
      border-radius: 12px;
      padding: 24px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.08);
      transition: transform 0.2s, box-shadow 0.2s;
    }}

    .stat-box:hover {{
      transform: translateY(-4px);
      box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }}

    .stat-box h3 {{
      font-size: 36px;
      color: var(--text);
      margin-bottom: 8px;
    }}

    .stat-box p {{
      color: var(--text-secondary);
      font-size: 14px;
      font-weight: 500;
    }}

    .stat-box.primary {{
      background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
    }}

    .stat-box.primary h3,
    .stat-box.primary p {{
      color: white;
    }}

    .stat-box.success {{
      background: linear-gradient(135deg, #059669 0%, var(--success) 100%);
    }}

    .stat-box.success h3,
    .stat-box.success p {{
      color: white;
    }}

    .stat-box.warning {{
      background: linear-gradient(135deg, #d97706 0%, var(--warning) 100%);
    }}

    .stat-box.warning h3,
    .stat-box.warning p {{
      color: white;
    }}

    .card {{
      background: white;
      border-radius: 16px;
      padding: 32px;
      box-shadow: 0 10px 40px rgba(0,0,0,0.1);
      margin-bottom: 30px;
    }}

    .card h2 {{
      font-size: 24px;
      color: var(--text);
      margin-bottom: 24px;
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .table-container {{
      overflow-x: auto;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
    }}

    thead {{
      background: var(--bg-light);
    }}

    th {{
      padding: 16px;
      text-align: left;
      font-weight: 600;
      color: var(--text);
      font-size: 14px;
      border-bottom: 2px solid var(--border);
    }}

    td {{
      padding: 16px;
      color: var(--text-secondary);
      border-bottom: 1px solid var(--border);
      font-size: 14px;
    }}

    tbody tr {{
      transition: background 0.2s;
    }}

    tbody tr:hover {{
      background: var(--bg-light);
    }}

    .doc-title {{
      color: var(--text);
      font-weight: 600;
      margin-bottom: 4px;
    }}

    .doc-entity {{
      color: var(--text-secondary);
      font-size: 13px;
    }}

    .chunk-badge {{
      display: inline-block;
      background: #dbeafe;
      color: #1e40af;
      padding: 4px 10px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 600;
    }}

    .type-badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
    }}

    .type-pdf {{
      background: #fef3c7;
      color: #92400e;
    }}

    .type-nota {{
      background: #e0e7ff;
      color: #3730a3;
    }}

    .url-link {{
      color: var(--primary);
      text-decoration: none;
      font-size: 13px;
    }}

    .url-link:hover {{
      text-decoration: underline;
    }}

    .footer {{
      text-align: center;
      padding: 20px;
      color: white;
      margin-top: 40px;
    }}

    .footer a {{
      color: white;
      text-decoration: none;
      font-weight: 600;
    }}

    .footer a:hover {{
      text-decoration: underline;
    }}

    .empty-state {{
      text-align: center;
      padding: 60px 20px;
      color: var(--text-secondary);
    }}

    .empty-state svg {{
      width: 80px;
      height: 80px;
      margin-bottom: 20px;
      opacity: 0.3;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>
        🗄️ Base de Datos RAG
        <span class="badge success">● Activa</span>
      </h1>
      <p>Visualización del estado de la base de datos vectorial</p>
    </div>

    <div class="stats">
      <div class="stat-box primary">
        <h3>{len(docs)}</h3>
        <p>📄 Documentos</p>
      </div>
      <div class="stat-box success">
        <h3>{total_chunks}</h3>
        <p>🧩 Chunks Totales</p>
      </div>
      <div class="stat-box warning">
        <h3>{promedio}</h3>
        <p>📊 Promedio/Doc</p>
      </div>
      <div class="stat-box">
        <h3>{DB_BACKEND.upper()}</h3>
        <p>💾 Backend</p>
      </div>
    </div>

    <div class="card">
      <h2>📚 Documentos Procesados</h2>

      {f'''<div class="table-container">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Documento</th>
              <th>Tipo</th>
              <th>Chunks</th>
              <th>URL/Fuente</th>
            </tr>
          </thead>
          <tbody>
            {table_rows}
          </tbody>
        </table>
      </div>''' if docs else '''
        <div class="empty-state">
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
          </svg>
          <h3>No hay documentos procesados</h3>
          <p>Ingesta documentos para comenzar a usar el sistema RAG</p>
        </div>
      '''}
    </div>

    <div class="footer">
      <p>
        🤖 Sistema RAG | Backend: <strong>{DB_BACKEND}</strong> | LLM: <strong>{LLM_PROVIDER or 'openai'}</strong>
      </p>
      <p style="margin-top:8px">
        <a href="/">← Volver al Dashboard Principal</a>
      </p>
    </div>
  </div>
</body>
</html>
"""

    return html


# =========================
# Endpoints SECOP II
# =========================
@app.get("/secop/contratos")
def consultar_contratos_secop(
    entidad: Optional[str] = Query(None, description="Nombre de la entidad"),
    objeto: Optional[str] = Query(None, description="Objeto del contrato"),
    fecha_desde: Optional[str] = Query(None, description="Fecha desde (YYYY-MM-DD)"),
    fecha_hasta: Optional[str] = Query(None, description="Fecha hasta (YYYY-MM-DD)"),
    limite: int = Query(50, le=1000, description="Límite de resultados")
):
    """
    Consulta contratos reales del SECOP II

    Ejemplos:
    - /secop/contratos?entidad=Bogotá
    - /secop/contratos?objeto=tecnología&limite=10
    - /secop/contratos?fecha_desde=2024-01-01&fecha_hasta=2024-12-31
    """
    contratos = buscar_contratos(
        entidad=entidad,
        objeto_contratar=objeto,
        fecha_desde=fecha_desde,
        fecha_hasta=fecha_hasta,
        limite=limite
    )

    return {
        "total": len(contratos),
        "filtros": {
            "entidad": entidad,
            "objeto": objeto,
            "fecha_desde": fecha_desde,
            "fecha_hasta": fecha_hasta
        },
        "contratos": contratos
    }


@app.get("/secop/estadisticas/{entidad}")
def estadisticas_entidad(entidad: str):
    """
    Obtiene estadísticas de contratación de una entidad

    Ejemplo: /secop/estadisticas/SENA
    """
    return obtener_estadisticas_entidad(entidad)


@app.get("/secop/proveedores")
def proveedores_por_sector(sector: str = Query(..., description="Sector o palabra clave")):
    """
    Busca proveedores activos en un sector específico

    Ejemplo: /secop/proveedores?sector=software
    """
    proveedores = buscar_proveedores_por_sector(sector)

    return {
        "sector": sector,
        "total_proveedores": len(proveedores),
        "proveedores": proveedores[:20]  # Limitar a top 20
    }


# =========================
# Endpoints Contratos RAG
# =========================
@app.get("/rag/contratos")
def listar_contratos_rag(
    limit: int = Query(50, le=500),
    offset: int = Query(0)
):
    """Lista contratos cargados en el sistema RAG"""
    contratos = list_contratos(limit=limit, offset=offset)
    total = count_contratos()
    return {
        "ok": True,
        "total": total,
        "limit": limit,
        "offset": offset,
        "contratos": contratos
    }


@app.get("/rag/contratos/{codigo_unico}")
def obtener_contrato_rag(codigo_unico: str):
    """Obtiene un contrato por su código único"""
    contrato = get_contrato_by_codigo(codigo_unico)
    if not contrato:
        raise HTTPException(status_code=404, detail="Contrato no encontrado")
    return {"ok": True, "contrato": contrato}


@app.post("/rag/cargar")
def cargar_contratos_rag(
    entidad: Optional[str] = Query(None),
    objeto: Optional[str] = Query(None),
    limite: int = Query(100, le=1000)
):
    """Carga contratos desde SECOP II al sistema RAG"""
    contratos = buscar_contratos(
        entidad=entidad,
        objeto_contratar=objeto,
        limite=limite
    )

    if not contratos:
        return {"ok": False, "error": "No se encontraron contratos", "cargados": 0}

    cargados = 0
    for i, contrato in enumerate(contratos, 1):
        try:
            insert_contrato(contrato, i)
            cargados += 1
        except Exception:
            pass

    return {
        "ok": True,
        "encontrados": len(contratos),
        "cargados": cargados,
        "total_en_bd": count_contratos()
    }


@app.get("/rag/stats")
def estadisticas_rag():
    """Estadísticas del sistema RAG"""
    total_contratos = count_contratos()
    embeddings = fetch_all_contrato_embeddings()
    codigos_con_emb = len(set(e[0] for e in embeddings))

    return {
        "ok": True,
        "total_contratos": total_contratos,
        "contratos_con_embeddings": codigos_con_emb,
        "total_embeddings": len(embeddings)
    }


@app.get("/test/results")
def obtener_resultados_pruebas():
    """
    Devuelve los resultados de las pruebas sistemáticas desde test_results.json
    """
    try:
        test_file = Path(__file__).parent / "test_results.json"
        if not test_file.exists():
            return {
                "ok": False,
                "error": "No se han ejecutado pruebas aún. Ejecuta: ./venv/bin/python test_suite.py"
            }

        with open(test_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        return {
            "ok": True,
            **results
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }


# =========================
# LLM Helpers
# =========================
def build_context_for_answer(sims: List[Tuple[float, int, int, str, str]], max_chars: int = 4000) -> str:
    paras, seen = [], set()
    for _, _doc, _ord, text, _tit in sims:
        for p in (text or "").split("\n"):
            p = p.strip()
            if not p: continue
            key = " ".join(p.split())
            if key in seen: continue
            seen.add(key)
            paras.append(p)
            if sum(len(x) + 1 for x in paras) > max_chars:
                return "\n".join(paras)
    return "\n".join(paras)


def answer_with_openai(question: str, context: str) -> Optional[str]:
    if not OPENAI_API_KEY: return None
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "messages": [
                {"role": "system",
                 "content": "Responde en español de forma breve, directa y sustentada SOLO en el contexto dado. Si falta información, dilo y no inventes."},
                {"role": "user", "content": f"Pregunta: {question}\n\nContexto:\n{context}"}
            ]
        }
        r = requests.post(url, headers=headers, json=payload, timeout=45)
        if r.status_code == 200:
            return (r.json()["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return None
    return None


def heuristic_answer(question: str, sims: List[Tuple[float, int, int, str, str]]) -> str:
    ctx = build_context_for_answer(sims, max_chars=1200)
    sent, seen = [], set()
    for part in ctx.split("\n"):
        p = (part or "").strip()
        if not p: continue
        key = " ".join(p.split()).lower()
        if key in seen: continue
        seen.add(key)
        sent.append(p)
        if len(sent) >= 5: break
    if not sent:
        return "No encontré texto suficiente en la fuente para responder. Por favor, verifica el documento original."
    return " ".join(sent)


# =========================
# LÓGICA DE RESPUESTA PRINCIPAL
# =========================
class AskIn(BaseModel):
    query: str
    top_k: int = 1


@app.post("/ask")
def ask_ep(payload: AskIn):
    try:
        q = (payload.query or "").strip()
        if not q:
            return {"ok": True, "matches": [], "answer": "Por favor, escribe una pregunta."}

        _auto_ingest_from_web(q, min_docs=1)

        items = fetch_all_vectors()
        if not items:
            msg = "No hay documentos en la base de datos para responder. La búsqueda automática no encontró fuentes relevantes."
            return {"ok": True, "matches": [], "answer": msg}

        qvec = np.asarray(embed_text(q), dtype=np.float32)
        q_norm = np.linalg.norm(qvec) + 1e-9
        sims = []
        for _cid, doc_id, ord_, text, emb, titulo in items:
            emb_arr = np.asarray(emb, dtype=np.float32)
            sim = np.dot(qvec, emb_arr) / (q_norm * (np.linalg.norm(emb_arr) + 1e-9))
            sims.append((float(sim), doc_id, ord_, text, titulo))
        sims.sort(reverse=True, key=lambda x: x[0])

        if not sims:
            return {"ok": True, "matches": [], "answer": "No se encontraron resultados relevantes en los documentos."}

        best_doc_id = sims[0][1]
        top_doc_chunks = [s for s in sims if s[1] == best_doc_id][:10]

        # Construir contexto de documentos guía
        context_text = build_context_for_answer(top_doc_chunks)

        # Detectar si la pregunta requiere datos de SECOP II
        q_lower = q.lower()
        keywords_datos = ["cuánto", "cuántos", "cuantos", "estadística", "estadistica",
                         "contratos de", "gasto", "gastó", "empresas que", "proveedores"]
        necesita_datos_secop = any(kw in q_lower for kw in keywords_datos)

        # Agregar contexto de SECOP II si es necesario
        context_secop = ""
        if necesita_datos_secop:
            try:
                # Extraer palabras clave para buscar
                contratos = []
                if "sena" in q_lower:
                    contratos = buscar_contratos(entidad="SENA", limite=5)
                elif "tecnología" in q_lower or "tecnologia" in q_lower or "software" in q_lower:
                    contratos = buscar_contratos(objeto_contratar="tecnología", limite=5)
                elif "obra" in q_lower or "construcción" in q_lower:
                    contratos = buscar_contratos(objeto_contratar="obra", limite=5)
                else:
                    # Búsqueda general
                    contratos = buscar_contratos(limite=5)

                if contratos:
                    context_secop = "\n\n=== DATOS RECIENTES DE SECOP II ===\n"
                    for i, c in enumerate(contratos[:3], 1):
                        entidad = c.get('nombre_entidad', 'N/A')
                        objeto = c.get('descripcion_del_proceso', 'N/A')[:100]
                        valor = c.get('valor_del_contrato', 'N/A')
                        context_secop += f"\n{i}. Entidad: {entidad}\n   Objeto: {objeto}\n   Valor: ${valor}\n"
            except Exception:
                pass

        # Contexto completo
        full_context = context_text + context_secop

        # Generar respuesta
        answer = None
        if LLM_PROVIDER == "openai":
            answer = answer_with_openai(q, full_context)

        if not answer:
            answer = heuristic_answer(q, top_doc_chunks)

        best_chunk = top_doc_chunks[0]
        match = {
            "score": best_chunk[0],
            "doc_id": int(best_chunk[1]),
            "titulo": best_chunk[4],
            "chunk_ord": int(best_chunk[2]),
            "text_preview": (best_chunk[3] or "")[:600]
        }

        response = {
            "ok": True,
            "matches": [match],
            "answer": answer
        }

        # Indicar si se usaron datos de SECOP II
        if context_secop:
            response["secop_data_included"] = True
            response["secop_contracts_count"] = len(contratos)

        return response

    except Exception as e:
        msg = f"Ocurrió un error inesperado: {type(e).__name__}"
        try:
            doc_id = create_synthetic_doc(payload.query, msg)
            match = {"score": 0.0, "doc_id": doc_id, "titulo": "Nota de Error", "chunk_ord": 0, "text_preview": msg}
            return {"ok": True, "matches": [match], "answer": msg}
        except Exception:
            return {"ok": True, "matches": [], "answer": msg}


# =========================
# PDF Descargas
# =========================
def _original_path_for(doc_id: int) -> Path:
    return ORIG_DIR / f"doc_{doc_id}.pdf"


@app.get("/download")
def download(doc_id: int, q: Optional[str] = None, a: Optional[str] = None):
    row = get_document(doc_id)
    if not row: raise HTTPException(status_code=404, detail="Documento no encontrado.")

    local_path = _original_path_for(doc_id)
    if local_path.exists():
        return StreamingResponse(open(local_path, "rb"), media_type="application/pdf",
                                 headers={"Content-Disposition": f'attachment; filename="{local_path.name}"'})

    # Fallback si el PDF original no está, se reconstruye con el texto de la DB.
    full_text = fetch_doc_text(doc_id)
    pdf_bytes = build_pdf_bytes(row["titulo"], q, a, full_text)
    return StreamingResponse(BytesIO(pdf_bytes), media_type="application/pdf",
                             headers={"Content-Disposition": f'attachment; filename="Reconstruido_{doc_id}.pdf"'})


def build_pdf_bytes(title: str, q: Optional[str], a: Optional[str], full_text: str) -> bytes:
    """Genera un PDF real usando ReportLab."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)
    width, height = LETTER
    y = height - 50

    # Título
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, title or "Documento")
    y -= 30

    # Pregunta
    if q:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Pregunta:")
        y -= 20
        c.setFont("Helvetica", 10)
        lines = (q or "").split('\n')
        for line in lines[:3]:
            if y < 50:
                c.showPage()
                y = height - 50
            c.drawString(50, y, line[:80])
            y -= 15
        y -= 10

    # Respuesta
    if a:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Respuesta:")
        y -= 20
        c.setFont("Helvetica", 10)
        lines = (a or "").split('\n')
        for line in lines[:5]:
            if y < 50:
                c.showPage()
                y = height - 50
            c.drawString(50, y, line[:80])
            y -= 15
        y -= 10

    # Separador
    c.line(50, y, width - 50, y)
    y -= 20

    # Texto completo
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Contenido del Documento:")
    y -= 20
    c.setFont("Helvetica", 9)

    for line in (full_text or "").split('\n'):
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(50, y, line[:100])
        y -= 12

    c.save()
    buffer.seek(0)
    return buffer.read()


# =========================
# Ejecución
# =========================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8001, reload=True)