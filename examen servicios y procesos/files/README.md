#  Proyecto RAG — Programación de Servicios y Procesos

Proyecto con 4 ejercicios sobre RAG (Retrieval Augmented Generation),
ChromaDB, embeddings y CSS 3D.

---

## 📁 Estructura del proyecto

```
rag_proyecto/
├── rag_basico.py              
├── rag_ia.py                  
├── app_flask/
│   ├── app.py                 
│   └── templates/
│       └── index.html         
├── css3d/
│   └── index.html            
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalación

### 1. Crear entorno virtual (recomendado)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Configurar API Key de Anthropic (para ejercicios 2 y 3)
```bash
# Windows (PowerShell):
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# Mac/Linux:
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

##  Ejecución

### Ejercicio 1 — RAG Básico
```bash
python rag_basico.py
```
**Qué hace:**
- Divide un texto largo en chunks con solapamiento
- Genera embeddings con `sentence-transformers`
- Los mete en ChromaDB (base de datos vectorial persistente)
- Hace queries semánticas y devuelve los chunks más similares

---

### Ejercicio 2 — RAG + IA
```bash
python rag_ia.py
```
**Qué hace:**
- Pasa los chunks recuperados a Claude (Anthropic)
- La IA genera una respuesta humana y comprensible
- **Requiere:** `ANTHROPIC_API_KEY` configurada

---

### Ejercicio 3 — RAG Empaquetado (Flask)
```bash
cd app_flask
python app.py
# Abrir en el navegador: http://localhost:5000
```
**Qué hace:**
- Interfaz web completa para que cualquier usuario entrene su propio RAG
- Permite pegar texto o subir archivos (.txt, .pdf)
- El usuario puede hacer preguntas desde la web
- Devuelve: chunks crudos + respuesta generada por IA
- **Requiere:** `ANTHROPIC_API_KEY` configurada (para la respuesta IA)

---

### Ejercicio 4 — CSS 3D
```bash
# Simplemente abrir en el navegador:
css3d/index.html
# O con Live Server en VSCode: clic derecho → "Open with Live Server"
```
**Qué hace:**
- Entorno 3D CSS puro con `transform-style: preserve-3d`
- Cubo central que representa ChromaDB / Knowledge Base
- 3 anillos orbitales con nodos temáticos (Docs, Query, LLM, Output, Chunks, Cosine)
- Partículas flotantes y streams de datos
- HUD estilo sci-fi con contadores en tiempo real
- **Interactivo:** arrastra con el ratón para rotar la escena

---

##  Conceptos implementados

| Concepto | Archivo |
|---|---|
| Chunking con solapamiento | `rag_basico.py` |
| Embeddings (sentence-transformers) | `rag_basico.py` |
| ChromaDB PersistentClient | `rag_basico.py` |
| Similitud coseno | `rag_basico.py` |
| Paso de chunks a LLM | `rag_ia.py` |
| Prompt engineering para RAG | `rag_ia.py` |
| API REST con Flask | `app_flask/app.py` |
| Upload de archivos | `app_flask/app.py` |
| Sesiones Flask | `app_flask/app.py` |
| CSS `transform-style: preserve-3d` | `css3d/index.html` |
| CSS `perspective` y `rotateX/Y/Z` | `css3d/index.html` |
| CSS `animation` keyframes | `css3d/index.html` |
| Drag-to-rotate con JS | `css3d/index.html` |
