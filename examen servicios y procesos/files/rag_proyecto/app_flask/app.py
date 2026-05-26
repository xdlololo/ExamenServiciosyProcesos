# ============================================================
# EJERCICIO 3: RAG EMPAQUETADO - Interfaz Flask
# - El usuario puede subir sus propios documentos (.txt, .pdf)
# - Se procesan, se crean embeddings y se meten en ChromaDB
# - El usuario puede hacer preguntas desde la interfaz web
# - Las respuestas se generan con IA (Claude)
# ============================================================

import os
import uuid
import chromadb
from chromadb.utils import embedding_functions
from flask import Flask, render_template, request, jsonify, session
import anthropic

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ── Configuración ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "TU_API_KEY_AQUI")
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("./chromadb_flask", exist_ok=True)

# ── Cliente ChromaDB ──────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path="./chromadb_flask")
embedding_fn = embedding_functions.DefaultEmbeddingFunction()


# ── Utilidades ────────────────────────────────────────────────────────────────

def dividir_en_chunks(texto: str, tamanio: int = 100, solapamiento: int = 25) -> list[str]:
    """Divide texto en chunks con solapamiento."""
    palabras = texto.split()
    chunks, inicio = [], 0
    while inicio < len(palabras):
        chunk = " ".join(palabras[inicio:inicio + tamanio]).strip()
        if len(chunk) > 40:
            chunks.append(chunk)
        inicio += tamanio - solapamiento
    return chunks


def leer_archivo(filepath: str) -> str:
    """Lee el contenido de un archivo .txt o .pdf."""
    if filepath.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(filepath) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        except ImportError:
            return "Error: instala pdfplumber con 'pip install pdfplumber' para leer PDFs."
    else:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def obtener_o_crear_coleccion(nombre_coleccion: str) -> chromadb.Collection:
    """Obtiene una colección existente o la crea vacía."""
    try:
        return chroma_client.get_collection(
            name=nombre_coleccion,
            embedding_function=embedding_fn
        )
    except Exception:
        return chroma_client.create_collection(
            name=nombre_coleccion,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )


def generar_respuesta_ia(pregunta: str, chunks: list[str]) -> str:
    """Genera respuesta humanizada usando Claude."""
    if ANTHROPIC_API_KEY == "TU_API_KEY_AQUI":
        return "[Configura ANTHROPIC_API_KEY para obtener respuestas de IA]"

    cliente_ia = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    contexto = "\n\n---\n\n".join(chunks)

    prompt = f"""Eres un asistente experto. Se te proporcionan fragmentos de texto recuperados
de un documento del usuario mediante búsqueda semántica (RAG).

FRAGMENTOS RECUPERADOS DEL DOCUMENTO:
{contexto}

PREGUNTA DEL USUARIO:
{pregunta}

Responde de manera clara y natural usando ÚNICAMENTE la información proporcionada.
Si la información no es suficiente, indícalo honestamente."""

    respuesta = cliente_ia.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return respuesta.content[0].text


# ── Rutas Flask ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Página principal."""
    return render_template("index.html")


@app.route("/entrenar", methods=["POST"])
def entrenar():
    """
    Recibe un archivo o texto, lo divide en chunks
    y los mete en ChromaDB bajo una colección única del usuario.
    """
    # Crear ID de sesión único para este entrenamiento
    coleccion_id = f"rag_{uuid.uuid4().hex[:8]}"
    texto = ""

    # Opción A: texto pegado directamente
    if "texto" in request.form and request.form["texto"].strip():
        texto = request.form["texto"].strip()

    # Opción B: archivo subido
    elif "archivo" in request.files and request.files["archivo"].filename:
        archivo = request.files["archivo"]
        filename = archivo.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        archivo.save(filepath)
        texto = leer_archivo(filepath)

    if not texto:
        return jsonify({"error": "No se proporcionó texto ni archivo."}), 400

    # Dividir en chunks e indexar
    chunks = dividir_en_chunks(texto)
    if not chunks:
        return jsonify({"error": "El texto es demasiado corto."}), 400

    coleccion = obtener_o_crear_coleccion(coleccion_id)
    coleccion.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    session["coleccion_id"] = coleccion_id

    return jsonify({
        "ok": True,
        "coleccion_id": coleccion_id,
        "num_chunks": len(chunks),
        "preview": texto[:300] + "..." if len(texto) > 300 else texto
    })


@app.route("/preguntar", methods=["POST"])
def preguntar():
    """
    Recibe una pregunta, busca chunks relevantes en ChromaDB
    y devuelve: chunks crudos + respuesta generada por IA.
    """
    datos = request.get_json()
    pregunta = datos.get("pregunta", "").strip()
    coleccion_id = datos.get("coleccion_id") or session.get("coleccion_id")

    if not pregunta:
        return jsonify({"error": "No se proporcionó pregunta."}), 400
    if not coleccion_id:
        return jsonify({"error": "Primero debes entrenar el RAG con un documento."}), 400

    try:
        coleccion = chroma_client.get_collection(
            name=coleccion_id,
            embedding_function=embedding_fn
        )
    except Exception:
        return jsonify({"error": "Colección no encontrada. Vuelve a entrenar."}), 404

    # Buscar chunks relevantes
    resultados = coleccion.query(query_texts=[pregunta], n_results=3)
    chunks = resultados["documents"][0]
    distancias = resultados["distances"][0]

    chunks_info = [
        {"texto": chunk, "similitud": round((1 - dist) * 100, 1)}
        for chunk, dist in zip(chunks, distancias)
    ]

    # Generar respuesta con IA
    respuesta_ia = generar_respuesta_ia(pregunta, chunks)

    return jsonify({
        "ok": True,
        "pregunta": pregunta,
        "chunks": chunks_info,
        "respuesta_ia": respuesta_ia
    })


@app.route("/colecciones")
def listar_colecciones():
    """Lista todas las colecciones disponibles."""
    colecciones = chroma_client.list_collections()
    return jsonify({"colecciones": [c.name for c in colecciones]})


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  🚀 RAG Flask - Servidor iniciado")
    print("  📌 Abre: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
