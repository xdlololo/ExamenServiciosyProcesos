
# ============================================================
# EJERCICIO 3: RAG EMPAQUETADO - Interfaz Flask
# ============================================================
 
import os
import uuid
import traceback
 
import chromadb
from chromadb.utils import embedding_functions
from flask import Flask, render_template, request, jsonify, session
 
app = Flask(__name__)
app.secret_key = "clave-secreta-rag-2024"   # fija para que la sesión no se rompa al reiniciar
 
# ── Configuración ─────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
CHROMA_PATH   = os.path.join(os.path.dirname(__file__), "chromadb_flask")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_PATH,   exist_ok=True)
 
# ── ChromaDB (se inicializa una sola vez) ─────────────────────
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_fn  = embedding_functions.DefaultEmbeddingFunction()
 
 
# ── Utilidades ────────────────────────────────────────────────
 
def dividir_en_chunks(texto: str, tamanio: int = 100, solapamiento: int = 25):
    palabras = texto.split()
    chunks, inicio = [], 0
    while inicio < len(palabras):
        chunk = " ".join(palabras[inicio : inicio + tamanio]).strip()
        if len(chunk) > 40:
            chunks.append(chunk)
        inicio += tamanio - solapamiento
    return chunks
 
 
def leer_archivo(filepath: str) -> str:
    if filepath.lower().endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(filepath) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        except ImportError:
            return "ERROR_PDF"
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
 
 
def obtener_o_crear_coleccion(nombre: str):
    try:
        return chroma_client.get_collection(name=nombre, embedding_function=embedding_fn)
    except Exception:
        return chroma_client.create_collection(
            name=nombre,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
 
 
def generar_respuesta_ia(pregunta: str, chunks: list) -> str:
    if not ANTHROPIC_API_KEY:
        return None   # sin API key → solo chunks crudos
 
    try:
        import anthropic
        cliente = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        contexto = "\n\n---\n\n".join(chunks)
        prompt = (
            "Se te proporcionan fragmentos de un documento recuperados por búsqueda semántica.\n\n"
            f"FRAGMENTOS:\n{contexto}\n\n"
            f"PREGUNTA: {pregunta}\n\n"
            "Responde de forma clara y natural usando ÚNICAMENTE la información de los fragmentos. "
            "Si no hay suficiente información, indícalo."
        )
        resp = cliente.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    except Exception as e:
        return f"[Error al llamar a la IA: {e}]"
 
 
# ── Rutas ─────────────────────────────────────────────────────
 
@app.route("/")
def index():
    return render_template("index.html")
 
 
@app.route("/entrenar", methods=["POST"])
def entrenar():
    try:
        texto = ""
 
        # Opción A: texto pegado
        if request.form.get("texto", "").strip():
            texto = request.form["texto"].strip()
 
        # Opción B: archivo subido
        elif "archivo" in request.files and request.files["archivo"].filename:
            archivo  = request.files["archivo"]
            filepath = os.path.join(UPLOAD_FOLDER, archivo.filename)
            archivo.save(filepath)
            texto = leer_archivo(filepath)
            if texto == "ERROR_PDF":
                return jsonify({"error": "Para PDFs instala: pip install pdfplumber"}), 400
 
        if not texto:
            return jsonify({"error": "No se recibió texto ni archivo."}), 400
 
        chunks = dividir_en_chunks(texto)
        if not chunks:
            return jsonify({"error": "El texto es demasiado corto para indexar."}), 400
 
        coleccion_id = f"rag_{uuid.uuid4().hex[:8]}"
        coleccion    = obtener_o_crear_coleccion(coleccion_id)
        coleccion.add(
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))],
        )
 
        session["coleccion_id"] = coleccion_id
 
        return jsonify({
            "ok":           True,
            "coleccion_id": coleccion_id,
            "num_chunks":   len(chunks),
        })
 
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500
 
 
@app.route("/preguntar", methods=["POST"])
def preguntar():
    try:
        datos        = request.get_json(force=True) or {}
        pregunta     = datos.get("pregunta", "").strip()
        coleccion_id = datos.get("coleccion_id") or session.get("coleccion_id")
 
        if not pregunta:
            return jsonify({"error": "Escribe una pregunta."}), 400
        if not coleccion_id:
            return jsonify({"error": "Primero entrena el RAG con un documento."}), 400
 
        try:
            coleccion = chroma_client.get_collection(
                name=coleccion_id, embedding_function=embedding_fn
            )
        except Exception:
            return jsonify({"error": "Colección no encontrada. Vuelve a entrenar."}), 404
 
        resultados = coleccion.query(query_texts=[pregunta], n_results=3)
        chunks     = resultados["documents"][0]
        distancias = resultados["distances"][0]
 
        chunks_info = [
            {"texto": c, "similitud": round((1 - d) * 100, 1)}
            for c, d in zip(chunks, distancias)
        ]
 
        respuesta_ia = generar_respuesta_ia(pregunta, chunks)
 
        return jsonify({
            "ok":          True,
            "pregunta":    pregunta,
            "chunks":      chunks_info,
            "respuesta_ia": respuesta_ia,
        })
 
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500
 
 
@app.route("/colecciones")
def listar_colecciones():
    cols = chroma_client.list_collections()
    return jsonify({"colecciones": [c.name for c in cols]})
 
 
# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  🚀 RAG Flask — http://localhost:5000")
    if ANTHROPIC_API_KEY:
        print("  🤖 API Key de Anthropic detectada ✔")
    else:
        print("  ⚠  Sin API Key — solo chunks crudos (sin IA)")
    print("=" * 50)
    app.run(debug=True, port=5000)
 