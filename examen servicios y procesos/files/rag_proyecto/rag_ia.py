import os
import chromadb
from chromadb.utils import embedding_functions
import anthropic


ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "TU_API_KEY_AQUI")

TEXTO_LARGO = """
La inteligencia artificial (IA) es la simulación de procesos de inteligencia humana
por parte de máquinas, especialmente sistemas informáticos. Sus aplicaciones incluyen
el aprendizaje automático, el procesamiento del lenguaje natural y la visión artificial.

El aprendizaje automático es una rama de la inteligencia artificial que permite a los
sistemas aprender y mejorar automáticamente a partir de la experiencia sin ser
programados explícitamente. Se centra en el desarrollo de programas de computadora
que pueden acceder a datos y utilizarlos para aprender por sí mismos.

Las redes neuronales son sistemas de computación inspirados en las redes neuronales
biológicas del cerebro humano. Consisten en capas de nodos interconectados que
procesan información usando respuestas computacionales o de conexiones. Cada nodo
o neurona artificial puede transmitir una señal a otras neuronas.

El procesamiento del lenguaje natural (PLN) es una subdisciplina de la lingüística,
la informática y la inteligencia artificial que se ocupa de las interacciones entre
las computadoras y el lenguaje humano, en particular cómo programar computadoras para
procesar y analizar grandes cantidades de datos de lenguaje natural.

Los transformers son un tipo de arquitectura de red neuronal que ha revolucionado el
procesamiento del lenguaje natural. Introducidos en 2017, usan mecanismos de atención
para procesar secuencias de datos. Modelos como GPT y BERT están basados en transformers.

ChromaDB es una base de datos vectorial de código abierto diseñada para almacenar y
buscar embeddings de manera eficiente. Es especialmente útil en aplicaciones de
Retrieval Augmented Generation (RAG) donde se necesita recuperar fragmentos de texto
relevantes basándose en similitud semántica.

Los embeddings son representaciones numéricas (vectores) de texto que capturan el
significado semántico. Dos textos con significados similares tendrán embeddings
cercanos en el espacio vectorial. Esto permite hacer búsquedas por similitud semántica
en lugar de solo por palabras clave exactas.

RAG (Retrieval Augmented Generation) es una técnica que combina la recuperación de
información con la generación de texto. Primero recupera fragmentos relevantes de una
base de conocimiento y luego los usa como contexto para que un modelo de lenguaje
genere una respuesta más precisa y fundamentada.
"""


def dividir_en_chunks(texto: str, tamanio: int = 80, solapamiento: int = 20) -> list[str]:
    palabras = texto.split()
    chunks, inicio = [], 0
    while inicio < len(palabras):
        chunk = " ".join(palabras[inicio:inicio + tamanio]).strip()
        if len(chunk) > 30:
            chunks.append(chunk)
        inicio += tamanio - solapamiento
    return chunks


def obtener_coleccion() -> chromadb.Collection:
    """Carga o crea la colección ChromaDB con los chunks del texto."""
    cliente = chromadb.PersistentClient(path="./chromadb_data")
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    try:
        # Intentar cargar colección existente
        coleccion = cliente.get_collection(
            name="mi_coleccion_rag",
            embedding_function=embedding_fn
        )
        print("✅ Colección cargada desde disco.")
    except Exception:
        # Si no existe, crearla
        print(" Creando nueva colección...")
        try:
            cliente.delete_collection("mi_coleccion_rag")
        except Exception:
            pass

        coleccion = cliente.create_collection(
            name="mi_coleccion_rag",
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        chunks = dividir_en_chunks(TEXTO_LARGO)
        coleccion.add(
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )
        print(f"✅ Colección creada con {len(chunks)} chunks.")

    return coleccion


def recuperar_chunks(coleccion: chromadb.Collection, pregunta: str, n: int = 3) -> list[str]:
    """Recupera los n chunks más relevantes para la pregunta."""
    resultados = coleccion.query(query_texts=[pregunta], n_results=n)
    return resultados["documents"][0]


def generar_respuesta_ia(pregunta: str, chunks: list[str]) -> str:
    """
    Pasa la pregunta + chunks recuperados a Claude.
    La IA convierte los chunks crudos en una respuesta natural y comprensible.
    """
    cliente_ia = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    contexto = "\n\n---\n\n".join(chunks)

    prompt = f"""Eres un asistente experto. Se te proporcionan fragmentos de texto recuperados
de una base de conocimiento mediante búsqueda semántica (RAG).

FRAGMENTOS RECUPERADOS:
{contexto}

PREGUNTA DEL USUARIO:
{pregunta}

Usando ÚNICAMENTE la información de los fragmentos proporcionados, responde a la pregunta
de manera clara, natural y comprensible para cualquier persona. Si la información no es
suficiente para responder completamente, indícalo. No inventes información."""

    mensaje = cliente_ia.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return mensaje.content[0].text


def rag_con_ia(pregunta: str, n_chunks: int = 3):
    """Pipeline completo: RAG → chunks → IA → respuesta humana."""
    print(f"\n{'='*60}")
    print(f"🔍 Pregunta: {pregunta}")
    print("=" * 60)

    # Paso 1: Recuperar chunks relevantes
    coleccion = obtener_coleccion()
    chunks = recuperar_chunks(coleccion, pregunta, n=n_chunks)

    print(f"\n Chunks crudos recuperados ({len(chunks)}):")
    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}] {chunk[:120]}...")

    # Paso 2: Pasar chunks a la IA
    print("\n Generando respuesta con IA...\n")
    respuesta = generar_respuesta_ia(pregunta, chunks)

    print(" Respuesta generada por IA:")
    print("-" * 60)
    print(respuesta)
    print("-" * 60)

    return respuesta


# ── Ejecución principal ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("      RAG + IA - Chunks → Respuesta Humanizada")
    print("=" * 60)

    if ANTHROPIC_API_KEY == "TU_API_KEY_AQUI":
        print("\n  AVISO: No has configurado tu ANTHROPIC_API_KEY.")
        print("   Ejecuta: export ANTHROPIC_API_KEY='sk-ant-...'")
        print("   O edita la variable ANTHROPIC_API_KEY en este archivo.\n")
    else:
        preguntas = [
            "¿Qué son los embeddings y para qué sirven?",
            "¿Cómo funciona RAG?",
            "¿Qué diferencia hay entre el aprendizaje automático y las redes neuronales?",
        ]

        for pregunta in preguntas:
            rag_con_ia(pregunta)
            print()
