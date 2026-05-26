import chromadb
from chromadb.utils import embedding_functions
import textwrap

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


def dividir_en_chunks(texto: str, tamanio: int = 200, solapamiento: int = 50) -> list[str]:
    """
    Divide el texto en fragmentos (chunks) de tamaño aproximado.
    El solapamiento ayuda a no perder contexto entre chunks.
    """
    palabras = texto.split()
    chunks = []
    inicio = 0

    while inicio < len(palabras):
        fin = inicio + tamanio
        chunk = " ".join(palabras[inicio:fin])
        chunks.append(chunk.strip())
        inicio += tamanio - solapamiento  

    return [c for c in chunks if len(c) > 30]  


def crear_base_de_datos(chunks: list[str]) -> chromadb.Collection:
    """
    Crea un cliente ChromaDB, una colección y añade los chunks con sus embeddings.
    Usa la función de embedding por defecto de ChromaDB (all-MiniLM-L6-v2).
    """
    # Cliente persistente: guarda los datos en disco
    cliente = chromadb.PersistentClient(path="./chromadb_data")

    # Si ya existe la colección, la borramos para empezar limpio
    try:
        cliente.delete_collection("mi_coleccion_rag")
    except Exception:
        pass

    # Función de embedding: usa sentence-transformers por defecto
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    # Crear colección
    coleccion = cliente.create_collection(
        name="mi_coleccion_rag",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}  # similitud coseno
    )

    # Insertar chunks en la base de datos
    print(f"\n📦 Insertando {len(chunks)} chunks en ChromaDB...")
    coleccion.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print("✅ Base de datos creada correctamente.\n")

    return coleccion


def hacer_query(coleccion: chromadb.Collection, pregunta: str, n_resultados: int = 3):
    """
    Realiza una búsqueda semántica en ChromaDB.
    Devuelve los n_resultados chunks más similares a la pregunta.
    """
    resultados = coleccion.query(
        query_texts=[pregunta],
        n_results=n_resultados
    )

    print(f"\n🔍 Pregunta: {pregunta}")
    print("=" * 60)
    print(f"📄 Top {n_resultados} chunks más relevantes:\n")

    documentos = resultados["documents"][0]
    distancias = resultados["distances"][0]

    for i, (doc, dist) in enumerate(zip(documentos, distancias)):
        similitud = 1 - dist  # convertir distancia coseno a similitud
        print(f"  [{i+1}] Similitud: {similitud:.2%}")
        print(f"  {textwrap.fill(doc, width=70, initial_indent='  ', subsequent_indent='  ')}")
        print()

    return documentos


# ── Ejecución principal ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("         🧠 RAG BÁSICO - ChromaDB + Embeddings")
    print("=" * 60)

    # 1. Dividir texto en chunks
    chunks = dividir_en_chunks(TEXTO_LARGO, tamanio=80, solapamiento=20)
    print(f"\n✂️  Texto dividido en {len(chunks)} chunks.")

    # 2. Crear base de datos vectorial
    coleccion = crear_base_de_datos(chunks)

    # 3. Hacer consultas de prueba
    preguntas = [
        "¿Qué son los embeddings?",
        "¿Cómo funcionan las redes neuronales?",
        "¿Qué es RAG y para qué sirve?",
    ]

    for pregunta in preguntas:
        hacer_query(coleccion, pregunta, n_resultados=2)
        print("-" * 60)
