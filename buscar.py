import chromadb
import requests

OLLAMA_URL = "http://host.docker.internal:11434/api/embeddings"
MODELO_EMBED = "nomic-embed-text"

def obtener_embedding(texto):
    respuesta = requests.post(
        OLLAMA_URL,
        json={"model": MODELO_EMBED, "prompt": texto},
    )
    return respuesta.json()["embedding"]

# 1. Conectar a la Chroma que ya existe en disco
cliente = chromadb.PersistentClient(path="./chroma_db")
coleccion = cliente.get_collection(name="k8s")

# 2. La pregunta (en español, aunque la doc esté en inglés)
pregunta = "¿por qué necesito Kubernetes?"

# 3. Calcular el embedding de la pregunta
embedding_pregunta = obtener_embedding(pregunta)

# 4. Buscar los 3 chunks más cercanos
resultado = coleccion.query(
    query_embeddings=[embedding_pregunta],
    n_results=3,
)

# 5. Mostrar los chunks recuperados
print(f"PREGUNTA: {pregunta}\n")
print("=" * 60)
for i, chunk in enumerate(resultado["documents"][0]):
    print(f"\n--- CHUNK {i+1} ---")
    print(chunk)
