import chromadb
import requests

OLLAMA_EMBED = "http://host.docker.internal:11434/api/embeddings"
OLLAMA_GEN = "http://host.docker.internal:11434/api/generate"
MODELO_EMBED = "nomic-embed-text"
MODELO_CHAT = "qwen2.5:7b"

def obtener_embedding(texto):
    r = requests.post(OLLAMA_EMBED, json={"model": MODELO_EMBED, "prompt": texto})
    return r.json()["embedding"]

def preguntar_llm(prompt):
    r = requests.post(OLLAMA_GEN, json={"model": MODELO_CHAT, "prompt": prompt, "stream": False})
    return r.json()["response"]

# 1. Conectar a la vector DB
cliente = chromadb.PersistentClient(path="./chroma_db")
coleccion = cliente.get_collection(name="k8s")

# 2. La pregunta
pregunta = "¿por qué necesito Kubernetes? explicámelo simple"

# 3. RETRIEVAL: buscar los chunks relevantes
embedding_pregunta = obtener_embedding(pregunta)
resultado = coleccion.query(query_embeddings=[embedding_pregunta], n_results=3)
chunks = resultado["documents"][0]

# 4. Armar el contexto uniendo los chunks
contexto = "\n\n".join(chunks)

# 5. AUGMENTED: armar el prompt con el contexto
prompt = f"""Usá el siguiente contexto para responder la pregunta en español.
Si el contexto no tiene la respuesta, decilo.

CONTEXTO:
{contexto}

PREGUNTA:
{pregunta}
"""

# 6. GENERATION: el LLM responde
respuesta = preguntar_llm(prompt)

print(f"PREGUNTA: {pregunta}\n")
print("=" * 60)
print(respuesta)
