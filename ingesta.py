import chromadb
import requests
import glob
import os

OLLAMA_URL = "http://host.docker.internal:11434/api/embeddings"
MODELO_EMBED = "nomic-embed-text"

def obtener_embedding(texto):
    r = requests.post(OLLAMA_URL, json={"model": MODELO_EMBED, "prompt": texto})
    return r.json()["embedding"]

def partir_en_chunks(texto, tam=500):
    return [texto[i:i+tam] for i in range(0, len(texto), tam)]

# Conectar a Chroma por red (el servicio en Kubernetes)
CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
cliente = chromadb.HttpClient(host=CHROMA_HOST, port=8000)

# Borrar la colección vieja y recrearla desde cero (para no duplicar)
try:
    cliente.delete_collection(name="k8s")
except Exception:
    pass
coleccion = cliente.create_collection(name="k8s")

# Buscar TODOS los archivos .md dentro de docs/
archivos = glob.glob("docs/*.md")
print(f"Archivos encontrados: {archivos}")

id_global = 0
for archivo in archivos:
    with open(archivo, "r", encoding="utf-8") as f:
        contenido = f.read()

    chunks = partir_en_chunks(contenido)
    nombre = os.path.basename(archivo)
    print(f"\n{nombre}: {len(chunks)} chunks")

    for chunk in chunks:
        embedding = obtener_embedding(chunk)
        coleccion.add(
            ids=[str(id_global)],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"fuente": nombre}],
        )
        id_global += 1
    print(f"{nombre}: guardado")

print(f"\n¡Ingesta completa! Total: {id_global} chunks")
