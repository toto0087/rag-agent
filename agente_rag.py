from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import chromadb
import requests

OLLAMA_EMBED = "http://host.docker.internal:11434/api/embeddings"
MODELO_EMBED = "nomic-embed-text"

# Conexión a la vector DB (una vez)
cliente_chroma = chromadb.PersistentClient(path="./chroma_db")
coleccion = cliente_chroma.get_collection(name="k8s")


def obtener_embedding(texto):
    r = requests.post(OLLAMA_EMBED, json={"model": MODELO_EMBED, "prompt": texto})
    return r.json()["embedding"]


# LA TOOL: tu RAG convertido en herramienta
@tool
def buscar_en_docs(pregunta: str) -> str:
    """Busca información en la documentación de Kubernetes.
    Usá esta herramienta cuando el usuario pregunte sobre Kubernetes, pods,
    deployments, services, o cualquier concepto de K8s."""
    embedding = obtener_embedding(pregunta)
    resultado = coleccion.query(query_embeddings=[embedding], n_results=3)
    chunks = resultado["documents"][0]
    return "\n\n".join(chunks)


# El modelo
modelo = ChatOllama(model="qwen2.5:7b", base_url="http://host.docker.internal:11434")

# El agente, ahora con la tool de RAG
agente = create_react_agent(modelo, [buscar_en_docs])


def preguntar(texto):
    print(f"\n{'='*60}\nPREGUNTA: {texto}\n{'='*60}")
    resultado = agente.invoke({"messages": [("user", texto)]})
    for mensaje in resultado["messages"]:
        mensaje.pretty_print()


# Probamos DOS preguntas: una que necesita la doc, otra que no
preguntar("¿qué es un pod en kubernetes?")
preguntar("hola, ¿cómo estás?")
