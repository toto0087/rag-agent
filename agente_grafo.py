from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
import chromadb
import requests

OLLAMA_EMBED = "http://host.docker.internal:11434/api/embeddings"
MODELO_EMBED = "nomic-embed-text"

cliente_chroma = chromadb.PersistentClient(path="./chroma_db")
coleccion = cliente_chroma.get_collection(name="k8s")


def obtener_embedding(texto):
    r = requests.post(OLLAMA_EMBED, json={"model": MODELO_EMBED, "prompt": texto})
    return r.json()["embedding"]


@tool
def buscar_en_docs(pregunta: str) -> str:
    """Busca información en la documentación de Kubernetes.
    Usá esta herramienta cuando el usuario pregunte sobre Kubernetes, pods,
    deployments, services, o cualquier concepto de K8s."""
    embedding = obtener_embedding(pregunta)
    resultado = coleccion.query(query_embeddings=[embedding], n_results=3)
    return "\n\n".join(resultado["documents"][0])


# ---- 1. EL ESTADO ----
class Estado(TypedDict):
    messages: Annotated[list, add_messages]


# ---- El modelo, con la tool atada ----
tools = [buscar_en_docs]
modelo = ChatOllama(model="qwen2.5:7b", base_url="http://host.docker.internal:11434")
modelo_con_tools = modelo.bind_tools(tools)
tools_por_nombre = {t.name: t for t in tools}


# ---- 2. LOS NODOS ----
def nodo_modelo(estado: Estado):
    # Lee TODA la conversación del estado y se la manda al modelo
    respuesta = modelo_con_tools.invoke(estado["messages"])
    # Devuelve la respuesta para que se agregue al estado
    return {"messages": [respuesta]}


def nodo_tools(estado: Estado):
    # Mira el último mensaje (la decisión del modelo)
    ultimo = estado["messages"][-1]
    resultados = []
    # Por cada tool que el modelo pidió, la ejecuta
    for llamada in ultimo.tool_calls:
        tool = tools_por_nombre[llamada["name"]]
        salida = tool.invoke(llamada["args"])
        resultados.append(
            ToolMessage(content=str(salida), tool_call_id=llamada["id"])
        )
    return {"messages": resultados}


# ---- 3. LA ARISTA CONDICIONAL ----
def decidir(estado: Estado):
    ultimo = estado["messages"][-1]
    # ¿El modelo pidió una tool?
    if ultimo.tool_calls:
        return "tools"   # sí -> ir al nodo tools
    return END           # no -> terminar


# ---- ARMAR EL GRAFO ----
grafo = StateGraph(Estado)
grafo.add_node("modelo", nodo_modelo)
grafo.add_node("tools", nodo_tools)

grafo.add_edge(START, "modelo")           # arranca en el modelo
grafo.add_conditional_edges("modelo", decidir)  # después del modelo, decidir
grafo.add_edge("tools", "modelo")         # después de tools, volver al modelo (el loop)

app = grafo.compile()


# ---- PROBAR ----
def preguntar(texto):
    print(f"\n{'='*60}\nPREGUNTA: {texto}\n{'='*60}")
    resultado = app.invoke({"messages": [("user", texto)]})
    for m in resultado["messages"]:
        m.pretty_print()


preguntar("¿qué es un pod en kubernetes?")
preguntar("hola, ¿cómo estás?")
