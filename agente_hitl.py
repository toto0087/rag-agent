from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
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


@tool
def crear_deployment(nombre: str, imagen: str) -> str:
    """Crea un deployment de Kubernetes con el nombre y la imagen dados.
    Usá esta herramienta cuando el usuario pida crear o desplegar algo."""
    # Simulado: en la vida real acá se llamaría a la API de Kubernetes
    return f"✅ Deployment '{nombre}' creado con la imagen '{imagen}'"


# Tools que requieren confirmación humana antes de ejecutarse
TOOLS_SENSIBLES = {"crear_deployment"}


class Estado(TypedDict):
    messages: Annotated[list, add_messages]


tools = [buscar_en_docs, crear_deployment]
modelo = ChatOllama(model="qwen2.5:7b", base_url="http://host.docker.internal:11434")
modelo_con_tools = modelo.bind_tools(tools)
tools_por_nombre = {t.name: t for t in tools}


def nodo_modelo(estado: Estado):
    respuesta = modelo_con_tools.invoke(estado["messages"])
    return {"messages": [respuesta]}


def nodo_tools(estado: Estado):
    ultimo = estado["messages"][-1]
    resultados = []
    for llamada in ultimo.tool_calls:
        nombre = llamada["name"]

        # Si la tool es sensible, PAUSAR y pedir confirmación
        if nombre in TOOLS_SENSIBLES:
            decision = interrupt({
                "tool": nombre,
                "args": llamada["args"],
            })
            if decision != "si":
                resultados.append(
                    ToolMessage(
                        content=f"Acción '{nombre}' CANCELADA por el usuario.",
                        tool_call_id=llamada["id"],
                    )
                )
                continue

        # Ejecutar la tool (si no era sensible, o si fue confirmada)
        tool = tools_por_nombre[nombre]
        salida = tool.invoke(llamada["args"])
        resultados.append(
            ToolMessage(content=str(salida), tool_call_id=llamada["id"])
        )
    return {"messages": resultados}


def decidir(estado: Estado):
    if estado["messages"][-1].tool_calls:
        return "tools"
    return END


grafo = StateGraph(Estado)
grafo.add_node("modelo", nodo_modelo)
grafo.add_node("tools", nodo_tools)
grafo.add_edge(START, "modelo")
grafo.add_conditional_edges("modelo", decidir)
grafo.add_edge("tools", "modelo")

# El checkpointer da memoria al grafo (necesario para pausar/reanudar)
memoria = MemorySaver()
app = grafo.compile(checkpointer=memoria)


def preguntar(texto):
    print(f"\n{'='*60}\nPREGUNTA: {texto}\n{'='*60}")
    config = {"configurable": {"thread_id": "conversacion-1"}}

    # Arrancar el grafo
    resultado = app.invoke({"messages": [("user", texto)]}, config=config)

    # ¿El grafo se pausó pidiendo confirmación?
    while "__interrupt__" in resultado:
        pedido = resultado["__interrupt__"][0].value
        print(f"\n🔔 EL AGENTE QUIERE EJECUTAR UNA ACCIÓN:")
        print(f"   Tool: {pedido['tool']}")
        print(f"   Argumentos: {pedido['args']}")
        respuesta = input("   ¿Confirmás? (si/no): ").strip()

        # Reanudar el grafo con la decisión del usuario
        resultado = app.invoke(Command(resume=respuesta), config=config)

    # Mostrar la conversación final
    for m in resultado["messages"]:
        m.pretty_print()


# Probamos: una acción sensible (pide confirmación)
preguntar("creá un deployment llamado nginx con la imagen nginx:latest")
