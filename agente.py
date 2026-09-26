from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# 1. La misma tool de suma de antes
@tool
def sumar(a: int, b: int) -> int:
    """Suma dos números enteros y devuelve el resultado."""
    return a + b

# 2. El modelo local
modelo = ChatOllama(model="qwen2.5:7b", base_url="http://host.docker.internal:11434")

# 3. Crear el agente: le pasás el modelo y la lista de tools.
#    LangGraph arma el grafo (modelo <-> tools) con el loop por dentro.
agente = create_react_agent(modelo, [sumar])

# 4. Preguntar. El agente maneja el ciclo completo solo.
print("=== PREGUNTA: 'cuánto es 25 + 17' ===")
resultado = agente.invoke({"messages": [("user", "cuánto es 25 + 17")]})

# 5. Mostrar TODA la conversación, para ver el ciclo paso a paso
for mensaje in resultado["messages"]:
    mensaje.pretty_print()
