from langchain_ollama import ChatOllama
from langchain_core.tools import tool

# 1. Definir una tool: es solo una función con un decorador
@tool
def sumar(a: int, b: int) -> int:
    """Suma dos números enteros y devuelve el resultado."""
    return a + b

# 2. Conectar al modelo local (qwen en tu Ollama) y darle la tool
modelo = ChatOllama(model="qwen2.5:7b", base_url="http://host.docker.internal:11434")
modelo_con_tools = modelo.bind_tools([sumar])

# 3. Pregunta que SÍ necesita la tool
print("=== PREGUNTA 1: 'cuánto es 25 + 17' ===")
r1 = modelo_con_tools.invoke("cuánto es 25 + 17")
print("¿El modelo pidió usar una tool?:", r1.tool_calls)
print("Texto de respuesta:", r1.content)

print()

# 4. Pregunta que NO necesita la tool
print("=== PREGUNTA 2: 'hola, cómo estás' ===")
r2 = modelo_con_tools.invoke("hola, cómo estás")
print("¿El modelo pidió usar una tool?:", r2.tool_calls)
print("Texto de respuesta:", r2.content)
