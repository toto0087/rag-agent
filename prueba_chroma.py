import chromadb

# 1. Crear el cliente de Chroma (en memoria, no guarda en disco por ahora)
cliente = chromadb.Client()

# 2. Crear una "colección" (como una tabla donde viven los embeddings)
coleccion = cliente.create_collection(name="prueba")

# 3. Guardar unas frases. Chroma calcula el embedding de cada una solo.
coleccion.add(
    documents=[
        "El gato duerme en el sofá",
        "La bolsa de valores subió hoy",
        "Los felinos descansan durante el día",
    ],
    ids=["1", "2", "3"],
)

# 4. Buscar: le paso una pregunta y le pido las 2 frases más parecidas
resultado = coleccion.query(
    query_texts=["¿dónde descansan los animales?"],
    n_results=2,
)

# 5. Mostrar qué encontró
print(resultado["documents"])
