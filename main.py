from fastapi import FastAPI
from pydantic import BaseModel
import requests
import chromadb
import os

app = FastAPI()

OLLAMA_EMBED = "http://host.docker.internal:11434/api/embeddings"
OLLAMA_GEN = "http://host.docker.internal:11434/api/generate"
MODELO_EMBED = "nomic-embed-text"
MODELO_CHAT = "qwen2.5:7b"

CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
cliente_chroma = chromadb.HttpClient(host=CHROMA_HOST, port=8000)
coleccion = cliente_chroma.get_or_create_collection(name="k8s")


class ChatRequest(BaseModel):
    message: str


class AskRequest(BaseModel):
    question: str


def obtener_embedding(texto):
    r = requests.post(OLLAMA_EMBED, json={"model": MODELO_EMBED, "prompt": texto})
    return r.json()["embedding"]


def preguntar_llm(prompt):
    r = requests.post(OLLAMA_GEN, json={"model": MODELO_CHAT, "prompt": prompt, "stream": False})
    return r.json()["response"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    respuesta = requests.post(
        OLLAMA_GEN,
        json={"model": MODELO_CHAT, "prompt": req.message, "stream": False},
    )
    return {"response": respuesta.json()["response"]}


@app.post("/ask")
def ask(req: AskRequest):
    embedding_pregunta = obtener_embedding(req.question)
    resultado = coleccion.query(query_embeddings=[embedding_pregunta], n_results=3)
    chunks = resultado["documents"][0]

    contexto = "\n\n".join(chunks)
    prompt = f"""Usá el siguiente contexto para responder la pregunta en español.
Si el contexto no tiene la respuesta, decilo.

CONTEXTO:
{contexto}

PREGUNTA:
{req.question}
"""

    respuesta = preguntar_llm(prompt)

    return {
        "answer": respuesta,
        "chunks_usados": chunks,
    }
