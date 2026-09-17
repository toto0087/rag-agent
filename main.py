from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    respuesta = requests.post(
        "http://host.docker.internal:11434/api/generate",
        json={
            "model": "qwen2.5:7b",
            "prompt": req.message,
            "stream": False,
        },
    )
    data = respuesta.json()
    return {"response": data["response"]}
