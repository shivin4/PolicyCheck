import os
import requests
from fastapi import FastAPI

app = FastAPI()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("LLM_MODEL", "codellama:latest")


@app.get("/")
def home():
    return {
        "service": "PolicyCheck LLM Service",
        "status": "running"
    }


@app.post("/generate")
def generate(prompt: str):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()

    return {
        "response": data["response"]
    }
