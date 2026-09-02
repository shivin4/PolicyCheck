import json
import requests
import math
from fastapi import FastAPI

app = FastAPI()

EMBEDDING_URL = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "nomic-embed-text"

with open("embeddings.json", "r", encoding="utf-8") as f:
    documents = json.load(f)


def get_embedding(text):
    response = requests.post(
        EMBEDDING_URL,
        json={
            "model": EMBEDDING_MODEL,
            "prompt": text
        }
    )
    response.raise_for_status()
    return response.json()["embedding"]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))

    if mag_a == 0 or mag_b == 0:
        return 0

    return dot / (mag_a * mag_b)


@app.get("/")
def home():
    return {"service": "PolicyCheck Retrieval Service", "status": "running"}


@app.post("/retrieve")
def retrieve(question: str):

    query_embedding = get_embedding(question)

    results = []

    for document in documents:
        score = cosine_similarity(
            query_embedding,
            document["embedding"]
        )

        results.append({
            "score": score,
            "source": document["source"],
            "text": document["text"]
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return {
        "question": question,
        "results": results[:3]
    }