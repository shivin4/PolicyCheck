import json
import os
import requests
import math

EMBEDDING_URL = "http://localhost:11434/api/embeddings"
GENERATE_URL = "http://localhost:11434/api/generate"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = os.getenv("LLM_MODEL", "codellama:latest")

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
    dot_product = sum(x * y for x, y in zip(a, b))

    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product / (magnitude_a * magnitude_b)


def retrieve(question, top_k=3):
    query_embedding = get_embedding(question)

    scored_documents = []

    for document in documents:
        score = cosine_similarity(
            query_embedding,
            document["embedding"]
        )

        scored_documents.append({
            "score": score,
            "source": document["source"],
            "text": document["text"]
        })

    scored_documents.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return scored_documents[:top_k]


def generate_answer(question, results):

    context = "\n\n".join(
        f"Source: {result['source']}\n{result['text']}"
        for result in results
    )

    prompt = f"""
You are PolicyCheck AI, a policy information assistant.

Answer the user's question using ONLY the provided context.

If the answer is not available in the context, say:
"I could not find this information in the provided policy documents."

Context:
{context}

Question:
{question}

Answer:
"""

    response = requests.post(
        GENERATE_URL,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    response.raise_for_status()

    return response.json()["response"]


if __name__ == "__main__":

    question = input("Ask PolicyCheck: ")

    results = retrieve(question)

    print("\nRetrieved Context:")
    print("------------------")

    for i, result in enumerate(results, 1):
        print(
            f"{i}. {result['source']} "
            f"(similarity: {result['score']:.4f})"
        )

    answer = generate_answer(question, results)

    print("\nPolicyCheck AI Response:")
    print("----------------------")
    print(answer)