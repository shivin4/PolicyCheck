from fastapi import FastAPI
import requests

app = FastAPI()

RETRIEVAL_URL = "http://host.docker.internal:8001/retrieve"
LLM_URL = "http://host.docker.internal:8002/generate"


@app.get("/")
def home():
    return {
        "service": "PolicyCheck Orchestration Service",
        "status": "running"
    }


@app.post("/ask")
def ask(question: str):

    # Step 1: Retrieve relevant information
    retrieval_response = requests.post(
        RETRIEVAL_URL,
        params={"question": question}
    )

    retrieval_data = retrieval_response.json()

    results = retrieval_data["results"]

    # Step 2: Build context
    context = "\n\n".join(
        result["text"] for result in results
    )

    # Step 3: Send context + question to LLM
    prompt = f"""
You are PolicyCheck AI.

Answer the user's question using ONLY the provided context.
If the answer is not present in the context, say that the
information is not available in the knowledge base.

Context:
{context}

Question:
{question}

Answer:
"""

    llm_response = requests.post(
        LLM_URL,
        params={"prompt": prompt}
    )

    llm_data = llm_response.json()

    return {
        "question": question,
        "retrieved_results": results,
        "answer": llm_data["response"]
    }
