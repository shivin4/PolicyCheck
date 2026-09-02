import json
import requests
import time

INPUT_FILE = "chunks.json"
OUTPUT_FILE = "embeddings.json"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

results = []

for i, chunk in enumerate(chunks):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": chunk["text"]
        }
    )

    response.raise_for_status()

    embedding = response.json()["embedding"]

    results.append({
        "id": chunk["id"],
        "source": chunk["source"],
        "text": chunk["text"],
        "embedding": embedding
    })

    print(f"Embedded {i + 1}/{len(chunks)}")

    # Small delay to avoid overloading Ollama
    time.sleep(0.05)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f)

print(f"\nSaved embeddings to: {OUTPUT_FILE}")
print(f"Total embeddings: {len(results)}")
