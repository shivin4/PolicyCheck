# 🛡️ PolicyCheck AI

> **Policy Intelligence Assistant** · Grounded RAG Pipeline · Microservices Architecture · FastAPI · Ollama

PolicyCheck AI is an enterprise policy intelligence platform that uses **Retrieval-Augmented Generation (RAG)** to analyze and answer questions grounded strictly in uploaded legal and policy documents (e.g., GST Act, IT Act, Labour Laws, Consumer Protection Act).

---

## ✨ UI Showcase & Interface Highlights

PolicyCheck AI is designed using the **Clean Enterprise Light** design system, combining executive aesthetics with real-time technical telemetry:

| Tab / Module | Feature Description | Key Capabilities |
| :--- | :--- | :--- |
| **💬 Ask Question** | **Interactive 9-Step RAG Pipeline** | Real-time step rendering (*Normalization → 768d Embedding → Cosine Similarity Search → Top-K Retrieval → Context Assembly → LLM Generation → Sources & Grounding*). |
| **⚖️ Compare** | **RAG vs. No-RAG Benchmark** | Side-by-side comparison proving how ungrounded direct LLM generation fails/hallucinates vs. how grounded RAG guarantees accuracy. |
| **📚 Knowledge Base** | **PDF Ingestion & Chunk Inspector** | Ingests PDF policies, splits them into 1500-character chunks (200 overlap), and displays vector embedding metrics & document status cards. |
| **🏗️ Architecture** | **System Topology Inspector** | Interactive topology diagram mapping microservice communication channels and port assignments (`:8080`, `:8000`, `:8001`, `:8002`, `:11434`). |
| **⚡ System Status** | **Microservice Health Telemetry** | Active heartbeat monitoring with pulse indicators across UI App, Docker Orchestration, Retrieval Service, LLM Service, and Ollama. |

---

## 🏛️ Microservice Architecture Overview

PolicyCheck AI is built as a decoupled microservices architecture containerized via Docker Compose:

```text
[ User Browser ]
       │
       ▼
 [ UI / App Service ] (app.py · Port 8080 / 8000)
       │
       ├───► [ Retrieval Service ] (retrieval_service.py · Port 8001) ───► [ Ollama Embeddings ] (Port 11434)
       │            │                                                            │ (nomic-embed-text)
       │            ▼                                                            ▼
       │     Vector Search (NumPy Cosine) ───────────────► Matched Policy Chunks
       │
       └───► [ LLM Service ] (llm_service.py · Port 8002) ───────────────► [ Ollama Generation ] (Port 11434)
                    │                                                            │ (llama3.1 / qwen2.5)
                    ▼                                                            ▼
             Grounded RAG Answer ◄───────────────────────── Final Verified Output
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Pull Ollama Models
```bash
# Vector embedding model (768 dimensions)
ollama pull nomic-embed-text

# Recommended LLM model (or llama3.1:8b for RTX GPU)
ollama pull qwen2.5:0.5b
```

### 3. Run Locally
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open [http://localhost:8000/](http://localhost:8000/) in your browser.

---

## 🐳 Docker Compose Microservices Launch

To run all decoupled microservices simultaneously:

```bash
sudo docker compose up --build -d
```

---

## 📄 Document Processing & NLP Specs

- **Chunking Strategy**: 1500 characters per chunk with 200 character overlap.
- **Embedding Dimensions**: 768-dimensional continuous vector space (`nomic-embed-text`).
- **Vector Distance Metric**: Cosine Similarity ($\ge 0.50$ threshold for knowledge base relevance).
