# PolicyCheck AI

> **Policy Intelligence Assistant** · Grounded RAG Pipeline + Microservice Architecture + FastAPI + Ollama

PolicyCheck AI is an enterprise policy intelligence platform that uses Retrieval-Augmented Generation (RAG) to answer questions grounded strictly in uploaded policy documents.

---

## 🏛 Architecture Overview

PolicyCheck AI is built as a microservices architecture orchestrated via Docker Compose:

- **UI / Application Service (`app.py` · Port 8080)**: Serves the modern web UI, handles document uploads, and provides API endpoints.
- **Orchestration Service (`orchestration_service.py` · Port 8000)**: Coordinates retrieval and generation pipelines across microservices inside Docker containers.
- **Retrieval Service (`retrieval_service.py` · Port 8001)**: Generates query embeddings via `nomic-embed-text` and computes cosine similarity scores across knowledge base chunks.
- **LLM Service (`llm_service.py` · Port 8002)**: Connects to Ollama hosting `codellama:latest` to generate grounded answers.
- **Vector Storage**: 768-dimensional embeddings generated with `nomic-embed-text`.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Locally
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open [http://localhost:8000/](http://localhost:8000/) in your browser.

---

## 📄 Knowledge Base Document Support

Documents are automatically ingested, chunked (1500 characters with 200 character overlap), embedded, and stored in the vector index for immediate RAG retrieval.
