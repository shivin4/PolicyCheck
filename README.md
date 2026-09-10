# PolicyCheck AI

Policy Intelligence Assistant · Grounded RAG Pipeline · Microservices Architecture · FastAPI · Ollama

PolicyCheck AI is an enterprise policy intelligence platform that uses Retrieval-Augmented Generation (RAG) to analyze and answer questions grounded strictly in uploaded legal and policy documents (GST Act, IT Act, Labour Laws, Consumer Protection Act).

---

## UI Overview

| Tab | Description | Key Capabilities |
| :--- | :--- | :--- |
| Architecture | System topology and pipeline overview | Full RAG pipeline diagram, service descriptions, tech stack, knowledge base build pipeline |
| Knowledge Base | PDF ingestion and chunk inspector | Ingests PDFs, splits into 1500-char chunks (200 overlap), displays vector embedding metrics |
| Ask Question | Interactive 9-step RAG pipeline | Real-time step rendering: Normalization, 768-dim Embedding, Cosine Similarity Search, Top-K Retrieval, Context Assembly, LLM Generation, Sources and Grounding |
| Compare | RAG vs No-RAG benchmark | Side-by-side comparison showing how ungrounded LLM generation hallucinates vs grounded RAG |
| System Status | Microservice health telemetry | Active heartbeat monitoring across UI, Orchestration, Retrieval, LLM, and Ollama services |
| Model Evaluation | Week 4 multi-model evaluation dashboard | Category-wise comparison of CodeLlama, Qwen 2.5, and TinyLLaMA across 25 benchmark questions |

---

## Microservice Architecture

PolicyCheck AI is built as a decoupled microservices architecture containerized via Docker Compose.

```
User Browser
    |
    v
UI / App Service          (app.py                    · Port 8080)
    |
    v
Orchestration Service     (orchestration_service.py  · Port 8000)
    |
    |-----> Retrieval Service  (retrieval_service.py  · Port 8001)
    |           |
    |           v
    |       Ollama Embeddings  (nomic-embed-text       · Port 11434)
    |           |
    |           v
    |       Vector Search (NumPy Cosine Similarity against embeddings.json)
    |
    |-----> LLM Service        (llm_service.py         · Port 8002)
                |
                v
            Ollama Generation  (qwen2.5:0.5b / tinyllama:1.1b / codellama:latest · Port 11434)
                |
                v
            Grounded RAG Answer
```

---

## Knowledge Base Build Pipeline (One-time Setup)

```
PDF Documents (CGST Act, IT Act, Consumer Protection Act, Essential Commodities Act)
    |
    v  chunk_documents.py  (1500-char chunks, 200-char overlap)
    |
chunks.json  (590 text chunks)
    |
    v  create_embeddings.py  (nomic-embed-text via Ollama)
    |
embeddings.json  (590 x 768-dimensional vectors)
    |
    v  loaded into memory at startup
    |
Retrieval Service  (ready to search at query time)
```

---

## Tech Stack

| Component | Technology |
| :--- | :--- |
| Language | Python 3.12 |
| Web Framework | FastAPI + Uvicorn |
| LLM Runtime | Ollama (local inference) |
| Embedding Model | nomic-embed-text (768-dimensional) |
| Vector Search | NumPy cosine similarity |
| Containerisation | Docker + Docker Compose |
| PDF Processing | pypdf (1500-char chunks, 200-char overlap) |
| Deployment | AWS EC2 (Ubuntu) |

---

## Document Processing Specs

- Chunking Strategy: 1500 characters per chunk with 200 character overlap
- Embedding Dimensions: 768-dimensional continuous vector space (nomic-embed-text)
- Vector Distance Metric: Cosine Similarity (threshold >= 0.50 for knowledge base relevance)
- Total Chunks: 590 across 4 policy documents

---

## Quick Start (Local)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Pull Ollama Models

```bash
# Embedding model (768 dimensions)
ollama pull nomic-embed-text

# LLM generation model
ollama pull qwen2.5:0.5b
```

### 3. Run Locally (Without Docker)

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Open http://localhost:8080 in your browser.

---

## Docker Compose (All Microservices)

```bash
docker compose up --build -d
```

Services started:
- UI Service: http://localhost:8080
- Orchestration Service: http://localhost:8000
- Retrieval Service: http://localhost:8001
- LLM Service: http://localhost:8002

---

## Week 4 — Multi-Model Evaluation

PolicyCheck AI was evaluated across 3 LLM models using 25 benchmark questions divided into 6 task categories.

### Models Evaluated

| Model | Parameters | Type |
| :--- | :--- | :--- |
| codellama:latest | 7B | Code-focused |
| qwen2.5:0.5b | 0.5B | Instruction-tuned |
| tinyllama:1.1b | 1.1B | Compact general |

### Task Categories

| Category | Questions | Description |
| :--- | :--- | :--- |
| Explanation | Q01, Q02, Q05, Q10, Q16, Q21 | Explain what a concept or act is |
| Retrieval | Q03, Q07, Q08, Q11, Q20 | Retrieve a specific clause or fact |
| Dependency Understanding | Q04, Q12, Q17, Q22 | Multi-condition or cross-section reasoning |
| Bug Analysis | Q06, Q18, Q19, Q23, Q24 | Identify consequences, penalties, liability |
| RAG-based Question | Q13, Q14, Q15 | Only answerable using retrieved context |
| Hallucination Trap | Q09, Q25 | Out-of-scope questions testing refusal accuracy |

### Category-wise Results Summary

| Category | Best Accuracy | Fastest | Least Hallucinations |
| :--- | :--- | :--- | :--- |
| Explanation | CodeLlama + TinyLLaMA (1.83/2) | TinyLLaMA | All zero |
| Retrieval | Qwen 2.5 (1.80/2) | Qwen 2.5 | All zero |
| Dependency Understanding | TinyLLaMA (1.25/2) | Qwen 2.5 | All zero |
| Bug Analysis | Qwen 2.5 (1.60/2) | Qwen 2.5 | All zero |
| RAG-based Question | TinyLLaMA (1.67/2) | Qwen 2.5 | All zero |
| Hallucination Trap | TinyLLaMA (0.50/2, 1 hallucination) | Qwen 2.5 | TinyLLaMA |

### Overall Wins

- TinyLLaMA: 3 category wins (highest overall accuracy, best hallucination resistance)
- Qwen 2.5: 2 category wins (fastest across all categories)
- CodeLlama: 1 category win (tied for Explanation)

### Metrics Defined

| Metric | Calculation |
| :--- | :--- |
| Correctness | Keyword overlap between response and ground truth. >= 0.40 = 2, >= 0.20 = 1, else 0 |
| Relevance | Token overlap ratio between question keywords and response tokens (0.0-1.0) |
| Retrieval Quality | 1 if expected source PDF appears in top-3 retrieved chunks, 0 if not |
| Hallucination | True if model gives a specific fabricated answer to an out-of-scope question |
| Response Latency | End-to-end time in milliseconds from request to response |
| LLM Latency | Time spent strictly inside Ollama model inference |

### Key Findings

Qwen 2.5 (0.5B) provides the optimal speed-accuracy trade-off for cloud deployment, averaging 7.7s response time with strong relevance (0.75). TinyLLaMA (1.1B) achieves the highest correctness and best hallucination resistance. CodeLlama (7B) suffers excessive latency (41.4s average) due to its larger parameter count, and its code-focused training reduces relevance on legal text tasks.

---

## Evaluation Files

| File | Description |
| :--- | :--- |
| evaluation_dataset.json | 25 benchmark questions with ground truth, task categories, and expected sources |
| evaluation_results.json | 75 evaluation records (25 questions x 3 models) with full metrics |
| evaluation_results.csv | Tabular version of evaluation results |
| category_analysis.json | Pre-computed per-category per-model aggregated metrics |
| run_evaluation.py | Evaluation runner script |
| generate_category_analysis.py | Category analysis generator from evaluation results |

---

## Project Structure

```
PolicyCheck/
├── app.py                          # UI service (Port 8080)
├── orchestration_service.py        # Orchestration service (Port 8000)
├── retrieval_service.py            # Retrieval service (Port 8001)
├── llm_service.py                  # LLM service (Port 8002)
├── rag.py                          # Shared RAG utilities
├── chunk_documents.py              # One-time PDF chunking script
├── create_embeddings.py            # One-time embedding generation script
├── run_evaluation.py               # Week 4 evaluation runner
├── generate_category_analysis.py   # Category-wise analysis generator
├── chunks.json                     # 590 text chunks
├── embeddings.json                 # 590 x 768-dim vectors
├── evaluation_dataset.json         # 25 benchmark questions
├── evaluation_results.json         # 75 evaluation records
├── evaluation_results.csv          # Tabular evaluation results
├── category_analysis.json          # Category-wise analysis
├── Dockerfile                      # Container image definition
├── docker-compose.yml              # Multi-service orchestration
├── requirements.txt                # Python dependencies
├── templates/
│   └── index.html                  # Full UI (all tabs, CSS, JavaScript)
└── Dataset/
    └── pdfs/
        ├── GST/cgst_act.pdf
        ├── General/it_act.pdf
        ├── Labour/essential_commodities.pdf
        └── WelfareScheme/consumer_act.pdf
```
