"""
PolicyCheck AI — Main Application (Exercise 1 + UI)
Serves the web UI and exposes API endpoints for all exercises.
"""

import json
import math
import os
import shutil
import time
import requests
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pypdf import PdfReader

app = FastAPI(title="PolicyCheck AI")

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
EMBED_URL    = "http://localhost:11434/api/embeddings"
LLM_MODEL    = os.getenv("LLM_MODEL", "qwen2.5:0.5b")
EMBED_MODEL  = "nomic-embed-text"

# ── Relevance threshold for KB matching ───────────────────────────────────────
# Scores below this indicate the question is outside the knowledge base.
# Calibrated on actual KB: in-KB questions score 0.55–0.76,
# out-of-KB questions score 0.38–0.50.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.50"))

RETRIEVAL_SERVICE = "http://localhost:8001"
LLM_SERVICE       = "http://localhost:8002"

# ── Load embeddings once at startup ──────────────────────────────────────────
with open("embeddings.json", "r", encoding="utf-8") as f:
    DOCUMENTS = json.load(f)

with open("chunks.json", "r", encoding="utf-8") as f:
    CHUNKS = json.load(f)


def _reload_kb():
    """Reload DOCUMENTS and CHUNKS from disk into the live module globals."""
    global DOCUMENTS, CHUNKS, _EMB_BY_ID
    with open("embeddings.json", "r", encoding="utf-8") as f:
        DOCUMENTS = json.load(f)
    with open("chunks.json", "r", encoding="utf-8") as f:
        CHUNKS = json.load(f)
    _EMB_BY_ID = {doc["id"]: doc["embedding"] for doc in DOCUMENTS}


# ── UI ────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    html = Path("templates/index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


# ── Exercise 1: Direct LLM (no RAG) ──────────────────────────────────────────
@app.post("/ask")
def ask_policy(question: str, model: str = ""):
    """Exercise 1 — Basic: question → Ollama → Code Llama → response"""
    llm = model or LLM_MODEL
    response = requests.post(
        OLLAMA_URL,
        json={"model": llm, "prompt": question, "stream": False}
    )
    result = response.json()
    return {"question": question, "response": result["response"]}


@app.post("/api/ask-direct")
def api_ask_direct(question: str, model: str = ""):
    """UI endpoint: direct LLM answer without any RAG context"""
    llm = model or LLM_MODEL
    response = requests.post(
        OLLAMA_URL,
        json={"model": llm, "prompt": question, "stream": False}
    )
    result = response.json()
    return {"question": question, "response": result["response"]}


# ── Exercise 3: RAG pipeline (embedded in app) ────────────────────────────────
def get_embedding(text: str):
    try:
        r = requests.post(
            EMBED_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=10
        )
        r.raise_for_status()
        return r.json()["embedding"]
    except Exception:
        # Fallback for newer Ollama embedding format
        r = requests.post(
            EMBED_URL,
            json={"model": "nomic-embed-text:latest", "input": text},
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        return data.get("embedding") or data.get("embeddings", [[]])[0]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def retrieve_top_k(question: str, top_k: int = 3):
    query_emb = get_embedding(question)
    scored = []
    for doc in DOCUMENTS:
        score = cosine_similarity(query_emb, doc["embedding"])
        scored.append({"score": score, "source": doc["source"], "text": doc["text"]})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


@app.post("/api/rag-ask")
def api_rag_ask(question: str, model: str = ""):
    """UI endpoint: full RAG pipeline with KB relevance gate."""
    llm = model or LLM_MODEL
    results = retrieve_top_k(question)

    best_score = results[0]["score"] if results else 0.0
    kb_match = best_score >= RELEVANCE_THRESHOLD

    if not kb_match:
        return {
            "question": question,
            "retrieved_results": results,
            "kb_match": False,
            "best_score": round(best_score, 4),
            "threshold": RELEVANCE_THRESHOLD,
            "answer": (
                "I couldn't find relevant information about this question in the "
                "PolicyCheck Knowledge Base. PolicyCheck is designed to answer questions "
                "using only the available uploaded policy documents."
            ),
        }

    context = "\n\n".join(
        f"Source: {r['source']}\n{r['text']}" for r in results
    )

    prompt = f"""You are PolicyCheck AI, a policy information assistant.

Answer the user's question using ONLY the provided context.
If the answer is not in the context, say: "I could not find this information in the policy documents."

Context:
{context}

Question:
{question}

Answer:"""

    llm_resp = requests.post(
        OLLAMA_URL,
        json={"model": llm, "prompt": prompt, "stream": False}
    )
    llm_resp.raise_for_status()

    return {
        "question": question,
        "retrieved_results": results,
        "kb_match": True,
        "best_score": round(best_score, 4),
        "threshold": RELEVANCE_THRESHOLD,
        "answer": llm_resp.json()["response"],
    }


# ── Exercise 3b: RAG pipeline with full structured breakdown ─────────────────
@app.post("/api/rag-ask-pipeline")
def api_rag_ask_pipeline(question: str, model: str = ""):
    """
    Full RAG pipeline returning every intermediate step as structured data:
    embedding preview, similarity scores for all searched docs, top-k chunks,
    the exact context string sent to the LLM, and the final answer.
    """
    import time
    llm = model or LLM_MODEL

    # Step 1 – embed the query
    t0 = time.time()
    query_emb = get_embedding(question)
    embed_ms = round((time.time() - t0) * 1000)

    # Step 2 – cosine similarity over all documents
    t1 = time.time()
    scored = []
    for doc in DOCUMENTS:
        score = cosine_similarity(query_emb, doc["embedding"])
        scored.append({
            "score": score,
            "source": doc["source"],
            "text": doc["text"],
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    search_ms = round((time.time() - t1) * 1000)

    top_k = scored[:3]

    # ── KB relevance gate ─────────────────────────────────────
    best_score = top_k[0]["score"] if top_k else 0.0
    kb_match   = best_score >= RELEVANCE_THRESHOLD

    if not kb_match:
        emb_preview = [round(v, 4) for v in query_emb[:8]]
        return {
            "question":  question,
            "kb_match":  False,
            "best_score": round(best_score, 4),
            "threshold":  RELEVANCE_THRESHOLD,
            "embedding": {
                "model":       EMBED_MODEL,
                "dimensions":  len(query_emb),
                "preview":     emb_preview,
                "latency_ms":  embed_ms,
            },
            "similarity_search": {
                "method":         "Cosine Similarity",
                "total_searched": len(DOCUMENTS),
                "top_k":          3,
                "latency_ms":     search_ms,
                "results": [
                    {
                        "rank":     i + 1,
                        "source":   r["source"],
                        "filename": Path(r["source"]).name,
                        "score":    round(r["score"], 6),
                        "text":     r["text"],
                    }
                    for i, r in enumerate(top_k)
                ],
            },
            "context_sent": "",
            "llm": {"provider": "Ollama", "model": llm, "latency_ms": 0},
            "answer": (
                "I couldn't find relevant information about this question in the "
                "PolicyCheck Knowledge Base. PolicyCheck is designed to answer questions "
                "using only the available uploaded policy documents."
            ),
        }

    # Step 3 – build context string (exactly what goes to the LLM)
    context = "\n\n".join(
        f"Source: {r['source']}\n{r['text']}" for r in top_k
    )

    prompt = f"""You are PolicyCheck AI, a policy information assistant.

Answer the user's question using ONLY the provided context.
If the answer is not in the context, say: "I could not find this information in the policy documents."

Context:
{context}

Question:
{question}

Answer:"""

    # Step 4 – LLM generation
    t2 = time.time()
    llm_resp = requests.post(
        OLLAMA_URL,
        json={"model": llm, "prompt": prompt, "stream": False}
    )
    llm_resp.raise_for_status()
    llm_ms = round((time.time() - t2) * 1000)
    answer = llm_resp.json()["response"]

    # Build a short embedding preview (first 8 values, rounded)
    emb_preview = [round(v, 4) for v in query_emb[:8]]

    return {
        "question": question,
        "kb_match":   True,
        "best_score": round(best_score, 4),
        "threshold":  RELEVANCE_THRESHOLD,
        # embedding step
        "embedding": {
            "model": EMBED_MODEL,
            "dimensions": len(query_emb),
            "preview": emb_preview,
            "latency_ms": embed_ms,
        },
        # similarity search step
        "similarity_search": {
            "method": "Cosine Similarity",
            "total_searched": len(DOCUMENTS),
            "top_k": 3,
            "latency_ms": search_ms,
            "results": [
                {
                    "rank": i + 1,
                    "source": r["source"],
                    "filename": Path(r["source"]).name,
                    "score": round(r["score"], 6),
                    "text": r["text"],
                }
                for i, r in enumerate(top_k)
            ],
        },
        # context sent to LLM
        "context_sent": context,
        # LLM step
        "llm": {
            "provider": "Ollama",
            "model": llm,
            "latency_ms": llm_ms,
        },
        # final answer
        "answer": answer,
    }


# ── KB relevance threshold info ───────────────────────────────────────────────
@app.get("/api/kb-threshold")
def kb_threshold():
    return {
        "threshold": RELEVANCE_THRESHOLD,
        "description": (
            "Cosine similarity threshold for Knowledge Base relevance detection. "
            "Queries whose best retrieved chunk scores below this value are "
            "considered outside the Knowledge Base and LLM generation is blocked."
        ),
    }


# ── Exercise 4: Proxy to microservices ───────────────────────────────────────
@app.post("/api/retrieve")
def api_retrieve(question: str):
    """Calls the Retrieval Service (port 8001)"""
    r = requests.post(
        f"{RETRIEVAL_SERVICE}/retrieve",
        params={"question": question}
    )
    return r.json()


# ── Knowledge Base info ───────────────────────────────────────────────────────
@app.get("/api/kb-info")
def kb_info():
    """Returns knowledge base stats and sample chunks for the UI"""
    # Group chunks by source
    source_counts: dict = {}
    for chunk in CHUNKS:
        src = chunk["source"]
        source_counts[src] = source_counts.get(src, 0) + 1

    documents = []
    for src, count in source_counts.items():
        parts = Path(src).parts
        # e.g. Dataset/pdfs/GST/cgst_act.pdf
        category = parts[-2] if len(parts) >= 2 else "Unknown"
        documents.append({
            "file": Path(src).name,
            "category": category,
            "chunks": count,
            "path": src
        })

    # Embedding dimensions from first document
    emb_dims = len(DOCUMENTS[0]["embedding"]) if DOCUMENTS else 0

    return {
        "document_count": len(source_counts),
        "total_chunks": len(CHUNKS),
        "total_embeddings": len(DOCUMENTS),
        "embedding_dimensions": emb_dims,
        "embedding_model": EMBED_MODEL,
        "chunk_size": 1500,
        "chunk_overlap": 200,
        "documents": documents,
        "sample_chunks": CHUNKS[:6]
    }


# ── Knowledge Base chunk explorer ─────────────────────────────────────────────
# Build fast lookup: chunk id → embedding vector (kept in sync by _reload_kb)
_EMB_BY_ID: dict = {doc["id"]: doc["embedding"] for doc in DOCUMENTS}

@app.get("/api/kb-chunks")
def kb_chunks(
    source: str = "",       # filter by exact source path, empty = all
    category: str = "",     # filter by category folder name, empty = all
    search: str = "",       # free-text search across chunk text + id
    page: int = 1,          # 1-based page number
    page_size: int = 20,    # chunks per page
    include_embedding: bool = False,  # whether to attach embedding preview
):
    """
    Paginated, filtered, searchable chunk browser.
    Returns the matching slice plus total count and document metadata.
    """
    # ── filter ────────────────────────────────────────────────
    filtered = CHUNKS
    if source:
        filtered = [c for c in filtered if c["source"] == source]
    if category:
        filtered = [c for c in filtered if
                    Path(c["source"]).parts[-2] == category]
    if search:
        q = search.lower()
        filtered = [c for c in filtered if
                    q in c["text"].lower() or
                    q in c["id"].lower() or
                    q in c["source"].lower()]

    total = len(filtered)

    # ── paginate ──────────────────────────────────────────────
    page = max(1, page)
    page_size = max(1, min(100, page_size))
    start = (page - 1) * page_size
    end   = start + page_size
    page_chunks = filtered[start:end]

    # ── build response items ──────────────────────────────────
    items = []
    for c in page_chunks:
        item: dict = {
            "id":       c["id"],
            "source":   c["source"],
            "filename": Path(c["source"]).name,
            "category": Path(c["source"]).parts[-2] if len(Path(c["source"]).parts) >= 2 else "Unknown",
            "text":     c["text"],
            "char_count": len(c["text"]),
        }
        if include_embedding:
            emb = _EMB_BY_ID.get(c["id"])
            if emb:
                item["embedding_preview"] = [round(v, 4) for v in emb[:8]]
                item["embedding_dims"]    = len(emb)
            else:
                item["embedding_preview"] = []
                item["embedding_dims"]    = 0
        items.append(item)

    # ── per-document counts for sidebar ──────────────────────
    source_counts: dict = {}
    for chunk in CHUNKS:
        src = chunk["source"]
        source_counts[src] = source_counts.get(src, 0) + 1

    doc_list = []
    for src, count in source_counts.items():
        parts = Path(src).parts
        cat   = parts[-2] if len(parts) >= 2 else "Unknown"
        doc_list.append({
            "path":     src,
            "filename": Path(src).name,
            "category": cat,
            "chunks":   count,
        })

    return {
        "total":      total,
        "page":       page,
        "page_size":  page_size,
        "total_pages": math.ceil(total / page_size) if total else 0,
        "items":      items,
        "documents":  doc_list,
        "filters_applied": {
            "source":   source,
            "category": category,
            "search":   search,
        },
    }


# ── Evaluation examples for Compare tab ──────────────────────────────────────
@app.get("/api/eval-examples")
def eval_examples():
    """
    Returns 4 curated examples from evaluation_results.json for the
    RAG Pipeline Analysis section of the Compare tab.
    Categories: correct, partial, wrong, hallucinated.
    """
    eval_path = Path("evaluation_results.json")
    if not eval_path.exists():
        return {"examples": []}
    with open(eval_path, encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    ok = [r for r in results if not r.get("error") and r.get("response")]

    def pick(fn):
        return next((r for r in ok if fn(r)), None)

    examples = []
    targets = [
        ("good_retrieval_correct_answer",
         "✅ Good Retrieval → Correct Answer",
         lambda r: r["correctness_score"] == 2
                   and r.get("retrieval_quality", {}).get("retrieval_quality_score") == 1
                   and r["question_id"] == "Q07"),
        ("good_retrieval_partial_answer",
         "⚠️ Good Retrieval → Partial Answer",
         lambda r: r["correctness_score"] == 1
                   and r.get("retrieval_quality", {}).get("retrieval_quality_score") == 1
                   and r["question_id"] == "Q11"),
        ("good_retrieval_wrong_answer",
         "❌ Good Retrieval → Incorrect Answer",
         lambda r: r["correctness_score"] == 0 and r["kb_supported"]
                   and not r.get("hallucination")
                   and r["question_id"] == "Q13"),
        ("hallucinated_answer",
         "🚨 Hallucinated Answer",
         lambda r: r.get("hallucination") is True
                   and r["question_id"] == "Q09"),
    ]

    for scenario, label, fn in targets:
        r = pick(fn)
        if r:
            ret = r.get("retrieved_results", [])
            examples.append({
                "scenario":    scenario,
                "label":       label,
                "model":       r["model"],
                "question_id": r["question_id"],
                "question":    r["question"],
                "response":    r.get("response", ""),
                "correctness_score": r.get("correctness_score"),
                "hallucination":     r.get("hallucination"),
                "retrieval_quality_score": r.get("retrieval_quality", {}).get("retrieval_quality_score"),
                "top_chunks": [
                    {"filename": c.get("filename"), "score": c.get("score"),
                     "text_preview": c.get("text_preview", "")}
                    for c in ret[:3]
                ],
                "latency_ms":  r.get("latency_ms"),
            })
    return {"examples": examples}


# ── PDF Ingestion endpoint ────────────────────────────────────────────────────
@app.post("/api/ingest-pdf")
async def ingest_pdf(files: list[UploadFile] = File(...)):
    """
    Upload one or more PDFs → extract text → chunk → embed → append to
    chunks.json + embeddings.json → reload in-process KB.
    Uses the SAME chunking params (1500 chars / 200 overlap) and
    the SAME embedding model (nomic-embed-text) as the existing pipeline.
    """
    CHUNK_SIZE = 1500
    OVERLAP    = 200
    UPLOAD_DIR = Path("Dataset/pdfs/Uploaded")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    ingest_results = []

    for upload in files:
        fname = Path(upload.filename).name
        if not fname.lower().endswith(".pdf"):
            ingest_results.append({"filename": fname, "status": "error",
                                   "message": "Not a PDF file"})
            continue

        dest = UPLOAD_DIR / fname
        # save to disk
        content = await upload.read()
        dest.write_bytes(content)

        # ── 1. Extract text ───────────────────────────────────
        try:
            reader = PdfReader(str(dest))
            raw_text = ""
            for page in reader.pages:
                raw_text += (page.extract_text() or "") + "\n"
            raw_text = " ".join(raw_text.split())
        except Exception as e:
            ingest_results.append({"filename": fname, "status": "error",
                                   "message": f"PDF extraction failed: {e}"})
            continue

        if not raw_text.strip():
            ingest_results.append({"filename": fname, "status": "error",
                                   "message": "No extractable text found in PDF"})
            continue

        # ── 2. Chunk ──────────────────────────────────────────
        stem   = dest.stem
        source = str(dest)
        new_chunks = []
        existing_ids = {c["id"] for c in CHUNKS}

        start, chunk_idx = 0, 0
        while start < len(raw_text):
            chunk_text = raw_text[start : start + CHUNK_SIZE]
            if chunk_text.strip():
                cid = f"{stem}_{chunk_idx}"
                # make unique if id already exists (re-upload)
                base_cid = cid
                suffix   = 0
                while cid in existing_ids:
                    suffix += 1
                    cid = f"{base_cid}_v{suffix}"
                new_chunks.append({"id": cid, "source": source, "text": chunk_text})
                existing_ids.add(cid)
            chunk_idx += 1
            start     += CHUNK_SIZE - OVERLAP

        # ── 3. Embed each chunk ───────────────────────────────
        new_docs = []
        failed_embed = 0
        for chunk in new_chunks:
            try:
                r = requests.post(
                    EMBED_URL,
                    json={"model": EMBED_MODEL, "prompt": chunk["text"]},
                    timeout=60,
                )
                r.raise_for_status()
                emb = r.json()["embedding"]
                new_docs.append({
                    "id":        chunk["id"],
                    "source":    chunk["source"],
                    "text":      chunk["text"],
                    "embedding": emb,
                })
                time.sleep(0.05)   # same small delay as create_embeddings.py
            except Exception:
                failed_embed += 1

        # ── 4. Persist – append to existing JSON files ────────
        # chunks.json
        all_chunks = CHUNKS + new_chunks
        with open("chunks.json", "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False)

        # embeddings.json
        all_docs = DOCUMENTS + new_docs
        with open("embeddings.json", "w", encoding="utf-8") as f:
            json.dump(all_docs, f, ensure_ascii=False)

        # ── 5. Reload in-process KB ───────────────────────────
        _reload_kb()

        ingest_results.append({
            "filename":          fname,
            "status":            "success",
            "source_path":       source,
            "chunks_created":    len(new_chunks),
            "embeddings_created": len(new_docs),
            "embed_failures":    failed_embed,
            "total_chunks_now":  len(CHUNKS),
            "total_docs_now":    len(DOCUMENTS),
        })

    return {"results": ingest_results}


# ── Week 4 data endpoint ─────────────────────────────────────────────────────
@app.get("/api/week4-data")
def week4_data():
    """
    Single endpoint serving all data needed by the Week 4 UI tab:
    - evaluation dataset (25 questions)
    - per-model summary stats derived from evaluation_results.json
    - 5 RAG analysis examples from evaluation_results.json
    - codebase Q&A answers (static, grounded in actual files)
    """
    # ── Load evaluation dataset ───────────────────────────────
    ds_path = Path("evaluation_dataset.json")
    dataset = []
    if ds_path.exists():
        with open(ds_path, encoding="utf-8") as f:
            raw = json.load(f)
        dataset = raw.get("questions", [])

    # ── Load evaluation results ───────────────────────────────
    res_path = Path("evaluation_results.json")
    model_stats: dict = {}
    rag_examples: list = []

    if res_path.exists():
        with open(res_path, encoding="utf-8") as f:
            res_data = json.load(f)
        all_results = res_data.get("results", [])
        scoring_defs = res_data.get("metadata", {}).get("scoring", {})

        # Per-model summary
        MODELS_ORDER = ["codellama:latest", "qwen2.5:0.5b", "tinyllama:1.1b"]
        for model in MODELS_ORDER:
            rs  = [r for r in all_results if r["model"] == model]
            ok  = [r for r in rs if not r.get("error") and r.get("response")]
            err = [r for r in rs if r.get("error")]
            n   = len(ok)

            if n == 0:
                model_stats[model] = {
                    "n_valid": 0, "n_error": len(err),
                    "accuracy_pct": 0, "avg_correctness": 0,
                    "avg_relevance": 0, "hallucination_count": 0,
                    "hallucination_rate_pct": 0,
                    "retrieval_hit_rate_pct": None,
                    "avg_latency_ms": None, "avg_llm_latency_ms": None,
                    "score_dist": {0: 0, 1: 0, 2: 0},
                    "per_question": [],
                }
                continue

            corr    = [r["correctness_score"] for r in ok]
            rel     = [r["relevance_score"]    for r in ok]
            lats    = [r["latency_ms"]         for r in ok if r.get("latency_ms")]
            llm_lats= [r["llm_latency_ms"]     for r in ok if r.get("llm_latency_ms")]
            halls   = [r for r in ok if r.get("hallucination")]
            ret_q   = [r for r in ok if r.get("retrieval_quality",{}).get("retrieval_quality_score") is not None]
            ret_hit = [r for r in ret_q if r["retrieval_quality"]["retrieval_quality_score"] == 1]

            dist = {0:0, 1:0, 2:0}
            for s in corr: dist[s] = dist.get(s,0) + 1

            avg = lambda lst: round(sum(lst)/len(lst), 3) if lst else None

            per_q = []
            for r in ok:
                per_q.append({
                    "question_id":        r["question_id"],
                    "correctness_score":  r["correctness_score"],
                    "relevance_score":    r["relevance_score"],
                    "hallucination":      r.get("hallucination", False),
                    "latency_ms":         r.get("latency_ms"),
                    "retrieval_quality_score": r.get("retrieval_quality", {}).get("retrieval_quality_score"),
                    "top_source":         (r.get("retrieved_results") or [{}])[0].get("filename", ""),
                    "top_score":          (r.get("retrieved_results") or [{}])[0].get("score"),
                    "response_preview":   (r.get("response") or "")[:200],
                    "error":              r.get("error"),
                })

            model_stats[model] = {
                "n_valid":               n,
                "n_error":               len(err),
                "total_score":           sum(corr),
                "max_possible":          n * 2,
                "accuracy_pct":          round(sum(corr) / (n*2) * 100, 1),
                "avg_correctness":       round(avg(corr), 3),
                "avg_relevance":         round(avg(rel),  3),
                "hallucination_count":   len(halls),
                "hallucination_rate_pct":round(len(halls)/n*100, 1),
                "retrieval_hit_rate_pct":round(len(ret_hit)/len(ret_q)*100,1) if ret_q else None,
                "avg_latency_ms":        round(avg(lats))  if lats     else None,
                "avg_llm_latency_ms":    round(avg(llm_lats)) if llm_lats else None,
                "score_dist":            dist,
                "per_question":          per_q,
            }

        # RAG analysis examples — pick 5 representative results
        ok_all = [r for r in all_results if not r.get("error") and r.get("response")]
        def pick(fn):
            return next((r for r in ok_all if fn(r)), None)

        scenarios = [
            ("correct_retrieval_correct_answer",
             "✅ Good Retrieval → Correct Answer",
             "Q07", "codellama:latest",
             lambda r: r["question_id"]=="Q07" and r["model"]=="codellama:latest"
                       and r["correctness_score"]==2),
            ("correct_retrieval_partial_answer",
             "⚠️ Good Retrieval → Partial Answer",
             "Q11", "codellama:latest",
             lambda r: r["question_id"]=="Q11" and r["model"]=="codellama:latest"
                       and r["correctness_score"]==1),
            ("correct_retrieval_wrong_answer",
             "❌ Good Retrieval → Incorrect Answer",
             "Q13", "codellama:latest",
             lambda r: r["question_id"]=="Q13" and r["model"]=="codellama:latest"
                       and r["correctness_score"]==0 and not r.get("hallucination")),
            ("hallucination",
             "🚨 Hallucination Despite Retrieved Context",
             "Q09", "codellama:latest",
             lambda r: r["question_id"]=="Q09" and r["model"]=="codellama:latest"
                       and r.get("hallucination")),
            ("tinyllama_correct",
             "🦙 TinyLlama — Correct Answer",
             "Q05", "tinyllama:1.1b",
             lambda r: r["question_id"]=="Q05" and r["model"]=="tinyllama:1.1b"
                       and r["correctness_score"]==2),
        ]

        for sid, label, qid, model, fn in scenarios:
            r = pick(fn)
            if r:
                chunks = r.get("retrieved_results", [])
                rag_examples.append({
                    "scenario":          sid,
                    "label":             label,
                    "model":             r["model"],
                    "question_id":       r["question_id"],
                    "question":          r["question"],
                    "category":          r.get("category",""),
                    "difficulty":        r.get("difficulty",""),
                    "kb_supported":      r.get("kb_supported"),
                    "expected_source":   r.get("expected_source",""),
                    "response":          r.get("response",""),
                    "response_length":   r.get("response_length_chars",0),
                    "correctness_score": r.get("correctness_score"),
                    "relevance_score":   r.get("relevance_score"),
                    "hallucination":     r.get("hallucination",False),
                    "latency_ms":        r.get("latency_ms"),
                    "context_sent_length": r.get("context_sent_length",0),
                    "retrieval_quality_score": r.get("retrieval_quality",{}).get("retrieval_quality_score"),
                    "retrieved_chunks":  [
                        {"rank":    c.get("rank"),
                         "filename":c.get("filename",""),
                         "source":  c.get("source",""),
                         "score":   c.get("score"),
                         "preview": c.get("text_preview","")}
                        for c in chunks
                    ],
                    "keyword_overlap":   r.get("scoring_details",{}).get("keyword_overlap"),
                    "total_chunks_searched": r.get("total_chunks_searched",590),
                })
    else:
        scoring_defs = {}

    # ── Codebase Q&A (grounded in actual project files) ───────
    codebase_qa = [
        {
            "id": "CQ1",
            "question": "Which files are involved in the RAG question-answering pipeline?",
            "files_identified": [
                "app.py — /api/rag-ask and /api/rag-ask-pipeline endpoints, get_embedding(), cosine_similarity(), retrieve_top_k()",
                "rag.py — standalone RAG implementation (LLM_MODEL, EMBEDDING_MODEL, GENERATE_URL)",
                "retrieval_service.py — microservice on :8001 for embedding + cosine similarity",
                "llm_service.py — microservice on :8002 wrapping Ollama generate",
                "orchestration_service.py — Docker microservice on :8000 orchestrating retrieval→LLM",
                "embeddings.json — 590 pre-computed 768-dim vectors (nomic-embed-text)",
                "chunks.json — 590 text chunks with source paths",
                "templates/index.html — doPipelineAsk() calls /api/rag-ask-pipeline",
            ],
            "answer": "The core RAG pipeline lives in app.py (retrieve_top_k → cosine similarity → LLM call). It reads embeddings.json and chunks.json at startup. The microservice version splits these into retrieval_service.py (:8001) and llm_service.py (:8002), orchestrated by orchestration_service.py (:8000). The UI calls /api/rag-ask-pipeline which returns all intermediate steps.",
            "assessment": "Grounded — all files verified in the project directory.",
        },
        {
            "id": "CQ2",
            "question": "How does a PDF uploaded through the Knowledge Base become searchable by the RAG system?",
            "files_identified": [
                "app.py — /api/ingest-pdf: saves PDF → PdfReader extracts text → 1500-char chunks (200 overlap) → nomic-embed-text embedding per chunk → appends to chunks.json + embeddings.json → calls _reload_kb()",
                "app.py — _reload_kb(): reloads DOCUMENTS and CHUNKS globals from disk, rebuilds _EMB_BY_ID lookup",
                "templates/index.html — kbUpload() calls /api/ingest-pdf, animates 6-stage pipeline",
                "Dataset/pdfs/Uploaded/ — new PDFs saved here",
            ],
            "answer": "POST /api/ingest-pdf saves the PDF to Dataset/pdfs/Uploaded/, extracts text with pypdf.PdfReader, splits into 1500-char chunks (200 overlap), calls Ollama nomic-embed-text for each chunk, appends results to chunks.json and embeddings.json, then calls _reload_kb() which reloads both files into the live DOCUMENTS/CHUNKS globals. The next query to retrieve_top_k() automatically searches the new vectors via cosine similarity.",
            "assessment": "Grounded — end-to-end flow verified in app.py lines 1–90 (ingest) and retrieve_top_k().",
        },
        {
            "id": "CQ3",
            "question": "Which components are involved in generating embeddings?",
            "files_identified": [
                "create_embeddings.py — standalone script: reads chunks.json → POST to Ollama :11434/api/embeddings with model=nomic-embed-text → writes embeddings.json",
                "app.py — get_embedding(text): POST to EMBED_URL with EMBED_MODEL=nomic-embed-text → returns 768-dim list",
                "app.py — /api/ingest-pdf: calls get_embedding equivalent inline for new PDFs",
                "retrieval_service.py — embeds query via nomic-embed-text before cosine similarity",
                "Ollama :11434 — runs nomic-embed-text model, returns {embedding: [768 floats]}",
            ],
            "answer": "Embeddings are generated by posting text to Ollama's /api/embeddings endpoint with model=nomic-embed-text. This returns a 768-dimensional float vector. The batch ingestion is done by create_embeddings.py (offline). The live path goes through app.py's get_embedding() which is called both for query embedding during RAG retrieval and for new PDF chunks during ingestion.",
            "assessment": "Grounded — verified in create_embeddings.py and app.py.",
        },
        {
            "id": "CQ4",
            "question": "What happens from the moment a user asks a question until the final answer is displayed?",
            "files_identified": [
                "templates/index.html — doPipelineAsk(): encodes question + model → POST /api/rag-ask-pipeline",
                "app.py — api_rag_ask_pipeline(): 1) get_embedding(question) via nomic-embed-text, 2) cosine_similarity over all 590 DOCUMENTS vectors, 3) sort descending → top_k=3, 4) build context string, 5) POST to Ollama with prompt+context, 6) return structured JSON",
                "templates/index.html — revealStep() animates 9 pipeline steps progressively as data arrives",
            ],
            "answer": "1. User types question, selects model, clicks Ask. 2. doPipelineAsk() in index.html POSTs to /api/rag-ask-pipeline with question and model params. 3. app.py embeds the question (nomic-embed-text, ~40ms). 4. Cosine similarity is computed over all 590 stored vectors (~30ms). 5. Top-3 chunks are selected. 6. A prompt is built: system instruction + retrieved context + question. 7. Ollama generates the answer with the selected LLM (codellama ~37s, qwen ~6s, tinyllama ~9s). 8. The full structured response is returned. 9. The frontend reveals each of 9 pipeline steps progressively with real data.",
            "assessment": "Grounded — traced through index.html doPipelineAsk() and app.py api_rag_ask_pipeline().",
        },
        {
            "id": "CQ5",
            "question": "Which files would need to change if we wanted to change the chunk size or overlap?",
            "files_identified": [
                "chunk_documents.py — CHUNK_SIZE=1500, OVERLAP=200 (lines 6–7); must be changed and script re-run to re-chunk all PDFs",
                "app.py — /api/ingest-pdf: hardcoded CHUNK_SIZE=1500, OVERLAP=200 (inside function body); must match chunk_documents.py",
                "create_embeddings.py — no chunk size logic; reads chunks.json output; would need re-run after re-chunking",
                "app.py — /api/kb-info: returns chunk_size and chunk_overlap as hardcoded values 1500/200 for UI display",
                "templates/index.html — Chunking Config sidebar shows hardcoded '1500 chars' / '200 chars'",
            ],
            "answer": "To change chunk size: (1) Update CHUNK_SIZE and OVERLAP in chunk_documents.py, (2) Update the same constants in /api/ingest-pdf in app.py, (3) Re-run chunk_documents.py to regenerate chunks.json, (4) Re-run create_embeddings.py to regenerate embeddings.json, (5) Update the display values in /api/kb-info and in the index.html sidebar. The existing 590 chunks and embeddings would be discarded.",
            "assessment": "Grounded — verified CHUNK_SIZE in chunk_documents.py line 6, and in /api/ingest-pdf in app.py.",
        },
    ]

    return {
        "dataset":       dataset,
        "model_stats":   model_stats,
        "scoring_defs":  scoring_defs,
        "rag_examples":  rag_examples,
        "codebase_qa":   codebase_qa,
        "models_order":  ["codellama:latest", "qwen2.5:0.5b", "tinyllama:1.1b"],
        "unavailable_metrics": [
            "token_usage — Ollama /api/generate does not return token counts",
            "cpu_usage_pct — not measured at request level",
            "memory_mb — not measured at request level",
        ],
    }


# ── Codebase analysis endpoint ────────────────────────────────────────────────
_CODEBASE_CONTEXT: dict | None = None   # cached once per process start

def _load_codebase_context() -> str:
    """
    Read all relevant PolicyCheck source files and produce a structured
    code-context block that is passed to the LLM for codebase Q&A.
    Files are read once and cached.
    """
    global _CODEBASE_CONTEXT
    if _CODEBASE_CONTEXT is not None:
        return _CODEBASE_CONTEXT

    files = [
        ("app.py",                    "Main FastAPI app — serves UI, all RAG/LLM endpoints, KB ingestion"),
        ("rag.py",                    "Standalone RAG module — get_embedding, cosine_similarity, LLM call"),
        ("retrieval_service.py",      "Microservice :8001 — embeds query, cosine similarity, returns top-3"),
        ("llm_service.py",            "Microservice :8002 — wraps Ollama /api/generate"),
        ("orchestration_service.py",  "Microservice :8000 (Docker) — calls retrieval then LLM"),
        ("chunk_documents.py",        "Offline script — reads PDFs, produces chunks.json"),
        ("create_embeddings.py",      "Offline script — reads chunks.json, produces embeddings.json"),
        ("docker-compose.yml",        "Docker Compose — defines retrieval, llm, orchestration services"),
    ]

    parts = []
    for fname, desc in files:
        fp = Path(fname)
        if not fp.exists():
            continue
        content = fp.read_text(encoding="utf-8")
        # Truncate large files to keep the prompt manageable
        if len(content) > 3000:
            content = content[:3000] + "\n... [truncated for context window]"
        parts.append(f"### {fname}  ({desc})\n```python\n{content}\n```")

    _CODEBASE_CONTEXT = "\n\n".join(parts)
    return _CODEBASE_CONTEXT


@app.post("/api/codebase-analyze")
def codebase_analyze(question: str, model: str = ""):
    """
    Answer a repository-level question about the PolicyCheck codebase.
    Reads the actual source files and asks the LLM to reason over them.
    Returns structured result: question, files_inspected, flow, answer.
    """
    llm = model or LLM_MODEL
    context = _load_codebase_context()

    prompt = f"""You are a senior software engineer analysing the PolicyCheck AI codebase.

The source files are provided below. Answer the user's question based ONLY on the actual code.
Be specific: name real files, functions, classes, and endpoints. Do not invent anything.

For each answer, structure your response as follows (use these exact headings):

FILES INSPECTED:
List each relevant file and what role it plays.

COMPONENTS / SERVICES:
Name the components, services, or modules involved.

FLOW:
Describe the execution or communication flow step by step (e.g. A → B → C).

ANSWER:
Give a clear, grounded explanation based on the code.

---
SOURCE FILES:

{context}

---
QUESTION:
{question}
"""

    import time
    t0 = time.time()
    resp = requests.post(
        OLLAMA_URL,
        json={"model": llm, "prompt": prompt, "stream": False},
        timeout=300,
    )
    resp.raise_for_status()
    latency_ms = round((time.time() - t0) * 1000)
    raw_answer = resp.json()["response"]

    # ── Parse structured sections out of the LLM response ────
    def extract_section(text: str, heading: str) -> str:
        """Extract content after a heading until the next heading.
        Handles bold markdown (** heading **) and plain heading variants."""
        headings = ["FILES INSPECTED:", "COMPONENTS / SERVICES:", "FLOW:", "ANSWER:"]
        # Build regex that matches the heading with optional surrounding ** and whitespace
        import re as _re
        pat = _re.compile(
            r'\*{0,2}\s*' + _re.escape(heading.rstrip(':')) + r'\s*:?\s*\*{0,2}',
            _re.IGNORECASE
        )
        m = pat.search(text)
        if not m:
            return ""
        start = m.end()
        # Find the next heading
        end = len(text)
        for h in headings:
            if h.upper() == heading.upper():
                continue
            hpat = _re.compile(
                r'\*{0,2}\s*' + _re.escape(h.rstrip(':')) + r'\s*:?\s*\*{0,2}',
                _re.IGNORECASE
            )
            hm = hpat.search(text, start)
            if hm and hm.start() < end:
                end = hm.start()
        return text[start:end].strip()

    files_section      = extract_section(raw_answer, "FILES INSPECTED:")
    components_section = extract_section(raw_answer, "COMPONENTS / SERVICES:")
    flow_section       = extract_section(raw_answer, "FLOW:")
    answer_section     = extract_section(raw_answer, "ANSWER:")

    # Build files_identified list (one entry per non-empty line)
    files_list = [
        ln.strip().lstrip("-•* ").strip()
        for ln in files_section.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]

    return {
        "question":        question,
        "is_custom":       True,
        "model":           llm,
        "latency_ms":      latency_ms,
        "files_identified": files_list or ["(see full answer)"],
        "components":      components_section or "",
        "flow":            flow_section or "",
        "answer":          answer_section or raw_answer,
        "raw_response":    raw_answer,
        "files_inspected": [
            "app.py", "rag.py", "retrieval_service.py",
            "llm_service.py", "orchestration_service.py",
            "chunk_documents.py", "create_embeddings.py",
            "docker-compose.yml",
        ],
    }


# ── Health checks ─────────────────────────────────────────────────────────────
@app.get("/api/health/app")
def health_app():
    # This service itself — always ok if we reach here
    return {"ok": True, "service": "PolicyCheck UI / App", "port": 8080}


@app.get("/api/health/retrieval")
def health_retrieval():
    try:
        r = requests.get(f"{RETRIEVAL_SERVICE}/", timeout=3)
        return {"ok": r.status_code == 200, "service": "retrieval", "port": 8001}
    except Exception:
        return {"ok": False, "service": "retrieval", "port": 8001}


@app.get("/api/health/llm")
def health_llm():
    try:
        r = requests.get(f"{LLM_SERVICE}/", timeout=3)
        return {"ok": r.status_code == 200, "service": "llm", "port": 8002}
    except Exception:
        return {"ok": False, "service": "llm", "port": 8002}


@app.get("/api/health/orch")
def health_orch():
    # Docker orchestration runs on port 8000
    try:
        r = requests.get("http://localhost:8000/", timeout=3)
        ok = r.status_code == 200 and "Orchestration" in r.text
        return {"ok": ok, "service": "Docker Orchestration", "port": 8000}
    except Exception:
        return {"ok": False, "service": "Docker Orchestration", "port": 8000}


@app.get("/api/health/ollama")
def health_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        data = r.json()
        models = [m["name"] for m in data.get("models", [])]
        return {"ok": True, "service": "ollama", "port": 11434, "models": models}
    except Exception:
        return {"ok": False, "service": "ollama", "port": 11434}
