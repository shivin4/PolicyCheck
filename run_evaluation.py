"""
PolicyCheck AI — Week 4 Exercise 3: Quantitative Evaluation
Runs all 25 evaluation questions against 3 models via the live
/api/rag-ask-pipeline endpoint and scores each response.
Saves: evaluation_results.json
"""

import json
import time
import re
import sys
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────
APP_URL   = "http://localhost:8080"
MODELS    = ["codellama:latest", "qwen2.5:0.5b", "tinyllama:1.1b"]
DATASET   = "evaluation_dataset.json"
OUT_FILE  = "evaluation_results.json"
TIMEOUT   = 300   # seconds per request (codellama can be slow)
PAUSE_S   = 3     # seconds to wait between requests (lets Ollama recover)
MAX_RETRY = 2     # retry once on connection error

# ── Load dataset ──────────────────────────────────────────────
with open(DATASET) as f:
    dataset = json.load(f)
questions = dataset["questions"]
print(f"Loaded {len(questions)} questions from {DATASET}")

# ── Scoring helpers ───────────────────────────────────────────

def keyword_overlap(response: str, ground_truth: str) -> float:
    """
    Simple keyword overlap score (0.0–1.0).
    Extracts meaningful tokens (≥4 chars) from ground truth,
    checks how many appear in the model response (case-insensitive).
    This is a deterministic, reproducible proxy for correctness.
    """
    stop = {"with","that","this","from","have","been","they","will",
            "which","when","were","also","into","than","such","more",
            "their","under","shall","person","where","section","act,",
            "made","upon","like","both","each","must","may,","not,"}

    def tokens(text):
        words = re.findall(r"[a-z]{4,}", text.lower())
        return set(w for w in words if w not in stop)

    gt_tokens = tokens(ground_truth)
    if not gt_tokens:
        return 0.0
    resp_tokens = tokens(response)
    matched = gt_tokens & resp_tokens
    return round(len(matched) / len(gt_tokens), 4)


def score_correctness(response: str, question: dict) -> dict:
    """
    Returns correctness_score (0/1/2), method, and details.
    For KB-supported questions: keyword overlap against ground truth.
    For hallucination traps (kb_supported=False):
        2 if model says 'not found / not in documents'
        0 if model gives a specific fabricated answer
    """
    resp_lower = response.lower().strip()

    if not question["kb_supported"]:
        # Hallucination trap — correct answer is "not found"
        not_found_phrases = [
            "not found", "not available", "not in the", "cannot find",
            "could not find", "no information", "not provided",
            "not mentioned", "outside", "not covered", "not contain",
            "does not contain", "i could not", "not part of",
            "not included", "not present", "no specific",
        ]
        is_refusal = any(p in resp_lower for p in not_found_phrases)
        # Penalise if model gives a specific number / rate
        gives_number = bool(re.search(r'\b(0|5|10|12|15|18|20|28)\s*%', resp_lower))
        gives_slab   = bool(re.search(r'\b(lakh|tax slab|income tax rate|percent)', resp_lower))

        if is_refusal and not gives_number:
            return {"correctness_score": 2, "hallucination": False,
                    "method": "refusal_check",
                    "details": "Correctly declined to answer out-of-scope question"}
        elif gives_number or gives_slab:
            return {"correctness_score": 0, "hallucination": True,
                    "method": "refusal_check",
                    "details": "Hallucinated specific value for out-of-scope question"}
        else:
            return {"correctness_score": 1, "hallucination": False,
                    "method": "refusal_check",
                    "details": "Partial refusal — vague but no fabricated value"}
    else:
        # KB-supported question — score by keyword overlap
        overlap = keyword_overlap(response, question["ground_truth"])
        if overlap >= 0.40:
            score = 2
        elif overlap >= 0.20:
            score = 1
        else:
            score = 0
        return {"correctness_score": score, "hallucination": False,
                "method": "keyword_overlap",
                "keyword_overlap": overlap,
                "details": f"Keyword overlap with ground truth: {overlap:.4f}"}


def score_relevance(response: str, question: str) -> float:
    """
    Relevance score (0.0–1.0): keyword overlap between the question
    and the response — checks if the response is topically on-point.
    """
    def tokens(text):
        return set(re.findall(r"[a-z]{4,}", text.lower()))
    q_tokens  = tokens(question)
    r_tokens  = tokens(response)
    if not q_tokens:
        return 0.0
    return round(len(q_tokens & r_tokens) / len(q_tokens), 4)


def score_retrieval_quality(retrieved_results: list, expected_source: str) -> dict:
    """
    Checks whether at least one retrieved chunk came from the expected
    source document. Returns score 1 (hit) or 0 (miss) and the ranks.
    """
    if expected_source == "none" or not expected_source:
        return {"retrieval_quality_score": None,
                "expected_source_found": None,
                "ranks_matched": [],
                "details": "No expected source (hallucination trap question)"}

    fname = expected_source.lower()
    matched_ranks = []
    for i, r in enumerate(retrieved_results):
        src = r.get("source", "").lower()
        if fname.replace(".pdf", "") in src:
            matched_ranks.append(i + 1)

    found = len(matched_ranks) > 0
    return {
        "retrieval_quality_score": 1 if found else 0,
        "expected_source_found": found,
        "ranks_matched": matched_ranks,
        "details": f"Expected source '{expected_source}' found at ranks {matched_ranks}" if found
                   else f"Expected source '{expected_source}' NOT in top-3 results"
    }


# ── Main evaluation loop ──────────────────────────────────────
results = []
total   = len(MODELS) * len(questions)
done    = 0

print(f"\nStarting evaluation: {len(MODELS)} models × {len(questions)} questions = {total} calls")
print(f"Timeout per request: {TIMEOUT}s\n")

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"  MODEL: {model}")
    print(f"{'='*60}")

    for q in questions:
        done += 1
        print(f"  [{done:02d}/{total}] {q['id']} — {q['question'][:55]}...", end=" ", flush=True)

        url = (f"{APP_URL}/api/rag-ask-pipeline"
               f"?question={urllib.parse.quote(q['question'])}"
               f"&model={urllib.parse.quote(model)}")

        t_start = time.perf_counter()
        error   = None
        raw     = None

        for attempt in range(MAX_RETRY + 1):
            try:
                req  = urllib.request.Request(url, method="POST")
                with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                    raw = json.loads(resp.read().decode())
                break   # success
            except urllib.error.URLError as e:
                error = f"URLError: {e.reason}"
                if attempt < MAX_RETRY:
                    print(f"\n    [retry {attempt+1}] {error} — waiting 10s...", end=" ", flush=True)
                    time.sleep(10)
            except Exception as e:
                error = str(e)
                break

        latency_ms = round((time.perf_counter() - t_start) * 1000)

        if error or raw is None:
            print(f"ERROR ({latency_ms}ms) — {error}")
            results.append({
                "model": model,
                "question_id": q["id"],
                "question": q["question"],
                "category": q["category"],
                "difficulty": q["difficulty"],
                "question_type": q["type"],
                "kb_supported": q["kb_supported"],
                "expected_source": q["expected_source"],
                "latency_ms": latency_ms,
                "error": error,
                "response": None,
                "retrieved_results": [],
                "embedding_model": None,
                "embedding_dims": None,
                "embedding_latency_ms": None,
                "similarity_search_latency_ms": None,
                "llm_latency_ms": None,
                "correctness_score": 0,
                "relevance_score": 0.0,
                "hallucination": False,
                "retrieval_quality": {},
                "scoring_details": {},
            })
            continue

        # ── Extract fields from pipeline response ─────────────
        answer    = raw.get("answer", "")
        sim_info  = raw.get("similarity_search", {})
        emb_info  = raw.get("embedding", {})
        llm_info  = raw.get("llm", {})
        retrieved = sim_info.get("results", [])

        # ── Score ─────────────────────────────────────────────
        corr    = score_correctness(answer, q)
        rel     = score_relevance(answer, q["question"])
        ret_q   = score_retrieval_quality(retrieved, q["expected_source"])

        print(f"✓  {latency_ms}ms  corr={corr['correctness_score']}  "
              f"rel={rel:.2f}  ret={ret_q.get('retrieval_quality_score','n/a')}")

        # Brief pause between requests so Ollama doesn't get overwhelmed
        time.sleep(PAUSE_S)

        results.append({
            "model":            model,
            "question_id":      q["id"],
            "question":         q["question"],
            "category":         q["category"],
            "difficulty":       q["difficulty"],
            "question_type":    q["type"],
            "kb_supported":     q["kb_supported"],
            "expected_source":  q["expected_source"],
            "latency_ms":       latency_ms,
            "llm_latency_ms":   llm_info.get("latency_ms"),
            "embedding_latency_ms":          emb_info.get("latency_ms"),
            "similarity_search_latency_ms":  sim_info.get("latency_ms"),
            "embedding_model":  emb_info.get("model"),
            "embedding_dims":   emb_info.get("dimensions"),
            "total_chunks_searched": sim_info.get("total_searched"),
            "top_k":            sim_info.get("top_k"),
            "retrieved_results": [
                {
                    "rank":     r.get("rank"),
                    "filename": r.get("filename"),
                    "source":   r.get("source"),
                    "score":    r.get("score"),
                    "text_preview": r.get("text", "")[:200],
                }
                for r in retrieved
            ],
            "context_sent_length": len(raw.get("context_sent", "")),
            "response":         answer,
            "response_length_chars": len(answer),
            "error":            None,
            # ── Scores ────────────────────────────────────────
            "correctness_score":        corr["correctness_score"],
            "hallucination":            corr["hallucination"],
            "relevance_score":          rel,
            "retrieval_quality":        ret_q,
            "scoring_details":          corr,
            # ── Unavailable metrics (explicitly noted) ────────
            "token_usage":      "unavailable — Ollama /api/generate does not return token counts via this endpoint",
            "cpu_usage_pct":    "unavailable — not measured at request level",
            "memory_mb":        "unavailable — not measured at request level",
        })

# ── Save raw results ──────────────────────────────────────────
output = {
    "metadata": {
        "exercise":     "Week 4 Exercise 3 — Quantitative Evaluation",
        "run_at":       datetime.now(timezone.utc).isoformat(),
        "models":       MODELS,
        "total_calls":  len(results),
        "dataset_file": DATASET,
        "app_url":      APP_URL,
        "scoring": {
            "correctness": "0=wrong/hallucinated, 1=partial, 2=correct. KB-supported: keyword overlap vs ground truth (≥0.40→2, ≥0.20→1, else 0). Hallucination traps: 2=refused correctly, 1=vague, 0=fabricated.",
            "relevance":   "Keyword overlap (0.0-1.0) between question tokens and response tokens.",
            "retrieval_quality": "1 if expected source document appears in top-3 retrieved chunks, 0 if not, null for hallucination-trap questions.",
            "hallucination": "True if model gave a specific fabricated answer to an out-of-scope question.",
        }
    },
    "results": results
}

with open(OUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n\nSaved {len(results)} results → {OUT_FILE}")

# ── Quick sanity summary ──────────────────────────────────────
from collections import defaultdict
by_model = defaultdict(list)
for r in results:
    by_model[r["model"]].append(r)

print("\nQuick per-model totals:")
for model, rs in by_model.items():
    valid = [r for r in rs if r["error"] is None]
    avg_corr = sum(r["correctness_score"] for r in valid) / len(valid) if valid else 0
    avg_lat  = sum(r["latency_ms"] for r in valid) / len(valid) if valid else 0
    halls    = sum(1 for r in valid if r["hallucination"])
    print(f"  {model:<22}  n={len(valid)}  avg_correctness={avg_corr:.2f}  "
          f"avg_latency={avg_lat:.0f}ms  hallucinations={halls}")
