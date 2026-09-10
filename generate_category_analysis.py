"""
PolicyCheck AI — Category-wise Analysis Generator
Reads evaluation_results.json + evaluation_dataset.json
Computes per-task-category metrics for each model
Saves: category_analysis.json
"""

import json
from collections import defaultdict

# ── Load data ─────────────────────────────────────────────────
with open("evaluation_results.json", "r", encoding="utf-8") as f:
    eval_data = json.load(f)

with open("evaluation_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

results = eval_data["results"]
questions = dataset["questions"]

# Build lookup: question_id → task_category
q_to_task = {q["id"]: q["task_category"] for q in questions}
q_to_text  = {q["id"]: q["question"]      for q in questions}

MODELS = ["codellama:latest", "qwen2.5:0.5b", "tinyllama:1.1b"]

TASK_CATEGORIES = [
    "Explanation",
    "Retrieval",
    "Dependency Understanding",
    "Bug Analysis",
    "RAG-based Question",
    "Hallucination Trap",
]

# ── Per-category per-model analysis ──────────────────────────
category_analysis = {}

for task_cat in TASK_CATEGORIES:
    # Question IDs that belong to this category
    cat_qids = [q["id"] for q in questions if q["task_category"] == task_cat]
    category_analysis[task_cat] = {
        "question_ids": cat_qids,
        "question_count": len(cat_qids),
        "models": {}
    }

    for model in MODELS:
        # Get results for this model + this category
        model_cat_results = [
            r for r in results
            if r["model"] == model and r["question_id"] in cat_qids
        ]

        if not model_cat_results:
            category_analysis[task_cat]["models"][model] = {}
            continue

        n = len(model_cat_results)

        avg_correctness = sum(r.get("correctness_score") or 0 for r in model_cat_results) / n
        avg_relevance   = sum(r.get("relevance_score")   or 0 for r in model_cat_results) / n
        avg_latency     = sum(r.get("latency_ms")        or 0 for r in model_cat_results) / n
        avg_llm_latency = sum(r.get("llm_latency_ms")    or 0 for r in model_cat_results) / n
        hallucination_count = sum(1 for r in model_cat_results if r.get("hallucination"))

        retrieval_scores = [
            r["retrieval_quality"]["retrieval_quality_score"]
            for r in model_cat_results
            if isinstance(r.get("retrieval_quality"), dict)
            and r["retrieval_quality"].get("retrieval_quality_score") is not None
        ]
        retrieval_hit_rate = (sum(retrieval_scores) / len(retrieval_scores)) if retrieval_scores else None

        # Per-question detail
        per_question = []
        for r in model_cat_results:
            per_question.append({
                "question_id":      r["question_id"],
                "question":         q_to_text.get(r["question_id"], ""),
                "correctness_score": r.get("correctness_score") or 0,
                "relevance_score":  r.get("relevance_score") or 0,
                "latency_ms":       r.get("latency_ms") or 0,
                "hallucination":    r.get("hallucination") or False,
                "retrieval_hit":    (
                    r["retrieval_quality"].get("retrieval_quality_score") == 1
                    if isinstance(r.get("retrieval_quality"), dict) else None
                ),
                "response_preview": (r.get("response") or "")[:200],
                "retrieved_sources": [
                    x.get("filename", "") for x in (r.get("retrieved_results") or [])
                ],
            })

        category_analysis[task_cat]["models"][model] = {
            "n":                    n,
            "avg_correctness":      round(avg_correctness, 3),
            "avg_relevance":        round(avg_relevance, 3),
            "avg_latency_ms":       round(avg_latency, 1),
            "avg_llm_latency_ms":   round(avg_llm_latency, 1),
            "hallucination_count":  hallucination_count,
            "retrieval_hit_rate":   round(retrieval_hit_rate, 3) if retrieval_hit_rate is not None else None,
            "per_question":         per_question,
        }

# ── Best model per category ───────────────────────────────────
best_model_per_category = {}

for task_cat, data in category_analysis.items():
    best = {"correctness": None, "latency": None, "hallucination": None}

    scores = {
        m: data["models"][m].get("avg_correctness", 0)
        for m in MODELS if data["models"].get(m)
    }
    latencies = {
        m: data["models"][m].get("avg_latency_ms", 99999)
        for m in MODELS if data["models"].get(m)
    }
    halls = {
        m: data["models"][m].get("hallucination_count", 99)
        for m in MODELS if data["models"].get(m)
    }

    if scores:
        best["correctness"] = max(scores, key=scores.get)
    if latencies:
        best["latency"]     = min(latencies, key=latencies.get)
    if halls:
        best["hallucination"] = min(halls, key=halls.get)

    best_model_per_category[task_cat] = best

# ── Overall winner summary ────────────────────────────────────
wins = defaultdict(int)
for task_cat, best in best_model_per_category.items():
    if best["correctness"]:
        wins[best["correctness"]] += 1

# ── Save output ───────────────────────────────────────────────
output = {
    "metadata": {
        "generated_from": ["evaluation_results.json", "evaluation_dataset.json"],
        "models": MODELS,
        "task_categories": TASK_CATEGORIES,
        "total_questions": len(questions),
        "total_results": len(results),
    },
    "category_analysis": category_analysis,
    "best_model_per_category": best_model_per_category,
    "category_wins": dict(wins),
}

with open("category_analysis.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("✓ Saved category_analysis.json\n")

# ── Print readable summary ────────────────────────────────────
print("=" * 70)
print("CATEGORY-WISE MODEL COMPARISON SUMMARY")
print("=" * 70)

for task_cat in TASK_CATEGORIES:
    data = category_analysis[task_cat]
    print(f"\n📂 {task_cat.upper()} ({data['question_count']} questions: {', '.join(data['question_ids'])})")
    print(f"  {'Model':<22} {'Correctness':>12} {'Relevance':>10} {'Latency':>10} {'Hallucinations':>15} {'Retrieval Hit':>14}")
    print(f"  {'-'*22} {'-'*12} {'-'*10} {'-'*10} {'-'*15} {'-'*14}")
    for model in MODELS:
        m = data["models"].get(model, {})
        if not m:
            continue
        rhr = f"{m['retrieval_hit_rate']*100:.1f}%" if m.get("retrieval_hit_rate") is not None else "N/A"
        print(
            f"  {model:<22} "
            f"{m['avg_correctness']:>10.2f}/2 "
            f"{m['avg_relevance']:>10.2f} "
            f"{m['avg_latency_ms']:>8.0f}ms "
            f"{m['hallucination_count']:>15} "
            f"{rhr:>14}"
        )
    best = best_model_per_category[task_cat]
    print(f"  → Best accuracy: {best['correctness']}  |  Fastest: {best['latency']}  |  Least hallucinations: {best['hallucination']}")

print("\n" + "=" * 70)
print("CATEGORY WINS (most accurate model per category)")
print("=" * 70)
for model, w in sorted(wins.items(), key=lambda x: -x[1]):
    print(f"  {model}: {w} category win(s)")
