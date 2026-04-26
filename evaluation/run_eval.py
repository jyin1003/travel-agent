"""
Run full evaluation across all variants and queries.
Usage: python -m evaluation.run_eval
Output: evaluation/results.csv
"""
import csv, json, os, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from evaluation.variants import run_s0, run_s1, run_s2, run_s3, run_s4
from evaluation.metrics import compute_mrr, llm_judge

# ── Test queries ──────────────────────────────────────────────────────────────

QUERIES = {
    "Q1":  "How much did I spend on transport in Japan?",
    "Q2":  "What hotel did I stay at in Lisbon?",
    "Q3":  "How long was my Vietnam trip?",
    "Q4":  "Find a photo showing a street food stall I visited.",
    "Q5":  "What was I eating when I visited the night market?",
    "Q6":  "Which photos match the place I spent the most at dinner?",
    "Q7":  "Which destination gave me the best value for money?",
    "Q8":  "Plan a trip like my Portugal one but cheaper.",
    "Q9":  "I want somewhere warm to travel.",
    "Q10": "Under $2000.",   # requires session memory from Q9
    "Q11": "Somewhere vegetarian-friendly like my Bali trip.",  # requires Q9+Q10 memory
}

FAMILIES = {
    "Q1": "factual", "Q2": "factual", "Q3": "factual",
    "Q4": "cross_modal", "Q5": "cross_modal", "Q6": "cross_modal",
    "Q7": "multi_hop", "Q8": "multi_hop",
    "Q9": "conversational", "Q10": "conversational", "Q11": "conversational",
}

VARIANTS = {
    "S0": lambda q, mem: run_s0(q),
    "S1": lambda q, mem: run_s1(q),
    "S2": lambda q, mem: run_s2(q),
    "S3": lambda q, mem: run_s3(q, mem),
    "S4": lambda q, mem: run_s4(q, mem),
}

# ── Load ground truth ─────────────────────────────────────────────────────────

GT_PATH = Path(__file__).parent / "ground_truth.json"
with open(GT_PATH) as f:
    GROUND_TRUTH = json.load(f)

# ── Runner ────────────────────────────────────────────────────────────────────

def run_evaluation():
    rows = []

    for variant_name, run_fn in VARIANTS.items():
        print(f"\n=== {variant_name} ===")
        session_memory = {}  # reset per variant; carries state across Q9→Q10→Q11

        for qid, query in QUERIES.items():
            print(f"  {qid}: {query[:50]}...")
            try:
                result = run_fn(query, session_memory)
            except Exception as e:
                print(f"    ERROR: {e}")
                result = {"answer": "", "latency": 0, "tool_calls": [], "retrieved_docs": []}

            # Update session memory for conversational queries (S3/S4 only)
            if variant_name in ("S3", "S4"):
                session_memory.update(result.get("memory", {}))

            # Retrieved doc IDs for MRR
            retrieved_ids = [d["id"] for d in result.get("retrieved_docs", [])]
            gold_ids = GROUND_TRUTH.get(qid, {}).get("gold_ids", [])
            mrr = compute_mrr(retrieved_ids, gold_ids) if gold_ids else None

            # LLM-as-judge
            context_preview = " ".join(
                d.get("document", "")[:100] for d in result.get("retrieved_docs", [])[:3]
            )
            scores = llm_judge(query, result.get("answer", ""), context_preview)

            rows.append({
                "variant":          variant_name,
                "query_id":         qid,
                "family":           FAMILIES[qid],
                "query":            query,
                "answer":           result.get("answer", "")[:200],
                "latency_s":        result.get("latency", 0),
                "tool_call_count":  len(result.get("tool_calls", [])),
                "mrr":              mrr if mrr is not None else "N/A",
                "judge_accuracy":   scores.get("factual_accuracy", 0),
                "judge_grounded":   scores.get("groundedness", 0),
                "judge_relevance":  scores.get("relevance", 0),
                "judge_complete":   scores.get("completeness", 0),
                "judge_mean":       scores.get("mean", 0),
            })

    # Write CSV
    out_path = Path(__file__).parent / "results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {out_path}")
    return rows

if __name__ == "__main__":
    run_evaluation()