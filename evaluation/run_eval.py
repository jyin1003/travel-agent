"""
evaluation/run_eval.py

Run full evaluation across all variants and queries.
Output: evaluation/results.csv

Usage
-----
PowerShell (Windows):
    $env:EVAL_VARIANTS="S4"; $env:EVAL_QUERY_IDS="Q1,Q2,Q3"; python -m evaluation.run_eval
    # Clean up after:
    Remove-Item Env:EVAL_VARIANTS; Remove-Item Env:EVAL_QUERY_IDS

Bash / Mac / Linux:
    EVAL_VARIANTS=S4 EVAL_QUERY_IDS=Q1,Q2,Q3 python -m evaluation.run_eval

Full run (all variants, all queries):
    python -m evaluation.run_eval

Rate limiting notes
-------------------
- Each S3/S4 query runs the full LangGraph agent (~6-8 LLM calls internally).
- The LLM judge adds a 2.5s sleep per call (24 RPM effective, within Groq's 30 RPM limit).
- Total judge calls: 5 variants × 11 queries = 55 calls.
- Estimated total wall time: 30-45 minutes for a full run.
"""
import csv
import json
import os
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from evaluation.variants import run_s0, run_s1, run_s2, run_s3, run_s4
from evaluation.metrics import compute_mrr, compute_recall_at_k, llm_judge

RECALL_K = 5  # Recall@5 — matches the k=5 used in retrieval baselines

# ── Test queries ──────────────────────────────────────────────────────────────

ALL_QUERIES = {
    "Q1":  "How much did I spend on transport in Japan?",
    "Q2":  "What hotel did I stay at in Tokyo?",
    "Q3":  "How long was my Copenhagen trip?",
    "Q4":  "Find a photo showing scenery from my travels.",
    "Q5":  "What was I eating when I visited the market?",
    "Q6":  "Where and what did I spend the most on dinner?",
    "Q7":  "Which destination gave me the best value for money?",
    "Q8":  "Plan a trip like my Paris one but cheaper.",
    "Q9":  "I want somewhere warm to travel.",
    "Q10": "Under $2000.",
    "Q11": "Somewhere public transport friendly like my Japan trip.",
}

FAMILIES = {
    "Q1": "factual",      "Q2": "factual",      "Q3": "factual",
    "Q4": "cross_modal",  "Q5": "cross_modal",  "Q6": "cross_modal",
    "Q7": "multi_hop",    "Q8": "multi_hop",
    "Q9": "conversational", "Q10": "conversational", "Q11": "conversational",
}

ALL_VARIANTS = {
    "S0": lambda q, mem: run_s0(q),
    "S1": lambda q, mem: run_s1(q),
    "S2": lambda q, mem: run_s2(q),
    "S3": lambda q, mem: run_s3(q, mem),
    "S4": lambda q, mem: run_s4(q, mem),
}

# ── Env-based subsetting (for quick runs) ────────────────────────────────────

def _selected_variants() -> dict:
    env = os.getenv("EVAL_VARIANTS", "")
    if not env:
        return ALL_VARIANTS
    keys = [k.strip() for k in env.split(",") if k.strip()]
    return {k: ALL_VARIANTS[k] for k in keys if k in ALL_VARIANTS}


def _selected_queries() -> dict:
    env = os.getenv("EVAL_QUERY_IDS", "")
    if not env:
        return ALL_QUERIES
    keys = [k.strip() for k in env.split(",") if k.strip()]
    return {k: ALL_QUERIES[k] for k in keys if k in ALL_QUERIES}


# ── Load ground truth ─────────────────────────────────────────────────────────

GT_PATH = Path(__file__).parent / "ground_truth.json"

def _load_ground_truth() -> dict:
    if not GT_PATH.exists():
        print(f"[eval] Warning: {GT_PATH} not found — MRR will be 'N/A' for all queries")
        return {}
    with open(GT_PATH) as f:
        return json.load(f)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_run(run_fn, query: str, session_memory: dict) -> dict:
    """Run a variant function, catching all exceptions."""
    try:
        return run_fn(query, session_memory)
    except Exception as e:
        traceback.print_exc()
        return {
            "answer":         f"[ERROR: {e}]",
            "latency":        0,
            "tool_calls":     [],
            "retrieved_docs": [],
            "memory":         {},
        }


# ── Main runner ───────────────────────────────────────────────────────────────

def run_evaluation() -> list[dict]:
    ground_truth = _load_ground_truth()
    variants     = _selected_variants()
    queries      = _selected_queries()

    rows: list[dict] = []

    for variant_name, run_fn in variants.items():
        print(f"\n{'='*60}")
        print(f"Variant: {variant_name}")
        print(f"{'='*60}")

        # Session memory resets per variant; carries state across Q9→Q10→Q11
        session_memory: dict = {}

        for qid, query in queries.items():
            print(f"\n  [{qid}] {query[:60]}...")

            t_start = time.perf_counter()
            result  = _safe_run(run_fn, query, session_memory)
            wall_s  = round(time.perf_counter() - t_start, 2)

            # Update session memory for conversational queries (S3/S4 only)
            if variant_name in ("S3", "S4"):
                session_memory.update(result.get("memory", {}))

            # ── MRR + Recall@k ────────────────────────────────────────────
            # FIX: ground_truth.json uses "ground_truth_ids", not "ground_truth_ids"
            retrieved_ids = [d["id"] for d in result.get("retrieved_docs", [])]
            ground_truth_ids      = ground_truth.get(qid, {}).get("ground_truth_ids", [])
            mrr           = compute_mrr(retrieved_ids, ground_truth_ids)           if ground_truth_ids else None
            recall        = compute_recall_at_k(retrieved_ids, ground_truth_ids, k=RECALL_K) if ground_truth_ids else None

            # ── LLM-as-judge ─────────────────────────────────────────────────
            context_preview = " ".join(
                d.get("document", "")[:100]
                for d in result.get("retrieved_docs", [])[:3]
            )
            print(f"    judging answer ({len(result.get('answer',''))} chars)...")
            scores = llm_judge(query, result.get("answer", ""), context_preview)

            answer_preview = result.get("answer", "")[:200].replace("\n", " ")
            print(f"    MRR={mrr}  Recall@{RECALL_K}={recall}  judge_mean={scores.get('mean')}  latency={wall_s}s")

            rows.append({
                "variant":         variant_name,
                "query_id":        qid,
                "family":          FAMILIES.get(qid, "?"),
                "query":           query,
                "answer_preview":  answer_preview,
                "latency_s":       wall_s,
                "tool_call_count": len(result.get("tool_calls", [])),
                "mrr":             mrr      if mrr      is not None else "N/A",
                f"recall_at_{RECALL_K}": recall if recall is not None else "N/A",
                "judge_accuracy":  scores.get("factual_accuracy", 0),
                "judge_grounded":  scores.get("groundedness", 0),
                "judge_relevance": scores.get("relevance", 0),
                "judge_complete":  scores.get("completeness", 0),
                "judge_mean":      scores.get("mean", 0),
            })

            # Small sleep between agent queries to avoid hammering Groq RPM
            # (judge sleep is handled inside llm_judge itself)
            time.sleep(1.0)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    out_path = Path(__file__).parent / "results.csv"
    if rows:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n{'='*60}")
        print(f"Results written to: {out_path}")
        _print_summary(rows)
    else:
        print("\n[eval] No rows — nothing to write.")

    return rows


def _print_summary(rows: list[dict]) -> None:
    """Print a quick per-variant summary table."""
    from collections import defaultdict

    by_variant: dict[str, list] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)

    print("\nSummary (mean across queries):")
    print(f"{'Variant':<10} {'Judge Mean':>10} {'MRR':>8} {'Latency(s)':>12} {'Tool calls':>12}")
    print("-" * 56)
    for variant, vrows in by_variant.items():
        judge_means = [r["judge_mean"] for r in vrows if isinstance(r["judge_mean"], (int, float))]
        mrrs        = [r["mrr"]        for r in vrows if isinstance(r["mrr"],        float)]
        latencies   = [r["latency_s"]  for r in vrows if isinstance(r["latency_s"],  (int, float))]
        tool_counts = [r["tool_call_count"] for r in vrows]

        avg_judge   = round(sum(judge_means) / len(judge_means), 2)  if judge_means else "N/A"
        avg_mrr     = round(sum(mrrs)        / len(mrrs),        4)  if mrrs        else "N/A"
        avg_latency = round(sum(latencies)   / len(latencies),   2)  if latencies   else "N/A"
        avg_tools   = round(sum(tool_counts) / len(tool_counts), 1)  if tool_counts else "N/A"

        print(f"{variant:<10} {str(avg_judge):>10} {str(avg_mrr):>8} {str(avg_latency):>12} {str(avg_tools):>12}")


if __name__ == "__main__":
    run_evaluation()