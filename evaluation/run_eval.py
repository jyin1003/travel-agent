"""
evaluation/run_eval.py

Run full evaluation across all variants and queries.
Output: evaluation/results.csv

Checkpoint / resume
-------------------
Every completed (variant, query) cell is immediately written to:
    evaluation/results.csv          ← append-on-completion, always up to date
    evaluation/checkpoint.json      ← tracks which cells are done

If the run is interrupted (token limit, crash, keyboard interrupt), restarting
will skip already-completed cells and pick up exactly where it left off.

To start fresh, delete evaluation/checkpoint.json (and optionally results.csv).

Usage
-----
Set EVAL_VARIANTS and EVAL_QUERY_IDS in .env or leave blank for the entire suite.

    python -m evaluation.run_eval

Rate limiting
-------------
When a 429 / RateLimitError is caught the runner sleeps for the wait time
extracted from the error message (+ 30s buffer), then retries the same cell
automatically. On TPD (tokens-per-day) exhaustion the run is checkpointed and
the process exits cleanly — restart tomorrow.
"""

import csv
import json
import os
import re
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from evaluation.variants import run_s0, run_s1, run_s2, run_s3, run_s4
from evaluation.metrics import compute_mrr, compute_recall_at_k, llm_judge

RECALL_K = 5

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

# ── Paths ─────────────────────────────────────────────────────────────────────

_EVAL_DIR       = Path(__file__).parent
_RESULTS_PATH   = _EVAL_DIR / "results.csv"
_CHECKPOINT_PATH= _EVAL_DIR / "checkpoint.json"
_GT_PATH        = _EVAL_DIR / "ground_truth.json"

# ── CSV field names ───────────────────────────────────────────────────────────

_FIELDNAMES = [
    "variant", "query_id", "family", "query", "answer_preview",
    "latency_s", "tool_call_count", "mrr", f"recall_at_{RECALL_K}",
    "judge_accuracy", "judge_grounded", "judge_relevance",
    "judge_complete", "judge_mean",
]

# ── Env-based subsetting ──────────────────────────────────────────────────────

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


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_checkpoint() -> set[str]:
    """Return set of 'VARIANT|QID' keys that are already done."""
    if not _CHECKPOINT_PATH.exists():
        return set()
    with open(_CHECKPOINT_PATH) as f:
        data = json.load(f)
    return set(data.get("completed", []))


def _save_checkpoint(completed: set[str]) -> None:
    with open(_CHECKPOINT_PATH, "w") as f:
        json.dump({"completed": sorted(completed)}, f, indent=2)


def _checkpoint_key(variant: str, qid: str) -> str:
    return f"{variant}|{qid}"


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _ensure_csv_header() -> None:
    """Write the header row only if the file doesn't exist yet."""
    if not _RESULTS_PATH.exists():
        with open(_RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            writer.writeheader()


def _append_row(row: dict) -> None:
    """Append a single result row immediately."""
    with open(_RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writerow(row)


# ── Rate-limit parsing ────────────────────────────────────────────────────────

def _parse_wait_seconds(error_str: str) -> float:
    """
    Try to extract the 'Please try again in Xm Ys' wait from the error message.
    Returns seconds to sleep (+ 30s buffer), or 300 if unparseable.
    """
    # e.g. "Please try again in 7m43.967999999s"
    match = re.search(r"try again in\s+(?:(\d+)m\s*)?(\d+(?:\.\d+)?)s", error_str)
    if match:
        minutes = float(match.group(1) or 0)
        seconds = float(match.group(2))
        return minutes * 60 + seconds + 30  # +30s buffer
    # e.g. "Please try again in 7m12.864s" already handled above
    match2 = re.search(r"try again in\s+(\d+(?:\.\d+)?)\s*minute", error_str)
    if match2:
        return float(match2.group(1)) * 60 + 30
    return 300.0  # default: wait 5 minutes


def _is_tpd_error(error_str: str) -> bool:
    """Tokens-per-DAY exhaustion — can't retry until tomorrow."""
    return "tokens per day" in error_str.lower() or "TPD" in error_str


# ── Safe runner with 429 retry ────────────────────────────────────────────────

MAX_RETRIES = 5

def _safe_run(run_fn, query: str, session_memory: dict) -> dict:
    """
    Run a variant function. On 429 / RateLimitError:
      - If TPD exhausted → re-raise so the outer loop can checkpoint + exit.
      - If RPM limit     → sleep for the extracted wait time, then retry.
    Other exceptions are caught and returned as error rows.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return run_fn(query, session_memory)
        except Exception as e:
            error_str = str(e)

            # Check for any rate-limit flavour
            is_rate_limit = (
                "429" in error_str
                or "rate_limit_exceeded" in error_str.lower()
                or "RateLimitError" in type(e).__name__
            )

            if is_rate_limit:
                if _is_tpd_error(error_str):
                    # Can't recover today — bubble up
                    raise
                # RPM or TPM limit — sleep and retry
                wait = _parse_wait_seconds(error_str)
                print(f"\n  [rate-limit] RPM/TPM hit — sleeping {wait:.0f}s before retry "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            # Any other error — log and return an error row
            traceback.print_exc()
            return {
                "answer":         f"[ERROR: {e}]",
                "latency":        0,
                "tool_calls":     [],
                "retrieved_docs": [],
                "memory":         {},
            }

    return {
        "answer":         "[ERROR: max retries exceeded]",
        "latency":        0,
        "tool_calls":     [],
        "retrieved_docs": [],
        "memory":         {},
    }


# ── Load ground truth ─────────────────────────────────────────────────────────

def _load_ground_truth() -> dict:
    if not _GT_PATH.exists():
        print(f"[eval] Warning: {_GT_PATH} not found — MRR will be 'N/A' for all queries")
        return {}
    with open(_GT_PATH) as f:
        return json.load(f)


# ── Main runner ───────────────────────────────────────────────────────────────

def run_evaluation() -> list[dict]:
    ground_truth = _load_ground_truth()
    variants     = _selected_variants()
    queries      = _selected_queries()
    completed    = _load_checkpoint()

    _ensure_csv_header()

    # Count skipped cells up front
    total_cells  = len(variants) * len(queries)
    skipped      = sum(
        1 for v in variants for q in queries
        if _checkpoint_key(v, q) in completed
    )
    if skipped:
        print(f"[eval] Resuming — {skipped}/{total_cells} cells already done, skipping them.")

    rows: list[dict] = []

    for variant_name, run_fn in variants.items():
        print(f"\n{'='*60}")
        print(f"Variant: {variant_name}")
        print(f"{'='*60}")

        # Session memory carries state across Q9→Q10→Q11 within a variant.
        # When resuming mid-variant we can't recover previous memory, so we
        # just start with an empty dict — conversational queries may degrade
        # slightly but correctness is preserved for all other query families.
        session_memory: dict = {}

        for qid, query in queries.items():
            ck = _checkpoint_key(variant_name, qid)

            if ck in completed:
                print(f"  [{qid}] skipping (already done)")
                continue

            print(f"\n  [{qid}] {query[:60]}...")

            try:
                t_start = time.perf_counter()
                result  = _safe_run(run_fn, query, session_memory)
                wall_s  = round(time.perf_counter() - t_start, 2)

            except Exception as e:
                # TPD exhaustion or unrecoverable error — checkpoint and exit
                error_str = str(e)
                if _is_tpd_error(error_str):
                    wait = _parse_wait_seconds(error_str)
                    print(
                        f"\n[eval] *** TPD limit reached — checkpointing progress. ***\n"
                        f"       Completed {len(completed)}/{total_cells} cells.\n"
                        f"       Groq asks you to wait ~{wait/60:.0f} minutes.\n"
                        f"       Re-run the script tomorrow (or after the reset) to resume."
                    )
                else:
                    print(f"\n[eval] Unrecoverable error on {variant_name}/{qid}: {e}")
                _save_checkpoint(completed)
                return rows

            # Update session memory for conversational queries (S3/S4 only)
            if variant_name in ("S3", "S4"):
                session_memory.update(result.get("memory", {}))

            # ── MRR + Recall@k ────────────────────────────────────────────
            retrieved_ids = [d["id"] for d in result.get("retrieved_docs", [])]
            print(f"    [eval] retrieved ids: {retrieved_ids[:5]}")
            ground_truth_ids = ground_truth.get(qid, {}).get("ground_truth_ids", [])
            mrr              = compute_mrr(retrieved_ids, ground_truth_ids)           if ground_truth_ids else None
            recall           = compute_recall_at_k(retrieved_ids, ground_truth_ids, k=RECALL_K) if ground_truth_ids else None

            # ── LLM-as-judge ──────────────────────────────────────────────
            context_preview = " ".join(
                d.get("document", "")[:100]
                for d in result.get("retrieved_docs", [])[:3]
            )
            print(f"    judging answer ({len(result.get('answer',''))} chars)...")
            scores = llm_judge(query, result.get("answer", ""), context_preview)

            answer_preview = result.get("answer", "")[:200].replace("\n", " ")
            print(f"    MRR={mrr}  Recall@{RECALL_K}={recall}  "
                  f"judge_mean={scores.get('mean')}  latency={wall_s}s")

            row = {
                "variant":              variant_name,
                "query_id":             qid,
                "family":               FAMILIES.get(qid, "?"),
                "query":                query,
                "answer_preview":       answer_preview,
                "latency_s":            wall_s,
                "tool_call_count":      len(result.get("tool_calls", [])),
                "mrr":                  mrr      if mrr      is not None else "N/A",
                f"recall_at_{RECALL_K}":recall   if recall   is not None else "N/A",
                "judge_accuracy":       scores.get("factual_accuracy", 0),
                "judge_grounded":       scores.get("groundedness", 0),
                "judge_relevance":      scores.get("relevance", 0),
                "judge_complete":       scores.get("completeness", 0),
                "judge_mean":           scores.get("mean", 0),
            }

            # ── Write immediately ─────────────────────────────────────────
            _append_row(row)
            rows.append(row)

            completed.add(ck)
            _save_checkpoint(completed)

            # Small inter-query sleep to avoid hammering Groq RPM
            time.sleep(1.0)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Results written to: {_RESULTS_PATH}")
    if rows:
        _print_summary(rows)

    # Clean up checkpoint if everything is done
    if len(completed) >= total_cells:
        print("[eval] All cells complete — removing checkpoint file.")
        _CHECKPOINT_PATH.unlink(missing_ok=True)

    return rows


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(rows: list[dict]) -> None:
    from collections import defaultdict

    by_variant: dict[str, list] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)

    print("\nSummary (mean across queries this run):")
    print(f"{'Variant':<10} {'Judge Mean':>10} {'MRR':>8} {'Latency(s)':>12} {'Tool calls':>12}")
    print("-" * 56)
    for variant, vrows in by_variant.items():
        judge_means = [r["judge_mean"]      for r in vrows if isinstance(r["judge_mean"], (int, float))]
        mrrs        = [r["mrr"]             for r in vrows if isinstance(r["mrr"], float)]
        latencies   = [r["latency_s"]       for r in vrows if isinstance(r["latency_s"], (int, float))]
        tool_counts = [r["tool_call_count"] for r in vrows]

        avg_judge   = round(sum(judge_means) / len(judge_means), 2) if judge_means else "N/A"
        avg_mrr     = round(sum(mrrs)        / len(mrrs),        4) if mrrs        else "N/A"
        avg_latency = round(sum(latencies)   / len(latencies),   2) if latencies   else "N/A"
        avg_tools   = round(sum(tool_counts) / len(tool_counts), 1) if tool_counts else "N/A"

        print(f"{variant:<10} {str(avg_judge):>10} {str(avg_mrr):>8} "
              f"{str(avg_latency):>12} {str(avg_tools):>12}")


if __name__ == "__main__":
    run_evaluation()