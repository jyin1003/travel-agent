"""
evaluation/metrics.py

Metrics:
    compute_mrr(retrieved_ids, ground_truth_ids)            → float
    compute_recall_at_k(retrieved_ids, ground_truth_ids, k) → float
    llm_judge(query, answer, context)               → dict

Ground truth notes
------------------
Some queries have multiple valid ground truth documents (noted as "filtered for N"
in ground_truth.json). The correct metrics for these are:

    MRR        → reciprocal rank of the *first* gold doc hit (measures ranking quality)
    Recall@k   → fraction of gold docs that appear anywhere in top-k (measures coverage)

For Q3 (Copenhagen trip dates), both the 2025-11-19 and 2025-11-22 docs must be
retrieved for a complete answer. Recall@5 = 1.0 only if both appear in top-5.

Rate limits (Groq free tier): ~30 RPM, ~14,400 TPM.
Judge sleep of 2.5s enforced after every call → effective ~24 RPM.
"""

import json
import os
import re
import time
import requests

from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

_JUDGE_MODEL   = os.getenv("EVAL_JUDGE_MODEL", "mistral")
_OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434")
_JUDGE_MAX_TOK = int(os.getenv("EVAL_JUDGE_MAX_TOKENS", "512"))
_JUDGE_SLEEP_S = float(os.getenv("EVAL_JUDGE_SLEEP", "0.5"))  # no rate limit locally


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def compute_mrr(retrieved_ids: list[str], ground_truth_ids: list[str]) -> float:
    """
    Mean Reciprocal Rank: 1/rank of the first gold doc in the retrieved list.

    Measures how highly ranked the first relevant result is.
    For multi-doc ground truths, scores the *best-ranked* gold doc.
    """
    if not ground_truth_ids or not retrieved_ids:
        return 0.0
    gold_set = set(ground_truth_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_set:
            return round(1.0 / rank, 4)
    return 0.0


def compute_recall_at_k(
    retrieved_ids: list[str],
    ground_truth_ids: list[str],
    k: int = 5,
) -> float:
    """
    Recall@k: fraction of gold docs that appear in the top-k retrieved results.

    Recall@k = |gold ∩ top-k retrieved| / |gold|

    This is the right metric when ground truth has multiple documents
    (e.g. Q1 needs 4 transport docs, Q3 needs both trip-boundary docs).

    Examples:
        gold = [A, B, C, D],  top-5 retrieves [A, B, X, Y, Z]  → 2/4 = 0.5
        gold = [A, B],        top-5 retrieves [A, B, X, Y, Z]  → 2/2 = 1.0
    """
    if not ground_truth_ids or not retrieved_ids:
        return 0.0
    gold_set  = set(ground_truth_ids)
    top_k_set = set(retrieved_ids[:k])
    hits      = gold_set & top_k_set
    return round(len(hits) / len(gold_set), 4)


# ── LLM-as-judge ─────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are an evaluation judge. Score the answer on four dimensions, "
    "each from 1 to 5. Return ONLY a JSON object with no explanation, "
    "no reasoning, no markdown, no preamble. Just the raw JSON."
)

_ZERO_SCORES = {
    "factual_accuracy": 0,
    "groundedness":     0,
    "relevance":        0,
    "completeness":     0,
    "mean":             0.0,
}



def llm_judge(query: str, answer: str, context: str = "") -> dict:
    if not answer or not answer.strip():
        time.sleep(_JUDGE_SLEEP_S)
        return _ZERO_SCORES.copy()

    context_line = f"Context (excerpt): {context[:400]}" if context else ""

    prompt = (
        f"Question: {query}\n"
        f"{context_line}\n"
        f"Answer: {answer[:600]}\n\n"
        "Score each 1-5:\n"
        "- factual_accuracy\n"
        "- groundedness\n"
        "- relevance\n"
        "- completeness\n\n"
        "Output ONLY this JSON and nothing else:\n"
        '{"factual_accuracy": N, "groundedness": N, "relevance": N, "completeness": N, "mean": N.N}'
    )

    for attempt in range(3):
        try:
            response = requests.post(
                f"{_OLLAMA_URL}/api/chat",
                json={
                    "model": _JUDGE_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are an evaluation judge. Return ONLY valid JSON, no explanation, no markdown."},
                        {"role": "user",   "content": prompt},
                    ],
                    "stream": False,
                    "options": {"num_predict": _JUDGE_MAX_TOK, "temperature": 0},
                },
                timeout=60,
            )
            response.raise_for_status()
            raw = response.json()["message"]["content"] or ""
            print(f"    [judge] raw response: {raw[:200]}")

            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")

            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                scores = json.loads(match.group())
                nums = [
                    float(scores.get(k, 0))
                    for k in ("factual_accuracy", "groundedness", "relevance", "completeness")
                ]
                scores["mean"] = round(sum(nums) / len(nums), 1) if nums else 0.0
                return scores
            else:
                print(f"    [judge] no JSON found. Raw:\n{raw[:300]}")
                break

        except Exception as e:
            print(f"    [judge] attempt {attempt+1} error: {e}")
            if attempt < 2:
                time.sleep(5)

        finally:
            time.sleep(_JUDGE_SLEEP_S)

    return _ZERO_SCORES.copy()
