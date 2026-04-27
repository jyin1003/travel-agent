"""
evaluation/metrics.py

Metrics:
    compute_mrr(retrieved_ids, gold_ids)            → float
    compute_recall_at_k(retrieved_ids, gold_ids, k) → float
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

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

_JUDGE_SLEEP_S = float(os.getenv("EVAL_JUDGE_SLEEP",    "2.5"))
_JUDGE_MODEL   = os.getenv("EVAL_JUDGE_MODEL",           "qwen/qwen3-32b")
_JUDGE_MAX_TOK = int(os.getenv("EVAL_JUDGE_MAX_TOKENS", "256"))

_groq_client: Groq | None = None


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set — cannot run LLM judge")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def compute_mrr(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """
    Mean Reciprocal Rank: 1/rank of the first gold doc in the retrieved list.

    Measures how highly ranked the first relevant result is.
    For multi-doc ground truths, scores the *best-ranked* gold doc.
    """
    if not gold_ids or not retrieved_ids:
        return 0.0
    gold_set = set(gold_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_set:
            return round(1.0 / rank, 4)
    return 0.0


def compute_recall_at_k(
    retrieved_ids: list[str],
    gold_ids: list[str],
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
    if not gold_ids or not retrieved_ids:
        return 0.0
    gold_set  = set(gold_ids)
    top_k_set = set(retrieved_ids[:k])
    hits      = gold_set & top_k_set
    return round(len(hits) / len(gold_set), 4)


# ── LLM-as-judge ─────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are an evaluation judge. Score the answer on four dimensions, "
    "each from 1 to 5. Return ONLY valid JSON with no extra text or markdown."
)

_JUDGE_PROMPT = """\
Question: {query}
{context_line}
Answer: {answer}

Score each dimension 1–5 (1=very poor, 5=excellent):
- factual_accuracy: are stated facts correct?
- groundedness: are claims supported by the context (if provided)?
- relevance: does the answer address the question?
- completeness: does it cover all parts of the question?
- mean: average of the four scores (float, 1 decimal place)

Return ONLY valid JSON:
{{"factual_accuracy": N, "groundedness": N, "relevance": N, "completeness": N, "mean": N.N}}"""

_ZERO_SCORES = {
    "factual_accuracy": 0,
    "groundedness":     0,
    "relevance":        0,
    "completeness":     0,
    "mean":             0.0,
}


def llm_judge(query: str, answer: str, context: str = "") -> dict:
    """
    Score an answer using qwen/qwen3-32b as judge.

    Sleeps _JUDGE_SLEEP_S after the call to stay within Groq RPM limits.
    Context truncated to 400 chars, answer to 600 chars to manage TPM.
    """
    if not answer or not answer.strip():
        time.sleep(_JUDGE_SLEEP_S)
        return _ZERO_SCORES.copy()

    context_line = f"Context (excerpt): {context[:400]}" if context else ""

    prompt = _JUDGE_PROMPT.format(
        query=query,
        context_line=context_line,
        answer=answer[:600],
    )

    client = _get_groq_client()
    try:
        response = client.chat.completions.create(
            model=_JUDGE_MODEL,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_tokens=_JUDGE_MAX_TOK,
        )
        raw = response.choices[0].message.content or ""

        # Strip <think>...</think> blocks (qwen3 chain-of-thought output)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            scores = json.loads(match.group())
            # Recompute mean from the four scores (don't trust model's arithmetic)
            nums = [
                float(scores.get(k, 0))
                for k in ("factual_accuracy", "groundedness", "relevance", "completeness")
            ]
            scores["mean"] = round(sum(nums) / len(nums), 1) if nums else 0.0
            return scores

    except Exception as e:
        print(f"    [judge] error: {e}")

    finally:
        time.sleep(_JUDGE_SLEEP_S)

    return _ZERO_SCORES.copy()