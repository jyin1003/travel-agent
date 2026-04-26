import os
import anthropic

def compute_mrr(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """MRR: reciprocal rank of the first gold doc in retrieved list."""
    if not gold_ids or not retrieved_ids:
        return 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_ids:
            return round(1.0 / rank, 4)
    return 0.0

def llm_judge(query: str, answer: str, context: str = "") -> dict:
    """
    Score an answer on 4 dimensions using Claude Haiku as judge.
    Returns dict with scores and mean.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    prompt = f"""Score the following answer on four dimensions, each from 1 to 5.
Return ONLY valid JSON with keys: factual_accuracy, groundedness, relevance, completeness, mean.

Question: {query}
{"Context provided to system: " + context[:500] if context else ""}
Answer: {answer}

Scoring guide (1=very poor, 5=excellent):
- factual_accuracy: are stated facts correct?
- groundedness: are claims supported by the context (if provided)?
- relevance: does the answer address the question?
- completeness: does it cover all parts of the question?
- mean: average of the four scores"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    import json, re
    text = response.content[0].text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"factual_accuracy": 0, "groundedness": 0, "relevance": 0, "completeness": 0, "mean": 0}