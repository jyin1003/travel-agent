"""
agent/tools.py

LangChain tool wrappers around the retriever module.

Four tools mirror the CLIP tutorial node graph:

    search_text_tool    → mode="text"   (dense + BM25 hybrid)
    search_images_tool  → mode="image"  (CLIP + caption hybrid)
    hybrid_search_tool  → mode="full"   (all four paths fused)
    memory_tool         → read / write persistent user preferences

The tool_executor node in nodes.py calls these directly.
They are also bound to the LLM via .bind_tools() so the model can
invoke them in structured tool-call mode if needed.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain.tools import tool

from src.retrieval.retriever import retrieve


# ─────────────────────────────────────────────────────────────────────────────
# In-process memory store (replace with Redis / SQLite for production)
# ─────────────────────────────────────────────────────────────────────────────

_MEMORY_STORE: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval tools
# ─────────────────────────────────────────────────────────────────────────────

@tool
def search_text_tool(
    query: str,
    k: int = 5,
    destination: Optional[str] = None,
    source_type: Optional[str] = None,
) -> str:
    """
    Factual text retrieval using dense (sentence-transformer) + BM25 hybrid search.

    Use for direct fact lookup: costs, dates, durations, place names, itinerary
    details stored as text in the knowledge base.

    Args:
        query:        Natural-language question or keyword string.
        k:            Number of results to return (default 5).
        destination:  Optional destination filter, e.g. "Tokyo".
        source_type:  Optional source type filter: "blog", "receipt",
                      "itinerary", "map", "note".
    """
    where: Optional[dict] = None
    if destination and source_type:
        where = {"$and": [{"destination": {"$eq": destination}},
                          {"source_type":  {"$eq": source_type}}]}
    elif destination:
        where = {"destination": destination}
    elif source_type:
        where = {"source_type": source_type}

    results = retrieve(query=query, mode="text", k=k, text_where=where)
    return _format_results(results)


@tool
def search_images_tool(
    query: str,
    k: int = 5,
    destination: Optional[str] = None,
) -> str:
    """
    Cross-modal image retrieval using CLIP + caption-mediated hybrid search.

    Use when the query references a visual scene, photo content, or asks to
    find images matching a description (e.g. "beach at sunset", "crowded market").

    Args:
        query:        Visual description or scene query.
        k:            Number of results to return (default 5).
        destination:  Optional destination filter, e.g. "Kyoto".
    """
    where: Optional[dict] = {"destination": destination} if destination else None
    results = retrieve(query=query, mode="image", k=k, image_where=where)
    return _format_results(results)


@tool
def hybrid_search_tool(
    query: str,
    k: int = 5,
    destination: Optional[str] = None,
) -> str:
    """
    Full multimodal hybrid retrieval across all four paths:
    dense text + BM25 + CLIP image + caption-mediated image.

    Use for multi-hop or ambiguous queries where the answer might come from
    text documents, receipts, blog posts, OR photos.

    Args:
        query:        Natural-language query.
        k:            Number of results to return (default 5).
        destination:  Optional destination filter applied to both text and
                      image paths.
    """
    where: Optional[dict] = {"destination": destination} if destination else None
    results = retrieve(
        query=query,
        mode="full",
        k=k,
        text_where=where,
        image_where=where,
    )
    return _format_results(results)


# ─────────────────────────────────────────────────────────────────────────────
# Memory tool
# ─────────────────────────────────────────────────────────────────────────────

@tool
def memory_tool(action: str, key: Optional[str] = None, value: Optional[str] = None) -> str:
    """
    Read or write persistent user preferences (allergies, budget, travel style).

    Actions:
        "get_all"  → return all stored preferences as a JSON string
        "get"      → return the value for `key`
        "set"      → store `value` under `key`
        "delete"   → remove `key`

    Args:
        action:  One of "get_all", "get", "set", "delete".
        key:     Preference key (required for get / set / delete).
        value:   Preference value (required for set).

    Examples:
        memory_tool(action="set", key="dietary_restriction", value="vegetarian")
        memory_tool(action="get_all")
        memory_tool(action="get", key="budget_aud")
    """
    if action == "get_all":
        return json.dumps(_MEMORY_STORE) if _MEMORY_STORE else "{}"

    if action == "get":
        if not key:
            return "Error: key is required for action='get'"
        return str(_MEMORY_STORE.get(key, f"(no value stored for '{key}')"))

    if action == "set":
        if not key or value is None:
            return "Error: both key and value are required for action='set'"
        _MEMORY_STORE[key] = value
        return f"Stored: {key} = {value}"

    if action == "delete":
        if not key:
            return "Error: key is required for action='delete'"
        removed = _MEMORY_STORE.pop(key, None)
        return f"Deleted '{key}'" if removed is not None else f"Key '{key}' not found"

    return f"Error: unknown action '{action}'. Use get_all / get / set / delete."


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_results(results: list[dict]) -> str:
    """Format retriever output as a readable string for LLM consumption."""
    if not results:
        return "No relevant documents found."

    lines = []
    for i, r in enumerate(results, 1):
        meta   = r.get("metadata", {})
        doc_id = r.get("id", "?")
        text   = r.get("document", "")[:400]
        source = meta.get("source_type", "unknown")
        dest   = meta.get("destination", "")
        score  = r.get("rrf_score", r.get("score", 0))

        header = f"[{i}] id={doc_id} | source={source}"
        if dest:
            header += f" | destination={dest}"
        header += f" | score={score:.4f}"

        lines.append(f"{header}\n{text}")

    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience exports
# ─────────────────────────────────────────────────────────────────────────────

# All tools as a list — used by ToolNode and llm.bind_tools()
ALL_TOOLS = [search_text_tool, search_images_tool, hybrid_search_tool, memory_tool]

# Raw memory store accessor for nodes that read memory directly
def get_memory_store() -> dict:
    return dict(_MEMORY_STORE)

def update_memory_store(updates: dict) -> None:
    _MEMORY_STORE.update(updates)