"""
evaluation/variants.py

Five system variants for ablation study:

    S0: Plain LLM         — no retrieval, no agent
    S1: Text-only RAG     — BM25 + dense, flat pipeline, no images
    S2: Text + image RAG  — full multimodal retrieval, flat pipeline
    S3: Agent, no memory  — full agent, memory disabled
    S4: Full system       — all components active
"""
import os
import time

from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


def _get_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )


# ── S0: Plain LLM ────────────────────────────────────────────────────────────

def run_s0(query: str) -> dict:
    """Plain LLM — no retrieval, no graph. Establishes the floor."""
    llm = _get_llm()
    t0 = time.perf_counter()
    response = llm.invoke([
        SystemMessage(content="You are a travel assistant. Answer from general knowledge only."),
        HumanMessage(content=query),
    ])
    return {
        "answer":        response.content.strip(),
        "latency":       round(time.perf_counter() - t0, 2),
        "tool_calls":    [],
        "retrieved_docs": [],
        "memory":        {},
    }


# ── S1: Text-only RAG ────────────────────────────────────────────────────────

_S1_SYSTEM = (
    "You are a travel assistant. Answer the question using ONLY the provided "
    "context. Be specific — cite amounts, dates, and place names where available. "
    "If the context does not contain enough information, say so."
)


def run_s1(query: str) -> dict:
    """
    Text-only RAG — BM25 + dense retrieval over text index only.
    Flat single-step pipeline: retrieve → generate. No agent, no images.

    NOTE: retrieval_mode is forced to 'text' so the image index is never
    queried. This is the correct ablation for 'text-only'.
    """
    from src.retrieval.retriever import retrieve

    t0 = time.perf_counter()

    docs = retrieve(query, mode="text", k=5, auto_expand=True)

    context = "\n\n".join(
        f"[{i+1}] {d.get('document', '')[:300]}"
        for i, d in enumerate(docs)
    )

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=_S1_SYSTEM),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ])

    return {
        "answer":         response.content.strip(),
        "latency":        round(time.perf_counter() - t0, 2),
        "tool_calls":     ["retrieve(mode=text)"],
        "retrieved_docs": docs,
        "memory":         {},
    }


# ── S2: Text + image RAG ─────────────────────────────────────────────────────

_S2_SYSTEM = (
    "You are a travel assistant. Answer the question using ONLY the provided "
    "context, which may include text records AND photo descriptions. "
    "Be specific — cite amounts, dates, place names, and photo details where available."
)


def run_s2(query: str) -> dict:
    """
    Text + image RAG — full multimodal retrieval (CLIP + captions), flat pipeline.
    No agent routing or planning.
    """
    from src.retrieval.retriever import retrieve

    t0 = time.perf_counter()

    docs = retrieve(query, mode="full", k=5, auto_expand=True)

    context = "\n\n".join(
        f"[{i+1}] ({d.get('metadata', {}).get('source_type', '?')}) "
        f"{d.get('document', '')[:300]}"
        for i, d in enumerate(docs)
    )

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=_S2_SYSTEM),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ])

    return {
        "answer":         response.content.strip(),
        "latency":        round(time.perf_counter() - t0, 2),
        "tool_calls":     ["retrieve(mode=full)"],
        "retrieved_docs": docs,
        "memory":         {},
    }


# ── S3: Agent, no memory ─────────────────────────────────────────────────────

def run_s3(query: str, session_memory: dict = None) -> dict:
    """
    Full LangGraph agent with all indices, but memory is disabled.
    Tests the value of routing + planning without personalisation.
    """
    # Ensure temporal correlation is on; memory passed as empty dict
    os.environ["AGENT_TEMPORAL"] = "1"

    from agent.graph import run_query

    t0 = time.perf_counter()
    result = run_query(query, session_memory={})
    result["latency"] = round(time.perf_counter() - t0, 2)
    return result


# ── S4: Full system ──────────────────────────────────────────────────────────

def run_s4(query: str, session_memory: dict = None) -> dict:
    """Full system — multimodal indices, LangGraph agent with memory."""
    os.environ["AGENT_TEMPORAL"] = "1"

    from agent.graph import run_query

    t0 = time.perf_counter()
    result = run_query(query, session_memory=session_memory or {})
    result["latency"] = round(time.perf_counter() - t0, 2)
    return result