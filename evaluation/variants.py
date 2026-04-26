import os, time
from langchain_core.messages import HumanMessage, SystemMessage
from agent.graph import run_query

def _get_llm():
    from langchain_groq import ChatGroq
    from dotenv import load_dotenv
    load_dotenv()
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

def run_s0(query: str) -> dict:
    """Plain LLM — no retrieval, no graph."""
    llm = _get_llm()
    t0 = time.perf_counter()
    response = llm.invoke([
        SystemMessage(content="You are a travel assistant. Answer from general knowledge only."),
        HumanMessage(content=query),
    ])
    return {
        "answer": response.content.strip(),
        "latency": round(time.perf_counter() - t0, 2),
        "tool_calls": [],
        "retrieved_docs": [],
    }

def run_s1(query: str) -> dict:
    """Text-only RAG — force retrieval_mode=text, skip image index."""
    os.environ["AGENT_TEMPORAL"] = "0"
    t0 = time.perf_counter()
    result = run_query(query)
    # Override: re-retrieve with text mode only
    from src.retrieval.retriever import retrieve
    docs = retrieve(query, mode="text", k=5)
    result["retrieved_docs"] = docs
    result["latency"] = round(time.perf_counter() - t0, 2)
    os.environ["AGENT_TEMPORAL"] = "1"
    return result

def run_s2(query: str) -> dict:
    """Text + image RAG — full retrieval, flat pipeline (no agent planning)."""
    os.environ["AGENT_TEMPORAL"] = "0"
    t0 = time.perf_counter()
    from src.retrieval.retriever import retrieve
    docs = retrieve(query, mode="full", k=5)
    llm = _get_llm()
    context = "\n\n".join(d.get("document", "")[:300] for d in docs)
    response = llm.invoke([
        SystemMessage(content="Answer using only the provided context."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ])
    os.environ["AGENT_TEMPORAL"] = "1"
    return {
        "answer": response.content.strip(),
        "latency": round(time.perf_counter() - t0, 2),
        "tool_calls": ["retrieve(mode=full)"],
        "retrieved_docs": docs,
    }

def run_s3(query: str, session_memory: dict = None) -> dict:
    """Full agent, memory disabled."""
    os.environ["AGENT_TEMPORAL"] = "1"
    t0 = time.perf_counter()
    # Pass empty memory to disable persistence effect
    result = run_query(query, session_memory={})
    result["latency"] = round(time.perf_counter() - t0, 2)
    return result

def run_s4(query: str, session_memory: dict = None) -> dict:
    """Full system."""
    os.environ["AGENT_TEMPORAL"] = "1"
    t0 = time.perf_counter()
    result = run_query(query, session_memory=session_memory or {})
    result["latency"] = round(time.perf_counter() - t0, 2)
    return result