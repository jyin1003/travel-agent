"""
agent/graph.py

Graph topology:

  query_router
       |
  memory_manager
       |
  retrieval_planner
       |
  tool_executor
       |
  temporal_correlator
       |
    analyser
       |  conditional on intent
  ┌────┴────────┐
  direct     creative
  responder  responder

"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph

from .nodes import (
    analyser,
    creative_responder,
    direct_responder,
    memory_manager,
    query_router,
    retrieval_planner,
    temporal_correlator,
    tool_executor,
    transaction_enricher,
)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # Input
    query: str

    # Conversation history
    messages: Annotated[list[BaseMessage], operator.add]

    # Router outputs
    intent:        str            # "lookup" | "generative"
    query_type:    str            # "factual" | "cross_modal" | "multi_hop" | "conversational"
    image_paths: list[str]        # list of image paths
    memory_lookup: bool
    memory_write:  bool

    # Planner outputs
    sub_queries:    list[str]
    retrieval_mode: str
    text_filter:    Optional[dict]
    image_filter:   Optional[dict]

    # Retrieved evidence
    retrieved_docs:   list[dict]
    retrieved_images: list[dict]

    # Temporal correlation outputs
    temporal_context: str
    trip_windows:     list[dict]

    # Analyser output
    analysis: str

    # Memory
    memory: dict

    # Final output
    answer:             str
    grounded:           bool
    ungrounded_claims:  list[str]
    tool_calls:         list[str]


# ---------------------------------------------------------------------------
# Routing edges
# ---------------------------------------------------------------------------
def _route_after_router(state: AgentState) -> str:
    return "memory_manager"   # always load memory; node itself guards writes

def _route_after_analyser(state: AgentState) -> str:
    """Route to creative_responder for generative intent, direct_responder otherwise."""
    return "creative_responder" if state.get("intent") == "generative" else "direct_responder"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("query_router",        query_router)
    graph.add_node("memory_manager",      memory_manager)
    graph.add_node("retrieval_planner",   retrieval_planner)
    graph.add_node("tool_executor",       tool_executor)
    graph.add_node("transaction_enricher", transaction_enricher)
    graph.add_node("temporal_correlator", temporal_correlator)
    graph.add_node("analyser",            analyser)
    graph.add_node("direct_responder",    direct_responder)
    graph.add_node("creative_responder",  creative_responder)

    graph.set_entry_point("query_router")

    graph.add_edge("query_router", "memory_manager")
    graph.add_edge("memory_manager", "retrieval_planner")
    graph.add_edge("retrieval_planner",   "tool_executor")
    graph.add_edge("tool_executor",        "transaction_enricher")
    graph.add_edge("transaction_enricher", "temporal_correlator")
    graph.add_edge("temporal_correlator", "analyser")

    graph.add_conditional_edges(
        "analyser",
        _route_after_analyser,
        {
            "direct_responder":   "direct_responder",
            "creative_responder": "creative_responder",
        },
    )

    graph.add_edge("direct_responder",   END)
    graph.add_edge("creative_responder", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Compiled app
# ---------------------------------------------------------------------------

app = build_graph()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_query(query: str, session_memory: dict | None = None, image_paths: list[str] | None = None) -> dict:
    """
    Run a single query through the agent.

    Returns full AgentState. Key fields:
        answer           : final answer text
        intent           : "lookup" or "generative"
        analysis         : structured analysis from the analyser node
        trip_windows     : temporal windows built from retrieved docs
        grounded         : verification result
        tool_calls       : full execution trace
    """
    initial: AgentState = {
        "query":             query,
        "image_paths":       image_paths or [],
        "messages":          [HumanMessage(content=query)],
        "intent":            "lookup",
        "query_type":        "",
        "memory_lookup":     False,
        "memory_write":      False,
        "sub_queries":       [],
        "retrieval_mode":    "full",
        "text_filter":       None,
        "image_filter":      None,
        "retrieved_docs":    [],
        "retrieved_images":  [],
        "temporal_context":  "",
        "trip_windows":      [],
        "analysis":          "",
        "memory":            session_memory or {},
        "answer":            "",
        "grounded":          False,
        "ungrounded_claims": [],
        "tool_calls":        [],
    }
    return app.invoke(initial)


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Travel Knowledge-Base Agent  (type 'exit' to quit)\n")
    session: dict = {}

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        result = run_query(user_input, session_memory=session)
        session.update(result.get("memory", {}))

        print(f"\nAgent: {result['answer']}")
        if not result["grounded"] and result["ungrounded_claims"]:
            print(f"  ⚠  Ungrounded: {result['ungrounded_claims']}")
        print(f"  [intent={result.get('intent')}]  [trace] {' → '.join(result['tool_calls'])}\n")