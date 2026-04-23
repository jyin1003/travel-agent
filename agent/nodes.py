"""
agent/nodes.py

Node pipeline:
  query_router -> [memory_manager] -> retrieval_planner
                                            |
                                      tool_executor
                                            |
                                    temporal_correlator
                                            |
                                       analyser          <- extracts facts/patterns from evidence
                                            |
                               ┌────────────┴────────────┐
                               ▼                         ▼
                        (lookup path)            (generative path)
                      direct_responder         creative_responder
                               └────────────┬────────────┘
                                            ▼
                                           END

Intent types
------------
  lookup     : answer lives directly in the data (factual, cross-modal)
               e.g. "how much did I spend in Tokyo", "show me photos from Berlin"
               -> analyser extracts the fact -> direct_responder returns it

  generative : answer requires reasoning BEYOND the data to produce something new
               e.g. "recommend somewhere new", "suggest a trip", "what should I do next"
               -> analyser extracts patterns/constraints -> creative_responder generates
                  something the user has NOT done/seen before

The router classifies intent; the analyser does the heavy reasoning;
the responders are thin formatters focused on their specific output style.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

from .tools import (
    get_memory_store,
    hybrid_search_tool,
    search_images_tool,
    search_text_tool,
    update_memory_store,
)
from src.retrieval.retriever import retrieve

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [NODES] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _ts(label: str) -> None:
    """Emit a timestamped step marker."""
    log.info(f"{label}")

# ---------------------------------------------------------------------------
# Ablation controls
# ---------------------------------------------------------------------------

ENABLE_VERIFY: bool = False
ENABLE_TEMPORAL_CORRELATION: bool = True
TRIP_WINDOW_GAP_DAYS: int = 3

# ---------------------------------------------------------------------------
# LLM instances
# ---------------------------------------------------------------------------

_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
_llm = ChatGroq(model=_GROQ_MODEL, temperature=0, api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_JUNK_TOKENS = {
    "factual", "cross_modal", "multi_hop", "conversational", "lookup",
    "generative", "text", "image", "full", "dense_only", "bm25_only",
    "clip_only", "caption_only", "sub_query", "query", "none",
}

_VALID_FILTER_KEYS = {"destination", "year", "source_type"}

_PREFERENCE_SIGNALS = {
    "allergic", "allergy", "vegetarian", "vegan", "halal", "kosher",
    "budget", "prefer", "hate", "love", "always", "never", "dislike",
    "don't like", "i like", "i want", "my preference", "hostel", "hotel",
}

# Keywords that hard-classify a query as generative regardless of LLM output.
# Small models frequently misclassify recommendation queries as "lookup".
_GENERATIVE_SIGNALS = {
    "recommend", "recommendation", "suggest", "suggestion",
    "where should i", "where to go", "where to travel", "where to visit",
    "what should i do", "plan me", "plan a trip", "plan my trip",
    "somewhere new", "somewhere to go", "somewhere to travel",
    "what's next", "what is next", "next trip", "new destination",
    "haven't been", "have not been", "never been", "never visited",
}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def _sanitise_query(value, fallback: str) -> str:
    if isinstance(value, str):
        s = value.strip()
    elif isinstance(value, dict):
        for key in ("sub_query", "query", "text", "q"):
            if key in value and isinstance(value[key], str):
                s = value[key].strip()
                break
        else:
            return fallback
    else:
        return fallback
    if s.lower() in _JUNK_TOKENS or len(s) < 6:
        return fallback
    return s


def _docs_to_context(docs: list[dict], max_docs: int = 8) -> str:
    lines = []
    for i, d in enumerate(docs[:max_docs], 1):
        meta   = d.get("metadata", {})
        doc_id = d.get("id", "?")
        text   = d.get("document", "")[:200]
        source = meta.get("source_type", "text")
        dest   = meta.get("destination", "")
        header = f"[{i}] id={doc_id} source={source}"
        if dest:
            header += f" destination={dest}"
        lines.append(f"{header}\n{text}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Temporal helpers
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+\-]\d{2}:\d{2}",
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
    r"\d{4}-\d{2}-\d{2}",
]


def _extract_date(doc: dict) -> datetime | None:
    meta = doc.get("metadata", {})
    for field in ("date", "timestamp", "datetime", "created_at"):
        val = meta.get(field)
        if val:
            try:
                dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
                return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
            except (ValueError, TypeError):
                pass
    text = doc.get("document", "")
    for pattern in _DATE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            try:
                dt = datetime.fromisoformat(match.group().replace("Z", "+00:00"))
                return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
            except ValueError:
                pass
    return None


def _build_trip_windows(docs: list[dict], gap_days: int = TRIP_WINDOW_GAP_DAYS) -> list[dict]:
    dated, undated = [], []
    for doc in docs:
        dt = _extract_date(doc)
        (dated if dt is not None else undated).append((dt, doc) if dt else doc)

    if not dated:
        return []

    dated.sort(key=lambda x: x[0])
    gap     = timedelta(days=gap_days)
    windows = [[dated[0]]]
    for dt, doc in dated[1:]:
        if dt - windows[-1][-1][0] <= gap:
            windows[-1].append((dt, doc))
        else:
            windows.append([(dt, doc)])

    result = []
    for window in windows:
        start_dt, end_dt = window[0][0], window[-1][0]
        days   = max(1, (end_dt.date() - start_dt.date()).days + 1)
        bucket = defaultdict(list)
        for _, doc in window:
            src = doc.get("metadata", {}).get("source_type", "unknown")
            bucket["photos" if src == "photo_caption" else src].append(doc)
        result.append({
            "start":        start_dt.date().isoformat(),
            "end":          end_dt.date().isoformat(),
            "days":         days,
            "transactions": bucket["transaction"],
            "map":          bucket["map"],
            "photos":       bucket["photos"],
            "all_docs":     [doc for _, doc in window],
        })
    return result


def _windows_to_context(windows: list[dict], undated: list[dict]) -> str:
    lines = []
    for i, w in enumerate(windows, 1):
        lines.append(f"=== Trip window {i}: {w['start']} to {w['end']} ({w['days']} days) ===")
        if w["transactions"]:
            lines.append("  Transactions:")
            for doc in w["transactions"][:5]:
                lines.append(f"    - {doc.get('document', '')[:150]}")
        if w["map"]:
            lines.append("  Map / places:")
            for doc in w["map"][:3]:
                lines.append(f"    - {doc.get('document', '')[:150]}")
        if w["photos"]:
            lines.append("  Photos:")
            for doc in w["photos"][:3]:
                lines.append(f"    - {doc.get('document', '')[:150]}")
        lines.append("")
    if undated:
        lines.append("=== Undated records ===")
        for doc in undated[:5]:
            src  = doc.get("metadata", {}).get("source_type", "?")
            lines.append(f"  [{src}] {doc.get('document', '')[:150]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node 1 -- Query Router
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM = """You are a query classifier for a personal travel knowledge base.

Classify the INTENT of the query:

  "lookup"     - the answer exists directly in stored data; fetch and return it
                 Examples: "how much did I spend in Tokyo", "when did I visit Berlin",
                 "show me photos from my Japan trip", "what restaurants did I go to"

  "generative" - the answer requires reasoning FROM the data to produce something NEW
                 that the user hasn't necessarily done/seen before
                 Examples: "recommend somewhere to travel", "suggest a new destination",
                 "what should I do next", "plan me a trip", "where should I go for X"

Also classify query_type (for retrieval planning):
  "factual"        - direct fact lookup
  "cross_modal"    - needs photos/visual content
  "multi_hop"      - combines multiple evidence sources
  "conversational" - references preferences or earlier turns

And memory flags:
  memory_lookup = true  when stored preferences help answer
  memory_write  = true  ONLY when query explicitly states a NEW personal preference

Respond with ONLY valid JSON:
{
  "intent": "lookup" or "generative",
  "query_type": "...",
  "memory_lookup": true/false,
  "memory_write": true/false
}"""


def query_router(state: dict) -> dict:
    """Classify intent and query type, set routing flags."""
    _ts("query_router: start")
    response = _llm.invoke([
        SystemMessage(content=_ROUTER_SYSTEM),
        HumanMessage(content=state["query"]),
    ])
    parsed     = _parse_json(response.content)
    intent     = parsed.get("intent", "lookup")
    query_type = parsed.get("query_type", "factual")
    mem_lookup = bool(parsed.get("memory_lookup", False))
    mem_write  = bool(parsed.get("memory_write", False))

    if query_type == "conversational":
        mem_lookup = True
    if mem_write:
        mem_write = any(kw in state["query"].lower() for kw in _PREFERENCE_SIGNALS)

    # Hard override: if query contains generative keywords, force intent=generative
    # Small models frequently misclassify recommendation queries as "lookup"
    q_lower = state["query"].lower()
    if any(kw in q_lower for kw in _GENERATIVE_SIGNALS):
        intent = "generative"

    log.info(f"[router] intent={intent}  query_type={query_type}  memory_lookup={mem_lookup}  memory_write={mem_write}")
    _ts(f"query_router: done  (intent={intent}  type={query_type})")
    return {
        "intent":        intent,
        "query_type":    query_type,
        "memory_lookup": mem_lookup,
        "memory_write":  mem_write,
        "tool_calls":    state.get("tool_calls", []) + [f"router(intent={intent}, type={query_type})"],
    }


# ---------------------------------------------------------------------------
# Node 2 -- Memory Manager
# ---------------------------------------------------------------------------

def memory_manager(state: dict) -> dict:
    """Load persistent preferences; extract new ones only when a preference signal is present."""
    _ts("memory_manager: start")
    if state.get("memory_write"):
        if any(kw in state["query"].lower() for kw in _PREFERENCE_SIGNALS):
            resp = _llm.invoke([HumanMessage(content=(
                "Extract NEW user travel preferences from the query. "
                "Return ONLY a flat JSON object of string key-value pairs, or {}.\n\n"
                f"Query: {state['query']}"
            ))])
            new_prefs = _parse_json(resp.content)
            if new_prefs:
                update_memory_store(new_prefs)
    _ts("memory_manager: done")
    return {
        "memory":     get_memory_store(),
        "tool_calls": state.get("tool_calls", []) + ["memory_manager"],
    }


# ---------------------------------------------------------------------------
# Node 3 -- Retrieval Planner
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """You are a retrieval planner for a personal travel knowledge base.

Source types available:
  transaction   - bank/card transactions (merchant, amount, category, date, destination)
  map           - saved places, routes, points of interest from Google Maps
  photo_caption - AI-generated captions describing travel photos (scene, location, date)

Retrieval modes:
  "text"  - transactions + map entries
  "image" - photo captions only
  "full"  - all three (DEFAULT — use unless clearly single-source)

The system correlates all three sources by timestamp into trip windows after retrieval,
so fetching "full" gives the richest cross-modal context.

For GENERATIVE queries (recommendations, suggestions, planning):
  Fetch evidence about PAST behaviour to inform what's NEW:
  - Past destinations and costs (to know what's already been done and typical budget)
  - Saved map places (to understand interests and travel style)
  - Photo scenes (to understand what environments the user enjoys)

For LOOKUP queries:
  Fetch the specific evidence needed to answer directly.

Sub-query rules:
  - Plain natural language strings only, NOT SQL
  - 2-5 sub-queries targeting different evidence angles
  - Filters: destination (str), year (int), source_type (str)

Examples:

Query: recommend somewhere new to travel for less than $3000 [intent=generative]
{"sub_queries": ["past trip total costs by destination", "destinations previously visited transactions", "saved map places countries", "travel photo locations visited"],
 "retrieval_mode": "full", "text_filter": {}, "image_filter": {}}

Query: what did I spend in London [intent=lookup]
{"sub_queries": ["London spending transactions", "London transport food accommodation costs"],
 "retrieval_mode": "text", "text_filter": {"destination": "London", "source_type": "transaction"}, "image_filter": {}}

Query: show me photos from my Munich trip [intent=lookup]
{"sub_queries": ["Munich travel photos scenes"],
 "retrieval_mode": "image", "text_filter": {}, "image_filter": {"destination": "Munich"}}

Query: what was my London trip like [intent=lookup]
{"sub_queries": ["London transactions spending", "London map places", "London photos scenes"],
 "retrieval_mode": "full", "text_filter": {"destination": "London"}, "image_filter": {"destination": "London"}}

Respond ONLY with valid JSON:"""


def retrieval_planner(state: dict) -> dict:
    """Decompose the query into sub-queries and select retrieval parameters."""
    _ts("retrieval_planner: start")
    memory_str = json.dumps(state.get("memory", {})) if state.get("memory") else "{}"

    response = _llm.invoke([
        SystemMessage(content=_PLANNER_SYSTEM),
        HumanMessage(content=(
            f"Intent: {state.get('intent', 'lookup')}\n"
            f"Query type: {state.get('query_type', 'factual')}\n"
            f"User preferences: {memory_str}\n"
            f"Query: {state['query']}"
        )),
    ])
    parsed = _parse_json(response.content)
    log.info(f"[planner] parsed: {parsed}")

    valid_modes = {"text", "image", "full", "dense_only", "bm25_only", "clip_only", "caption_only"}
    mode = parsed.get("retrieval_mode", "full")
    if mode not in valid_modes:
        mode = "full"

    raw_sub     = parsed.get("sub_queries") or [state["query"]]
    sub_queries = list(dict.fromkeys(
        _sanitise_query(sq, state["query"]) for sq in raw_sub if sq
    ))
    if not sub_queries:
        sub_queries = [state["query"]]

    text_filter  = parsed.get("text_filter")  or None
    image_filter = parsed.get("image_filter") or None
    if text_filter  == {}: text_filter  = None
    if image_filter == {}: image_filter = None
    if text_filter:
        text_filter  = {k: v for k, v in text_filter.items()  if k in _VALID_FILTER_KEYS} or None
    if image_filter:
        image_filter = {k: v for k, v in image_filter.items() if k in _VALID_FILTER_KEYS} or None

    # Generative queries need broad evidence — enforce minimum 3 sub-queries
    # so the analyser has enough signal about past trips, costs, and style
    if state.get("intent") == "generative" and len(sub_queries) < 3:
        log.info("[planner] generative query with too few sub-queries — expanding")
        base = sub_queries[0] if sub_queries else state["query"]
        sub_queries = list(dict.fromkeys([
            base,
            "past trip total costs by destination",
            "destinations previously visited transactions",
            "saved map places points of interest",
            "travel photo locations scenes visited",
        ]))

    log.info(f"[planner] mode={mode}  n_sub={len(sub_queries)}  text_filter={text_filter}")
    log.info(f"[planner] sub_queries={sub_queries}")
    _ts("retrieval_planner: done")
    return {
        "sub_queries":    sub_queries,
        "retrieval_mode": mode,
        "text_filter":    text_filter,
        "image_filter":   image_filter,
        "tool_calls":     state.get("tool_calls", []) + [f"planner(mode={mode}, n_sub={len(sub_queries)})"],
    }


# ---------------------------------------------------------------------------
# Node 4 -- Tool Executor
# ---------------------------------------------------------------------------

_ABLATION_MODES = {"dense_only", "bm25_only", "clip_only", "caption_only"}
_MODE_TO_TOOL   = {"text": search_text_tool, "image": search_images_tool, "full": hybrid_search_tool}


def tool_executor(state: dict) -> dict:
    """Run retrieve() for each sub-query and de-duplicate results by doc ID."""
    _ts("tool_executor: start")
    mode         = state.get("retrieval_mode", "full")
    raw_queries  = state.get("sub_queries", [state["query"]])
    text_filter  = state.get("text_filter")
    image_filter = state.get("image_filter")
    tool_trace   = list(state.get("tool_calls", []))

    sub_queries = list(dict.fromkeys(
        _sanitise_query(sq, state["query"]) for sq in raw_queries if sq
    )) or [state["query"]]

    accumulated: dict[str, dict] = {}
    for sq in sub_queries:
        _ts(f"tool_executor: retrieve  q={sq[:50]!r}  mode={mode}")
        try:
            results = retrieve(query=sq, mode=mode, k=5,
                               text_where=text_filter, image_where=image_filter)
            tool_fn = _MODE_TO_TOOL.get(mode, hybrid_search_tool)
            tool_trace.append(f"{tool_fn.name}(q={sq[:40]!r})")
            log.info(f"[tool_executor] {len(results)} docs for q={sq[:40]!r}")
        except Exception as e:
            tool_trace.append(f"retrieve_error({sq[:40]!r}: {e})")
            log.warning(f"[tool_executor] retrieve failed: {e}")
            results = []
        for r in results:
            accumulated[r["id"]] = r

    all_docs   = list(accumulated.values())
    image_docs = [d for d in all_docs if d.get("metadata", {}).get("source_type") == "photo_caption"]
    log.info(f"[tool_executor] total unique docs={len(all_docs)}  images={len(image_docs)}")
    _ts(f"tool_executor: done  ({len(all_docs)} unique docs)")
    return {
        "retrieved_docs":   all_docs,
        "retrieved_images": image_docs,
        "tool_calls":       tool_trace,
    }


# ---------------------------------------------------------------------------
# Node 5 -- Temporal Correlator
# ---------------------------------------------------------------------------

def temporal_correlator(state: dict) -> dict:
    """Group retrieved docs into trip windows by timestamp proximity."""
    _ts("temporal_correlator: start")
    if not ENABLE_TEMPORAL_CORRELATION:
        _ts("temporal_correlator: disabled (ablation)")
        return {
            "temporal_context": "",
            "trip_windows":     [],
            "tool_calls":       state.get("tool_calls", []) + ["temporal_correlator(disabled)"],
        }

    docs = state.get("retrieved_docs", [])
    if not docs:
        _ts("temporal_correlator: done (no docs)")
        return {
            "temporal_context": "",
            "trip_windows":     [],
            "tool_calls":       state.get("tool_calls", []) + ["temporal_correlator(no_docs)"],
        }

    dated_docs, undated_docs = [], []
    for doc in docs:
        (dated_docs if _extract_date(doc) is not None else undated_docs).append(doc)

    windows  = _build_trip_windows(dated_docs, gap_days=TRIP_WINDOW_GAP_DAYS)
    n_txn    = sum(len(w["transactions"]) for w in windows)
    n_map    = sum(len(w["map"])          for w in windows)
    n_photo  = sum(len(w["photos"])       for w in windows)

    log.info(f"[temporal_correlator] {len(windows)} windows | txn={n_txn}  map={n_map}  photo={n_photo}  undated={len(undated_docs)}")
    for i, w in enumerate(windows, 1):
        log.info(f"[temporal_correlator] window {i}: {w['start']} -> {w['end']} ({w['days']}d)  txn={len(w['transactions'])}  map={len(w['map'])}  photo={len(w['photos'])}")

    temporal_context = _windows_to_context(windows, undated_docs)
    _ts(f"temporal_correlator: done  ({len(windows)} windows)")
    return {
        "temporal_context": temporal_context,
        "trip_windows":     windows,
        "tool_calls":       state.get("tool_calls", []) + [f"temporal_correlator({len(windows)} windows)"],
    }


# ---------------------------------------------------------------------------
# Node 6 -- Analyser
# ---------------------------------------------------------------------------

_ANALYSER_SYSTEM = """You are an analyst extracting structured insights from a user's personal travel history.

Your job is to analyse the evidence and produce a clear summary that can be used to answer the user's question.
Do NOT answer the question yet — just analyse the evidence.

Extract and summarise:
1. Destinations visited (list each with approximate date and total spend if available)
2. Spending patterns (which categories: food, transport, accommodation, shopping, activities — and rough amounts)
3. Travel style signals (budget level, trip length, types of places saved or photographed)
4. What the user seems to enjoy (inferred from map saves and photo scenes)
5. Any budget or preference constraints relevant to the question

Be specific. Use numbers where available. Flag anything that seems notable or relevant to the question."""


def analyser(state: dict) -> dict:
    """
    Analyse retrieved evidence to extract facts, patterns, and constraints.

    Output is a structured analysis string stored in state["analysis"].
    This separates factual extraction from answer generation, giving the
    downstream responder clean structured input rather than raw documents.
    """
    _ts("analyser: start")
    docs             = state.get("retrieved_docs", [])
    temporal_context = state.get("temporal_context", "")
    query            = state["query"]
    memory           = state.get("memory", {})

    context_str = temporal_context if temporal_context else _docs_to_context(docs)
    memory_str  = json.dumps(memory) if memory else "(none stored)"

    if not docs:
        _ts("analyser: done (no docs)")
        return {
            "analysis":  "(no evidence retrieved)",
            "tool_calls": state.get("tool_calls", []) + ["analyser(no_docs)"],
        }

    response = _llm.invoke([
        SystemMessage(content=_ANALYSER_SYSTEM),
        HumanMessage(content=(
            f"User preferences: {memory_str}\n\n"
            f"Evidence:\n{context_str}\n\n"
            f"Question being answered: {query}"
        )),
    ])
    analysis = response.content.strip()
    log.info(f"[analyser] analysis length={len(analysis)} chars")
    _ts("analyser: done")
    return {
        "analysis":   analysis,
        "tool_calls": state.get("tool_calls", []) + ["analyser"],
    }


# ---------------------------------------------------------------------------
# Node 7a -- Direct Responder  (lookup path)
# ---------------------------------------------------------------------------

_DIRECT_SYSTEM = """You are a travel assistant answering questions about the user's personal travel history.

You have been given a structured analysis of the relevant evidence.
Answer the user's question directly and precisely from this analysis.

Guidelines:
- Be specific — use destinations, amounts, dates from the analysis
- For spending: group by category, give totals or estimates
- For factual lookups: give the direct answer, don't pad
- Do NOT mention "analysis", "evidence", "documents", or any technical process
- Write as if you know the user's travel history personally"""


def direct_responder(state: dict) -> dict:
    """
    Lookup path: answer directly from the analysis.
    Used when intent="lookup" — the answer exists in the data.
    """
    _ts("direct_responder: start")
    analysis = state.get("analysis", "(no analysis)")
    memory   = state.get("memory", {})
    query    = state["query"]

    response = _llm.invoke([
        SystemMessage(content=_DIRECT_SYSTEM),
        HumanMessage(content=(
            f"User preferences: {json.dumps(memory) if memory else '(none)'}\n\n"
            f"Analysis of travel history:\n{analysis}\n\n"
            f"Question: {query}"
        )),
    ])
    answer = response.content.strip()
    _ts("direct_responder: done")
    return _build_output(state, answer)


# ---------------------------------------------------------------------------
# Node 7b -- Creative Responder  (generative path)
# ---------------------------------------------------------------------------

_CREATIVE_SYSTEM = """You are a travel assistant making personalised recommendations based on the user's travel history.

You have been given a structured analysis of the user's past trips, spending patterns, and travel style.
Use this analysis to generate a recommendation for something NEW — not somewhere they have already been.

Guidelines:
- Identify destinations already visited from the analysis and explicitly EXCLUDE them
- Reason from their budget (typical spend), travel style, and what they seem to enjoy
- Suggest 2-3 specific new destinations with justification tied to their past behaviour
- Give an estimated budget based on their typical spending patterns
- Be concrete and personal — reference their actual history in your reasoning
- Do NOT mention "analysis", "evidence", "documents", or any technical process
- If you cannot determine past destinations clearly, say so and give a best-effort recommendation"""


def creative_responder(state: dict) -> dict:
    """
    Generative path: produce a new recommendation/plan from the analysis.
    Used when intent="generative" — the answer must go beyond the data.
    """
    _ts("creative_responder: start")
    analysis = state.get("analysis", "(no analysis)")
    memory   = state.get("memory", {})
    query    = state["query"]

    response = _llm.invoke([
        SystemMessage(content=_CREATIVE_SYSTEM),
        HumanMessage(content=(
            f"User preferences: {json.dumps(memory) if memory else '(none)'}\n\n"
            f"Analysis of past travel history:\n{analysis}\n\n"
            f"Request: {query}"
        )),
    ])
    answer = response.content.strip()
    _ts("creative_responder: done")
    return _build_output(state, answer)


def _build_output(state: dict, answer: str) -> dict:
    """Shared output builder for both responder nodes."""
    grounded   = True
    ungrounded = []

    if ENABLE_VERIFY or os.getenv("AGENT_VERIFY", "0") == "1":
        _ts("verifier: start")
        context_str = state.get("temporal_context") or _docs_to_context(state.get("retrieved_docs", []))
        try:
            ver_response = _llm.invoke([
                SystemMessage(content=(
                    "Check whether each factual claim in the answer is supported by the evidence. "
                    "Respond ONLY with valid JSON: "
                    '{"grounded": true/false, "ungrounded_claims": ["..."]}'
                )),
                HumanMessage(content=f"Evidence:\n{context_str}\n\nAnswer:\n{answer}"),
            ])
            v          = _parse_json(ver_response.content)
            grounded   = bool(v.get("grounded", True))
            ungrounded = v.get("ungrounded_claims", [])
            _ts(f"verifier: done  (grounded={grounded})")
        except Exception:
            pass

    return {
        "answer":            answer,
        "grounded":          grounded,
        "ungrounded_claims": ungrounded,
        "messages":          state.get("messages", []) + [AIMessage(content=answer)],
        "tool_calls":        state.get("tool_calls", []) + [f"synthesis(grounded={grounded})"],
    }