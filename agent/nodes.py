"""
agent/nodes.py

Node pipeline:
  query_router -> [memory_manager] -> retrieval_planner
                                            |
                                      tool_executor
                                            |
                                    temporal_correlator
                                            |
                                       analyser
                                            |
                               ┌────────────┴────────────┐
                               ▼                         ▼
                        (lookup path)            (generative path)
                      direct_responder         creative_responder
                               └────────────┬────────────┘
                                            ▼
                                           END

Retrieval improvements (v2)
---------------------------
1. Country-level expansion in retrieval_planner:
   - The planner now always outputs a `geo_scope` field with the inferred
     country (e.g. "Japan" for a query about "Tokyo").
   - text_filter and image_filter are set at COUNTRY level when the query
     is country-scoped, city level only when hyper-local precision is needed.
   - This fixes Q1-style failures where documents are tagged with cities
     (e.g. destination="Tokyo") but the query uses the country name ("Japan").

2. Broad retrieval fallback in tool_executor:
   - After the main retrieval pass, if total unique docs < SPARSE_THRESHOLD,
     tool_executor runs a second "broad" pass: no metadata filter, high k,
     then post-filters in Python using the geo_terms set built from the query.
   - This catches city-tagged documents that city-only filters miss.

3. Query expansion for conversational queries (Q4):
   - When query_type == "conversational", tool_executor calls the LLM once
     to expand the query into 3-5 semantically related sub-queries, then
     runs each through retrieval and merges results.
   - This dramatically broadens recall for preference-sensitive questions.

4. Default k raised from 15 → 25 per sub-query.

5. Caption fallback retained and triggered earlier (photo_count < 2).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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
from src.retrieval.retriever import (
    retrieve,
    build_geo_terms,
    resolve_country,
    broad_retrieve,
)

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
    log.info(f"{label}")


# ---------------------------------------------------------------------------
# Ablation controls
# ---------------------------------------------------------------------------

ENABLE_VERIFY: bool = os.getenv("AGENT_VERIFY", "0") == "1"
ENABLE_TEMPORAL_CORRELATION: bool = os.getenv("AGENT_TEMPORAL", "1") == "1"
TRIP_WINDOW_GAP_DAYS: int = 3

# Minimum unique docs before triggering broad fallback in tool_executor
SPARSE_THRESHOLD: int = int(os.getenv("AGENT_SPARSE_THRESHOLD", "8"))

# Default k per sub-query (raised from 15 for broader recall)
DEFAULT_K: int = int(os.getenv("AGENT_K", "25"))

# ---------------------------------------------------------------------------
# LLM / VLM instances
# ---------------------------------------------------------------------------

_GROQ_MODEL     = os.getenv("GROQ_MODEL",     "llama-3.3-70b-versatile")
_GROQ_VLM_MODEL = os.getenv("GROQ_VLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

_llm = ChatGroq(model=_GROQ_MODEL,     temperature=0, api_key=os.getenv("GROQ_API_KEY"))
_vlm = ChatGroq(model=_GROQ_VLM_MODEL, temperature=0, api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------------------------------------------------------
# Image encoding helper
# ---------------------------------------------------------------------------

_SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def _encode_images(image_paths: list[str]) -> list[dict]:
    blocks = []
    for path in image_paths:
        p = Path(path)
        if not p.exists():
            log.warning(f"[nodes] Image not found: {path}")
            continue
        if p.suffix.lower() not in _SUPPORTED_IMAGE_EXTS:
            log.warning(f"[nodes] Unsupported image extension: {path}")
            continue
        ext  = p.suffix.lower().lstrip(".")
        mime = "jpeg" if ext == "jpg" else ext
        data = base64.b64encode(p.read_bytes()).decode("utf-8")
        blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{mime};base64,{data}"},
        })
        log.info(f"[nodes] Encoded image: {p.name} ({p.stat().st_size // 1024} KB)")
    return blocks


def _build_user_content(text: str, image_blocks: list[dict]) -> list[dict] | str:
    if not image_blocks:
        return text
    return [{"type": "text", "text": text}] + image_blocks


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


def _docs_to_context(docs: list[dict], max_docs: int = 20) -> str:
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
        lines.append(
            f"=== Trip window {i}: {w['start']} → {w['end']} "
            f"({w['days']} day{'s' if w['days'] != 1 else ''}) ==="
        )
        dated_records = []
        for doc in w["transactions"]:
            dt = _extract_date(doc)
            dated_records.append((dt, "transaction", doc))
        for doc in w["photos"]:
            dt = _extract_date(doc)
            dated_records.append((dt, "photo", doc))
        dated_records.sort(key=lambda x: (x[0] or datetime.min.replace(tzinfo=timezone.utc)))

        if dated_records:
            lines.append("  Chronological activity:")
            prev_date = None
            for dt, rec_type, doc in dated_records:
                date_str = dt.date().isoformat() if dt else "unknown date"
                if date_str != prev_date:
                    lines.append(f"  [{date_str}]")
                    prev_date = date_str
                text = doc.get("document", "")[:180]
                lines.append(f"    {rec_type.upper()}: {text}")

        if w["map"]:
            lines.append("  Saved places (map):")
            for doc in w["map"][:8]:
                lines.append(f"    MAP: {doc.get('document', '')[:150]}")
        lines.append("")

    if undated:
        lines.append("=== Records without dates ===")
        for doc in undated[:10]:
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

If one or more images are attached, treat them as additional query context.
A query with images is almost always "cross_modal" or "factual".

Memory flags:
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

    image_paths   = state.get("image_paths", [])
    image_blocks  = _encode_images(image_paths)
    model         = _vlm if image_blocks else _llm
    user_content  = _build_user_content(state["query"], image_blocks)

    if image_blocks:
        log.info(f"[router] {len(image_blocks)} image(s) attached — using VLM")

    response   = model.invoke([
        SystemMessage(content=_ROUTER_SYSTEM),
        HumanMessage(content=user_content),
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
    q_lower = state["query"].lower()
    if any(kw in q_lower for kw in _GENERATIVE_SIGNALS):
        intent = "generative"

    # Hard override: if query contains preference keywords, force memory_write
    if any(kw in state["query"].lower() for kw in _PREFERENCE_SIGNALS):
        mem_write = True


    # If images are attached and no generative signal, lean toward cross_modal
    if image_blocks and query_type not in ("cross_modal", "multi_hop"):
        query_type = "cross_modal"

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

Source types:
    transaction   - bank/card transactions: merchant, amount, category, date, destination
    map           - saved Google Maps places: name, address, type, rating, opening hours
    photo_caption - AI captions of travel photos: scene description, location, date, GPS

Retrieval modes:
    "text"  - transactions + map entries only
    "image" - photo captions only
    "full"  - all three (use this by DEFAULT for almost all queries)

CRITICAL RULE — COUNTRY-LEVEL SCOPE:
Data may be tagged at the CITY level (e.g. destination="Tokyo"), NOT country level.
If the query mentions or implies a country (Japan, Italy, Denmark, etc.), you MUST:

1. Set `geo_scope` to the COUNTRY name (e.g. "Japan").
2. Set `text_filter` and `image_filter` to {} (EMPTY — no filter).
   This is because filtering by country name would miss city-tagged docs.
   The tool_executor will do a broad post-filter using all known cities for that country.
3. Include sub-queries using CITY NAMES of that country, not just the country name.
   E.g. for Japan: use "Tokyo", "Osaka", "Kyoto", "Hiroshima" in sub-queries.

If the query mentions a SPECIFIC CITY and you want precise results, you MAY set:
    text_filter: {"destination": "<city>"}
But only do this when the query is very specifically about that one city AND
you are confident the data uses that exact city name.

For ANY query where you are unsure of the exact destination tag in the data,
use empty filters and rely on broad sub-queries.

CRITICAL RULE — PARALLEL SOURCE COVERAGE:
For every query, generate sub-queries covering ALL THREE source types:
    - Transaction sub-queries (what was bought/paid)
    - Map sub-queries (what places were saved/visited)
    - Photo sub-queries (what scenes/activities were photographed)

Photo sub-queries must describe VISUAL SCENES:
    Good: "street food market stalls eating local cuisine"
    Bad:  "travel photo locations"

Sub-query rules:
    - Plain natural language strings only, NOT SQL
    - 6-10 sub-queries covering all three source types
    - Use specific city names where relevant (e.g. "Tokyo train station", "Osaka food")
    - At least 2 transaction, 2 map, and 3 photo scene sub-queries

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query: how much did I spend on transport in Japan [intent=lookup]
→ geo_scope="Japan", filters EMPTY (city-tagged docs need broad search)
{"geo_scope": "Japan",
 "sub_queries": [
    "Tokyo train subway metro ticket transport",
    "Osaka bus rail transport ticket fare",
    "Kyoto transport taxi bus train",
    "Japan IC card Suica Pasmo railway",
    "shinkansen bullet train ticket Japan",
    "Tokyo station platform transport map saved",
    "Osaka metro station saved places map",
    "train station platform Japan travel photo",
    "subway metro underground Japan photo",
    "bus transport street Japan travel photo"
 ],
 "retrieval_mode": "full", "text_filter": {}, "image_filter": {}}

Query: when did I visit Copenhagen [intent=lookup]
→ geo_scope="Denmark", filters EMPTY
{"geo_scope": "Denmark",
 "sub_queries": [
    "Copenhagen transaction payment spending",
    "Copenhagen Kobenhavn Koebenhavn date visit",
    "Denmark travel spending food transport",
    "Copenhagen saved places restaurants attractions map",
    "Copenhagen street canals colourful buildings photo",
    "Denmark Scandinavia travel photo",
    "Nyhavn harbour Copenhagen photo",
    "Danish food smørrebrød pastry photo"
 ],
 "retrieval_mode": "full", "text_filter": {}, "image_filter": {}}

Query: what do I spend money on while travelling [intent=lookup]
→ no specific location, no geo_scope, no filters
{"geo_scope": null,
 "sub_queries": [
    "restaurant food drink payment receipt",
    "transport taxi train bus ticket fare",
    "accommodation hotel hostel payment",
    "shopping market souvenir purchase",
    "street food market eating local cuisine photo",
    "transport station commuting travel photo",
    "hotel accommodation room stay photo",
    "tourist attraction museum sightseeing photo",
    "saved restaurants cafes food places map",
    "saved transport routes stations map"
 ],
 "retrieval_mode": "full", "text_filter": {}, "image_filter": {}}

Query: recommend somewhere new to travel [intent=generative]
{"geo_scope": null,
 "sub_queries": [
    "all destination transactions total costs",
    "accommodation food transport spending amounts",
    "saved map places countries points of interest",
    "outdoor nature landscape scenery travel photo",
    "urban city streets architecture cultural photo",
    "food dining restaurant local cuisine photo",
    "beach ocean coastal scenery photo",
    "mountain hiking outdoor adventure photo"
 ],
 "retrieval_mode": "full", "text_filter": {}, "image_filter": {}}

Respond ONLY with valid JSON:"""


def retrieval_planner(state: dict) -> dict:
    """Decompose the query into sub-queries and select retrieval parameters."""
    _ts("retrieval_planner: start")
    memory_str = json.dumps(state.get("memory", {})) if state.get("memory") else "{}"

    image_note = ""
    if state.get("image_paths"):
        n = len(state["image_paths"])
        image_note = f"\nNote: the user provided {n} image file(s). Bias retrieval_mode toward 'full' and include photo scene sub-queries.\n"

    response = _llm.invoke([
        SystemMessage(content=_PLANNER_SYSTEM),
        HumanMessage(content=(
            f"Intent: {state.get('intent', 'lookup')}\n"
            f"Query type: {state.get('query_type', 'factual')}\n"
            f"User preferences: {memory_str}\n"
            f"{image_note}"
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

    # ── geo_scope handling ────────────────────────────────────────────────────
    geo_scope = parsed.get("geo_scope")  # e.g. "Japan", "Denmark", None

    # Resolve geo_scope to a canonical country and build geo_terms
    geo_terms: set[str] = set()
    if geo_scope:
        country = resolve_country(geo_scope) or geo_scope
        geo_terms = build_geo_terms(country)
        log.info(f"[planner] geo_scope={geo_scope!r} → geo_terms={geo_terms}")

    # ── Filters: use EMPTY for country-scoped queries ─────────────────────────
    text_filter  = parsed.get("text_filter")  or None
    image_filter = parsed.get("image_filter") or None
    if text_filter  == {}: text_filter  = None
    if image_filter == {}: image_filter = None
    if text_filter:
        text_filter  = {k: v for k, v in text_filter.items()  if k in _VALID_FILTER_KEYS} or None
    if image_filter:
        image_filter = {k: v for k, v in image_filter.items() if k in _VALID_FILTER_KEYS} or None

    # If geo_scope was set but filters were not explicitly cleared, clear them
    # so we rely on broad post-filtering rather than restrictive ChromaDB pre-filters
    if geo_scope and (text_filter or image_filter):
        log.info("[planner] geo_scope set — overriding filters to empty for broad retrieval")
        text_filter  = None
        image_filter = None

    if state.get("intent") == "generative" and len(sub_queries) < 3:
        base = sub_queries[0] if sub_queries else state["query"]
        sub_queries = list(dict.fromkeys([
            base,
            "past trip total costs by destination",
            "destinations previously visited transactions",
            "saved map places points of interest",
            "travel photo locations scenes visited",
            "outdoor scenery nature travel photo",
            "city architecture urban culture photo",
        ]))

    log.info(f"[planner] mode={mode}  n_sub={len(sub_queries)}  text_filter={text_filter}  geo_scope={geo_scope}")
    log.info(f"[planner] sub_queries={sub_queries}")
    _ts("retrieval_planner: done")
    return {
        "sub_queries":    sub_queries,
        "retrieval_mode": mode,
        "text_filter":    text_filter,
        "image_filter":   image_filter,
        "geo_scope":      geo_scope,
        "geo_terms":      list(geo_terms),  # serialisable for state
        "tool_calls":     state.get("tool_calls", []) + [f"planner(mode={mode}, n_sub={len(sub_queries)}, geo={geo_scope})"],
    }


# ---------------------------------------------------------------------------
# Node 4 -- Tool Executor
# ---------------------------------------------------------------------------

_ABLATION_MODES = {"dense_only", "bm25_only", "clip_only", "caption_only"}
_MODE_TO_TOOL   = {"text": search_text_tool, "image": search_images_tool, "full": hybrid_search_tool}

# Query expansion prompt for conversational queries (Q4-style)
_EXPAND_SYSTEM = """You are expanding a conversational travel query into multiple search queries.

The user's query references personal preferences, dietary restrictions, past experiences,
or multi-turn context. Generate 4-6 diverse search queries that together will retrieve
all relevant information needed to answer it.

Rules:
- Queries must be plain natural language strings
- Cover different aspects: places, transactions, photos, preferences
- Vary terminology and specificity
- Do NOT include meta-instructions, just query strings

Return ONLY a JSON array of strings, e.g.:
["query 1", "query 2", "query 3"]"""


def _expand_query_for_conversational(query: str, memory: dict) -> list[str]:
    """
    Use the LLM to expand a conversational/preference query into multiple
    semantically diverse sub-queries for broader retrieval.
    """
    try:
        memory_str = json.dumps(memory) if memory else "{}"
        response = _llm.invoke([
            SystemMessage(content=_EXPAND_SYSTEM),
            HumanMessage(content=(
                f"User preferences from memory: {memory_str}\n\n"
                f"Query to expand: {query}"
            )),
        ])
        raw = response.content.strip()
        # Try to extract JSON array
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            candidates = json.loads(match.group())
            return [c for c in candidates if isinstance(c, str) and len(c) > 5]
    except Exception as e:
        log.warning(f"[tool_executor] query expansion failed: {e}")
    return []


def tool_executor(state: dict) -> dict:
    """
    Run retrieve() for each sub-query and de-duplicate results by doc ID.

    Improvements over v1:
    - k raised to DEFAULT_K (25) per sub-query
    - Conversational queries trigger LLM query expansion before retrieval
    - Country-scoped queries trigger a broad geo-filtered fallback pass
      when results are sparse (< SPARSE_THRESHOLD unique docs)
    - Caption fallback triggered earlier (photo_count < 2)
    """
    _ts("tool_executor: start")
    mode         = state.get("retrieval_mode", "full")
    raw_queries  = state.get("sub_queries", [state["query"]])
    text_filter  = state.get("text_filter")
    image_filter = state.get("image_filter")
    geo_scope    = state.get("geo_scope")
    geo_terms    = set(state.get("geo_terms", []))
    query_type   = state.get("query_type", "factual")
    tool_trace   = list(state.get("tool_calls", []))

    sub_queries = list(dict.fromkeys(
        _sanitise_query(sq, state["query"]) for sq in raw_queries if sq
    )) or [state["query"]]

    # ── Query expansion for conversational queries (Q4) ───────────────────────
    if query_type == "conversational":
        _ts("tool_executor: expanding conversational query")
        expanded = _expand_query_for_conversational(state["query"], state.get("memory", {}))
        if expanded:
            log.info(f"[tool_executor] query expansion added {len(expanded)} sub-queries")
            tool_trace.append(f"query_expansion({len(expanded)} queries)")
            # Prepend expanded queries, deduplicate preserving order
            combined = expanded + sub_queries
            sub_queries = list(dict.fromkeys(combined))

    # ── Main retrieval pass ───────────────────────────────────────────────────
    accumulated: dict[str, dict] = {}
    for sq in sub_queries:
        _ts(f"tool_executor: retrieve  q={sq[:50]!r}  mode={mode}")
        try:
            results = retrieve(
                query=sq,
                mode=mode,
                k=DEFAULT_K,
                text_where=text_filter,
                image_where=image_filter,
                auto_expand=True,  # retriever will auto-expand if needed
            )
            tool_fn = _MODE_TO_TOOL.get(mode, hybrid_search_tool)
            tool_trace.append(f"{tool_fn.name}(q={sq[:40]!r})")
            log.info(f"[tool_executor] {len(results)} docs for q={sq[:40]!r}")
        except Exception as e:
            tool_trace.append(f"retrieve_error({sq[:40]!r}: {e})")
            log.warning(f"[tool_executor] retrieve failed: {e}")
            results = []
        for r in results:
            accumulated[r["id"]] = r

    # ── Broad geo fallback pass ───────────────────────────────────────────────
    # Triggered when: geo_scope is set AND we have too few docs.
    # Runs a wide unfiltered search then post-filters by all known geo terms.
    if geo_scope and len(accumulated) < SPARSE_THRESHOLD:
        _ts(f"tool_executor: sparse results ({len(accumulated)}) — broad geo fallback for {geo_scope!r}")
        if not geo_terms:
            geo_terms = build_geo_terms(geo_scope)
        try:
            broad_results = broad_retrieve(
                query=state["query"],
                geo_terms=geo_terms,
                k=DEFAULT_K * 3,  # wide net
                mode=mode if mode != "image" else "full",
            )
            tool_trace.append(f"broad_geo_fallback(geo={geo_scope}, k={DEFAULT_K*3})")
            log.info(f"[tool_executor] broad geo fallback returned {len(broad_results)} docs")
            for r in broad_results:
                accumulated[r["id"]] = r
        except Exception as e:
            log.warning(f"[tool_executor] broad geo fallback failed: {e}")

    # Additional broad pass with individual city sub-queries for geo-scoped queries
    if geo_scope and geo_terms and len(accumulated) < SPARSE_THRESHOLD * 2:
        _ts("tool_executor: running city-level sub-queries for geo scope")
        city_queries = [
            f"{city} transport ticket fare train bus"
            for city in sorted(geo_terms)[:6]
            if len(city) > 3
        ]
        for cq in city_queries:
            try:
                city_results = retrieve(query=cq, mode="text", k=10, auto_expand=False)
                tool_trace.append(f"city_sub(q={cq[:30]!r})")
                new_count = sum(1 for r in city_results if r["id"] not in accumulated)
                for r in city_results:
                    accumulated[r["id"]] = r
                if new_count > 0:
                    log.info(f"[tool_executor] city sub-query added {new_count} new docs: {cq[:40]!r}")
            except Exception as e:
                log.warning(f"[tool_executor] city sub-query failed: {e}")

    all_docs = list(accumulated.values())

    # ── Caption fallback (triggered if < 2 photos, not 0 as before) ───────────
    photo_count = sum(
        1 for d in all_docs
        if d.get("metadata", {}).get("source_type") == "photo_caption"
    )
    if photo_count < 2 and mode != "text":
        _ts("tool_executor: few photos — running caption fallback pass")
        try:
            fallback_results = retrieve(
                query=state["query"],
                mode="caption_only",
                k=DEFAULT_K,
                image_where=image_filter,
                auto_expand=True,
            )
            tool_trace.append(f"caption_fallback(q={state['query'][:40]!r})")
            log.info(f"[tool_executor] caption fallback returned {len(fallback_results)} docs")
            for r in fallback_results:
                accumulated[r["id"]] = r
        except Exception as e:
            log.warning(f"[tool_executor] caption fallback failed: {e}")

    all_docs   = list(accumulated.values())
    image_docs = [d for d in all_docs if d.get("metadata", {}).get("source_type") == "photo_caption"]
    log.info(f"[tool_executor] total unique docs={len(all_docs)}  photos={len(image_docs)}")
    _ts(f"tool_executor: done  ({len(all_docs)} unique docs, {len(image_docs)} photos)")
    return {
        "retrieved_docs":   all_docs,
        "retrieved_images": image_docs,
        "tool_calls":       tool_trace,
    }


# ---------------------------------------------------------------------------
# Rule-based category inference
# ---------------------------------------------------------------------------

_CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("accommodation", [
        "hotel", "hostel", "airbnb", "booking.com", "hotels.com", "inn",
        "lodge", "resort", "motel", "guesthouse", "bnb", "b&b", "suites",
        "apartments", "serviced apartment",
    ]),
    ("flights", [
        "ryanair", "easyjet", "british airways", "lufthansa", "emirates",
        "cathay", "qantas", "delta", "united airlines", "southwest",
        "air france", "klm", "wizz", "jet2", "flybe", "airport", "airways",
    ]),
    ("transport", [
        "uber", "lyft", "bolt", "grab", "ola", "taxi", "cab",
        "tfl", "oyster", "trainline", "national rail", "eurostar",
        "bus", "metro", "tube", "rail", "ferry", "tram", "mtr",
        "transport for london", "heathrow express",
    ]),
    ("food & drink", [
        "restaurant", "cafe", "coffee", "starbucks", "costa", "pret",
        "mcdonald", "kfc", "subway", "nandos", "wagamama", "itsu",
        "eat", "food", "dining", "bistro", "brasserie", "pub", "bar",
        "bakery", "deli", "sushi", "pizza", "burger", "chicken",
        "ramen", "noodle", "curry", "dim sum", "tapas", "brunch",
        "deliveroo", "uber eats", "just eat",
    ]),
    ("groceries", [
        "supermarket", "tesco", "sainsbury", "waitrose", "marks & spencer",
        "m&s", "asda", "morrisons", "lidl", "aldi", "co-op", "whole foods",
        "trader joe", "kroger", "carrefour", "rewe", "spar",
    ]),
    ("activities & attractions", [
        "museum", "gallery", "exhibition", "tour", "tickets", "entry",
        "admission", "zoo", "aquarium", "theme park", "cinema", "theatre",
        "concert", "event", "excursion", "experience", "escape room",
        "bowling", "climbing", "kayak", "surf", "ski", "dive",
    ]),
    ("shopping", [
        "amazon", "ebay", "asos", "zara", "h&m", "primark", "topshop",
        "john lewis", "selfridges", "harrods", "department store",
        "pharmacy", "boots", "duty free", "souvenir", "gift shop",
        "market stall", "mall", "outlet",
    ]),
    ("health & wellness", [
        "pharmacy", "chemist", "doctor", "clinic", "hospital", "dentist",
        "gym", "spa", "massage", "yoga", "fitness",
    ]),
    ("communication & data", [
        "sim card", "data plan", "roaming", "vodafone", "ee", "o2",
        "three", "at&t", "verizon", "t-mobile",
    ]),
    ("cash withdrawal", ["atm", "cash", "withdrawal", "cashpoint"]),
]


def _rule_infer_category(payee: str, description: str, address: str = "") -> str:
    haystack = " ".join([payee, description, address]).lower()
    for category, keywords in _CATEGORY_RULES:
        if any(kw in haystack for kw in keywords):
            return category
    return ""


# ---------------------------------------------------------------------------
# Node 4b -- Transaction Enricher
# ---------------------------------------------------------------------------

_ENRICHER_SYSTEM = """You are enriching bank transaction records with inferred activity context.

For each transaction you will be given:
  - The raw transaction (payee, amount, existing category if any, date)
  - Photos taken on the same or adjacent day (AI captions)
  - Map places saved on the same trip

Your job: infer the most specific activity label possible.

Rules:
  - Use photo captions to refine the activity label when plausible
  - Use map saves to refine place type
  - Never invent specifics not in the evidence
  - Labels: 3-8 words, concise

Return ONLY a JSON array:
[{"id": "txn_001", "inferred_activity": "street food at night market"}, ...]"""


def transaction_enricher(state: dict) -> dict:
    _ts("transaction_enricher: start")
    docs = state.get("retrieved_docs", [])
    if not docs:
        _ts("transaction_enricher: done (no docs)")
        return {
            "retrieved_docs": docs,
            "tool_calls": state.get("tool_calls", []) + ["transaction_enricher(no_docs)"],
        }

    transactions = [d for d in docs if d.get("metadata", {}).get("source_type") == "transaction"]
    photos       = [d for d in docs if d.get("metadata", {}).get("source_type") == "photo_caption"]
    map_saves    = [d for d in docs if d.get("metadata", {}).get("source_type") == "map"]

    if not transactions:
        _ts("transaction_enricher: done (no transactions)")
        return {
            "retrieved_docs": docs,
            "tool_calls": state.get("tool_calls", []) + ["transaction_enricher(no_txns)"],
        }

    # Pass 1: rule-based
    rule_enriched = 0
    for txn in transactions:
        meta = txn.get("metadata", {})
        if meta.get("inferred_activity"):
            continue
        inferred = _rule_infer_category(
            meta.get("payee", ""),
            meta.get("description", ""),
            meta.get("address", ""),
        )
        if inferred:
            meta["inferred_activity"] = inferred
            txn["document"] = txn.get("document", "") + f" [activity: {inferred}]"
            rule_enriched += 1

    log.info(f"[transaction_enricher] rule-based: {rule_enriched}/{len(transactions)} categorised")

    # Pass 2: LLM inference
    def _date_key(doc: dict) -> str:
        dt = _extract_date(doc)
        return dt.date().isoformat() if dt else "unknown"

    photos_by_date: dict[str, list[dict]] = {}
    map_by_date:    dict[str, list[dict]] = {}
    for p in photos:
        photos_by_date.setdefault(_date_key(p), []).append(p)
    for m in map_saves:
        map_by_date.setdefault(_date_key(m), []).append(m)

    to_enrich_llm = []
    for txn in transactions:
        meta     = txn.get("metadata", {})
        txn_date = _date_key(txn)
        has_photos       = bool(photos_by_date.get(txn_date) or photos_by_date.get("unknown"))
        already_specific = bool(meta.get("inferred_activity")) and not has_photos
        if not already_specific:
            to_enrich_llm.append(txn)

    llm_enriched = 0
    if to_enrich_llm:
        txn_blocks = []
        for txn in to_enrich_llm[:20]:
            meta     = txn.get("metadata", {})
            txn_date = _date_key(txn)
            same_day_photos = (
                photos_by_date.get(txn_date, []) + photos_by_date.get("unknown", [])
            )[:4]
            same_day_maps = map_by_date.get(txn_date, [])[:3]
            txn_blocks.append({
                "id":               txn["id"],
                "date":             txn_date,
                "payee":            meta.get("payee", ""),
                "description":      meta.get("description", ""),
                "amount":           meta.get("amount", ""),
                "current_category": meta.get("inferred_activity") or meta.get("category", ""),
                "photos_same_day":  [p.get("document", "")[:150] for p in same_day_photos],
                "map_saves":        [m.get("document", "")[:100] for m in same_day_maps],
            })

        try:
            response = _llm.invoke([
                SystemMessage(content=_ENRICHER_SYSTEM),
                HumanMessage(content=json.dumps(txn_blocks, indent=2)),
            ])
            raw = re.search(r"\[.*\]", response.content, re.DOTALL)
            if raw:
                for item in json.loads(raw.group()):
                    txn_id   = item.get("id")
                    activity = item.get("inferred_activity", "").strip()
                    if not txn_id or not activity:
                        continue
                    for doc in docs:
                        if doc["id"] == txn_id:
                            doc["metadata"]["inferred_activity"] = activity
                            if "[activity:" not in doc.get("document", ""):
                                doc["document"] = doc.get("document", "") + f" [activity: {activity}]"
                            else:
                                doc["document"] = re.sub(
                                    r"\[activity:[^\]]*\]",
                                    f"[activity: {activity}]",
                                    doc["document"],
                                )
                            llm_enriched += 1
                            break
        except Exception as e:
            log.warning(f"[transaction_enricher] LLM pass failed: {e}")

    log.info(f"[transaction_enricher] LLM pass: {llm_enriched}/{len(to_enrich_llm)} refined")
    _ts(f"transaction_enricher: done (rule={rule_enriched}, llm={llm_enriched})")
    return {
        "retrieved_docs": docs,
        "tool_calls": state.get("tool_calls", []) + [
            f"transaction_enricher(rule={rule_enriched}, llm={llm_enriched})"
        ],
    }


# ---------------------------------------------------------------------------
# Node 5 -- Temporal Correlator
# ---------------------------------------------------------------------------

def temporal_correlator(state: dict) -> dict:
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
        log.info(f"[temporal_correlator] window {i}: {w['start']} -> {w['end']} ({w['days']}d)")

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

_ANALYSER_SYSTEM = """You are an analyst reconstructing a user's travel activities from three linked data sources:

  TRANSACTIONS  — what they paid for (merchant, amount, category, date)
  MAP SAVES     — places they bookmarked (name, type, address, rating)
  PHOTOS        — what they were seeing/doing (AI caption, GPS, date)

Evidence arrives grouped into TRIP WINDOWS — clusters of records sharing overlapping dates.
Fuse all three sources within each window to reconstruct what actually happened.

If images are directly attached, treat them as first-hand photo evidence.

FUSION RULES:
0. Check pre-enriched activity labels on transactions — use them as primary activity descriptions.
1. Anchor to dates — photo + transaction on same/adjacent date = same activity unless contradicted.
2. Photos reveal the activity — let captions refine transaction labels.
3. Map saves provide intent and context.
4. No transaction for a photo activity → note as zero-cost activity.
5. No photo for a transaction → describe category only. Do NOT invent.

For each trip window output:
  - Date range and destination
  - Reconstructed daily activities (fuse transaction + photo + map)
  - Spending breakdown by actual activity
  - What photos reveal about travel style
  - Zero-cost activities visible in photos

Cross-window summary:
  - Overall spending patterns by activity type
  - Recurring preferences across trips
  - How photo evidence changes interpretation of transactions

Be specific: quote dates, amounts, place names, photo descriptions."""


def analyser(state: dict) -> dict:
    _ts("analyser: start")
    docs             = state.get("retrieved_docs", [])
    temporal_context = state.get("temporal_context", "")
    query            = state["query"]
    memory           = state.get("memory", {})
    image_paths      = state.get("image_paths", [])

    context_str = temporal_context if temporal_context else _docs_to_context(docs)
    memory_str  = json.dumps(memory) if memory else "(none stored)"

    if not docs and not image_paths:
        _ts("analyser: done (no docs, no images)")
        return {
            "analysis":   "(no evidence retrieved)",
            "tool_calls": state.get("tool_calls", []) + ["analyser(no_docs)"],
        }

    image_blocks = _encode_images(image_paths)
    model        = _vlm if image_blocks else _llm

    if image_blocks:
        log.info(f"[analyser] {len(image_blocks)} image(s) attached — using VLM")

    text_content = (
        f"User preferences: {memory_str}\n\n"
        f"Evidence:\n{context_str}\n\n"
        f"Question being answered: {query}"
    )
    user_content = _build_user_content(text_content, image_blocks)

    response = model.invoke([
        SystemMessage(content=_ANALYSER_SYSTEM),
        HumanMessage(content=user_content),
    ])
    analysis = response.content.strip()
    log.info(f"[analyser] analysis length={len(analysis)} chars")
    _ts("analyser: done")
    return {
        "analysis":   analysis,
        "tool_calls": state.get("tool_calls", []) + ["analyser"],
    }


# ---------------------------------------------------------------------------
# Node 7a -- Direct Responder
# ---------------------------------------------------------------------------

_DIRECT_SYSTEM = """You are a travel assistant answering questions about the user's personal travel history.

Answer directly and precisely from the analysis provided.

Guidelines:
- Be specific — use destinations, amounts, dates from the analysis
- For spending: group by category, give totals or estimates
- For date/period questions: give the earliest and latest dates found for that destination
- Do NOT mention "analysis", "evidence", "documents", or any technical process
- Write as if you know the user's travel history personally"""


def direct_responder(state: dict) -> dict:
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
# Node 7b -- Creative Responder
# ---------------------------------------------------------------------------

_CREATIVE_SYSTEM = """You are a travel assistant making personalised recommendations based on the user's travel history.

Use the analysis to generate recommendations for something NEW — not somewhere they've already been.

Guidelines:
- Identify already-visited destinations and EXCLUDE them
- Reason from budget, travel style, and what they enjoy
- Suggest 2-3 specific new destinations with justification tied to their history
- Give estimated budget based on typical spending patterns
- Reference their actual history in your reasoning
- Do NOT mention "analysis", "evidence", "documents", or any technical process"""


def creative_responder(state: dict) -> dict:
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