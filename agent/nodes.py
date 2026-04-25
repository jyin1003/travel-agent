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
               e.g. "how much did I spend in Tokyo", "when did I visit Berlin",
               "show me photos from my Japan trip", "what restaurants did I go to"
               -> analyser extracts the fact -> direct_responder returns it

  generative : answer requires reasoning BEYOND the data to produce something new
               e.g. "recommend somewhere new", "suggest a trip", "what should I do next"
               -> analyser extracts patterns/constraints -> creative_responder generates
                  something the user has NOT done/seen before

The router classifies intent; the analyser does the heavy reasoning;
the responders are thin formatters focused on their specific output style.

Image handling
--------------
  When image_paths are present in state, nodes that need visual context (_query_router,
  analyser) use _vlm (a vision-language model) instead of _llm.
  _encode_images() converts local file paths to base64 image_url content blocks
  compatible with the Groq vision API.
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
    """
    Encode local image files as base64 image_url content blocks for the Groq
    vision API.  Unsupported extensions and missing files are skipped with a
    warning so a bad path never crashes the pipeline.

    Returns a list of dicts:
        [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]
    """
    blocks = []
    for path in image_paths:
        p = Path(path)
        if not p.exists():
            log.warning(f"[nodes] Image not found, skipping: {path}")
            continue
        if p.suffix.lower() not in _SUPPORTED_IMAGE_EXTS:
            log.warning(f"[nodes] Unsupported image extension, skipping: {path}")
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
    """
    Return a multimodal content list when image_blocks are present,
    or a plain string when there are none (keeps non-vision calls clean).
    """
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

If one or more images are attached to this message, treat them as additional query
context. A query with images is almost always "cross_modal" or "factual".

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

    image_paths   = state.get("image_paths", [])
    image_blocks  = _encode_images(image_paths)
    model         = _vlm if image_blocks else _llm

    user_content  = _build_user_content(state["query"], image_blocks)

    if image_blocks:
        log.info(f"[router] {len(image_blocks)} image(s) attached — using VLM ({_GROQ_VLM_MODEL})")

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

Source types:
    transaction   - bank/card transactions: merchant, amount, category, date, destination
    map           - saved Google Maps places: name, address, type, rating, opening hours
    photo_caption - AI captions of travel photos: scene description, location, date, GPS

Retrieval modes:
    "text"  - transactions + map entries only
    "image" - photo captions only  
    "full"  - all three (use this by DEFAULT for almost all queries)

CRITICAL RULE — parallel source coverage:
For every query, generate at least ONE sub-query targeting EACH source type:
    - A transaction sub-query  (what was bought/paid)
    - A map sub-query          (what places were saved/visited)
    - A photo sub-query        (what scenes/activities were photographed)

This is essential because the system links all three by DATE into trip windows.
Photos reveal WHAT the user was doing when a transaction occurred.
Map saves reveal WHERE they were. Transactions reveal HOW MUCH they spent.
Together they reconstruct the full activity, not just the cost.

Photo sub-queries should describe VISUAL SCENES, not just locations:
    Good: "street food market stalls eating", "mountain hiking trail scenery"
    Bad:  "travel photo locations visited"

Sub-query rules:
    - Plain natural language strings only, NOT SQL
    - 4-6 sub-queries covering all three source types
    - At least 1 transaction, 1 map, and 2 photo scene sub-queries
    - Filters: destination (str), year (int), source_type (str)

Examples:

Query: what do I spend money on while travelling [intent=lookup]
{"sub_queries": [
    "restaurant food drink payment receipt",
    "transport taxi train bus ticket fare",
    "accommodation hotel hostel payment",
    "shopping market souvenir purchase",
    "street food market eating local cuisine photo",
    "transport station platform commuting photo",
    "hotel accommodation room stay photo",
    "tourist attraction museum sightseeing photo",
    "saved restaurants cafes food places map",
    "saved transport routes stations map"
    ],
    "retrieval_mode": "full", "text_filter": {}, "image_filter": {}}

Query: what was my Hong Kong trip like [intent=lookup]
{"sub_queries": [
    "Hong Kong transactions spending food transport",
    "Hong Kong saved places restaurants attractions map",
    "Hong Kong street scenes harbour skyline photo",
    "Hong Kong food markets local cuisine photo",
    "Hong Kong transport MTR ferry commute photo"
    ],
    "retrieval_mode": "full",
    "text_filter": {"destination": "Hong Kong"},
    "image_filter": {"destination": "Hong Kong"}}

Query: recommend somewhere new to travel [intent=generative]
{"sub_queries": [
    "all destination transactions total costs",
    "accommodation food transport spending amounts",
    "saved map places countries points of interest",
    "outdoor nature landscape scenery travel photo",
    "urban city streets architecture cultural photo",
    "food dining restaurant local cuisine photo"
    ],  
    "retrieval_mode": "full", "text_filter": {}, "image_filter": {}}

Respond ONLY with valid JSON:"""


def retrieval_planner(state: dict) -> dict:
    """Decompose the query into sub-queries and select retrieval parameters."""
    _ts("retrieval_planner: start")
    memory_str = json.dumps(state.get("memory", {})) if state.get("memory") else "{}"

    # If images were provided, note that to the planner so it biases toward "full"
    image_note = ""
    if state.get("image_paths"):
        n = len(state["image_paths"])
        image_note = f"\nNote: the user also provided {n} image file(s) directly. Bias retrieval_mode toward 'full' and include photo scene sub-queries.\n"

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

    text_filter  = parsed.get("text_filter")  or None
    image_filter = parsed.get("image_filter") or None
    if text_filter  == {}: text_filter  = None
    if image_filter == {}: image_filter = None
    if text_filter:
        text_filter  = {k: v for k, v in text_filter.items()  if k in _VALID_FILTER_KEYS} or None
    if image_filter:
        image_filter = {k: v for k, v in image_filter.items() if k in _VALID_FILTER_KEYS} or None

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
            results = retrieve(query=sq, mode=mode, k=15,
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

    all_docs = list(accumulated.values())
    photo_count = sum(
        1 for d in all_docs
        if d.get("metadata", {}).get("source_type") == "photo_caption"
    )
    if photo_count == 0 and mode != "text":
        _ts("tool_executor: no photos retrieved — running caption fallback pass")
        try:
            fallback_results = retrieve(
                query=state["query"],
                mode="caption_only",
                k=10,
                image_where=image_filter,
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
    ("cash withdrawal", [
        "atm", "cash", "withdrawal", "cashpoint",
    ]),
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
  - Photos taken on the same or adjacent day (AI captions describing what was seen)
  - Map places saved on the same trip (names, types, addresses)

Your job: infer the most specific activity label possible.

Rules:
  - If a photo caption describes an activity that plausibly matches the transaction
    payee/amount, use that activity as the label
  - If a map save matches the payee name or area, use the place type to refine
  - If no cross-modal evidence exists, infer from payee and description text alone
  - Never invent specifics not supported by the evidence
  - Keep labels concise: 3-8 words

Respond with ONLY a JSON array, one object per transaction:
[{"id": "txn_001", "inferred_activity": "street food at night market"}, ...]"""


def transaction_enricher(state: dict) -> dict:
    """
    Enrich transaction records with inferred activity labels:
      1. Rule-based inference from payee/description/address (fast, no LLM)
      2. LLM inference for ambiguous transactions cross-referencing same-day
         photos and map saves
    """
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

    # ── Pass 1: rule-based ────────────────────────────────────────
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

    # ── Pass 2: LLM inference ─────────────────────────────────────
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
        has_photos      = bool(photos_by_date.get(txn_date) or photos_by_date.get("unknown"))
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
                photos_by_date.get(txn_date, []) +
                photos_by_date.get("unknown", [])
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

_ANALYSER_SYSTEM = """You are an analyst reconstructing a user's travel activities from three linked data sources:

  TRANSACTIONS  — what they paid for (merchant, amount, category, date)
  MAP SAVES     — places they bookmarked (name, type, address, rating)
  PHOTOS        — what they were seeing/doing (AI caption, GPS, date)

The evidence arrives grouped into TRIP WINDOWS — clusters of records that share
overlapping dates (within a few days of each other). Your job is to fuse all three
sources within each window to reconstruct what actually happened.

If one or more images are directly attached to this message, treat them as
additional first-hand photo evidence. Describe what you see in each image and
incorporate it into the analysis alongside the retrieved documents.

FUSION RULES:
0. Check for pre-enriched activity labels — transactions may already carry an
   'inferred_activity' field. If present, use it as the primary activity description.
1. Anchor to dates — if a photo and a transaction share the same date or are within
   1 day of each other, treat them as the SAME activity unless content contradicts it.
2. Photos reveal the activity — always let the photo description refine the transaction label.
3. Map saves provide intent and context.
4. When no transaction matches a photo — note these as zero-cost activities.
5. When no photo matches a transaction — describe the transaction category only.
   Do not invent activities.

For each trip window, output:
  - Date range and destination
  - Reconstructed daily activities (fuse transaction + photo + map where possible)
  - Spending breakdown by actual activity
  - What the photos reveal about travel style and preferences
  - Zero-cost activities visible in photos but absent from transactions

End with a cross-window summary covering:
  - Overall spending patterns by activity type
  - Recurring preferences visible across multiple trips
  - Anything notable about how photo evidence changes the interpretation of transactions

Be specific. Quote dates, amounts, place names, and photo descriptions where available."""


def analyser(state: dict) -> dict:
    """
    Analyse retrieved evidence to extract facts, patterns, and constraints.
    When image_paths are present, encodes them and sends to the VLM so the
    images become part of the evidence alongside retrieved documents.
    """
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
        log.info(f"[analyser] {len(image_blocks)} image(s) attached — using VLM ({_GROQ_VLM_MODEL})")

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
    """Lookup path: answer directly from the analysis."""
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
    """Generative path: produce a new recommendation/plan from the analysis."""
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