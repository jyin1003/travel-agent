"""
retriever.py

Three retrieval paths are available:

    1. Dense text   — sentence-transformer cosine similarity on text_index
    2. Sparse text  — BM25 keyword matching on the same corpus
    3. CLIP image   — CLIP text encoder cosine similarity on image_index
    4. Caption text — sentence-transformer search on text_index pre-filtered
                      to source_type="photo" captions (caption-mediated path)

Fusion strategy
---------------
- Text queries:   RRF over [dense, BM25]          → text_hybrid_query()
- Image queries:  RRF over [CLIP, caption]         → image_hybrid_query()
- Full hybrid:    RRF over [dense, BM25, CLIP, caption] → hybrid_query()

The public `retrieve()` dispatcher is the single entry point for LangGraph
agent tools.  Pass `mode` to select the path; metadata pre-filters are
forwarded to ChromaDB BEFORE the vector search.

Broad retrieval strategy
------------------------
A key failure mode is city-tagged data being missed by country-level queries
(e.g. documents tagged destination="Tokyo" not found for query="Japan transport").

To address this:
- `infer_country_from_city()` maps known cities → countries so filters can be
  widened from city → country when needed.
- `broad_retrieve()` runs retrieve() with NO metadata filter at high k, then
  post-filters by country or city substring match.  Used as a fallback when
  filtered retrieval returns too few results.
- The `retrieve()` dispatcher accepts `auto_expand=True` (default) to
  automatically fall back to broad retrieval when filtered results are sparse.

Trade-off notes
---------------
- BM25 adds ~40-80 ms on first call (corpus build from ChromaDB).  Corpus
  is cached as a module-level singleton so subsequent calls are fast.
- Caption-mediated path gives text-model recall on images where CLIP's
  shared embedding space is weak.
- RRF avoids the need to tune a scalar weighting parameter between modalities.
- Metadata pre-filters reduce the candidate pool before vector search,
  improving precision.  Do NOT post-filter results — that silently reduces k.
  Exception: broad_retrieve() intentionally does a wide fetch then post-filters.
"""

from __future__ import annotations

import threading, logging
from typing import Optional, Literal
from datetime import datetime

from src import common

# Logging helper
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [RETRIEVER] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# City → Country expansion map
# ─────────────────────────────────────────────────────────────────────────────

# Maps lowercase city/suburb names to their country.
# Used to widen a city-level filter to a country-level search so that
# documents tagged with any city in that country are retrieved.
_CITY_TO_COUNTRY: dict[str, str] = {
    # Japan
    "tokyo": "Japan", "toukiyouto": "Japan", "osaka": "Japan",
    "kyoto": "Japan",
    # UK
    "london": "United Kingdom", "edinburgh": "United Kingdom",
    "oxford": "United Kingdom",
    # Italy
    "milan": "Italy", "rome": "Italy", "florence": "Italy",
    "Bologna": "Italy", "venice": "Italy",
    # France
    "paris": "France",
    # Germany
    "munich": "Germany",
    # Sweden
    "goteborg": "Sweden", "gothenburg": "Sweden",
    "skovde": "Sweden", "vastra frolunda": "Sweden",
    # Denmark
    "copenhagen": "Denmark", "koebenhavn": "Denmark",
    "kobenhavn": "Denmark", "kxbenhavn": "Denmark",
    # Belgium
    "brussels": "Belgium", "antwerpen": "Belgium", "antwerp": "Belgium",
    "gent": "Belgium", "ghent": "Belgium", "mechelen": "Belgium",
    # Netherlands
    "amsterdam": "Netherlands", "gouda": "Netherlands",
    # Australia
    "sydney": "Australia", "melbourne": "Australia",
    "brisbane": "Australia",
    # China
    "shanghai": "China", "beijing": "China", "shenzhen": "China",
    "chongqing": "China", "guangzhou": "China", "harbin": "China",
    "hangzhou": "China", "shenyang": "China",
}

# Maps lowercase country names / aliases to canonical country strings
# so that "japan", "uk", "england", etc. all resolve consistently.
_COUNTRY_ALIASES: dict[str, str] = {
    "japan": "Japan",
    "uk": "United Kingdom", "england": "United Kingdom",
    "scotland": "United Kingdom", "britain": "United Kingdom",
    "great britain": "United Kingdom",
    "italy": "Italy",
    "france": "France",
    "germany": "Germany",
    "sweden": "Sweden",
    "denmark": "Denmark",
    "belgium": "Belgium",
    "netherlands": "Netherlands", "holland": "Netherlands",
    "australia": "Australia",
    "china": "China",
}

# Reverse map: country → set of known city strings (lowercase)
_COUNTRY_TO_CITIES: dict[str, set[str]] = {}
for _city, _country in _CITY_TO_COUNTRY.items():
    _COUNTRY_TO_CITIES.setdefault(_country, set()).add(_city)


def infer_country_from_city(city: str) -> Optional[str]:
    """Return the canonical country for a city name, or None if unknown."""
    return _CITY_TO_COUNTRY.get(city.lower().strip())


def cities_for_country(country: str) -> set[str]:
    """Return all known city strings for a country (lowercase)."""
    canonical = _COUNTRY_ALIASES.get(country.lower().strip(), country)
    return _COUNTRY_TO_CITIES.get(canonical, set())


def resolve_country(term: str) -> Optional[str]:
    """
    Given any geographic term (city or country alias), return the canonical
    country string, or None if unrecognised.
    """
    t = term.lower().strip()
    if t in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[t]
    if t in _CITY_TO_COUNTRY:
        return _CITY_TO_COUNTRY[t]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# BM25 corpus — built lazily from ChromaDB, cached as module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_bm25_lock = threading.Lock()
_bm25_index = None
_bm25_corpus: list[dict] = []


def _get_bm25_index():
    """
    Load (or rebuild) the BM25 index from the text_index ChromaDB collection.

    The corpus is fetched once and cached.  Call _reset_bm25_cache() if you
    re-index and need to invalidate.

    Why BM25 alongside dense retrieval?
    Dense sentence-transformers capture semantic similarity well but under-weight
    rare tokens — exact city names ("Shibuya"), dates ("2024-03-15"), and proper
    nouns score low unless they appeared often in training.  BM25 gives exact
    lexical match on those terms.  RRF then merges both ranked lists without
    requiring a tuned scalar weight.
    """
    global _bm25_index, _bm25_corpus

    with _bm25_lock:
        if _bm25_index is not None:
            log.info(f"BM25 cache hit — {len(_bm25_corpus)} cached docs")
            return _bm25_index, _bm25_corpus

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError(
                "rank_bm25 is required for sparse retrieval. "
                "Install with: pip install rank-bm25"
            ) from e

        log.info("Building BM25 index from text collection...")
        collection = common.get_text_collection()
        total = collection.count()
        if total == 0:
            _bm25_corpus = []
            _bm25_index = BM25Okapi([[]])
            return _bm25_index, _bm25_corpus

        log.info(f"text collection count = {total}")

        raw = collection.get(include=["documents", "metadatas"])
        docs    = raw.get("documents", [])
        metas   = raw.get("metadatas", [])
        ids     = raw.get("ids", [])

        _bm25_corpus = [
            {"id": doc_id, "document": doc, "metadata": meta, "distance": 0.0, "score": 0.0}
            for doc_id, doc, meta in zip(ids, docs, metas)
        ]

        tokenised = [doc.lower().split() for doc in docs]
        _bm25_index = BM25Okapi(tokenised)

        log.info(f"BM25 index built — {len(_bm25_corpus)} docs")
        return _bm25_index, _bm25_corpus


def _reset_bm25_cache() -> None:
    """Invalidate the BM25 cache — call after re-indexing."""
    global _bm25_index, _bm25_corpus
    log.info("Resetting BM25 cache...")
    with _bm25_lock:
        _bm25_index = None
        _bm25_corpus = []


# ─────────────────────────────────────────────────────────────────────────────
# BM25 query
# ─────────────────────────────────────────────────────────────────────────────

def query_bm25(
    query: str,
    k: int = 10,
    where: Optional[dict] = None,
) -> list[dict]:
    """
    BM25 keyword search over the text_index corpus.

    `where` is applied as a post-filter on metadata here (unlike ChromaDB
    pre-filters).  BM25 has no native metadata filtering — the corpus is
    small enough that post-filtering is acceptable.
    """
    log.info(f"BM25 query start — query={query!r}, k={k}, where={where}")

    bm25, corpus = _get_bm25_index()
    if not corpus:
        return []

    tokens = query.lower().split()
    raw_scores = bm25.get_scores(tokens)

    scored = []
    for i, score in enumerate(raw_scores):
        if score <= 0:
            continue
        doc = corpus[i]
        if where and not _metadata_matches(doc["metadata"], where):
            continue
        scored.append({**doc, "bm25_score": round(float(score), 4)})

    scored.sort(key=lambda x: x["bm25_score"], reverse=True)
    log.info(f"BM25 matches after filtering = {len(scored)}, returning {len(scored[:k])}")
    return scored[:k]


def _metadata_matches(meta: dict, where: dict) -> bool:
    """
    Simple equality filter matching ChromaDB's `where` dict syntax.
    Supports: plain equality, $in, $eq, $ne, $and.
    """
    if "$and" in where:
        return all(_metadata_matches(meta, sub) for sub in where["$and"])
    for key, condition in where.items():
        val = meta.get(key)
        if isinstance(condition, dict):
            op, operand = next(iter(condition.items()))
            if op == "$in" and val not in operand:
                return False
            elif op == "$eq" and val != operand:
                return False
            elif op == "$ne" and val == operand:
                return False
        else:
            if val != condition:
                return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Fusion helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rrf_fuse(
    result_lists: list[tuple[list[dict], str]],
    rrf_k: int = 60,
    top_k: int = 10,
) -> list[dict]:
    """
    Reciprocal Rank Fusion over an arbitrary number of ranked lists.
    score(d) = Σ_i  1 / (rrf_k + rank_i(d))
    """
    scores:     dict[str, float] = {}
    result_map: dict[str, dict]  = {}
    sources:    dict[str, set]   = {}

    for results, label in result_lists:
        for rank, r in enumerate(results, start=1):
            uid = r["id"]
            scores[uid]  = scores.get(uid, 0.0) + 1.0 / (rrf_k + rank)
            sources.setdefault(uid, set()).add(label)
            if uid not in result_map:
                result_map[uid] = r

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    out = []
    for uid, rrf_score in ranked:
        entry = {
            **result_map[uid],
            "rrf_score": round(rrf_score, 6),
            "source":    "+".join(sorted(sources[uid])),
        }
        out.append(entry)
    return out


def _score_fuse(
    result_lists: list[tuple[list[dict], str, float]],
    top_k: int = 10,
) -> list[dict]:
    """Weighted score fusion (ablation variant)."""
    scores:     dict[str, float] = {}
    result_map: dict[str, dict]  = {}
    sources:    dict[str, set]   = {}

    for results, label, weight in result_lists:
        if not results:
            continue
        max_s = max((r.get("bm25_score", r.get("score", 0.0)) for r in results), default=1.0)
        if max_s == 0:
            max_s = 1.0
        for r in results:
            uid  = r["id"]
            raw  = r.get("bm25_score", r.get("score", 0.0))
            norm = raw / max_s
            scores[uid]  = scores.get(uid, 0.0) + weight * norm
            sources.setdefault(uid, set()).add(label)
            if uid not in result_map:
                result_map[uid] = r

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            **result_map[uid],
            "fused_score": round(s, 4),
            "source":      "+".join(sorted(sources[uid])),
        }
        for uid, s in ranked
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Single-path query functions
# ─────────────────────────────────────────────────────────────────────────────

def query_text_dense(query: str, k: int = 10, where: Optional[dict] = None) -> list[dict]:
    """Dense text retrieval via sentence-transformer cosine similarity."""
    return common.query_text_index(query, k=k, where=where)


def query_text_bm25(query: str, k: int = 10, where: Optional[dict] = None) -> list[dict]:
    """Sparse BM25 keyword retrieval."""
    return query_bm25(query, k=k, where=where)


def query_image_clip(query: str, k: int = 10, where: Optional[dict] = None) -> list[dict]:
    """Cross-modal CLIP retrieval."""
    return common.query_image_index(query, k=k, where=where)


def query_image_caption(query: str, k: int = 10, where: Optional[dict] = None) -> list[dict]:
    """Caption-mediated image retrieval via sentence-transformer on photo captions."""
    caption_filter: dict = {"source_type": "photo_caption"}
    if where:
        caption_filter = {"$and": [{"source_type": {"$eq": "photo_caption"}}, where]}
    return common.query_text_index(query, k=k, where=caption_filter)


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid query functions
# ─────────────────────────────────────────────────────────────────────────────

def text_hybrid_query(
    query: str,
    k: int = 10,
    where: Optional[dict] = None,
    fusion: Literal["rrf", "score"] = "rrf",
    rrf_k: int = 60,
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list[dict]:
    """Hybrid text retrieval: fuse dense + BM25."""
    dense_results = query_text_dense(query, k=k, where=where)
    bm25_results  = query_text_bm25(query,  k=k, where=where)
    log.info(f"Text hybrid — dense={len(dense_results)}, bm25={len(bm25_results)}")

    if fusion == "rrf":
        return _rrf_fuse(
            [(dense_results, "dense"), (bm25_results, "bm25")],
            rrf_k=rrf_k, top_k=k,
        )
    return _score_fuse(
        [(dense_results, "dense", dense_weight), (bm25_results, "bm25", bm25_weight)],
        top_k=k,
    )


def image_hybrid_query(
    query: str,
    k: int = 10,
    where: Optional[dict] = None,
    fusion: Literal["rrf", "score"] = "rrf",
    rrf_k: int = 60,
    clip_weight: float = 0.5,
    caption_weight: float = 0.5,
) -> list[dict]:
    """Hybrid image retrieval: fuse CLIP + caption-mediated paths."""
    clip_results    = query_image_clip(query,    k=k, where=where)
    caption_results = query_image_caption(query, k=k, where=where)
    log.info(f"Image hybrid — clip={len(clip_results)}, caption={len(caption_results)}")

    if fusion == "rrf":
        return _rrf_fuse(
            [(clip_results, "clip"), (caption_results, "caption")],
            rrf_k=rrf_k, top_k=k,
        )
    return _score_fuse(
        [(clip_results, "clip", clip_weight), (caption_results, "caption", caption_weight)],
        top_k=k,
    )


def hybrid_query(
    query: str,
    k: int = 10,
    text_where: Optional[dict] = None,
    image_where: Optional[dict] = None,
    fusion: Literal["rrf", "score"] = "rrf",
    rrf_k: int = 60,
) -> list[dict]:
    """Full multimodal hybrid retrieval: fuse all four paths."""
    dense_results   = query_text_dense(query,    k=k, where=text_where)
    bm25_results    = query_text_bm25(query,     k=k, where=text_where)
    clip_results    = query_image_clip(query,    k=k, where=image_where)
    caption_results = query_image_caption(query, k=k, where=image_where)
    log.info(
        f"Full hybrid — dense={len(dense_results)}, bm25={len(bm25_results)}, "
        f"clip={len(clip_results)}, caption={len(caption_results)}"
    )

    if fusion == "rrf":
        return _rrf_fuse(
            [
                (dense_results,   "dense"),
                (bm25_results,    "bm25"),
                (clip_results,    "clip"),
                (caption_results, "caption"),
            ],
            rrf_k=rrf_k, top_k=k,
        )
    return _score_fuse(
        [
            (dense_results,   "dense",   0.3),
            (bm25_results,    "bm25",    0.2),
            (clip_results,    "clip",    0.3),
            (caption_results, "caption", 0.2),
        ],
        top_k=k,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Broad retrieval — wide fetch + post-filter by geographic terms
# ─────────────────────────────────────────────────────────────────────────────

def _doc_matches_geo(doc: dict, terms: set[str]) -> bool:
    """
    Return True if the document's metadata or text contains any of the
    geographic terms (case-insensitive substring match).

    Checks: destination, city, country, suburb fields + document text.
    """
    meta = doc.get("metadata", {})
    haystack_parts = [
        str(meta.get("destination", "")),
        str(meta.get("city", "")),
        str(meta.get("country", "")),
        str(meta.get("suburb", "")),
        doc.get("document", ""),
    ]
    haystack = " ".join(haystack_parts).lower()
    return any(t.lower() in haystack for t in terms)


def broad_retrieve(
    query: str,
    geo_terms: set[str],
    k: int = 30,
    mode: str = "full",
) -> list[dict]:
    """
    Run retrieval with NO metadata filter at high k, then post-filter
    results to those mentioning any of the geographic terms.

    This catches documents where the destination field is a city but the
    query used the country name, or vice versa.

    geo_terms: set of strings to match (e.g. {"Japan", "Tokyo", "Osaka", ...})
    """
    log.info(f"[broad_retrieve] query={query!r}, geo_terms={geo_terms}, k={k}")
    results = retrieve(query=query, mode=mode, k=k, auto_expand=False)
    filtered = [r for r in results if _doc_matches_geo(r, geo_terms)]
    log.info(f"[broad_retrieve] {len(results)} raw → {len(filtered)} after geo filter")
    return filtered


def build_geo_terms(filter_value: str) -> set[str]:
    """
    Given a city or country string, return the full set of geographic terms
    to use for broad post-filtering.

    E.g. "Japan" → {"Japan", "tokyo", "osaka", "kyoto", ...}
         "Tokyo" → {"Tokyo", "Japan", "tokyo", ...}
    """
    terms: set[str] = {filter_value, filter_value.lower()}

    # If it's a country alias, resolve and add all cities
    country = resolve_country(filter_value)
    if country:
        terms.add(country)
        terms.update(cities_for_country(country))
        return terms

    # If it's a city, also add the country
    inferred = infer_country_from_city(filter_value)
    if inferred:
        terms.add(inferred)
        terms.update(cities_for_country(inferred))

    return terms


# ─────────────────────────────────────────────────────────────────────────────
# Public dispatcher
# ─────────────────────────────────────────────────────────────────────────────

RetrievalMode = Literal[
    "text",
    "image",
    "full",
    "dense_only",
    "bm25_only",
    "clip_only",
    "caption_only",
]

# Minimum number of results before we trigger the broad fallback
_SPARSE_RESULT_THRESHOLD = 5

def _extract_destination_value(filt: dict) -> Optional[str]:
    if "destination" not in filt:
        return None

    val = filt["destination"]

    # Case 1: plain string
    if isinstance(val, str):
        return val

    # Case 2: {"$eq": "Tokyo"}
    if isinstance(val, dict):
        if "$eq" in val:
            return val["$eq"]
        if "$in" in val and isinstance(val["$in"], list) and val["$in"]:
            return val["$in"][0]  # take first for expansion

    return None

def retrieve(
    query: str,
    mode: RetrievalMode = "full",
    k: int = 25,
    text_where: Optional[dict] = None,
    image_where: Optional[dict] = None,
    fusion: Literal["rrf", "score"] = "rrf",
    rrf_k: int = 60,
    auto_expand: bool = True,
) -> list[dict]:
    """
    Single entry point for all retrieval paths.  LangGraph agent tools
    should call this rather than the individual query functions.

    Parameters
    ----------
    query       : natural-language query string
    mode        : retrieval strategy (see RetrievalMode)
    k           : number of results to return (default raised to 25 for broader recall)
    text_where  : ChromaDB metadata pre-filter for text paths
    image_where : ChromaDB metadata pre-filter for image/caption paths
    fusion      : "rrf" (default) or "score" (ablation)
    rrf_k       : RRF constant (default 60)
    auto_expand : if True and filtered results < _SPARSE_RESULT_THRESHOLD,
                  automatically fall back to broad_retrieve() using geo terms
                  inferred from the filter value

    Returns
    -------
    Ranked list of result dicts with id, document, metadata, distance, score,
    source, rrf_score/fused_score.
    """
    log.info(
        f"retrieve() — query={query!r}, mode={mode}, k={k}, "
        f"fusion={fusion}, text_where={text_where}, image_where={image_where}"
    )

    if mode == "text":
        results = text_hybrid_query(query, k=k, where=text_where, fusion=fusion, rrf_k=rrf_k)

    elif mode == "image":
        results = image_hybrid_query(query, k=k, where=image_where, fusion=fusion, rrf_k=rrf_k)

    elif mode == "full":
        results = hybrid_query(
            query, k=k,
            text_where=text_where, image_where=image_where,
            fusion=fusion, rrf_k=rrf_k,
        )

    elif mode == "dense_only":
        results = query_text_dense(query, k=k, where=text_where)

    elif mode == "bm25_only":
        results = query_text_bm25(query, k=k, where=text_where)

    elif mode == "clip_only":
        results = query_image_clip(query, k=k, where=image_where)

    elif mode == "caption_only":
        results = query_image_caption(query, k=k, where=image_where)

    else:
        raise ValueError(f"Unknown retrieval mode '{mode}'.")

    # ── Auto-expand: broad fallback when filtered results are sparse ──────────
    if auto_expand and (text_where or image_where):
        if len(results) < _SPARSE_RESULT_THRESHOLD:
            # Extract the filter value to build geo terms
            filter_val = None
            for filt in (text_where, image_where):
                if filt:
                    # Handle simple {"destination": "X"} and {"$and": [...]} forms
                    # direct case
                    filter_val = _extract_destination_value(filt)
                    if filter_val:
                        break

                    # $and case
                    if "$and" in filt:
                        for sub in filt["$and"]:
                            if isinstance(sub, dict):
                                filter_val = _extract_destination_value(sub)
                                if filter_val:
                                    break
                if filter_val:
                    break

            if filter_val:
                geo_terms = build_geo_terms(filter_val)
                log.info(
                    f"[retrieve] sparse results ({len(results)}) — "
                    f"auto-expanding with geo_terms={geo_terms}"
                )
                broad = broad_retrieve(query, geo_terms=geo_terms, k=k * 2, mode=mode if mode != "image" else "full")
                # Merge: broad results that aren't already in results
                existing_ids = {r["id"] for r in results}
                extras = [r for r in broad if r["id"] not in existing_ids]
                results = results + extras
                log.info(f"[retrieve] after auto-expand: {len(results)} total results")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser(description="Retriever smoke test CLI")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--query", type=str, default="restaurants in Tokyo")
    parser.add_argument("--mode", type=str, default="text",
                        choices=["text","image","full","dense_only","bm25_only","clip_only","caption_only"])
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    if args.smoke_test:
        print(f"Smoke test: query={args.query!r}, mode={args.mode!r}, k={args.k}")
        results = retrieve(args.query, mode=args.mode, k=args.k)
        pprint(results)