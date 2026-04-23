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

Trade-off notes
---------------
- BM25 adds ~40-80 ms on first call (corpus build from ChromaDB).  Corpus
  is cached as a module-level singleton so subsequent calls are fast.
- Caption-mediated path gives text-model recall on images where CLIP's
  shared embedding space is weak (long descriptive captions vs. short
  visual queries).
- RRF avoids the need to tune a scalar weighting parameter between modalities;
  use fusion="score" + text_weight/image_weight for the ablation variant.
- Metadata pre-filters reduce the candidate pool before vector search,
  improving precision without sacrificing recall on the filtered subset.
  Do NOT post-filter results — that silently reduces k.
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

# BM25 corpus — built lazily from ChromaDB, cached as module-level singleton

_bm25_lock = threading.Lock()
_bm25_index = None        # rank_bm25.BM25Okapi instance
_bm25_corpus: list[dict] = []   # parallel list of ChromaDB result dicts


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
        
        if total > 0:
            log.info(f"text collection count = {total}")
        else:
            log.info("text collection empty — returning empty BM25 index")

        # Fetch entire text corpus from ChromaDB (no embedding needed)
        raw = collection.get(include=["documents", "metadatas"])
        docs    = raw.get("documents", [])
        metas   = raw.get("metadatas", [])
        ids     = raw.get("ids", [])

        _bm25_corpus = [
            {"id": doc_id, "document": doc, "metadata": meta, "distance": 0.0, "score": 0.0}
            for doc_id, doc, meta in zip(ids, docs, metas)
        ]

        # Tokenise: lowercase split — simple but effective for travel notes
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


# ---------------------------------------------------------------------------
# BM25 query
# ---------------------------------------------------------------------------

def query_bm25(
    query: str,
    k: int = 5,
    where: Optional[dict] = None,
) -> list[dict]:
    """
    BM25 keyword search over the text_index corpus.

    `where` is applied as a post-filter on metadata here (unlike ChromaDB
    pre-filters).  BM25 has no native metadata filtering — the corpus is
    small enough that post-filtering is acceptable.  If the filtered result
    set is smaller than k, all matching documents are returned.

    Returns results in the same dict schema as common._format_results():
        {id, document, metadata, distance, score, bm25_score}
    """
    log.info(f"BM25 query start — query={query!r}, k={k}, where={where}")
    
    bm25, corpus = _get_bm25_index()

    if not corpus:
        log.info("BM25 query skipped — corpus is empty")
        return []

    tokens = query.lower().split()
    raw_scores = bm25.get_scores(tokens)     # float array, length == len(corpus)

    # Pair each doc with its BM25 score and apply optional metadata filter
    scored = []
    for i, score in enumerate(raw_scores):
        if score <= 0:
            continue
        doc = corpus[i]
        if where and not _metadata_matches(doc["metadata"], where):
            continue
        scored.append({**doc, "bm25_score": round(float(score), 4)})

    # Sort descending, take top k
    scored.sort(key=lambda x: x["bm25_score"], reverse=True)
    
    log.info(f"BM25 matches after filtering = {len(scored)}")
    log.info(f"BM25 query complete — returned {len(scored[:k])} results")
    
    return scored[:k]


def _metadata_matches(meta: dict, where: dict) -> bool:
    """
    Simple equality filter matching ChromaDB's `where` dict syntax.
    Supports plain equality: {"source_type": "map"} and
    $in operator:           {"source_type": {"$in": ["map", "transaction"]}}
    """
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


# Fusion helpers

def _rrf_fuse(
    result_lists: list[tuple[list[dict], str]],   # [(results, source_label), ...]
    rrf_k: int = 60,
    top_k: int = 5,
) -> list[dict]:
    """
    Reciprocal Rank Fusion over an arbitrary number of ranked lists.

    score(d) = Σ_i  1 / (rrf_k + rank_i(d))

    `source_label` for each list is used to populate the 'source' field.
    If a document appears in multiple lists its source becomes "text+image"
    or the labels joined with "+".

    RRF is chosen over scalar weighting because it requires no tuning and
    is robust to different score scales across modalities.
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
    result_lists: list[tuple[list[dict], str, float]],  # [(results, label, weight), ...]
    top_k: int = 5,
) -> list[dict]:
    """
    Weighted score fusion.  Each list entry is (results, source_label, weight).
    Use this as the ablation variant to compare against RRF.

    score(d) = Σ_i  weight_i * cosine_similarity_i(d)

    Scores must be in [0, 1]; the 'score' field from _format_results() qualifies.
    BM25 scores are normalised to [0, 1] by dividing by the list maximum.
    """
    scores:     dict[str, float] = {}
    result_map: dict[str, dict]  = {}
    sources:    dict[str, set]   = {}

    for results, label, weight in result_lists:
        if not results:
            continue
        # Normalise BM25 scores to [0,1] so they're comparable with cosine sims
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


# Single-path query functions

def query_text_dense(
    query: str,
    k: int = 5,
    where: Optional[dict] = None,
) -> list[dict]:
    """
    Dense text retrieval via sentence-transformer cosine similarity.
    Pre-filters with `where` BEFORE the vector search (ChromaDB native).
    """
    return common.query_text_index(query, k=k, where=where)


def query_text_bm25(
    query: str,
    k: int = 5,
    where: Optional[dict] = None,
) -> list[dict]:
    """
    Sparse BM25 keyword retrieval.  `where` is applied as a post-filter.
    Good for exact entity matches: city names, dates, currency amounts.
    """
    return query_bm25(query, k=k, where=where)


def query_image_clip(
    query: str,
    k: int = 5,
    where: Optional[dict] = None,
) -> list[dict]:
    """
    Cross-modal CLIP retrieval: encodes the text query with CLIP's text
    encoder to retrieve visually matching images.
    Best for short visual descriptions: "beach at sunset", "crowded market".
    """
    return common.query_image_index(query, k=k, where=where)


def query_image_caption(
    query: str,
    k: int = 5,
    where: Optional[dict] = None,
) -> list[dict]:
    """
    Caption-mediated image retrieval: searches text_index filtered to
    source_type="photo".  Uses sentence-transformer on the caption text,
    not the CLIP embedding space.

    Trade-off vs CLIP path:
    - CLIP is better for short visual queries matching learned image features
    - Caption path is better for long descriptive queries and abstract concepts
        where CLIP's shared space is weaker ("a photo from my trip when I felt
        lost") — the rich caption text carries more semantic signal.
    """
    caption_filter: dict = {"source_type": "photo"}
    if where:
        # Merge caller's filter with source_type constraint
        caption_filter = {"$and": [{"source_type": {"$eq": "photo"}}, where]}
    return common.query_text_index(query, k=k, where=caption_filter)


# Hybrid query functions (fuse multiple paths)

def text_hybrid_query(
    query: str,
    k: int = 5,
    where: Optional[dict] = None,
    fusion: Literal["rrf", "score"] = "rrf",
    rrf_k: int = 60,
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list[dict]:
    """
    Hybrid text retrieval: fuse dense + BM25.

    Why this combination?
    - Dense captures semantic paraphrases ("eatery" ≈ "restaurant")
    - BM25 catches exact entity tokens missed by dense ("Fushimi Inari")
    - RRF default avoids tuning dense_weight/bm25_weight

    Use fusion="score" with explicit weights for ablation experiments.
    """  
    dense_results = query_text_dense(query, k=k, where=where)
    bm25_results  = query_text_bm25(query,  k=k, where=where)
    
    log.info(
        f"Text hybrid components — "
        f"dense={len(dense_results)}, bm25={len(bm25_results)}"
    )

    if fusion == "rrf":
        return _rrf_fuse(
            [(dense_results, "dense"), (bm25_results, "bm25")],
            rrf_k=rrf_k,
            top_k=k,
        )
    return _score_fuse(
        [(dense_results, "dense", dense_weight), (bm25_results, "bm25", bm25_weight)],
        top_k=k,
    )


def image_hybrid_query(
    query: str,
    k: int = 5,
    where: Optional[dict] = None,
    fusion: Literal["rrf", "score"] = "rrf",
    rrf_k: int = 60,
    clip_weight: float = 0.5,
    caption_weight: float = 0.5,
) -> list[dict]:
    """
    Hybrid image retrieval: fuse CLIP + caption-mediated paths.

    Why two paths?
    - CLIP excels at short visual queries aligned with its training distribution
    - Caption path excels at long/abstract descriptions and exact nouns
    - Fusing both provides better recall across query types

    Returns image results only (source_type="photo" entries from both paths).
    """
    clip_results    = query_image_clip(query,    k=k, where=where)
    caption_results = query_image_caption(query, k=k, where=where)
    
    log.info(
        f"Image hybrid components — "
        f"clip={len(clip_results)}, caption={len(caption_results)}"
    )

    if fusion == "rrf":
        return _rrf_fuse(
            [(clip_results, "clip"), (caption_results, "caption")],
            rrf_k=rrf_k,
            top_k=k,
        )
    return _score_fuse(
        [(clip_results, "clip", clip_weight), (caption_results, "caption", caption_weight)],
        top_k=k,
    )


def hybrid_query(
    query: str,
    k: int = 5,
    text_where: Optional[dict] = None,
    image_where: Optional[dict] = None,
    fusion: Literal["rrf", "score"] = "rrf",
    rrf_k: int = 60,
) -> list[dict]:
    """
    Full multimodal hybrid retrieval: fuse all four paths.

        dense  +  BM25  +  CLIP  +  caption

    Use this for queries where you don't know whether the answer is in
    text or images (the agent's default for ambiguous queries).

    Results carry a 'source' field showing which paths contributed:
        "bm25", "caption", "clip", "dense", or combinations joined with "+"
    """
    dense_results   = query_text_dense(query,    k=k, where=text_where)
    bm25_results    = query_text_bm25(query,     k=k, where=text_where)
    clip_results    = query_image_clip(query,    k=k, where=image_where)
    caption_results = query_image_caption(query, k=k, where=image_where)
    
    log.info(
        f"Full hybrid components — "
        f"dense={len(dense_results)}, bm25={len(bm25_results)}, "
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
            rrf_k=rrf_k,
            top_k=k,
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


# Public dispatcher — single entry point for LangGraph agent tools

RetrievalMode = Literal[
    "text",          # dense + BM25 hybrid (default for text queries)
    "image",         # CLIP + caption hybrid (for visual queries)
    "full",          # all four paths fused (for ambiguous / multi-modal queries)
    "dense_only",    # ablation: dense text only, no BM25
    "bm25_only",     # ablation: BM25 only
    "clip_only",     # ablation: CLIP only, no caption
    "caption_only",  # ablation: caption-mediated only
]


def retrieve(
    query: str,
    mode: RetrievalMode = "full",
    k: int = 5,
    text_where: Optional[dict] = None,
    image_where: Optional[dict] = None,
    fusion: Literal["rrf", "score"] = "rrf",
    rrf_k: int = 60,
) -> list[dict]:
    """
    Single entry point for all retrieval paths.  LangGraph agent tools
    should call this rather than the individual query functions.

    Parameters
    ----------
    query       : natural-language query string
    mode        : retrieval strategy (see RetrievalMode)
    k           : number of results to return
    text_where  : ChromaDB metadata pre-filter for text paths
                  e.g. {"destination": "Japan"} or {"source_type": "map"}
    image_where : ChromaDB metadata pre-filter for image/caption paths
    fusion      : "rrf" (default, no tuning needed) or "score" (ablation)
    rrf_k       : RRF constant; higher = less rank-sensitive (default 60)

    Returns
    -------
    Ranked list of result dicts.  Each dict contains:
        id, document, metadata, distance, score, source,
        rrf_score (if fusion="rrf") or fused_score (if fusion="score")

    Examples
    --------
    # Factual text query with destination filter
    retrieve("What did I spend at restaurants in Tokyo?",
                mode="text", text_where={"destination": "Tokyo"})

    # Cross-modal: find photos of a specific visual scene
    retrieve("street market with colourful stalls",
                mode="image")

    # Multi-hop / ambiguous: search everything
    retrieve("What can I cook with what I have from my Kyoto trip?",
                mode="full")

    # Ablation: dense-only baseline (no BM25)
    retrieve("Fushimi Inari hike", mode="dense_only")
    """
    log.info(
        f"Retrieve called — query={query!r}, mode={mode}, k={k}, "
        f"fusion={fusion}, text_where={text_where}, image_where={image_where}"
    )
    
    if mode == "text":
        return text_hybrid_query(
            query, k=k, where=text_where, fusion=fusion, rrf_k=rrf_k
        )

    elif mode == "image":
        return image_hybrid_query(
            query, k=k, where=image_where, fusion=fusion, rrf_k=rrf_k
        )

    elif mode == "full":
        return hybrid_query(
            query, k=k,
            text_where=text_where, image_where=image_where,
            fusion=fusion, rrf_k=rrf_k,
        )

    # --- Ablation modes (single-path, for evaluation) ---

    elif mode == "dense_only":
        return query_text_dense(query, k=k, where=text_where)

    elif mode == "bm25_only":
        return query_text_bm25(query, k=k, where=text_where)

    elif mode == "clip_only":
        return query_image_clip(query, k=k, where=image_where)

    elif mode == "caption_only":
        return query_image_caption(query, k=k, where=image_where)

    else:
        raise ValueError(
            f"Unknown retrieval mode '{mode}'. "
            f"Valid modes: {list(RetrievalMode.__args__)}"
        )

# Smoke test
if __name__ == "__main__":
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser(description="Retriever smoke test CLI")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a simple retrieval smoke test",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="restaurants in Tokyo",
        help="Query string for smoke test",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="text",
        choices=[
            "text",
            "image",
            "full",
            "dense_only",
            "bm25_only",
            "clip_only",
            "caption_only",
        ],
        help="Retrieval mode for smoke test",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Number of results to return",
    )

    args = parser.parse_args()

    if args.smoke_test:
        print(f"Running smoke test with query={args.query!r}, mode={args.mode!r}, k={args.k}")
        results = retrieve(args.query, mode=args.mode, k=args.k)
        pprint(results)