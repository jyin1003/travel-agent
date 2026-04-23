"""
common.py

Shared ChromaDB clients, model loaders, embedding helpers, and query
functions used by both ingestion/indexer.py and retrieval/retriever.py.

All heavy imports (torch, transformers, sentence_transformers) are deferred
to the first time they are actually needed — importing this module is instant.
"""

from __future__ import annotations

import os, logging
from typing import Optional

import chromadb
from chromadb.config import Settings

import config

# Logging helper
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [COMMON] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def _ts(label: str) -> None:
    """Emit a timestamped step marker."""
    log.info(f"{label}")


# Lazy singletons

_chroma_client: Optional[chromadb.PersistentClient] = None
_sentence_model = None
_clip_model = None
_clip_processor = None


# ChromaDB clients + collections

def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(config.CHROMA_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=config.CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_text_collection() -> chromadb.Collection:
    """cosine distance matches sentence-transformer recommendations."""
    return get_chroma_client().get_or_create_collection(
        name=config.TEXT_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def get_image_collection() -> chromadb.Collection:
    return get_chroma_client().get_or_create_collection(
        name=config.IMAGE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


# Model loaders (imports)

def get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        log.info(f"Loading sentence-transformers: {config.SENTENCE_MODEL}", flush=True)
        _sentence_model = SentenceTransformer(config.SENTENCE_MODEL)
        log.info("sentence-transformers ready.", flush=True)
    return _sentence_model


def get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        log.info(f"Loading CLIP: {config.CLIP_MODEL}", flush=True)
        _clip_model = CLIPModel.from_pretrained(config.CLIP_MODEL)
        _clip_processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
        _clip_model.eval()
        log.info("CLIP ready.", flush=True)
    return _clip_model, _clip_processor


# Embedding helpers

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Sentence-transformer embeddings — used for text_index."""
    model = get_sentence_model()
    embeddings = model.encode(
        texts,
        batch_size=config.BATCH_SIZE,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def embed_images_clip(image_paths: list[str]) -> list[list[float]]:
    """CLIP image embeddings — used for image_index."""
    import torch
    from PIL import Image

    model, processor = get_clip()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(image_paths), config.BATCH_SIZE):
        batch_paths = image_paths[i : i + config.BATCH_SIZE]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=pixel_values)

            # Handle case where output is a ModelOutput object, not a raw tensor
            if not isinstance(image_features, torch.Tensor):
                image_features = image_features.pooler_output

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        all_embeddings.extend(image_features.cpu().numpy().tolist())

    return all_embeddings


def embed_text_clip(texts: list[str]) -> list[list[float]]:
    """
    CLIP text embeddings — used at query time to search image_index.
    Encodes a natural-language query into the same space as CLIP image
    embeddings, enabling cross-modal retrieval (text → image search).
    """
    import torch

    model, processor = get_clip()
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

        # Same defensive unwrap
        if not isinstance(text_features, torch.Tensor):
            text_features = text_features.pooler_output

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().tolist()


# Query helpers (called by retrieval/retriever.py and agent tools)

def _format_results(raw: dict) -> list[dict]:
    """Flatten chromadb query() output into a list of dicts."""
    out = []
    docs  = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas",  [[]])[0]
    dists = raw.get("distances",  [[]])[0]
    ids   = raw.get("ids",        [[]])[0]
    for doc, meta, dist, doc_id in zip(docs, metas, dists, ids):
        out.append({
            "id":       doc_id,
            "document": doc,
            "metadata": meta,
            "distance": dist,
            "score":    round(1 - dist, 4),   # cosine distance → similarity
        })
    return out


def query_text_index(
    query: str,
    k: int = 5,
    where: Optional[dict] = None,
) -> list[dict]:
    """
    Search text_index with a sentence-transformer embedding.
    Pass `where` to pre-filter by metadata BEFORE the vector search.

    Example:
        query_text_index("restaurant in Kyoto", k=5,
                         where={"source_type": "map"})
    """
    collection = get_text_collection()
    embedding = embed_texts([query])[0]
    kwargs: dict = {
        "query_embeddings": [embedding],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where     # pre-filter, not post-filter
    return _format_results(collection.query(**kwargs))


def query_image_index(
    query: str,
    k: int = 5,
    where: Optional[dict] = None,
) -> list[dict]:
    """
    Search image_index using CLIP's text encoder.
    The query is encoded into the same space as CLIP image embeddings,
    so natural-language descriptions retrieve visually matching photos.

    Example:
        query_image_index("people on a boat on a river", k=3)
    """
    collection = get_image_collection()
    embedding = embed_text_clip([query])[0]
    kwargs: dict = {
        "query_embeddings": [embedding],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    return _format_results(collection.query(**kwargs))


def hybrid_query(
    query: str,
    k: int = 5,
    text_where: Optional[dict] = None,
    image_where: Optional[dict] = None,
    fusion: str = "rrf",
    rrf_k: int = 60,
) -> list[dict]:
    """
    Run text and image searches in parallel, then fuse results.

    fusion="rrf"   — Reciprocal Rank Fusion: score = Σ 1 / (rrf_k + rank)
    fusion="score" — weighted score merge (equal weights by default)

    Returns a unified ranked list with a 'source' field: "text", "image", or "both".
    """
    text_results  = query_text_index(query,  k=k, where=text_where)
    image_results = query_image_index(query, k=k, where=image_where)

    if fusion == "rrf":
        return _rrf_fuse(text_results, image_results, rrf_k=rrf_k, top_k=k)
    return _score_fuse(text_results, image_results, top_k=k)


# Fusion helpers

def _rrf_fuse(
    text_results: list[dict],
    image_results: list[dict],
    rrf_k: int = 60,
    top_k: int = 5,
) -> list[dict]:
    scores:     dict[str, float] = {}
    result_map: dict[str, dict]  = {}

    for rank, r in enumerate(text_results, start=1):
        uid = r["id"]
        scores[uid] = scores.get(uid, 0) + 1 / (rrf_k + rank)
        result_map[uid] = {**r, "source": "text"}

    for rank, r in enumerate(image_results, start=1):
        uid = r["id"]
        scores[uid] = scores.get(uid, 0) + 1 / (rrf_k + rank)
        if uid not in result_map:
            result_map[uid] = {**r, "source": "image"}
        else:
            result_map[uid]["source"] = "both"

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{**result_map[uid], "rrf_score": round(s, 6)} for uid, s in ranked]


def _score_fuse(
    text_results: list[dict],
    image_results: list[dict],
    text_weight: float = 0.5,
    image_weight: float = 0.5,
    top_k: int = 5,
) -> list[dict]:
    scores:     dict[str, float] = {}
    result_map: dict[str, dict]  = {}

    for r in text_results:
        uid = r["id"]
        scores[uid] = scores.get(uid, 0) + text_weight * r["score"]
        result_map[uid] = {**r, "source": "text"}

    for r in image_results:
        uid = r["id"]
        scores[uid] = scores.get(uid, 0) + image_weight * r["score"]
        if uid not in result_map:
            result_map[uid] = {**r, "source": "image"}
        else:
            result_map[uid]["source"] = "both"

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{**result_map[uid], "fused_score": round(s, 4)} for uid, s in ranked]