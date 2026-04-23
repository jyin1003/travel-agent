"""
indexer.py

Embeds and upserts into two persistent ChromaDB collections:

    text_index   — sentence-transformers embeddings
                    sources: transaction rows, map rows, photo JSON captions
    image_index  — CLIP image embeddings
                    sources: .jpg photo files

Cross-modal bridge: at query time, encode a text query with CLIP's *text*
encoder and search image_index.  The shared embedding space means
"soup with tofu" can retrieve matching images by cosine similarity.

ChromaDB note: pass metadata filters BEFORE the vector search via the
`where=` argument — do NOT post-filter results afterwards.  This avoids
the metadata pre-filtering trap mentioned in the reference slides.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime

from .chunker import TextChunk, ImageRecord, build_all_chunks

import config
from src import common

# Logging helper

def _log(msg: str) -> None:
    """Print msg with an ISO timestamp prefix."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


# Metadata serialisation helper

def _safe_metadata(d: dict) -> dict:
    """
    ChromaDB metadata values must be str | int | float | bool.
    Drop None values and coerce everything else to str.
    """
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[str(k)] = str(v)
    return out


# Indexing: text_index

def index_text_chunks(chunks: list[TextChunk], reset: bool = False) -> None:
    """
    Embed all TextChunk objects and upsert into text_index.

    Metadata stored per document (enables `where=` pre-filtering):
        source_type, source_file, date, destination, photo_id (if any)
        + any extra fields from the source row
    """
    collection = common.get_text_collection()

    if reset:
        _log("[indexer] Resetting text_index...")
        common.get_chroma_client().delete_collection(config.TEXT_COLLECTION)
        collection = common.get_text_collection()

    _log(f"[indexer] Embedding {len(chunks)} text chunks...")
    t_start = time.perf_counter()

    for i in range(0, len(chunks), config.BATCH_SIZE):
        batch = chunks[i : i + config.BATCH_SIZE]
        texts = [c.text for c in batch]
        ids = [c.doc_id for c in batch]
        metadatas = [
            _safe_metadata({
                "source_type": c.source_type,
                "source_file": c.source_file,
                "date": c.date,
                "destination": c.destination,
                "photo_id": c.photo_id,
                **c.extra,
            })
            for c in batch
        ]
        embeddings = common.embed_texts(texts)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        n_done = min(i + config.BATCH_SIZE, len(chunks))
        elapsed = time.perf_counter() - t_start
        _log(f"[indexer] text_index: {n_done}/{len(chunks)} upserted ({elapsed:.1f}s elapsed)")

    total_elapsed = time.perf_counter() - t_start
    _log(f"[indexer] text_index complete — {collection.count()} documents in {total_elapsed:.1f}s")


# Indexing: image_index

def index_images(records: list[ImageRecord], reset: bool = False) -> None:
    """
    CLIP-embed each photo and upsert into image_index.

    The document string stored is the image path (so callers can retrieve
    the actual file path from query results).
    """
    collection = common.get_image_collection()

    if reset:
        _log("[indexer] Resetting image_index...")
        common.get_chroma_client().delete_collection(config.IMAGE_COLLECTION)
        collection = common.get_image_collection()

    _log(f"[indexer] Embedding {len(records)} images with CLIP...")
    t_start = time.perf_counter()

    for i in range(0, len(records), config.BATCH_SIZE):
        batch = records[i : i + config.BATCH_SIZE]
        ids = [r.photo_id for r in batch]
        paths = [r.image_path for r in batch]
        metadatas = [
            _safe_metadata({
                "source_type": "photo",
                "image_path": r.image_path,
                "date": r.date,
                "destination": r.destination,
                **r.extra,
            })
            for r in batch
        ]
        embeddings = common.embed_images_clip(paths)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=paths,
            metadatas=metadatas,
        )
        n_done = min(i + config.BATCH_SIZE, len(records))
        elapsed = time.perf_counter() - t_start
        _log(f"[indexer] image_index: {n_done}/{len(records)} upserted ({elapsed:.1f}s elapsed)")

    total_elapsed = time.perf_counter() - t_start
    _log(f"[indexer] image_index complete — {collection.count()} documents in {total_elapsed:.1f}s")


# Entrypoint — build and persist both indices

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python indexer.py <transactions.csv> <maps_dir/> <photos_dir/> <metadata_dir/>")
        sys.exit(1)

    t_total = time.perf_counter()
    _log("[indexer] Starting index build...")

    text_chunks, image_records = build_all_chunks(
        transactions_csv=sys.argv[1],
        maps_dir=sys.argv[2],
        photos_dir=sys.argv[3],
        metadata_dir=sys.argv[4],
    )

    _log(f"[indexer] Chunking complete — {len(text_chunks)} text chunks, {len(image_records)} image records")

    index_text_chunks(text_chunks, reset=True)
    index_images(image_records, reset=True)

    total_elapsed = time.perf_counter() - t_total
    _log(f"[indexer] Index build complete in {total_elapsed:.1f}s")
    _log(f"[indexer]   text_index : {common.get_text_collection().count()} vectors")
    _log(f"[indexer]   image_index: {common.get_image_collection().count()} vectors")

    # Quick sanity check
    _log("[indexer] Sanity query: 'beach sunset'")
    for r in common.hybrid_query("beach sunset", k=3):
        _log(f"[indexer]   [{r.get('source')}] {r['id']} score={r.get('rrf_score', r.get('score'))}")