"""
fetch_q1_ground_truth.py

Fetches all transaction documents from the text_index where:
  - category is "Public Transport"
  - address contains TOUKIYOUTO or AICHI

Run from project root:
    python fetch_q1_ground_truth.py

Outputs a JSON file: q1_transport_docs.json
"""

import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src import common


def fetch_q1_transport_docs() -> list[dict]:
    collection = common.get_text_collection()

    # Fetch all transaction docs — ChromaDB doesn't support OR filters natively
    # so we fetch each address variant separately and merge
    results = []
    seen_ids = set()

    address_filters = [
        {"$and": [
            {"source_type": {"$eq": "transaction"}},
            {"destination": {"$eq": "TOUKIYOUTO"}},
        ]},
        {"$and": [
            {"source_type": {"$eq": "transaction"}},
            {"destination": {"$eq": "Aichi"}},
        ]},
        {"$and": [
            {"source_type": {"$eq": "transaction"}},
            {"destination": {"$eq": "AICHI"}},
        ]},
    ]

    for where_filter in address_filters:
        raw = collection.get(
            where=where_filter,
            include=["documents", "metadatas"],
        )
        docs    = raw.get("documents", [])
        metas   = raw.get("metadatas", [])
        ids     = raw.get("ids", [])

        for doc_id, doc, meta in zip(ids, docs, metas):
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            results.append({
                "id":          doc_id,
                "source_type": meta.get("source_type"),
                "destination": meta.get("destination"),
                "date":        meta.get("date"),
                "category":    meta.get("category"),
                "payee":       meta.get("payee"),
                "amount":      meta.get("amount"),
                "address":     meta.get("address"),
                "document":    doc,
                "metadata":    meta,
            })

    # Sort by date
    results.sort(key=lambda x: x.get("date") or "")

    return results


def main():
    print("Fetching Q1 transport docs from text_index...")
    docs = fetch_q1_transport_docs()

    print(f"\nFound {len(docs)} documents:\n")
    for d in docs:
        print(f"  id:          {d['id']}")
        print(f"  destination: {d['destination']}")
        print(f"  date:        {d['date']}")
        print(f"  payee:       {d['payee']}")
        print(f"  amount:      {d['amount']}")
        print(f"  category:    {d['category']}")
        print(f"  document:    {d['document'][:120]}")
        print()

    out_path = Path("q1_transport_docs.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)
    print(f"Written to {out_path}")

    # Print IDs in ground_truth.json format for easy copy-paste
    print("\nGround truth IDs (copy into ground_truth.json):")
    print(json.dumps([d["id"] for d in docs], indent=2))


if __name__ == "__main__":
    main()