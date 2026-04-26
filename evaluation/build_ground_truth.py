# evaluation/build_ground_truth.py
"""
Run once to auto-generate ground_truth.json by querying the index
and selecting the most plausible ground truth document per query.

After running, manual check to verify entries.
"""
import json
from pathlib import Path
from src.retrieval.retriever import retrieve

QUERIES = {
    "Q1": {"query": "transport spending Japan",          "mode": "text",  "source_type": "transaction", "destination": "Japan"},
    "Q2": {"query": "hotel accommodation Tokyo",        "mode": "text",  "source_type": "transaction", "destination": "Tokyo"},
    "Q3": {"query": "Copenhagen trip dates",            "mode": "text",  "source_type": "transaction", "destination": "Copenhagen"},
    "Q4": {"query": "scenery nature outdoor lake sunset mountain",       "mode": "image", "source_type": "photo_caption"},
    "Q5": {"query": "market eating food",          "mode": "full",  "source_type": None},
    "Q6": {"query": "dinner restaurant expensive meal",  "mode": "full",  "source_type": None},
}

# Q7-Q11 are multi-hop or conversational. No single ground truth  doc, skip
NO_GROUND_TRUTH = {
    "Q7":  "multi-hop — judge only",
    "Q8":  "multi-hop — judge only",
    "Q9":  "conversational — judge only",
    "Q10": "conversational — judge only",
    "Q11": "conversational — judge only",
}

def build():
    ground_truth = {}

    for qid, cfg in QUERIES.items():
        query       = cfg["query"]
        mode        = cfg["mode"]
        source_type = cfg.get("source_type")
        destination = cfg.get("destination")

        def _build_where(source_type, destination):
            filters = []
            if source_type:
                filters.append({"source_type": {"$eq": source_type}})
            if destination:
                filters.append({"destination": {"$eq": destination}})
            if len(filters) == 0:
                return None
            if len(filters) == 1:
                return filters[0]
            return {"$and": filters}

        text_where  = _build_where(source_type, destination)
        image_where = _build_where(source_type, destination)

        results = retrieve(
            query,
            mode=mode,
            k=5,
            text_where=text_where  or None,
            image_where=image_where or None,
        )

        if not results:
            ground_truth[qid] = {"ground_truth_ids": [], "candidates": [], "note": "no results found — check query/filters"}
            continue

        # For Q3 (trip duration) we want the earliest AND latest doc
        if qid == "Q3":
            ground_truth_ids = [results[0]["id"], results[-1]["id"]]
        # For Q5/Q6 we want one photo + one transaction
        elif qid in ("Q5", "Q6"):
            photos = [r for r in results if r.get("metadata", {}).get("source_type") == "photo_caption"]
            txns   = [r for r in results if r.get("metadata", {}).get("source_type") == "transaction"]
            ground_truth_ids = []
            if photos: ground_truth_ids.append(photos[0]["id"])
            if txns:   ground_truth_ids.append(txns[0]["id"])
        else:
            ground_truth_ids = [results[0]["id"]]

        # Store candidates so you can verify visually
        ground_truth[qid] = {
            "ground_truth_ids":   ground_truth_ids,
            "candidates": [
                {
                    "id":          r["id"],
                    "source_type": r.get("metadata", {}).get("source_type"),
                    "destination": r.get("metadata", {}).get("destination"),
                    "date":        r.get("metadata", {}).get("date"),
                    "preview":     r.get("document", "")[:120],
                }
                for r in results
            ],
            "note": f"auto-generated from top-{len(results)} results for: {query}",
        }

        print(f"\n{qid} — ground_truth_ids: {ground_truth_ids}")
        for c in ground_truth[qid]["candidates"]:
            marker = "✓" if c["id"] in ground_truth_ids else " "
            print(f"  [{marker}] {c['id']}  {c['source_type']}  {c['destination']}  {c['preview']}")

    for qid, note in NO_GROUND_TRUTH.items():
        ground_truth[qid] = {"ground_truth_ids": [], "note": note}

    out = Path(__file__).parent / "ground_truth.json"
    with open(out, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"\nWrote {out}")

if __name__ == "__main__":
    build()