"""
chunker.py
----------
Produces a uniform list of TextChunk objects from all data sources:
    - transactions CSV (row-level)
    - maps CSVs (row-level)
    - photos + JSON sidecars (one text chunk per photo JSON)

Also produces an ImageRecord list for photos (path + shared metadata).

Output is consumed by indexer.py.
"""

from __future__ import annotations

import json
import os
import re
import time
import json
import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from groq import Groq
from dotenv import load_dotenv
load_dotenv()

# Data classes

@dataclass
class TextChunk:
    doc_id: str            # uuid for photos; generated for other sources
    text: str
    source_type: str       # "itinerary" | "transaction" | "map" | "photo_caption"
    source_file: str       # relative path of origin file
    date: Optional[str]    # ISO date string where available
    destination: Optional[str]
    # photo-specific cross-modal link
    photo_id: Optional[str] = None
    # extra key-value metadata (kept flat for ChromaDB compatibility)
    extra: dict = field(default_factory=dict)


@dataclass
class ImageRecord:
    photo_id: str          # uuid from filename stem
    image_path: str        # absolute or relative path to .jpeg / .jpg
    date: Optional[str]
    destination: Optional[str]
    extra: dict = field(default_factory=dict)


# Helpers

def _make_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index:06d}"

# Destination inference - Google Geocoding API + disk cache

_GEO_CACHE_PATH = Path(__file__).parent / "geocode_cache.json"


def _load_geo_cache() -> dict:
    if _GEO_CACHE_PATH.exists():
        with open(_GEO_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}
 
 
def _save_geo_cache(cache: dict) -> None:
    with open(_GEO_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
 
 
_geo_cache: dict = _load_geo_cache()
 
 
def _geocode(text: str, api_key: str) -> Optional[str]:
    """
    Call the Google Geocoding API for a free-text string.
    Extracts the most specific locality from address_components:
      locality > postal_town > administrative_area_level_2 > administrative_area_level_1 > country
    Returns the long_name of the best match, or None on failure/no result.
    """
    import urllib.request
    import urllib.parse
 
    key = text.strip().lower()
    if key in _geo_cache:
        return _geo_cache[key]
 
    result = None
    try:
        params = urllib.parse.urlencode({"address": text, "key": api_key})
        url = f"https://maps.googleapis.com/maps/api/geocode/json?{params}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode())
 
        if data.get("status") == "OK" and data.get("results"):
            components = data["results"][0].get("address_components", [])
            # Priority order for what counts as "destination"
            priority = [
                "locality",
                "postal_town",
                "administrative_area_level_2",
                "administrative_area_level_1",
                "country",
            ]
            found: dict[str, str] = {}
            for comp in components:
                for ptype in priority:
                    if ptype in comp.get("types", []):
                        found[ptype] = comp["long_name"]
            for ptype in priority:
                if ptype in found:
                    result = found[ptype]
                    break
 
    except Exception:
        result = None
 
    _geo_cache[key] = result
    _save_geo_cache(_geo_cache)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Transaction enrichment (run at index time, results cached to disk)
# ─────────────────────────────────────────────────────────────────────────────

_TXN_ENRICH_CACHE_PATH = Path(__file__).parent / "txn_enrich_cache.json"


def _load_enrich_cache() -> dict:
    if _TXN_ENRICH_CACHE_PATH.exists():
        with open(_TXN_ENRICH_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_enrich_cache(cache: dict) -> None:
    with open(_TXN_ENRICH_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _rule_infer_category(payee: str, description: str, address: str = "") -> str:
    """
    Fast rule-based category inference from payee/description/address.
    Mirrors the logic in nodes.py but applied at index time so the category
    is embedded into the text chunk rather than inferred at query time.
    """
    _CATEGORY_RULES: list[tuple[str, list[str]]] = [
        ("accommodation", [
            "hotel", "hostel", "airbnb", "booking.com", "hotels.com", "inn",
            "lodge", "resort", "motel", "guesthouse", "bnb", "b&b", "suites",
        ]),
        ("flights", [
            "ryanair", "easyjet", "british airways", "lufthansa", "emirates",
            "cathay", "qantas", "delta", "united", "southwest", "air france",
            "klm", "wizz", "jet2", "airport", "airways",
        ]),
        ("transport", [
            "uber", "lyft", "bolt", "grab", "taxi", "cab", "tfl", "oyster",
            "trainline", "national rail", "eurostar", "bus", "metro", "tube",
            "rail", "ferry", "tram", "mtr",
        ]),
        ("food & drink", [
            "restaurant", "cafe", "coffee", "starbucks", "costa", "pret",
            "mcdonald", "kfc", "subway", "nandos", "wagamama", "itsu",
            "food", "dining", "bistro", "pub", "bar", "bakery", "sushi",
            "pizza", "burger", "ramen", "noodle", "curry", "dim sum",
            "deliveroo", "uber eats", "just eat",
        ]),
        ("groceries", [
            "supermarket", "tesco", "sainsbury", "waitrose", "m&s", "asda",
            "morrisons", "lidl", "aldi", "co-op", "whole foods", "carrefour",
        ]),
        ("activities & attractions", [
            "museum", "gallery", "tour", "tickets", "admission", "zoo",
            "aquarium", "theme park", "cinema", "theatre", "concert",
        ]),
        ("shopping", [
            "amazon", "ebay", "asos", "zara", "h&m", "primark", "pharmacy",
            "boots", "duty free", "souvenir", "gift shop", "mall",
        ]),
        ("cash withdrawal", ["atm", "cash", "withdrawal", "cashpoint"]),
    ]
    haystack = " ".join([payee, description, address]).lower()
    for category, keywords in _CATEGORY_RULES:
        if any(kw in haystack for kw in keywords):
            return category
    return ""

def _apply_cache(chunks: list[TextChunk], cache: dict) -> None:
    """Apply cached enrichment results to chunk text and metadata."""
    import re
    for chunk in chunks:
        txn_id = chunk.extra.get("transaction_id") or chunk.doc_id
        result = cache.get(txn_id)
        if not result:
            continue
        city     = result.get("inferred_city")
        activity = result.get("inferred_activity")
        if city and not chunk.destination:
            chunk.destination = city
            chunk.extra["inferred_city"] = city
        if activity:
            chunk.extra["inferred_activity"] = activity
            # Don't double-append if already present
            if "[activity:" not in chunk.text:
                chunk.text += f" [activity: {activity}]"
        if city and f"[city: {city}]" not in chunk.text:
            chunk.text += f" [city: {city}]"

def enrich_transactions_llm(
    chunks: list[TextChunk],
    batch_size: int = 30,
) -> list[TextChunk]:

    if not chunks:
        return chunks

    # ── Pass 1: rule-based ─────────────────────────────────────────────
    for chunk in chunks:
        category = _rule_infer_category(
            chunk.extra.get("payee", ""),
            chunk.extra.get("description", ""),
            chunk.extra.get("address", ""),
        )
        if category:
            chunk.extra["inferred_category"] = category
            chunk.text += f" [category: {category}]"

    # ── API key check ──────────────────────────────────────────────────
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[chunker] No GROQ_API_KEY — skipping LLM enrichment")
        return chunks

    # ── Model fallback chain ───────────────────────────────────────────
    MODEL_CHAIN = [
        "llama-3.3-70b-versatile",
        "qwen/qwen3-32b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
    ]

    model_index = 0

    cache = _load_enrich_cache()

    # ── Filter uncached transactions ───────────────────────────────────
    to_enrich = []
    for chunk in chunks:
        txn_id = chunk.extra.get("transaction_id") or chunk.doc_id
        if txn_id not in cache:
            to_enrich.append(chunk)

    if not to_enrich:
        print(f"[chunker] All {len(chunks)} transactions cached")
        _apply_cache(chunks, cache)
        return chunks

    print(f"[chunker] {len(to_enrich)} transactions to enrich")

    client = Groq(api_key=api_key)

    # ── Rate limiting ──────────────────────────────────────────────────
    REQUESTS_PER_MIN = 30
    SAFETY_MARGIN = 0.8
    sleep_per_request = 60 / (REQUESTS_PER_MIN * SAFETY_MARGIN)  # ~2.5s

    _ENRICH_SYSTEM = """You are enriching bank transaction records.

Return ONLY valid JSON (no markdown).

Each item:
{
"id": "...",
"inferred_city": "... or null",
"inferred_activity": "3-6 word activity or null"
}

Rules:
- Use only evidence from payee, description, address
- If global brand → city = null
- Keep activity concise (3-6 words)
- If unsure → null

Return a JSON array in the same order.
"""

    enriched_count = 0

    # ── Batch loop ─────────────────────────────────────────────────────
    for i in range(0, len(to_enrich), batch_size):
        batch = to_enrich[i : i + batch_size]

        payload = [
            {
                "id": chunk.extra.get("transaction_id") or chunk.doc_id,
                "payee": chunk.extra.get("payee", ""),
                "description": chunk.extra.get("description", ""),
                "amount": chunk.extra.get("amount", ""),
                "address": chunk.extra.get("address", ""),
                "date": chunk.date or "",
            }
            for chunk in batch
        ]

        success = False
        rpm_retries = 0

        # ── Model fallback loop ────────────────────────────────────────
        while model_index < len(MODEL_CHAIN):
            current_model = MODEL_CHAIN[model_index]

            try:
                response = client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": _ENRICH_SYSTEM},
                        {"role": "user", "content": json.dumps(payload)},
                    ],
                    temperature=0,
                )

                raw = response.choices[0].message.content
                raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```")

                # ── JSON parsing with recovery ───────────────────────
                try:
                    items = json.loads(raw)
                except:
                    match = re.search(r"\[.*\]", raw, re.DOTALL)
                    if match:
                        items = json.loads(match.group(0))
                    else:
                        raise ValueError("JSON parse failed")

                for item in items:
                    txn_id = item.get("id")
                    if txn_id:
                        cache[txn_id] = {
                            "inferred_city": item.get("inferred_city"),
                            "inferred_activity": item.get("inferred_activity"),
                        }
                        enriched_count += 1

                success = True
                break

            except Exception as e:
                error_str = str(e)

                # 🔴 Token limit → switch model permanently
                if "rate_limit_exceeded" in error_str and "tokens" in error_str:
                    print(f"[chunker] {current_model} TPD exhausted → switching model")
                    model_index += 1
                    rpm_retries = 0
                    continue

                # 🟡 RPM limit → retry SAME model
                elif "429" in error_str:
                    rpm_retries += 1
                    if rpm_retries > 3:
                        print(f"[chunker] {current_model} stuck on RPM → switching model")
                        model_index += 1
                        rpm_retries = 0
                        continue

                    print(f"[chunker] {current_model} RPM hit → sleeping 15s")
                    time.sleep(15)
                    continue

                # 🔵 Other error → skip model
                else:
                    print(f"[chunker] {current_model} failed → switching model ({e})")
                    model_index += 1
                    rpm_retries = 0
                    continue

        if not success:
            print("[chunker] All models exhausted — stopping early")
            break

        print(f"[chunker] {min(i + batch_size, len(to_enrich))}/{len(to_enrich)} processed")

        # ── Rate limiting between batches ──────────────────────────────
        if i + batch_size < len(to_enrich):
            time.sleep(sleep_per_request)

    # ── Save + apply cache ─────────────────────────────────────────────
    _save_enrich_cache(cache)
    _apply_cache(chunks, cache)

    print(f"[chunker] Done — {enriched_count} enriched")
    return chunks


def _infer_destination(text: str, api_key: Optional[str] = None) -> Optional[str]:
    """
    Infer a destination city/country from an unstructured string (e.g. a
    transaction payee or address) using the Google Geocoding API.
 
    Pass api_key explicitly, or set the GOOGLE_MAPS_API_KEY environment
    variable. If neither is available, returns None silently.
 
    Results are cached to geocode_cache.json — each unique string is only
    looked up once; subsequent runs are instant with no API calls.
 
    For sources that already have a structured city field (photo JSON,
    maps CSV), callers should use that directly and not call this function.
    """
    if not text:
        return None
 
    if api_key is None:
        api_key = os.getenv("GOOGLE_API_KEY")
 
    if not api_key:
        return None   # degrade gracefully rather than crash
 
    return _geocode(text, api_key)


# Per-source chunkers
def chunk_transactions_csv(csv_path: str) -> list[TextChunk]:
    """
    Each row becomes one text chunk.
 
    Columns: transaction_id, datetime, type, amount, payee,
             description, category, address, source_bank
    """
    import csv
    chunks: list[TextChunk] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            transaction_id = row.get("transaction_id", "").strip()
            datetime_val   = row.get("datetime", "").strip()
            txn_type       = row.get("type", "").strip()
            amount         = row.get("amount", "").strip()
            payee          = row.get("payee", "").strip()
            description    = row.get("description", "").strip()
            category       = row.get("category", "").strip()
            address        = row.get("address", "").strip()
            source_bank    = row.get("source_bank", "").strip()
 
            # ISO date only (drop time component if present)
            date_val = datetime_val.split("T")[0] if "T" in datetime_val else datetime_val.split(" ")[0]
 
            # Natural-language sentence for embedding
            parts = []
            if datetime_val:
                parts.append(f"On {datetime_val}")
            if txn_type:
                parts.append(f"{txn_type}")
            if amount:
                parts.append(f"of {amount}")
            if payee:
                parts.append(f"at {payee}")
            if description and description.lower() != payee.lower():
                parts.append(f"— {description}")
            if category:
                parts.append(f"({category})")
            if address:
                parts.append(f"at {address}")
            if source_bank:
                parts.append(f"[{source_bank}]")
 
            text = " ".join(parts) if parts else str(row)
 
            # Use transaction_id as doc_id if present, else generate one
            doc_id = f"txn_{transaction_id}" if transaction_id else _make_id("txn", idx)
 
            chunks.append(TextChunk(
                doc_id=doc_id,
                text=text,
                source_type="transaction",
                source_file=csv_path,
                date=date_val or None,
                destination=_infer_destination(address or payee or description or text),
                extra={
                    "transaction_id": transaction_id,
                    "datetime":        datetime_val,
                    "type":            txn_type,
                    "amount":          amount,
                    "payee":           payee,
                    "description":     description,
                    "category":        category,
                    "address":         address,
                    "source_bank":     source_bank,
                },
            ))
    return chunks


def chunk_maps_csv(csv_path: str) -> list[TextChunk]:
    """
    Each row (saved place / POI) becomes one text chunk.
 
    Columns: original_title, note, original_url, place_api_id,
             name, formatted_address, latitude, longitude, types,
             rating, rating_count, phone, website, summary, opening_hours
    """
    import csv
    # Derive a location tag from the CSV filename, e.g. "kyoto_places.csv" → "kyoto"
    location_tag = Path(csv_path).stem.lower()
 
    chunks: list[TextChunk] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            name             = row.get("name", "").strip()
            original_title   = row.get("original_title", "").strip()
            formatted_address= row.get("formatted_address", "").strip()
            types            = row.get("types", "").strip()
            summary          = row.get("summary", "").strip()
            note             = row.get("note", "").strip()
            rating           = row.get("rating", "").strip()
            rating_count     = row.get("rating_count", "").strip()
            phone            = row.get("phone", "").strip()
            opening_hours    = row.get("opening_hours", "").strip()
            place_api_id     = row.get("place_api_id", "").strip()
            lat              = row.get("latitude", "").strip()
            lon              = row.get("longitude", "").strip()
 
            display_name = name or original_title
 
            # Build a rich prose description for embedding
            parts = []
            if display_name:
                parts.append(display_name)
            if formatted_address:
                parts.append(f"located at {formatted_address}")
            if types:
                parts.append(f"[{types}]")
            if rating and rating_count:
                parts.append(f"rated {rating} from {rating_count} reviews")
            if summary:
                parts.append(summary)
            if note:
                parts.append(f"Note: {note}")
            if opening_hours:
                parts.append(f"Hours: {opening_hours}")
 
            text = " ".join(parts) if parts else str(row)
 
            # Prefer place_api_id as stable doc_id; fall back to generated
            doc_id = f"map_{place_api_id}" if place_api_id else _make_id(f"map_{location_tag}", idx)
 
            chunks.append(TextChunk(
                doc_id=doc_id,
                text=text,
                source_type="map",
                source_file=csv_path,
                date=None,           # maps data has no date column
                destination=formatted_address.split(",")[-1].strip() \
                    if formatted_address else _infer_destination(display_name),
                extra={
                    "name":             display_name,
                    "formatted_address":formatted_address,
                    "types":            types,
                    "rating":           rating,
                    "lat":              lat,
                    "lon":              lon,
                    "place_api_id":     place_api_id,
                    "location_tag":     location_tag,
                },
            ))
    return chunks
 

def chunk_photos(photos_dir: str, metadata_dir: str) -> tuple[list[TextChunk], list[ImageRecord]]:
    """
    For each <uuid>.json in metadata_dir + <uuid>.jpg in photos_dir:
        - Build a TextChunk from caption + location metadata (source_type="photo_caption")
        - Build an ImageRecord pointing to the .jpg (for CLIP indexing)
    Both share the same photo_id (= uuid field from JSON).

    JSON schema:
        uuid, original_filename, renamed_filename, filepath,
        indexed_at, timestamp, gps {lat, lon},
        location {country, city, suburb},
        width, height, aspect_ratio, orientation, file_size_kb,
        original_format, caption,
        text_embedding_id, image_embedding_id

    All fields except uuid may be None.
    """
    def _s(val) -> str:
        """Safely coerce any value to a stripped string."""
        return (val or "").strip()

    text_chunks: list[TextChunk] = []
    image_records: list[ImageRecord] = []

    json_paths = sorted([
        p for p in glob.glob(os.path.join(metadata_dir, "*.json"))
        if Path(p).stem != "manifest" # ignore manifest.json
    ])

    for json_path in json_paths:
        with open(json_path, encoding="utf-8") as f:
            meta = json.load(f)

        photo_id = meta.get("uuid") or Path(json_path).stem

        # Locate the image — prefer photos_dir over the absolute filepath in JSON
        jpg_path = os.path.join(photos_dir, f"{photo_id}.jpg")
        if not os.path.exists(jpg_path):
            jpg_path = os.path.join(photos_dir, f"{photo_id}.jpeg")
        if not os.path.exists(jpg_path):
            jpg_path = _s(meta.get("filepath"))
        if not os.path.exists(jpg_path):
            print(f"[chunker] Warning: image not found for {photo_id}, skipping.")
            continue

        # --- Extract fields (all may be None) ---
        caption   = _s(meta.get("caption"))
        timestamp = _s(meta.get("timestamp"))          # "2025:12:27 10:24:06" or None

        gps = meta.get("gps") or {}
        lat = gps.get("lat")   # float or None
        lon = gps.get("lon")   # float or None

        location = meta.get("location") or {}
        country  = _s(location.get("country"))
        city     = _s(location.get("city"))
        suburb   = _s(location.get("suburb"))

        original_filename = _s(meta.get("original_filename"))
        orientation       = _s(meta.get("orientation"))

        # Normalise timestamp → ISO date "YYYY-MM-DD"
        date_val = None
        if timestamp:
            try:
                date_val = timestamp[:10].replace(":", "-")   # "2025:12:27" → "2025-12-27"
            except Exception:
                date_val = None

        # Destination from structured field (more reliable than keyword scan)
        destination = city or country or _infer_destination(caption) or None

        # Build text blob for embedding
        caption_parts = []
        if caption:
            caption_parts.append(caption)
        loc_parts = [p for p in [suburb, city, country] if p]
        if loc_parts:
            caption_parts.append(f"Location: {', '.join(loc_parts)}.")
        if date_val:
            caption_parts.append(f"Date: {date_val}.")
        if lat is not None and lon is not None:
            try:
                caption_parts.append(f"GPS: {float(lat):.4f}, {float(lon):.4f}.")
            except (TypeError, ValueError):
                pass

        text = " ".join(caption_parts) if caption_parts else f"Photo {photo_id}"

        text_chunks.append(TextChunk(
            doc_id=photo_id,
            text=text,
            source_type="photo_caption",
            source_file=json_path,
            date=date_val,
            destination=destination,
            photo_id=photo_id,
            extra={
                "lat":              lat,
                "lon":              lon,
                "country":          country,
                "city":             city,
                "suburb":           suburb,
                "original_filename": original_filename,
                "orientation":      orientation,
            },
        ))

        image_records.append(ImageRecord(
            photo_id=photo_id,
            image_path=jpg_path,
            date=date_val,
            destination=destination,
            extra={
                "lat":     lat,
                "lon":     lon,
                "country": country,
                "city":    city,
                "caption": caption,
            },
        ))

    return text_chunks, image_records

# Top-level builder
def build_all_chunks(
    transactions_csv: str,
    maps_dir: str,          # directory containing per-location map CSVs
    photos_dir: str,        # directory containing uuid.jpg files
    metadata_dir: str,      # directory containing uuid.json sidecars
) -> tuple[list[TextChunk], list[ImageRecord]]:
    """
    Run all chunkers and return:
        - all_text_chunks: fed into indexer.py → text_index collection
        - all_image_records: fed into indexer.py → image_index collection
    """
    all_text: list[TextChunk] = []

    print("[chunker] Processing transactions CSV...")
    all_text += chunk_transactions_csv(transactions_csv)

    print("[chunker] Processing maps CSVs...")
    map_csvs = glob.glob(os.path.join(maps_dir, "*.csv"))
    for csv_path in sorted(map_csvs):
        all_text += chunk_maps_csv(csv_path)

    print("[chunker] Processing photos + JSON sidecars...")
    photo_text, image_records = chunk_photos(photos_dir, metadata_dir)
    all_text += photo_text

    print(f"[chunker] Done. {len(all_text)} text chunks, {len(image_records)} image records.")

    print("[chunker] Enriching transactions...")
    txn_chunks = chunk_transactions_csv(transactions_csv)
    txn_chunks = enrich_transactions_llm(txn_chunks)   # ← add this
    all_text += txn_chunks
    
    return all_text, image_records


# Smoke test
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print("Usage: python chunker.py <transactions.csv> <maps_dir/> <photos_dir/> <metadata_dir/>")
        sys.exit(1)

    chunks, images = build_all_chunks(
        transactions_csv=sys.argv[1],
        maps_dir=sys.argv[2],
        photos_dir=sys.argv[3],
        metadata_dir=sys.argv[4],
    )

    print("\n--- Sample text chunks ---")
    for c in chunks[:5]:
        print(f"  [{c.source_type}] {c.doc_id} | dest={c.destination} | {c.text[:80]}...")

    print("\n--- Sample image records ---")
    for r in images[:3]:
        print(f"  {r.photo_id} | {r.image_path} | dest={r.destination}")