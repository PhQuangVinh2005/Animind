"""Ingest anime data into Qdrant vector store.

Usage:
    cd backend
    python scripts/ingest.py [--limit N] [--batch-size N]

Reads:
    data/raw/anime.json

Process:
    1. Load and validate raw records
    2. Build document text: "{title}\\n\\n{description}"
    3. Embed in batches with text-embedding-3-small via ShopAIKey provider
    4. Upsert to Qdrant collection "anime" with full metadata payload

Requires SHOPAIKEY_API_KEY in backend/.env.
Run  python setup_env.py  from the project root to configure credentials.
"""

import argparse
import html as html_lib
import json
import re
import sys
from pathlib import Path

from loguru import logger
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

# Allow running as `python scripts/ingest.py` from backend/ root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings          # noqa: E402
from app.openai_client import make_openai_client  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "anime.json"
EMBEDDING_DIM = 1536        # text-embedding-3-small output dimension
EMBED_BATCH_SIZE = 100      # OpenAI allows up to 2048 inputs; 100 is safe + fast
UPSERT_BATCH_SIZE = 256     # Qdrant upsert batch


# ── Chunking helpers (Strategy 5: Hybrid Metadata Filter + Single Chunk) ─────

_SOURCE_CITE_RE = re.compile(
    r'\s*\(Source:[^)]*\)|\s*\[Written by[^\]]*\]',
    re.IGNORECASE,
)


def clean_html(text: str) -> str:
    """Strip HTML tags, decode entities, remove AniList source citations."""
    if not text:
        return ""
    text = re.sub(r'<br\s*/?>', ' ', text)        # <br> → space
    text = re.sub(r'<[^>]+>', '', text)            # strip remaining tags
    text = html_lib.unescape(text)                 # &amp; → & etc.
    text = _SOURCE_CITE_RE.sub('', text)           # drop (Source: ...)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_anime(anime_json: dict) -> tuple[str, dict]:
    """Convert raw AniList record → (chunk_text, qdrant_payload).

    chunk_text  — the text to embed (Strategy 5 format)
    qdrant_payload — rich metadata stored in Qdrant (NOT embedded)
    """
    is_adult: bool = anime_json.get("is_adult", False)

    # ── 1. Titles (max 5, deduplicated case-insensitively) ───────────────────
    seen_lower: set[str] = set()
    titles: list[str] = []

    def _add(t: str | None) -> None:
        if t and t.lower() not in seen_lower and len(titles) < 5:
            seen_lower.add(t.lower())
            titles.append(t)

    _add(anime_json.get("title_preferred"))
    _add(anime_json.get("title_english"))
    _add(anime_json.get("title_romaji"))
    _add(anime_json.get("title_native"))
    for syn in (anime_json.get("synonyms") or []):
        _add(syn)

    title_line = ". ".join(titles) + "." if titles else "Unknown Title."

    # ── 2. Genres ────────────────────────────────────────────────────────────
    genres: list[str] = anime_json.get("genres") or []
    genre_line = (", ".join(genres) + ".") if genres else ""

    # ── 3. Tags (rank >= 70, no Sexual Content unless adult) ─────────────────
    raw_tags: list[dict] = [
        t for t in (anime_json.get("tags") or []) if isinstance(t, dict)
    ]
    filtered_tags = [
        t for t in raw_tags
        if t.get("rank", 0) >= 70
        and (is_adult or t.get("category") != "Sexual Content")
    ]
    filtered_tags.sort(key=lambda t: t.get("rank", 0), reverse=True)
    tag_line = (", ".join(t["name"] for t in filtered_tags) + ".") if filtered_tags else ""

    # ── 4. Synopsis ──────────────────────────────────────────────────────────
    synopsis = clean_html(anime_json.get("description") or "")
    if not synopsis:
        synopsis = "No synopsis available."
    elif len(synopsis) > 1000:
        synopsis = synopsis[:1000]

    # ── 5. Metadata line ─────────────────────────────────────────────────────
    studios: list[str] = anime_json.get("studios") or []
    meta: list[str] = ["Studio: " + (", ".join(studios) if studios else "Unknown")]

    if anime_json.get("season_year"):
        meta.append(f"Year: {anime_json['season_year']}")
    if anime_json.get("season"):
        meta.append(f"Season: {anime_json['season'].capitalize()}")
    if anime_json.get("format"):
        meta.append(f"Format: {anime_json['format']}")
    if anime_json.get("episodes"):
        meta.append(f"Episodes: {anime_json['episodes']}")

    score: int | None = anime_json.get("average_score")
    if score is not None:
        meta.append(f"Score: {score / 10:.1f}")

    if anime_json.get("source"):
        meta.append(f"Source: {anime_json['source'].replace('_', ' ').title()}")

    meta_line = ". ".join(meta) + "."

    # ── Assemble chunk ───────────────────────────────────────────────────────
    sections = [title_line]
    if genre_line:
        sections.append(genre_line)
    if tag_line:
        sections.append(tag_line)
    sections.append(synopsis)
    sections.append(meta_line)
    chunk_text = "\n".join(sections)

    # ── Payload ──────────────────────────────────────────────────────────────
    all_tag_names = [t["name"] for t in raw_tags]   # ALL tags for filtering

    payload: dict = {
        # Identifiers
        "anilist_id":     anime_json["id"],
        "mal_id":         anime_json.get("id_mal"),
        # Display
        "title": {
            "preferred": anime_json.get("title_preferred"),
            "english":   anime_json.get("title_english"),
            "native":    anime_json.get("title_native"),
        },
        "cover_image":    anime_json.get("cover_image"),
        "banner_image":   anime_json.get("banner_image"),
        "site_url":       anime_json.get("site_url"),
        # Filterable
        "year":           anime_json.get("season_year"),
        "season":         anime_json.get("season"),
        "season_display": anime_json["season"].capitalize() if anime_json.get("season") else None,
        "genres":         genres,
        "tags":           all_tag_names,
        "studios":        studios,
        "format":         anime_json.get("format"),
        "status":         anime_json.get("status"),
        "episodes":       anime_json.get("episodes"),
        "duration":       anime_json.get("duration"),
        "source":         anime_json.get("source"),
        "is_adult":       is_adult,
        "score":          score,
        "score_display":  round(score / 10, 1) if score is not None else None,
        # Full data for reranker & LLM context
        "full_data":      anime_json,
    }

    return chunk_text, payload


def _build_doc_text(record: dict) -> str:
    """Build the text to embed for a single anime record.

    Combines all text-bearing fields that help semantic retrieval:
    title variants, synonyms, description, genres, tags, source,
    country, and top rankings — all human-readable sentences.
    """
    # ── Title block ────────────────────────────────────────────────────────
    titles: list[str] = []
    for key in ("title_english", "title_romaji", "title_native"):
        if record.get(key):
            titles.append(record[key])
    if record.get("synonyms"):
        titles.extend(record["synonyms"])
    title_line = " / ".join(dict.fromkeys(titles)) if titles else "Unknown Title"

    parts = [title_line]

    # ── Description ────────────────────────────────────────────────────────
    description = (record.get("description") or "").strip()
    if len(description) > 1200:
        description = description[:1200] + "…"
    if description:
        parts.append(description)

    # ── Genres + Tags ──────────────────────────────────────────────────────
    genres = record.get("genres") or []
    if genres:
        parts.append("Genres: " + ", ".join(genres))

    tag_names = [t["name"] if isinstance(t, dict) else t for t in (record.get("tags") or [])]
    if tag_names:
        parts.append("Tags: " + ", ".join(tag_names))

    # ── Source & Origin ────────────────────────────────────────────────────
    meta: list[str] = []
    if record.get("source"):
        source_display = record["source"].replace("_", " ").title()
        meta.append(f"Source: {source_display}")
    if record.get("country_of_origin"):
        meta.append(f"Country: {record['country_of_origin']}")
    if record.get("format"):
        meta.append(f"Format: {record['format']}")
    if record.get("status"):
        meta.append(f"Status: {record['status']}")
    if record.get("season") and record.get("season_year"):
        meta.append(f"Aired: {record['season'].title()} {record['season_year']}")
    if record.get("episodes"):
        meta.append(f"Episodes: {record['episodes']}")
    if record.get("studios"):
        meta.append("Studios: " + ", ".join(record["studios"]))
    if meta:
        parts.append(" | ".join(meta))

    # ── Scores & Rankings ──────────────────────────────────────────────────
    score_parts: list[str] = []
    if record.get("average_score"):
        score_parts.append(f"Score: {record['average_score']}/100")
    if record.get("popularity"):
        score_parts.append(f"Popularity: {record['popularity']:,} users")
    for rank in (record.get("rankings") or []):
        ctx = rank.get("context") if isinstance(rank, dict) else None
        rk  = rank.get("rank")    if isinstance(rank, dict) else None
        if ctx and rk:
            score_parts.append(f"#{rk} {ctx}")
    if score_parts:
        parts.append(" | ".join(score_parts))

    # ── Streaming ──────────────────────────────────────────────────────────
    streaming = [
        lk["site"] if isinstance(lk, dict) else lk
        for lk in (record.get("streaming_links") or [])
        if (lk.get("site") if isinstance(lk, dict) else lk)
    ]
    if streaming:
        parts.append("Streaming on: " + ", ".join(dict.fromkeys(streaming)))

    return "\n\n".join(parts)


def _build_payload(record: dict) -> dict:
    """Extract Qdrant metadata payload — all fields used for filtering & display."""
    return {
        # Identity
        "anilist_id":       record["id"],
        "mal_id":           record.get("id_mal"),
        "title_romaji":     record.get("title_romaji"),
        "title_english":    record.get("title_english"),
        "title_native":     record.get("title_native"),
        "title_preferred":  record.get("title_preferred"),
        "synonyms":         record.get("synonyms", []),
        # Content
        "description":      record.get("description", ""),
        "genres":           record.get("genres", []),
        "tags":             record.get("tags", []),       # [{name, rank, category}]
        "source":           record.get("source"),         # MANGA, LIGHT_NOVEL, ORIGINAL…
        "country_of_origin": record.get("country_of_origin"),
        "is_adult":         record.get("is_adult", False),
        # Format & Airing
        "format":           record.get("format"),
        "status":           record.get("status"),
        "episodes":         record.get("episodes"),
        "duration":         record.get("duration"),
        "season":           record.get("season"),
        "season_year":      record.get("season_year"),
        "start_year":       record.get("start_year"),
        "start_month":      record.get("start_month"),
        "start_day":        record.get("start_day"),
        "end_year":         record.get("end_year"),
        "end_month":        record.get("end_month"),
        "end_day":          record.get("end_day"),
        "next_airing_episode": record.get("next_airing_episode"),
        "next_airing_at":   record.get("next_airing_at"),
        # Scores
        "average_score":    record.get("average_score"),
        "mean_score":       record.get("mean_score"),
        "popularity":       record.get("popularity"),
        "favourites":       record.get("favourites"),
        "trending":         record.get("trending"),
        # Rankings
        "rankings":         record.get("rankings", []),
        # Production
        "studios":          record.get("studios", []),
        "studio_urls":      record.get("studio_urls", []),
        # Streaming
        "streaming_links":  record.get("streaming_links", []),
        # Assets
        "cover_image":      record.get("cover_image"),
        "cover_color":      record.get("cover_color"),
        "banner_image":     record.get("banner_image"),
        "trailer_site":     record.get("trailer_site"),
        "trailer_id":       record.get("trailer_id"),
        "trailer_thumbnail": record.get("trailer_thumbnail"),
        # Links
        "site_url":         record.get("site_url"),
        "updated_at":       record.get("updated_at"),
    }


# ── Embedding ─────────────────────────────────────────────────────────────────

async def embed_texts(
    openai_client: AsyncOpenAI,
    texts: list[str],
) -> list[list[float]]:
    """Embed a list of texts in batches; returns list of float vectors."""
    all_vectors: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        logger.debug(
            "Embedding batch {start}-{end} of {total} …",
            start=i + 1,
            end=min(i + EMBED_BATCH_SIZE, len(texts)),
            total=len(texts),
        )
        resp = await openai_client.embeddings.create(
            model=settings.openai_embedding_model,
            input=batch,
        )
        all_vectors.extend(item.embedding for item in resp.data)

    return all_vectors


# ── Qdrant setup ──────────────────────────────────────────────────────────────

async def ensure_collection(qdrant: AsyncQdrantClient) -> None:
    """Create the Qdrant collection if it doesn't already exist."""
    existing = {c.name for c in (await qdrant.get_collections()).collections}

    if settings.qdrant_collection in existing:
        logger.info(
            "Collection '{col}' already exists — skipping creation.",
            col=settings.qdrant_collection,
        )
        return

    await qdrant.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    logger.info(
        "Created Qdrant collection '{col}' (dim={dim}, cosine).",
        col=settings.qdrant_collection,
        dim=EMBEDDING_DIM,
    )


# ── Pre-flight checks ─────────────────────────────────────────────────────────

async def preflight(qdrant: AsyncQdrantClient, openai_client: AsyncOpenAI) -> None:
    """Verify both Qdrant and embedding API are reachable before starting.

    Raises SystemExit(1) with a clear message if either service is down,
    instead of silently hanging or printing raw Cloudflare HTML.
    """

    # ── Check Qdrant ──────────────────────────────────────────────────────────
    logger.info("Pre-flight: checking Qdrant at {url} …", url=settings.qdrant_url)
    try:
        await qdrant.get_collections()
        logger.success("Pre-flight: Qdrant OK ✓")
    except Exception as exc:
        logger.error(
            "Pre-flight FAIL: Cannot reach Qdrant at {url}\n"
            "  → Start it with:  docker compose up -d qdrant\n"
            "  → Error: {exc}",
            url=settings.qdrant_url, exc=exc,
        )
        raise SystemExit(1) from exc

    # ── Check embedding API ───────────────────────────────────────────────────
    # Use the real OpenAI client (same headers/auth as production) rather than
    # a raw httpx call — this is a true end-to-end check with a longer timeout.
    logger.info(
        "Pre-flight: checking embedding API at {url} …",
        url=settings.shopaikey_base_url,
    )
    try:
        resp = await openai_client.embeddings.create(
            model=settings.openai_embedding_model,
            input=["ping"],           # minimal test — 1 token
        )
        dim = len(resp.data[0].embedding)
        logger.success(
            "Pre-flight: Embedding API OK ✓  (model={model}, dim={dim})",
            model=settings.openai_embedding_model,
            dim=dim,
        )
    except Exception as exc:
        msg = str(exc)
        hint = (
            "Cloudflare 5xx detected — provider may be temporarily down. Retry in a few minutes."
            if any(code in msg for code in ("522", "521", "524", "502", "503"))
            else "Check SHOPAIKEY_API_KEY and SHOPAIKEY_BASE_URL in your .env"
        )
        logger.error(
            "Pre-flight FAIL: Embedding API error\n"
            "  → URL:   {url}\n"
            "  → Error: {exc}\n"
            "  → Hint:  {hint}",
            url=settings.shopaikey_base_url, exc=exc, hint=hint,
        )
        raise SystemExit(1) from exc


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def ingest(limit: int | None = None, reset: bool = False) -> None:
    """Full ingestion pipeline: load → embed → upsert."""

    # 1. Load raw data
    if not RAW_PATH.exists():
        logger.error(
            "Raw data not found at {path}. Run fetch_anilist.py first.",
            path=RAW_PATH,
        )
        raise SystemExit(1)

    records: list[dict] = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    logger.info("Loaded {n} records from {path}", n=len(records), path=RAW_PATH)

    if limit:
        records = records[:limit]
        logger.info("Limiting to first {n} records (--limit flag)", n=limit)

    # 2. Process records → chunk texts + payloads (Strategy 5)
    processed = [process_anime(r) for r in records]
    texts    = [chunk for chunk, _     in processed]
    payloads = [pl    for _,     pl    in processed]

    # 3. Connect to services
    openai_client = make_openai_client()   # ShopAIKey provider (base_url + User-Agent)
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)

    # ← Pre-flight: fail fast with clear messages instead of hanging
    await preflight(qdrant, openai_client)

    if reset:
        existing = {c.name for c in (await qdrant.get_collections()).collections}
        if settings.qdrant_collection in existing:
            await qdrant.delete_collection(settings.qdrant_collection)
            logger.warning(
                "Deleted collection '{col}' (--reset flag).",
                col=settings.qdrant_collection,
            )

    await ensure_collection(qdrant)

    # 4. Embed in batches
    logger.info(
        "Embedding {n} documents with {model} …",
        n=len(texts),
        model=settings.openai_embedding_model,
    )
    vectors = await embed_texts(openai_client, texts)
    logger.success("Embedding complete — {n} vectors generated.", n=len(vectors))

    # 5. Upsert to Qdrant in batches
    logger.info(
        "Upserting {n} points to collection '{col}' …",
        n=len(records),
        col=settings.qdrant_collection,
    )

    for i in range(0, len(records), UPSERT_BATCH_SIZE):
        batch_records  = records [i : i + UPSERT_BATCH_SIZE]
        batch_vectors  = vectors [i : i + UPSERT_BATCH_SIZE]
        batch_payloads = payloads[i : i + UPSERT_BATCH_SIZE]

        points = [
            PointStruct(
                id=record["id"],          # AniList ID as Qdrant point ID (uint64)
                vector=vector,
                payload=payload,
            )
            for record, vector, payload in zip(batch_records, batch_vectors, batch_payloads)
        ]

        await qdrant.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
        )
        logger.info(
            "Upserted {end}/{total} points",
            end=min(i + UPSERT_BATCH_SIZE, len(records)),
            total=len(records),
        )

    # 6. Verify
    info = await qdrant.get_collection(settings.qdrant_collection)
    logger.success(
        "Ingestion complete. Collection '{col}' now has {count} vectors.",
        col=settings.qdrant_collection,
        count=info.points_count,
    )

    await qdrant.close()


def main() -> None:
    import asyncio

    global EMBED_BATCH_SIZE  # must appear before any use of this name in the function

    parser = argparse.ArgumentParser(description="Ingest anime data into Qdrant.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only ingest first N records (useful for testing).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=EMBED_BATCH_SIZE,
        help=f"OpenAI embedding batch size (default: {EMBED_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the Qdrant collection before ingesting (clean slate).",
    )
    args = parser.parse_args()

    EMBED_BATCH_SIZE = args.batch_size

    asyncio.run(ingest(limit=args.limit, reset=args.reset))


if __name__ == "__main__":
    main()
