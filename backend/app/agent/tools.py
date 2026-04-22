"""Agent tools — Qdrant-backed search and detail lookup.

Both tools query the indexed Qdrant collection (1 250 records).
No live AniList API calls are made here; the AniList API is only
used during data ingestion (scripts/fetch_anilist.py).

Tools
-----
search_anime    — semantic search + reranking, returns top-5 results
get_anime_details — lookup by AniList ID or title similarity, returns full payload
"""

from __future__ import annotations

import html as html_lib
import re

from loguru import logger
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.config import settings
from app.rag.reranker import rerank
from app.rag.retriever import RetrievedDoc, retrieve

# ── Qdrant singleton (reuse from retriever when possible) ─────────────────────
_qdrant: QdrantClient | None = None


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=settings.qdrant_url)
    return _qdrant


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def doc_to_dict(doc: RetrievedDoc) -> dict:
    """Serialise RetrievedDoc → plain dict for LangGraph state storage."""
    return {
        "qdrant_id": doc.qdrant_id,
        "score": doc.score,
        "title": doc.title,
        "chunk_text": doc.chunk_text,
        "payload": doc.payload,
    }


def payload_to_detail(payload: dict) -> dict:
    """Extract clean detail fields from a raw Qdrant payload."""
    title = payload.get("title", {})
    if isinstance(title, dict):
        title_str = (
            title.get("preferred")
            or title.get("english")
            or title.get("native")
            or "Unknown"
        )
    else:
        title_str = str(title) if title else "Unknown"

    full = payload.get("full_data") or {}
    desc_raw = full.get("description") or payload.get("description") or ""
    desc = _clean_html(desc_raw)[:1500]

    return {
        "anilist_id": payload.get("anilist_id"),
        "mal_id": payload.get("mal_id"),
        "title": title_str,
        "year": payload.get("year"),
        "season": payload.get("season"),
        "format": payload.get("format"),
        "episodes": payload.get("episodes"),
        "duration": payload.get("duration"),
        "score": payload.get("score"),
        "status": payload.get("status"),
        "source": payload.get("source"),
        "is_adult": payload.get("is_adult", False),
        "genres": payload.get("genres") or [],
        "tags": (payload.get("tags") or [])[:10],
        "studios": payload.get("studios") or [],
        "site_url": payload.get("site_url"),
        "cover_image": payload.get("cover_image"),
        "description": desc,
    }


# ── Tool: search_anime ────────────────────────────────────────────────────────

async def search_anime(
    query: str,
    oai_client: AsyncOpenAI,
    *,
    genres: list[str] | None = None,
    year: int | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    format_: str | None = None,
    score_min: int | None = None,
) -> dict:
    """Semantic search over Qdrant + Qwen3 reranking.

    Args:
        query:      User's search query (or rewritten form).
        oai_client: AsyncOpenAI client for embedding.
        genres:     Optional genre filter list.
        year:       Exact year filter.
        year_min/year_max: Year range filter (used if year is None).
        format_:    "TV" | "MOVIE" | "OVA" | "ONA" | "SPECIAL".
        score_min:  Minimum raw score (0–100).

    Returns:
        {"results": [doc_dict, ...], "count": int, "query": str, "filters": dict}
    """
    filter_kwargs: dict = {}
    if genres:
        filter_kwargs["genres"] = genres
    if year is not None:
        filter_kwargs["year"] = year
    if year_min is not None:
        filter_kwargs["year_min"] = year_min
    if year_max is not None:
        filter_kwargs["year_max"] = year_max
    if format_ is not None:
        filter_kwargs["format_"] = format_
    if score_min is not None:
        filter_kwargs["score_min"] = score_min

    candidates: list[RetrievedDoc] = await retrieve(
        query,
        oai_client,
        top_k=settings.retriever_top_k,
        filter_kwargs=filter_kwargs or None,
    )

    if not candidates:
        logger.debug("search_anime: no candidates for query={q!r}", q=query)
        return {"results": [], "count": 0, "query": query, "filters": filter_kwargs}

    doc_texts = [doc.chunk_text for doc in candidates]
    ranked = await rerank(query, doc_texts, top_k=settings.reranker_top_k)
    top_docs = [candidates[r["index"]] for r in ranked]

    serialised = [doc_to_dict(d) for d in top_docs]
    logger.info(
        "search_anime: {n} results for {q!r} (filters={f})",
        n=len(serialised),
        q=query[:60],
        f=bool(filter_kwargs),
    )
    return {
        "results": serialised,
        "count": len(serialised),
        "query": query,
        "filters": filter_kwargs,
    }


# ── Tool: get_anime_details ───────────────────────────────────────────────────

async def get_anime_details(
    identifier: str | int,
    oai_client: AsyncOpenAI,
) -> dict:
    """Lookup a specific anime by AniList ID or title string.

    Priority:
        1. Exact anilist_id match (if identifier is numeric).
        2. Vector similarity search (top-1) as fallback.

    Args:
        identifier: AniList numeric ID (e.g. 21) or title string (e.g. "One Piece").
        oai_client: AsyncOpenAI client (used for fallback vector search).

    Returns:
        {"found": bool, "anime": payload_dict | None}
    """
    qdrant = _get_qdrant()

    # ── Strategy 1: exact AniList ID match ───────────────────────────────────
    is_numeric = (
        isinstance(identifier, int)
        or (isinstance(identifier, str) and identifier.strip().isdigit())
    )
    if is_numeric:
        anilist_id = int(str(identifier).strip())
        scroll_result = qdrant.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="anilist_id", match=MatchValue(value=anilist_id))]
            ),
            limit=1,
            with_payload=True,
        )
        points, _ = scroll_result
        if points:
            logger.debug("get_anime_details: found by anilist_id={id}", id=anilist_id)
            return {"found": True, "anime": payload_to_detail(points[0].payload or {})}

    # ── Strategy 2: vector similarity search by title string ─────────────────
    candidates = await retrieve(str(identifier), oai_client, top_k=1)
    if candidates:
        logger.debug(
            "get_anime_details: found {title!r} by vector similarity",
            title=candidates[0].title,
        )
        return {"found": True, "anime": payload_to_detail(candidates[0].payload)}

    logger.warning("get_anime_details: not found for identifier={id!r}", id=identifier)
    return {"found": False, "anime": None}
