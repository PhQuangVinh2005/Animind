"""Qdrant retriever — hybrid search (dense + BM25 sparse) with RRF fusion.

Pipeline position:
    User query → [embed dense + sparse] → [hybrid retrieve top-K] → reranker → top-5

Hybrid strategy:
    1. Dense: text-embedding-3-small via OpenAI API → cosine similarity
    2. Sparse: fastembed Qdrant/bm25 → exact keyword/title matching
    3. Fusion: Qdrant server-side RRF (Reciprocal Rank Fusion) via prefetch

Payload fields available for filtering (set by ingest.py::process_anime):
    year (int), season (str), genres (list[str]), tags (list[str]),
    format (str), status (str), score (int, 0-100), is_adult (bool)
"""

from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass, field
from uuid import UUID

from fastembed import SparseTextEmbedding
from loguru import logger
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    Prefetch,
    Range,
    SparseVector,
)

from app.config import settings

# Module-level singleton — one TCP connection pool, not one per retrieve() call
_qdrant_client: QdrantClient | None = None


def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=settings.qdrant_url)
    return _qdrant_client


# Module-level singleton for BM25 sparse model — loaded once on first use
_bm25_model: SparseTextEmbedding | None = None


def _get_bm25_model() -> SparseTextEmbedding:
    """Lazy-load the fastembed BM25 sparse encoder (CPU only, ~5MB)."""
    global _bm25_model
    if _bm25_model is None:
        _bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info("BM25 sparse model loaded (Qdrant/bm25).")
    return _bm25_model


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RetrievedDoc:
    """A single document returned from Qdrant."""

    qdrant_id: int | str
    score: float
    payload: dict = field(default_factory=dict)

    @property
    def title(self) -> str:
        t = self.payload.get("title", {})
        if isinstance(t, dict):
            return (
                t.get("preferred")
                or t.get("english")
                or t.get("native")
                or "Unknown"
            )
        # flat payload fallback
        return (
            self.payload.get("title_preferred")
            or self.payload.get("title_english")
            or self.payload.get("title_romaji")
            or "Unknown"
        )

    @property
    def chunk_text(self) -> str:
        """Reconstructed text for reranker input (from payload fields)."""
        parts: list[str] = [self.title]
        genres = self.payload.get("genres") or []
        if genres:
            parts.append(", ".join(genres) + ".")
        # synopsis from full_data if available, else description
        full = self.payload.get("full_data") or {}
        desc = full.get("description") or self.payload.get("description") or ""
        if desc:
            desc = re.sub(r"<[^>]+>", " ", desc)
            desc = html_lib.unescape(desc)
            desc = re.sub(r"\s+", " ", desc).strip()[:500]
            parts.append(desc)
        return "\n".join(parts)


# ── Metadata filter builder ───────────────────────────────────────────────────

def build_filter(
    *,
    genres: list[str] | None = None,
    year: int | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    format_: str | None = None,       # "TV", "MOVIE", "OVA", "ONA", "SPECIAL"
    score_min: int | None = None,     # 0–100 (Qdrant payload uses raw integer)
    is_adult: bool | None = None,
) -> Filter | None:
    """Build a Qdrant Filter from optional keyword arguments.

    Returns None if no conditions are set (meaning: no filter, full scan).
    """
    must: list[FieldCondition] = []

    if genres:
        must.append(FieldCondition(key="genres", match=MatchAny(any=genres)))

    if year is not None:
        must.append(FieldCondition(key="year", match=MatchValue(value=year)))
    elif year_min is not None or year_max is not None:
        must.append(
            FieldCondition(
                key="year",
                range=Range(
                    gte=year_min,
                    lte=year_max,
                ),
            )
        )

    if format_ is not None:
        must.append(FieldCondition(key="format", match=MatchValue(value=format_)))

    if score_min is not None:
        must.append(FieldCondition(key="score", range=Range(gte=score_min)))

    if is_adult is not None:
        must.append(FieldCondition(key="is_adult", match=MatchValue(value=is_adult)))

    return Filter(must=must) if must else None  # type: ignore[arg-type]


# ── Sparse query helper ──────────────────────────────────────────────────────

def _encode_sparse(query: str) -> SparseVector:
    """Encode a query string into a BM25 sparse vector for Qdrant."""
    model = _get_bm25_model()
    result = list(model.embed([query]))[0]
    return SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


# ── Core retriever ────────────────────────────────────────────────────────────

async def retrieve(
    query: str,
    oai_client: AsyncOpenAI,
    *,
    top_k: int | None = None,
    filter_kwargs: dict | None = None,
) -> list[RetrievedDoc]:
    """Hybrid search: dense + BM25 sparse with RRF fusion.

    Uses Qdrant's server-side prefetch + Reciprocal Rank Fusion to
    combine semantic similarity (dense) with keyword matching (BM25).
    This dramatically improves recall for exact title queries like
    "Clannad: After Story" which dense-only search confuses with "Clannad".

    Args:
        query:          The (possibly rewritten) user query string.
        oai_client:     AsyncOpenAI client (ShopAIKey, already configured).
        top_k:          Number of candidates to retrieve. Defaults to
                        settings.retriever_top_k (20).
        filter_kwargs:  Optional dict of keyword args forwarded to build_filter()
                        e.g. {"genres": ["Action"], "year": 2023}.

    Returns:
        List of RetrievedDoc sorted by RRF fusion score (highest first).
    """
    if top_k is None:
        top_k = settings.retriever_top_k

    # 1. Embed the query (dense)
    emb_resp = await oai_client.embeddings.create(
        model=settings.openai_embedding_model,
        input=query,
    )
    dense_vector: list[float] = emb_resp.data[0].embedding

    # 2. Encode BM25 sparse vector
    sparse_vector = _encode_sparse(query)

    # 3. Build optional metadata filter
    qdrant_filter: Filter | None = None
    if filter_kwargs:
        qdrant_filter = build_filter(**filter_kwargs)

    # 4. Hybrid search — prefetch from both vector spaces, fuse with RRF
    #    Each prefetch retrieves a larger candidate pool (3× top_k);
    #    RRF fusion then selects the best top_k from the combined pool.
    prefetch_limit = min(top_k * 3, 60)  # cap at 60 to avoid excess

    results = _get_qdrant().query_points(
        collection_name=settings.qdrant_collection,
        prefetch=[
            Prefetch(
                query=dense_vector,
                using="dense",
                limit=prefetch_limit,
                filter=qdrant_filter,
            ),
            Prefetch(
                query=sparse_vector,
                using="bm25",
                limit=prefetch_limit,
                filter=qdrant_filter,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    docs = [
        RetrievedDoc(
            # pt.id is int | str | UUID; coerce UUID → str to satisfy the dataclass
            qdrant_id=str(pt.id) if isinstance(pt.id, UUID) else pt.id,
            score=pt.score,
            payload=pt.payload or {},
        )
        for pt in results.points
    ]

    logger.debug(
        "Hybrid retrieved {n} docs for query={q!r} (filter={f})",
        n=len(docs),
        q=query[:60],
        f=bool(qdrant_filter),
    )
    return docs
