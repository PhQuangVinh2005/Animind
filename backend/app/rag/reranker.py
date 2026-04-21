"""Reranker client — Qwen3-Reranker-0.6B via vLLM OpenAI-compatible API."""

import httpx
from loguru import logger

from app.config import settings


async def rerank(
    query: str,
    documents: list[str],
    top_k: int | None = None,
) -> list[dict]:
    """Rerank documents using Qwen3-Reranker-0.6B via vLLM.

    Args:
        query: The search query.
        documents: List of document texts to rerank.
        top_k: Number of top results to return. Defaults to settings.reranker_top_k.

    Returns:
        List of dicts with 'index', 'relevance_score', sorted by score descending.
    """
    if top_k is None:
        top_k = settings.reranker_top_k

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{settings.reranker_url}/v1/rerank",
            json={
                "model": settings.reranker_model,
                "query": query,
                "documents": documents,
            },
        )
        resp.raise_for_status()
        results = resp.json()["results"]

    ranked = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
    logger.debug(
        "Reranked {total} docs → top-{k}, best score: {score:.4f}",
        total=len(documents),
        k=top_k,
        score=ranked[0]["relevance_score"] if ranked else 0,
    )
    return ranked[:top_k]
