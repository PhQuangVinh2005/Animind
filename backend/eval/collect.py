"""Collect pipeline outputs for evaluation.

Runs baseline and/or current RAG pipeline for all questions in test_set.json
and saves raw results (question + answer + contexts + latency) to JSON files.

Two pipelines
-------------
baseline : raw embed → retrieve(top-5, no filter) → generate
           (no query rewrite, no filter extraction, no reranker)
           Uses top-5 to match current pipeline doc count — isolates reranker quality.
current  : rewrite → extract_filters → retrieve(top-20, filtered)
           → Qwen3 rerank(top-5) → generate

Performance engineering notes (agent-evaluation + performance-engineer)
----------------------------------------------------------------------
- Sequential processing: one question at a time to respect API rate limits.
- Incremental saves: results written after every question — safe to resume.
- Per-question latency tracking via time.perf_counter().
- 0.5 s sleep between questions to stay within ShopAIKey rate limits.
- Retry: 3 attempts with 2 s backoff per question on transient errors.

Usage
-----
    cd backend
    conda run -n animind python eval/collect.py --pipeline both
    conda run -n animind python eval/collect.py --pipeline baseline --limit 5
    conda run -n animind python eval/collect.py --pipeline current --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import html as html_lib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

# ── Path setup — allow imports from backend/app ─────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.openai_client import make_openai_client  # noqa: E402
from app.rag.chain import extract_filters, rewrite_query  # noqa: E402, PLC2701
from app.rag.reranker import rerank  # noqa: E402
from app.rag.retriever import RetrievedDoc, retrieve  # noqa: E402


# ── Context formatter for RAGAS/FactScore ─────────────────────────────────────

def _doc_to_eval_context(doc: RetrievedDoc) -> str:
    """Format one doc with ALL payload fields for RAGAS/FactScore contexts.

    Mirrors what _build_context() sends to the LLM so the evaluator judges
    the same information: score, year, format, genres, episodes, studio,
    and synopsis. Storing only chunk_text (title+genres+synopsis) causes
    faithfulness/FactScore failures for score/episode/studio questions.
    """
    payload = doc.payload
    title = doc.title

    # Score
    score_raw = payload.get("score") or payload.get("average_score")
    score_str = f"{score_raw / 10:.1f}/10" if score_raw else "N/A"

    # Year + format
    year = payload.get("year") or payload.get("season_year")
    fmt = payload.get("format", "")

    # Genres
    genres = ", ".join(payload.get("genres") or [])

    # Episode count
    episodes = payload.get("episodes")

    # Studios
    studios = payload.get("studios") or []
    studio_str = ", ".join(studios) if isinstance(studios, list) else str(studios)

    # Synopsis — strip HTML tags, unescape entities, truncate
    full = payload.get("full_data") or {}
    desc_raw = full.get("description") or payload.get("description") or ""
    desc = re.sub(r"<[^>]+>", " ", desc_raw)
    desc = html_lib.unescape(desc)
    desc = re.sub(r"\s+", " ", desc).strip()[:600]

    # Build header line — use explicit labels so LLM can extract individual fields
    fields: list[str] = [title]
    if year:
        fields.append(f"Year: {year}")
    if fmt:
        fields.append(f"Format: {fmt}")
    fields.append(f"Score: {score_str}")
    if genres:
        fields.append(f"Genres: {genres}")
    if episodes:
        fields.append(f"Episodes: {episodes}")
    if studio_str:
        fields.append(f"Studio: {studio_str}")

    parts = [" | ".join(fields)]
    if desc:
        parts.append(desc)
    return "\n".join(parts)


def _doc_to_raw_payload(doc: RetrievedDoc) -> dict[str, Any]:
    """Serialize a RetrievedDoc to a JSON-serialisable dict with full payload.

    Stored in retrieved_docs[] alongside the formatted context string so that
    post-hoc analysis can inspect exactly what Qdrant returned — title,
    vector similarity score, genres, year, format, episodes, studio,
    synopsis, and any other payload keys.
    """
    return {
        "title": doc.title,
        "vector_score": round(doc.score, 6) if hasattr(doc, "score") and doc.score is not None else None,
        "payload": doc.payload,  # full Qdrant payload dict — all keys preserved
    }

# ── Paths ────────────────────────────────────────────────────────────────────
EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
TEST_SET_PATH = EVAL_DIR / "test_set.json"

# ── System prompt (same as chain.py — reused for answer generation) ──────────
_SYSTEM_PROMPT = """\
You are AniMind, an expert anime and manga assistant backed by a curated \
database of 1 250+ AniList entries. Answer using the numbered context passages \
provided. Cite sources using [N] notation.

When answering, include ALL relevant factual details available in the context \
for the queried anime: title, year it first aired, format (TV/Movie/OVA), \
AniList score, episode count, studio(s), genres, and a brief plot summary. \
This ensures a comprehensive, fact-rich response.

If the context does not contain the requested anime, say so honestly and do \
not fabricate information.\
"""

# ── Retry config ─────────────────────────────────────────────────────────────
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0   # seconds


# ══════════════════════════════════════════════════════════════════════════════
# Answer generator (shared by both pipelines)
# ══════════════════════════════════════════════════════════════════════════════

async def _generate_answer(
    query: str,
    docs: list[RetrievedDoc],
    oai_client: AsyncOpenAI,
) -> str:
    """Generate an answer using FULL metadata context (score, episodes, studio, etc.).

    Uses _doc_to_eval_context() instead of chain._build_context() so the LLM
    actually has score/episodes/studio fields and can answer factual questions.
    chunk_text = title+synopsis only — missing key metadata → causes refusals.
    """
    # Build numbered passages with all payload fields
    context_lines = [
        f"[{i}] {_doc_to_eval_context(doc)}"
        for i, doc in enumerate(docs, 1)
    ]
    context = "\n\n".join(context_lines)
    user_msg = f"Context passages:\n\n{context}\n\n---\n\nUser question: {query}"
    resp = await oai_client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=1500,
    )
    return resp.choices[0].message.content or ""


# ══════════════════════════════════════════════════════════════════════════════
# Baseline pipeline
# ══════════════════════════════════════════════════════════════════════════════

async def run_baseline(
    question: str,
    oai_client: AsyncOpenAI,
) -> tuple[str, list[str], list[RetrievedDoc]]:
    """Baseline: embed → retrieve top-5 (no filter, no rewrite) → generate.

    Uses top-5 (matching current pipeline doc count) so comparison isolates
    reranker quality rather than doc count difference.
    Returns (answer, contexts, docs) where contexts has all payload fields.
    """
    docs = await retrieve(question, oai_client, top_k=5, filter_kwargs=None)
    answer = await _generate_answer(question, docs, oai_client)
    contexts = [_doc_to_eval_context(doc) for doc in docs]
    return answer, contexts, docs


# ══════════════════════════════════════════════════════════════════════════════
# RAGv1 pipeline  (rewrite → filter → retrieve top-20 → rerank top-5 → generate)
# To add RAGv2: write run_ragv2() below and add it to PIPELINE_REGISTRY.
# ══════════════════════════════════════════════════════════════════════════════

async def run_ragv1(
    question: str,
    oai_client: AsyncOpenAI,
) -> tuple[str, list[str], list[RetrievedDoc]]:
    """RAGv1: rewrite → extract_filters → retrieve(top-20) → rerank(top-5) → generate.

    Returns (answer, contexts, top_docs) where contexts has all payload fields.
    """
    # 1. Rewrite query
    rewritten = await rewrite_query(question, oai_client)

    # 2. Extract metadata filters
    filter_params = await extract_filters(question, oai_client)
    filter_kwargs = filter_params.to_dict() if not filter_params.is_empty() else None

    # 3. Retrieve top-20
    docs = await retrieve(rewritten, oai_client, top_k=settings.retriever_top_k, filter_kwargs=filter_kwargs)

    if not docs:
        return "I couldn't find any relevant anime for your query.", [], []

    # 4. Rerank → top-5
    doc_texts = [doc.chunk_text for doc in docs]
    ranked = await rerank(query=rewritten, documents=doc_texts, top_k=settings.reranker_top_k)
    top_docs = [docs[r["index"]] for r in ranked]

    # 5. Generate
    answer = await _generate_answer(question, top_docs, oai_client)
    contexts = [_doc_to_eval_context(doc) for doc in top_docs]
    return answer, contexts, top_docs


# ── Pipeline registry ─────────────────────────────────────────────────────────
# Map pipeline name → runner coroutine.
# To add RAGv2: implement run_ragv2() above and register it here.
# Output file will automatically be raw_{name}.json.
# ─────────────────────────────────────────────────────────────────────────────
PIPELINE_REGISTRY: dict[str, Any] = {
    "baseline": run_baseline,
    "ragv1":    run_ragv1,
    # "ragv2": run_ragv2,  ← add future versions here
}


# ══════════════════════════════════════════════════════════════════════════════
# Per-question runner with retry + latency tracking
# ══════════════════════════════════════════════════════════════════════════════

async def _run_question(
    item: dict,
    pipeline: str,
    oai_client: AsyncOpenAI,
) -> dict[str, Any]:
    """Run one question through the specified pipeline, with retry on failure."""
    question = item["question"]
    runner = PIPELINE_REGISTRY[pipeline]

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            t0 = time.perf_counter()
            answer, contexts, docs = await runner(question, oai_client)
            latency_ms = int((time.perf_counter() - t0) * 1000)

            return {
                "id": item["id"],
                "question": question,
                "category": item["category"],
                "answer": answer,
                "contexts": contexts,                        # formatted strings for RAGAS/FactScore
                "retrieved_docs": [_doc_to_raw_payload(d) for d in docs],  # full payload for analysis
                "latency_ms": latency_ms,
                "pipeline": pipeline,
                "error": None,
            }

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Question {id} attempt {a}/{m} failed: {e}",
                id=item["id"], a=attempt, m=_MAX_RETRIES, e=exc,
            )
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_BACKOFF * attempt)
            else:
                logger.error("Question {id} failed after {m} attempts", id=item["id"], m=_MAX_RETRIES)
                return {
                    "id": item["id"],
                    "question": question,
                    "category": item["category"],
                    "answer": "",
                    "contexts": [],
                    "latency_ms": 0,
                    "pipeline": pipeline,
                    "error": str(exc),
                }
    # unreachable
    return {}  # pragma: no cover


# ══════════════════════════════════════════════════════════════════════════════
# Collector — sequential with incremental saves
# ══════════════════════════════════════════════════════════════════════════════

async def collect(pipeline: str, limit: int | None = None) -> None:
    """Run pipeline for all questions and save results incrementally."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_set: list[dict] = json.loads(TEST_SET_PATH.read_text())
    if limit:
        test_set = test_set[:limit]
        logger.info("Limited to first {} questions", limit)

    out_path = RESULTS_DIR / f"raw_{pipeline}.json"
    results: list[dict] = []

    # Resume support: load existing results and skip already-done IDs
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        done_ids = {r["id"] for r in existing}
        results = existing
        test_set = [q for q in test_set if q["id"] not in done_ids]
        logger.info("Resuming: {} already done, {} remaining", len(done_ids), len(test_set))
    else:
        done_ids = set()

    oai_client = make_openai_client()
    total = len(test_set)

    logger.info("Starting collection: pipeline={p}, questions={n}", p=pipeline, n=total)
    t_start = time.perf_counter()

    for i, item in enumerate(test_set, 1):
        logger.info("[{i}/{n}] {id}: {q:.60s}", i=i, n=total, id=item["id"], q=item["question"])

        result = await _run_question(item, pipeline, oai_client)
        results.append(result)

        # Incremental save after every question
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))

        if result["error"]:
            logger.warning("[{id}] FAILED: {e}", id=item["id"], e=result["error"])
        else:
            logger.info(
                "[{id}] OK — latency={ms}ms | answer_len={n} | contexts={c}",
                id=item["id"],
                ms=result["latency_ms"],
                n=len(result["answer"]),
                c=len(result["contexts"]),
            )

        # Rate-limit guard between questions
        if i < total:
            await asyncio.sleep(0.5)

    elapsed = time.perf_counter() - t_start
    errors = sum(1 for r in results if r.get("error"))
    latencies = [r["latency_ms"] for r in results if not r.get("error")]
    p95 = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    logger.info(
        "Collection done: pipeline={p} | total={n} | errors={e} | "
        "elapsed={t:.1f}s | p95_latency={p95}ms",
        p=pipeline, n=len(results), e=errors, t=elapsed, p95=p95,
    )
    logger.info("Saved to {}", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _registered = list(PIPELINE_REGISTRY.keys())
    parser = argparse.ArgumentParser(
        description="Collect RAG pipeline outputs for evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Registered pipelines: " + ", ".join(_registered) + "\n"
            "Use 'all' to run every registered pipeline.\n"
            "Output: eval/results/raw_{pipeline}.json"
        ),
    )
    parser.add_argument(
        "--pipeline",
        choices=_registered + ["all"],
        default="all",
        help="Which pipeline to run (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N questions (for smoke testing)",
    )
    args = parser.parse_args()

    pipelines = _registered if args.pipeline == "all" else [args.pipeline]

    for p in pipelines:
        logger.info("=== Pipeline: {} ===", p.upper())
        asyncio.run(collect(p, limit=args.limit))


if __name__ == "__main__":
    main()
