"""RAGAS evaluation module — faithfulness + answer relevancy (reference-free).

Metrics
-------
Faithfulness     : Are all claims in the answer supported by retrieved context?
                   Internally decomposes answer → atomic statements → verifies each.
AnswerRelevancy  : Does the answer actually address the question?
                   Internally generates questions from answer → semantic similarity.

Both metrics require NO ground truth — purely reference-free.

RAGAS LLM configuration
------------------------
Uses langchain_openai.ChatOpenAI and OpenAIEmbeddings pointed at ShopAIKey
(OpenAI-compatible endpoint) so no separate OpenAI key is needed.

RAGAS v0.2 API note
--------------------
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.metrics import Faithfulness, AnswerRelevancy
    evaluate(dataset, metrics=[...], llm=..., embeddings=...)
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger

# RAGAS v0.2 imports — try both known locations
try:
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
except ImportError:
    from ragas import EvaluationDataset, SingleTurnSample  # type: ignore[no-redef, attr-defined]

from ragas import evaluate as ragas_evaluate
from ragas.metrics import AnswerRelevancy, Faithfulness

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings


def _make_ragas_llm(judge_model: str) -> ChatOpenAI:
    """Create a langchain ChatOpenAI client pointed at ShopAIKey."""
    return ChatOpenAI(
        model=judge_model,
        api_key=settings.shopaikey_api_key,
        base_url=settings.shopaikey_base_url,
        temperature=0.0,
        max_retries=3,
    )


def _make_ragas_embeddings() -> OpenAIEmbeddings:
    """Create OpenAIEmbeddings client pointed at ShopAIKey."""
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.shopaikey_api_key,
        base_url=settings.shopaikey_base_url,
    )


def _build_dataset(raw_results: list[dict[str, Any]]) -> EvaluationDataset:
    """Convert raw pipeline results to a RAGAS EvaluationDataset."""
    samples = []
    for r in raw_results:
        if r.get("error") or not r.get("answer") or not r.get("contexts"):
            logger.warning("Skipping {} — missing answer or contexts", r.get("id"))
            continue
        samples.append(
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["contexts"],
            )
        )
    return EvaluationDataset(samples=samples)


async def run_ragas_eval(
    raw_results: list[dict[str, Any]],
    judge_model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Run RAGAS faithfulness + answer relevancy on raw pipeline results.

    Args:
        raw_results: List of dicts from collect.py (question, answer, contexts).
        judge_model: LLM model name for RAGAS judge (default gpt-4o-mini).

    Returns:
        Dict with per-question scores and aggregate statistics.
    """
    valid = [r for r in raw_results if not r.get("error") and r.get("answer") and r.get("contexts")]
    logger.info("RAGAS eval: {} valid samples (skipped {} errors)", len(valid), len(raw_results) - len(valid))

    if not valid:
        logger.error("No valid samples for RAGAS eval")
        return {"error": "no valid samples", "scores": []}

    dataset = _build_dataset(valid)
    llm = _make_ragas_llm(judge_model)
    embeddings = _make_ragas_embeddings()

    faithfulness_metric = Faithfulness(llm=llm)
    relevancy_metric = AnswerRelevancy(llm=llm, embeddings=embeddings)

    logger.info("Running RAGAS evaluate (faithfulness + answer_relevancy)...")

    # RAGAS evaluate is synchronous internally — run in thread pool to not
    # block the event loop if called from an async context.
    result = await asyncio.to_thread(
        ragas_evaluate,
        dataset=dataset,
        metrics=[faithfulness_metric, relevancy_metric],
    )

    # Extract per-sample scores — replace NaN with 0.0 (LLM judge parse failures)
    import math
    scores_df = result.to_pandas()
    per_question: list[dict] = []
    for i, row in scores_df.iterrows():
        faith_val = float(row.get("faithfulness", 0.0))
        relev_val = float(row.get("answer_relevancy", 0.0))
        # NaN from RAGAS parse failures → treat as 0.0 for aggregation
        if math.isnan(faith_val):
            faith_val = 0.0
        if math.isnan(relev_val):
            relev_val = 0.0
        per_question.append({
            "id": valid[i]["id"],
            "question": valid[i]["question"],
            "category": valid[i].get("category", "unknown"),
            "faithfulness": faith_val,
            "answer_relevancy": relev_val,
        })

    # Aggregate by category
    from collections import defaultdict
    by_category: dict[str, list] = defaultdict(list)
    for q in per_question:
        by_category[q["category"]].append(q)

    category_stats: dict[str, dict] = {}
    for cat, items in by_category.items():
        category_stats[cat] = {
            "n": len(items),
            "faithfulness_mean": round(sum(q["faithfulness"] for q in items) / len(items), 4),
            "answer_relevancy_mean": round(sum(q["answer_relevancy"] for q in items) / len(items), 4),
        }

    all_faith = [q["faithfulness"] for q in per_question]
    all_relev = [q["answer_relevancy"] for q in per_question]

    aggregate = {
        "n": len(per_question),
        "faithfulness_mean": round(sum(all_faith) / len(all_faith), 4) if all_faith else 0.0,
        "answer_relevancy_mean": round(sum(all_relev) / len(all_relev), 4) if all_relev else 0.0,
    }

    logger.info(
        "RAGAS done — faithfulness={f:.3f} | answer_relevancy={r:.3f}",
        f=aggregate["faithfulness_mean"],
        r=aggregate["answer_relevancy_mean"],
    )

    return {
        "judge_model": judge_model,
        "aggregate": aggregate,
        "by_category": category_stats,
        "per_question": per_question,
    }
