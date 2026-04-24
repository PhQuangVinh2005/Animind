"""Evaluation orchestrator — loads cached pipeline outputs, runs all metrics, saves report.

Workflow
--------
1. collect.py  → generates raw/raw_{pipeline}.json  (run separately)
2. evaluate.py → loads raw files, runs RAGAS + FactScore, writes reports

Tag semantics: --tag is the pipeline name (ragv1, ragv2, ...).
Each report compares baseline (ragv0) vs the specified pipeline.
Baseline scores are computed once and cached in reports/scores_baseline.json.

Output files
------------
results/reports/scores_{tag}.json   — full per-question scores (baseline + pipeline)
results/reports/report_{tag}.md     — human-readable comparison table

Usage
-----
    cd backend

    # Evaluate ragv1 pipeline (compares vs cached baseline)
    conda run -n animind python eval/evaluate.py --tag ragv1

    # Evaluate ragv2 pipeline
    conda run -n animind python eval/evaluate.py --tag ragv2

    # Force re-evaluate baseline (clears cache)
    conda run -n animind python eval/evaluate.py --tag ragv1 --rerun-baseline
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.factscore_eval import run_factscore
from eval.ragas_eval import run_ragas_eval

EVAL_DIR    = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
RAW_DIR     = RESULTS_DIR / "raw"
REPORTS_DIR = RESULTS_DIR / "reports"

# Baseline cache — computed once, reused across all pipeline evaluations
BASELINE_CACHE = REPORTS_DIR / "scores_baseline.json"

# ── Pipeline config metadata (for report generation) ─────────────────────────
PIPELINE_CONFIGS: dict[str, dict[str, str]] = {
    "baseline": {
        "Query rewrite": "❌",
        "Metadata filter": "❌",
        "Retrieval": "Top-5 dense only",
        "Reranker": "❌",
        "Self-correction": "❌",
        "Docs used for answer": "5",
    },
    "ragv1": {
        "Query rewrite": "✅ GPT-4o-mini",
        "Metadata filter": "✅ Auto-extracted",
        "Retrieval": "Top-20 hybrid (dense + BM25 RRF)",
        "Reranker": "✅ Qwen3-Reranker → top-5",
        "Self-correction": "❌",
        "Docs used for answer": "5 (reranked)",
    },
    "ragv2": {
        "Query rewrite": "✅ GPT-4o-mini",
        "Metadata filter": "✅ Auto-extracted",
        "Retrieval": "Top-20 hybrid (dense + BM25 RRF)",
        "Reranker": "✅ Qwen3-Reranker → top-5",
        "Self-correction": "✅ Threshold=0.4, retry without filters",
        "Docs used for answer": "5 (reranked)",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Report generator
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(val: float | None, decimals: int = 3) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def _generate_report(
    baseline_scores: dict,
    current_scores: dict,
    tag: str,
    judge_model: str,
) -> str:
    """Generate a Markdown comparison report: baseline vs {tag}."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# AniMind Evaluation Report — `{tag}` vs baseline",
        f"",
        f"**Generated:** {now}  ",
        f"**Judge model:** `{judge_model}`  ",
        f"**Strategy:** Reference-free (Strategy A) — no ground truth required  ",
        f"",
        "## Overall Scores",
        "",
        f"| Metric | Baseline | {tag} | Delta |",
        "|---|---|---|---|",
    ]

    def delta_str(b: float | None, c: float | None) -> str:
        if b is None or c is None:
            return "N/A"
        d = c - b
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.3f}"

    # RAGAS metrics
    b_ragas = baseline_scores.get("ragas", {}).get("aggregate", {})
    c_ragas = current_scores.get("ragas", {}).get("aggregate", {})
    b_faith = b_ragas.get("faithfulness_mean")
    c_faith = c_ragas.get("faithfulness_mean")
    b_relev = b_ragas.get("answer_relevancy_mean")
    c_relev = c_ragas.get("answer_relevancy_mean")

    # FactScore metrics
    b_fs = baseline_scores.get("factscore", {}).get("aggregate", {})
    c_fs = current_scores.get("factscore", {}).get("aggregate", {})
    b_fact = b_fs.get("factscore_mean")
    c_fact = c_fs.get("factscore_mean")

    lines += [
        f"| **Faithfulness** (RAGAS) | {_fmt(b_faith)} | {_fmt(c_faith)} | {delta_str(b_faith, c_faith)} |",
        f"| **Answer Relevancy** (RAGAS) | {_fmt(b_relev)} | {_fmt(c_relev)} | {delta_str(b_relev, c_relev)} |",
        f"| **FactScore** (custom) | {_fmt(b_fact)} | {_fmt(c_fact)} | {delta_str(b_fact, c_fact)} |",
        "",
        "> **Targets:** Faithfulness ≥ 0.80 | Answer Relevancy ≥ 0.80 | FactScore ≥ 0.75",
        "",
    ]

    # Category breakdown — Faithfulness
    lines += ["## Faithfulness by Category", ""]
    lines += [f"| Category | Baseline | {tag} | Delta |", "|---|---|---|---|"]

    b_cats = baseline_scores.get("ragas", {}).get("by_category", {})
    c_cats = current_scores.get("ragas", {}).get("by_category", {})
    all_cats = sorted(set(list(b_cats.keys()) + list(c_cats.keys())))
    for cat in all_cats:
        bv = b_cats.get(cat, {}).get("faithfulness_mean")
        cv = c_cats.get(cat, {}).get("faithfulness_mean")
        lines.append(f"| {cat} | {_fmt(bv)} | {_fmt(cv)} | {delta_str(bv, cv)} |")

    # Category breakdown — FactScore
    lines += ["", "## FactScore by Category", ""]
    lines += [f"| Category | Baseline | {tag} | Delta |", "|---|---|---|---|"]

    b_fscats = baseline_scores.get("factscore", {}).get("by_category", {})
    c_fscats = current_scores.get("factscore", {}).get("by_category", {})
    all_cats_fs = sorted(set(list(b_fscats.keys()) + list(c_fscats.keys())))
    for cat in all_cats_fs:
        bv = b_fscats.get(cat, {}).get("factscore_mean")
        cv = c_fscats.get(cat, {}).get("factscore_mean")
        lines.append(f"| {cat} | {_fmt(bv)} | {_fmt(cv)} | {delta_str(bv, cv)} |")

    # Pipeline config comparison
    baseline_cfg = PIPELINE_CONFIGS.get("baseline", {})
    current_cfg = PIPELINE_CONFIGS.get(tag, {})
    if baseline_cfg or current_cfg:
        lines += [
            "",
            "## Pipeline Configurations",
            "",
            f"| Setting | Baseline | {tag} |",
            "|---|---|---|",
        ]
        all_keys = list(dict.fromkeys(list(baseline_cfg.keys()) + list(current_cfg.keys())))
        for key in all_keys:
            lines.append(f"| {key} | {baseline_cfg.get(key, '—')} | {current_cfg.get(key, '—')} |")

    lines += [
        "",
        "## Notes",
        "",
        f"- Re-run `evaluate.py --tag {tag}` after each RAG change.",
        f"- Sample counts: baseline n={b_ragas.get('n', '?')} | {tag} n={c_ragas.get('n', '?')}",
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Baseline score loader / computer
# ══════════════════════════════════════════════════════════════════════════════

async def _get_baseline_scores(
    judge_model: str,
    force_rerun: bool = False,
) -> dict:
    """Load cached baseline scores or compute + cache them.

    Baseline never changes, so we compute RAGAS + FactScore once and reuse.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if BASELINE_CACHE.exists() and not force_rerun:
        cached = json.loads(BASELINE_CACHE.read_text())
        logger.info("Loaded cached baseline scores from {}", BASELINE_CACHE)
        return cached

    logger.info("Computing baseline scores (will be cached for future runs)...")

    baseline_path = RAW_DIR / "raw_baseline.json"
    if not baseline_path.exists():
        logger.error("Missing {}. Run: collect.py --pipeline baseline", baseline_path)
        raise FileNotFoundError(str(baseline_path))

    baseline_raw: list[dict] = json.loads(baseline_path.read_text())

    # RAGAS
    logger.info("=== Running RAGAS eval on baseline ===")
    baseline_ragas = await run_ragas_eval(baseline_raw, judge_model=judge_model)

    # FactScore
    logger.info("=== Running FactScore eval on baseline ===")
    baseline_fs = run_factscore("baseline", judge_model=judge_model)

    baseline_scores = {"ragas": baseline_ragas, "factscore": baseline_fs}

    # Cache for reuse
    BASELINE_CACHE.write_text(json.dumps(baseline_scores, ensure_ascii=False, indent=2))
    logger.info("Baseline scores cached → {}", BASELINE_CACHE)

    return baseline_scores


# ══════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ══════════════════════════════════════════════════════════════════════════════

async def evaluate(
    tag: str = "ragv1",
    judge_model: str = "gpt-4o-mini",
    rerun_baseline: bool = False,
) -> None:
    """Run evaluation for a pipeline (tag) and compare vs baseline.

    Args:
        tag:            Pipeline name ("ragv1", "ragv2", ...).
        judge_model:    LLM model for RAGAS + FactScore judge.
        rerun_baseline: Force re-compute baseline scores (clear cache).
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Get baseline scores (cached or compute once)
    baseline_scores = await _get_baseline_scores(judge_model, force_rerun=rerun_baseline)

    # 2. Load current pipeline raw data
    current_path = RAW_DIR / f"raw_{tag}.json"
    if not current_path.exists():
        logger.error("Missing {}. Run: collect.py --pipeline {}", current_path, tag)
        return

    current_raw: list[dict] = json.loads(current_path.read_text())
    logger.info("Loaded: {} — {} samples", tag, len(current_raw))

    # 3. Run RAGAS on current pipeline
    logger.info("=== Running RAGAS eval on {} ===", tag)
    current_ragas = await run_ragas_eval(current_raw, judge_model=judge_model)

    # 4. Run FactScore on current pipeline
    logger.info("=== Running FactScore eval on {} ===", tag)
    current_fs = run_factscore(tag, judge_model=judge_model)

    # 5. Combine scores
    current_scores = {"ragas": current_ragas, "factscore": current_fs}

    scores_out = {
        "tag": tag,
        "judge_model": judge_model,
        "baseline": baseline_scores,
        tag: current_scores,
    }

    scores_path = REPORTS_DIR / f"scores_{tag}.json"
    scores_path.write_text(json.dumps(scores_out, ensure_ascii=False, indent=2))
    logger.info("Scores saved → {}", scores_path)

    # 6. Generate Markdown report
    report = _generate_report(baseline_scores, current_scores, tag, judge_model)
    report_path = REPORTS_DIR / f"report_{tag}.md"
    report_path.write_text(report)
    logger.info("Report saved → {}", report_path)

    # 7. Print summary
    b_agg = baseline_scores.get("ragas", {}).get("aggregate", {})
    c_agg = current_scores.get("ragas", {}).get("aggregate", {})
    b_fs_agg = (baseline_scores.get("factscore") or {}).get("aggregate", {})
    c_fs_agg = (current_scores.get("factscore") or {}).get("aggregate", {})

    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY — baseline vs {tag}")
    print("=" * 60)
    print(f"{'Metric':<30} {'Baseline':>10} {tag:>10}")
    print("-" * 60)
    print(f"{'Faithfulness (RAGAS)':<30} {_fmt(b_agg.get('faithfulness_mean')):>10} {_fmt(c_agg.get('faithfulness_mean')):>10}")
    print(f"{'Answer Relevancy (RAGAS)':<30} {_fmt(b_agg.get('answer_relevancy_mean')):>10} {_fmt(c_agg.get('answer_relevancy_mean')):>10}")
    print(f"{'FactScore (custom)':<30} {_fmt(b_fs_agg.get('factscore_mean')):>10} {_fmt(c_fs_agg.get('factscore_mean')):>10}")
    print("=" * 60)
    print(f"\nFull report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AniMind RAG evaluation orchestrator")
    parser.add_argument(
        "--tag",
        default="ragv1",
        help="Pipeline name to evaluate: ragv1, ragv2, ... (default: ragv1)",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="LLM judge model (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--rerun-baseline",
        action="store_true",
        help="Force re-compute baseline scores (clears cache)",
    )
    args = parser.parse_args()
    asyncio.run(evaluate(tag=args.tag, judge_model=args.judge_model, rerun_baseline=args.rerun_baseline))


if __name__ == "__main__":
    main()
