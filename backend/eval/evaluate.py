"""Evaluation orchestrator — loads cached pipeline outputs, runs all metrics, saves report.

Workflow
--------
1. collect.py  → generates raw_baseline.json + raw_ragv1.json  (run separately)
2. evaluate.py → loads those files, runs RAGAS + FactScore, writes scores + report

This separation means you can tweak metric prompts and re-score without
re-running the expensive RAG pipeline API calls.

Output files
------------
eval/results/scores_<tag>.json   — full per-question scores for both pipelines
eval/results/report_<tag>.md     — human-readable comparison table

Usage
-----
    cd backend

    # Full evaluation
    conda run -n animind python eval/evaluate.py

    # With a custom tag (for tracking improvement versions)
    conda run -n animind python eval/evaluate.py --tag v2_chunking

    # Use a different judge model
    conda run -n animind python eval/evaluate.py --judge-model gpt-4o
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

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"


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
    """Generate a Markdown comparison report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# AniMind Evaluation Report — `{tag}`",
        f"",
        f"**Generated:** {now}  ",
        f"**Judge model:** `{judge_model}`  ",
        f"**Strategy:** Reference-free (Strategy A) — no ground truth required  ",
        f"",
        "## Overall Scores",
        "",
        "| Metric | Baseline | Current | Delta |",
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

    # Category breakdown
    lines += ["## Faithfulness by Category", ""]
    lines += ["| Category | Baseline | Current | Delta |", "|---|---|---|---|"]

    b_cats = baseline_scores.get("ragas", {}).get("by_category", {})
    c_cats = current_scores.get("ragas", {}).get("by_category", {})
    all_cats = sorted(set(list(b_cats.keys()) + list(c_cats.keys())))
    for cat in all_cats:
        bv = b_cats.get(cat, {}).get("faithfulness_mean")
        cv = c_cats.get(cat, {}).get("faithfulness_mean")
        lines.append(f"| {cat} | {_fmt(bv)} | {_fmt(cv)} | {delta_str(bv, cv)} |")

    lines += ["", "## FactScore by Category", ""]
    lines += ["| Category | Baseline | Current | Delta |", "|---|---|---|---|"]

    b_fscats = baseline_scores.get("factscore", {}).get("by_category", {})
    c_fscats = current_scores.get("factscore", {}).get("by_category", {})
    all_cats_fs = sorted(set(list(b_fscats.keys()) + list(c_fscats.keys())))
    for cat in all_cats_fs:
        bv = b_fscats.get(cat, {}).get("factscore_mean")
        cv = c_fscats.get(cat, {}).get("factscore_mean")
        lines.append(f"| {cat} | {_fmt(bv)} | {_fmt(cv)} | {delta_str(bv, cv)} |")

    # Pipeline config notes
    lines += [
        "",
        "## Pipeline Configurations",
        "",
        "| Setting | Baseline | Current |",
        "|---|---|---|",
        "| Query rewrite | ❌ | ✅ GPT-4o-mini |",
        "| Metadata filter | ❌ | ✅ GPT-4o-mini extraction |",
        "| Retrieval top-k | 20 | 20 |",
        "| Reranker | ❌ | ✅ Qwen3-Reranker-0.6B → top-5 |",
        "| Docs used for answer | 20 | 5 (reranked) |",
        "",
        "## Notes",
        "",
        "- Improvement implementation will be added after baseline is established.",
        "- Re-run `evaluate.py` with `--tag vN_description` after each RAG change.",
        f"- Sample counts: baseline n={b_ragas.get('n', '?')} | current n={c_ragas.get('n', '?')}",
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ══════════════════════════════════════════════════════════════════════════════

async def evaluate(tag: str = "v1", judge_model: str = "gpt-4o-mini") -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline_path = RESULTS_DIR / "raw_baseline.json"
    ragv1_path   = RESULTS_DIR / "raw_ragv1.json"

    if not baseline_path.exists():
        logger.error("Missing {}. Run collect.py --pipeline baseline first.", baseline_path)
        return
    if not ragv1_path.exists():
        logger.error("Missing {}. Run collect.py --pipeline ragv1 first.", ragv1_path)
        return

    baseline_raw: list[dict] = json.loads(baseline_path.read_text())
    ragv1_raw: list[dict] = json.loads(ragv1_path.read_text())

    logger.info("Loaded: baseline={} | ragv1={} samples", len(baseline_raw), len(ragv1_raw))

    # ── RAGAS ────────────────────────────────────────────────────────────────
    logger.info("=== Running RAGAS eval on baseline ===")
    baseline_ragas = await run_ragas_eval(baseline_raw, judge_model=judge_model)

    logger.info("=== Running RAGAS eval on ragv1 ===")
    ragv1_ragas = await run_ragas_eval(ragv1_raw, judge_model=judge_model)

    # ── FactScore ─────────────────────────────────────────────────────────────
    logger.info("=== Running FactScore eval on baseline ===")
    baseline_fs = run_factscore("baseline", tag=tag, judge_model=judge_model)

    logger.info("=== Running FactScore eval on ragv1 ===")
    ragv1_fs = run_factscore("ragv1", tag=tag, judge_model=judge_model)

    # ── Combine scores ────────────────────────────────────────────────────────
    baseline_scores = {"ragas": baseline_ragas, "factscore": baseline_fs}
    ragv1_scores    = {"ragas": ragv1_ragas,    "factscore": ragv1_fs}

    scores_out = {
        "tag": tag,
        "judge_model": judge_model,
        "baseline": baseline_scores,
        "ragv1": ragv1_scores,
    }

    scores_path = RESULTS_DIR / f"scores_{tag}.json"
    scores_path.write_text(json.dumps(scores_out, ensure_ascii=False, indent=2))
    logger.info("Scores saved → {}", scores_path)

    # ── Markdown report ───────────────────────────────────────────────────────
    report = _generate_report(baseline_scores, ragv1_scores, tag, judge_model)
    report_path = RESULTS_DIR / f"report_{tag}.md"
    report_path.write_text(report)
    logger.info("Report saved → {}", report_path)

    # Print summary to stdout
    b_agg = baseline_scores["ragas"].get("aggregate", {})
    c_agg = ragv1_scores["ragas"].get("aggregate", {})
    b_fs_agg = (baseline_scores.get("factscore") or {}).get("aggregate", {})
    c_fs_agg = (ragv1_scores.get("factscore") or {}).get("aggregate", {})

    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY — tag={tag}")
    print("=" * 60)
    print(f"{'Metric':<30} {'Baseline':>10} {'Current':>10}")
    print("-" * 60)
    print(f"{'Faithfulness (RAGAS)':<30} {_fmt(b_agg.get('faithfulness_mean')):>10} {_fmt(c_agg.get('faithfulness_mean')):>10}")
    print(f"{'Answer Relevancy (RAGAS)':<30} {_fmt(b_agg.get('answer_relevancy_mean')):>10} {_fmt(c_agg.get('answer_relevancy_mean')):>10}")
    print(f"{'FactScore (custom)':<30} {_fmt(b_fs_agg.get('factscore_mean')):>10} {_fmt(c_fs_agg.get('factscore_mean')):>10}")
    print("=" * 60)
    print(f"\nFull report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AniMind RAG evaluation orchestrator")
    parser.add_argument("--tag", default="v1", help="Version tag for output files (default: v1)")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="LLM judge model")
    args = parser.parse_args()
    asyncio.run(evaluate(tag=args.tag, judge_model=args.judge_model))


if __name__ == "__main__":
    main()
