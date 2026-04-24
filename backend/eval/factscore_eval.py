"""FActScore subprocess wrapper — runs in the `factscore` conda environment.

Calls factscore_runner.py via `conda run -n factscore` so the evaluation
uses the separate factscore env (Python 3.9 + old openai SDK).

Called from evaluate.py:
    from eval.factscore_eval import run_factscore
    results = run_factscore("baseline")
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from loguru import logger

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))
from app.config import settings  # noqa: E402

EVAL_DIR     = Path(__file__).parent
RESULTS_DIR  = EVAL_DIR / "results"
RAW_DIR      = RESULTS_DIR / "raw"
FACTSCORE_DIR = RESULTS_DIR / "factscore"
DB_PATH      = EVAL_DIR / "factscore_db" / "anime_kb.db"
RUNNER       = EVAL_DIR / "factscore_runner.py"


def run_factscore(
    pipeline: str,
    judge_model: str = "gpt-4o-mini",
    gamma: int = 0,
    retrieve_k: int = 5,
    limit: int | None = None,
    conda_env: str = "factscore",
) -> dict | None:
    """Run FActScore for one pipeline via subprocess in the factscore conda env.

    Args:
        pipeline:    "baseline", "ragv1", or "ragv2"
        judge_model: model for LLM decompose + verify
        gamma:       length penalty (0 = disabled, paper default = 10)
        retrieve_k:  passages retrieved per atomic fact from SQLite BM25
        limit:       evaluate first N questions only (None = all)
        conda_env:   name of the conda env with old openai SDK

    Returns:
        Parsed JSON dict from factscore_runner.py output, or None on failure.
    """
    FACTSCORE_DIR.mkdir(parents=True, exist_ok=True)

    input_path  = RAW_DIR / f"raw_{pipeline}.json"
    output_path = FACTSCORE_DIR / f"factscore_{pipeline}.json"

    # Cache hit — return existing results without re-running
    if output_path.exists():
        logger.info("FActScore [{}] — cache hit, loading {}", pipeline, output_path)
        data: dict = json.loads(output_path.read_text())
        agg = data.get("aggregate", {})
        logger.info(
            "FActScore [{}] — n={} | mean={} (cached)",
            pipeline, agg.get("n_evaluated"), agg.get("factscore_mean"),
        )
        return data

    if not input_path.exists():
        logger.error("Input not found: {} — run collect.py --pipeline {} first", input_path, pipeline)
        return None

    if not DB_PATH.exists():
        logger.error(
            "SQLite KB not found: {} — run build_factscore_db.py first", DB_PATH
        )
        return None

    cmd = [
        "conda", "run", "--no-capture-output", "-n", conda_env,
        "python", str(RUNNER),
        "--input",       str(input_path),
        "--output",      str(output_path),
        "--db",          str(DB_PATH),
        "--openai-key",  settings.shopaikey_api_key,
        "--openai-base", settings.shopaikey_base_url.rstrip("/"),
        "--judge-model", judge_model,
        "--gamma",       str(gamma),
        "--retrieve-k",  str(retrieve_k),
    ]
    if limit:
        cmd += ["--limit", str(limit)]

    logger.info("Running FActScore (pipeline={}, env={}) ...", pipeline, conda_env)
    logger.debug("CMD: {}", " ".join(cmd))

    proc = subprocess.run(
        cmd,
        capture_output=False,   # stream runner logs to terminal
        text=True,
    )

    if proc.returncode != 0:
        logger.error("factscore_runner.py exited with code {}", proc.returncode)
        return None

    if not output_path.exists():
        logger.error("Output file not written: {}", output_path)
        return None

    data: dict = json.loads(output_path.read_text())
    agg = data.get("aggregate", {})
    logger.info(
        "FActScore [{}] — n={} | mean={} | gamma_penalized={}",
        pipeline,
        agg.get("n_evaluated"),
        agg.get("factscore_mean"),
        agg.get("factscore_penalized_mean"),
    )
    return data
