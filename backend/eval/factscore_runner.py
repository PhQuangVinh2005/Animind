"""FActScore evaluation runner — runs in the `factscore` conda environment.

Implements the FActScore algorithm (Min et al., EMNLP 2023):
  1. Decompose each answer into atomic facts via LLM
  2. For each fact: BM25 retrieve top-k passages from SQLite KB
  3. For each fact: LLM verify — "Is this fact supported by these passages?"
  4. FActScore = mean(supported_flags) with optional length penalty (gamma)

Uses old openai SDK (v0.x) with ShopAIKey (openai.api_base).
Must be run in the `factscore` conda environment:
    conda run -n factscore python eval/factscore_runner.py [args]

Usage:
    conda run -n factscore python eval/factscore_runner.py \
        --input  eval/results/raw_baseline.json \
        --output eval/results/factscore_raw_baseline.json \
        --db     eval/factscore_db/anime_kb.db \
        --openai-key  YOUR_KEY \
        --openai-base https://shopaikey-endpoint/v1 \
        --judge-model gpt-4o-mini \
        --gamma 0 \
        --retrieve-k 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
from pathlib import Path

# Load .env from backend/.env (two levels up from eval/)
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if _ENV_FILE.exists():
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE, override=False)  # don't override already-set env vars

import openai  # old SDK: openai<1.0  # noqa: E402
from loguru import logger  # noqa: E402

# ── Prompts (faithful to FActScore paper) ────────────────────────────────────

_DECOMPOSE_SYSTEM = (
    "You are an expert at decomposing text into independent atomic facts. "
    "An atomic fact is the smallest self-contained, verifiable claim. "
    "Each fact must be standalone and include the subject explicitly. "
    "Return ONLY valid JSON."
)

_DECOMPOSE_USER = """\
Decompose the following answer into a list of independent atomic facts.

Rules:
- Each fact must be self-contained (include subject, e.g. "FMA Brotherhood has 64 episodes")
- Include specific numbers, names, dates, titles when present
- Skip opinions, hedging phrases, meta-commentary ("Based on the context...")
- If no verifiable facts exist, return an empty list

Answer:
{answer}

Return format:
{{"facts": ["atomic fact 1", "atomic fact 2", ...]}}
"""

_VERIFY_SYSTEM = (
    "You are a fact-checking assistant. "
    "Determine if a statement is directly supported by the provided context. "
    "Answer ONLY with 'Yes' or 'No'. No explanation."
)

_VERIFY_USER = """\
Context:
{context}

Statement: {statement}

Is the statement directly supported by the context? (Yes/No):"""


# ── SQLite BM25 retrieval ─────────────────────────────────────────────────────

def _retrieve_passages(query: str, conn: sqlite3.Connection, k: int = 5) -> list[str]:
    """BM25 retrieve top-k anime entries matching the query via SQLite FTS5."""
    # Replace non-alphanumeric chars with spaces so "Steins;Gate" → "Steins Gate"
    # isalnum() was too aggressive — it dropped entire tokens like "Steins;Gate"
    clean = re.sub(r"[^\w\s]", " ", query)
    words = [w for w in clean.split() if len(w) > 2]
    if not words:
        return []
    # Deduplicate while preserving order, cap at 20 terms
    seen: set[str] = set()
    unique = [w for w in words if not (w.lower() in seen or seen.add(w.lower()))]  # type: ignore[func-returns-value]
    fts_query = " OR ".join(unique[:20])

    try:
        cursor = conn.execute(
            """
            SELECT d.text
            FROM documents_fts
            JOIN documents d ON d.rowid = documents_fts.rowid
            WHERE documents_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, k),
        )
        return [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError as exc:
        logger.warning("BM25 retrieval failed for query '{}...': {}", query[:40], exc)
        return []


# ── LLM calls (old openai SDK v0.x) ──────────────────────────────────────────

def _chat(messages: list[dict], model: str, max_tokens: int = 512) -> str:
    """Call OpenAI ChatCompletion (old SDK). Retries once on rate limit."""
    for attempt in range(3):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except openai.error.RateLimitError:
            logger.warning("Rate limit hit, sleeping 10 s (attempt {})", attempt + 1)
            time.sleep(10)
        except openai.error.OpenAIError as exc:
            logger.warning("OpenAI error attempt {}: {}", attempt + 1, exc)
            time.sleep(2)
    return ""


def _decompose(answer: str, model: str) -> list[str]:
    """Decompose an answer into atomic facts. Returns [] on failure."""
    raw = _chat(
        messages=[
            {"role": "system", "content": _DECOMPOSE_SYSTEM},
            {"role": "user",   "content": _DECOMPOSE_USER.format(answer=answer)},
        ],
        model=model,
        max_tokens=600,
    )
    try:
        data = json.loads(raw)
        facts = data.get("facts") or []
        return [str(f).strip() for f in facts if str(f).strip()]
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Decompose JSON parse failed for answer: {}...", answer[:60])
        return []


def _verify(fact: str, passages: list[str], model: str) -> bool:
    """Verify one atomic fact against retrieved passages. Returns True if supported."""
    if not passages:
        return False
    context = "\n\n---\n\n".join(passages)[:4000]
    raw = _chat(
        messages=[
            {"role": "system", "content": _VERIFY_SYSTEM},
            {"role": "user",   "content": _VERIFY_USER.format(context=context, statement=fact)},
        ],
        model=model,
        max_tokens=5,
    )
    return raw.strip().lower().startswith("yes")


# ── Length penalty (FActScore paper, eq. 1) ───────────────────────────────────

def _apply_gamma(score: float, n_facts: int, gamma: int) -> float:
    """Apply FActScore length penalty. gamma=0 disables it."""
    if gamma == 0 or n_facts == 0:
        return score
    return score * min(1.0, n_facts / gamma)


# ── Per-question FActScore ────────────────────────────────────────────────────

def _score_one(
    item: dict,
    conn: sqlite3.Connection,
    model: str,
    gamma: int,
    retrieve_k: int,
) -> dict:
    question: str = item["question"]
    answer:   str = item.get("answer") or ""

    if not answer:
        return {
            "id": item["id"], "question": question,
            "category": item.get("category"), "factscore": None,
            "factscore_penalized": None, "n_facts": 0, "n_supported": 0,
            "facts": [], "error": "empty answer",
        }

    # Step 1 — decompose
    facts = _decompose(answer, model)
    if not facts:
        return {
            "id": item["id"], "question": question,
            "category": item.get("category"), "factscore": None,
            "factscore_penalized": None, "n_facts": 0, "n_supported": 0,
            "facts": [], "error": "no atomic facts extracted",
        }

    # Step 2+3 — retrieve + verify each fact
    fact_results: list[dict] = []
    for fact in facts:
        # BM25 retrieval uses (question + fact) for better recall
        retrieval_query = f"{question} {fact}"
        passages = _retrieve_passages(retrieval_query, conn, k=retrieve_k)
        supported = _verify(fact, passages, model)
        fact_results.append({
            "fact": fact,
            "supported": supported,
            "n_passages_retrieved": len(passages),
        })
        time.sleep(0.1)  # light rate-limit guard

    n_total = len(fact_results)
    n_supported = sum(1 for f in fact_results if f["supported"])
    raw_score = n_supported / n_total if n_total > 0 else 0.0
    penalized = _apply_gamma(raw_score, n_total, gamma)

    return {
        "id": item["id"],
        "question": question,
        "category": item.get("category"),
        "factscore": round(raw_score, 4),
        "factscore_penalized": round(penalized, 4),
        "n_facts": n_total,
        "n_supported": n_supported,
        "facts": fact_results,
        "error": None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="FActScore runner (factscore env)")
    parser.add_argument("--input",        required=True,  help="Path to raw_{pipeline}.json")
    parser.add_argument("--output",       required=True,  help="Path to write factscore results JSON")
    parser.add_argument("--db",           required=True,  help="Path to anime_kb.db SQLite file")
    parser.add_argument("--openai-key",   default=None,
                        help="OpenAI / ShopAIKey API key (default: SHOPAIKEY_API_KEY env var)")
    parser.add_argument("--openai-base",  default=None,
                        help="OpenAI API base URL (default: SHOPAIKEY_BASE_URL env var)")
    parser.add_argument("--judge-model",  default="gpt-4o-mini")
    parser.add_argument("--gamma",        type=int, default=0,
                        help="Length penalty gamma (0 = disabled, paper default = 10)")
    parser.add_argument("--retrieve-k",  type=int, default=5,
                        help="Passages to retrieve per atomic fact")
    parser.add_argument("--limit",        type=int, default=None,
                        help="Evaluate first N questions only (for testing)")
    args = parser.parse_args()

    # Resolve credentials: CLI arg → env var → error
    api_key  = args.openai_key  or os.environ.get("SHOPAIKEY_API_KEY",  "")
    api_base = args.openai_base or os.environ.get("SHOPAIKEY_BASE_URL", "")

    if not api_key:
        raise SystemExit("ERROR: No API key found. Set --openai-key or SHOPAIKEY_API_KEY in .env")
    if not api_base or not api_base.startswith("http"):
        raise SystemExit(
            f"ERROR: API base URL invalid: {repr(api_base)}. "
            "Set --openai-base or SHOPAIKEY_BASE_URL in .env"
        )

    openai.api_key  = api_key
    openai.api_base = api_base.rstrip("/")
    logger.info("Using model={} | base={}", args.judge_model, openai.api_base)

    # Load raw results
    raw: list[dict] = json.loads(Path(args.input).read_text())
    valid = [r for r in raw if not r.get("error") and r.get("answer")]
    if args.limit:
        valid = valid[:args.limit]
    logger.info("Loaded {} valid items from {}", len(valid), args.input)

    # Open SQLite
    conn = sqlite3.connect(args.db)
    n_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    logger.info("Opened KB: {} documents in {}", n_docs, args.db)

    # Score each item
    results: list[dict] = []
    for i, item in enumerate(valid, 1):
        logger.info("[{}/{}] {} — {:.50s}", i, len(valid), item["id"], item["question"])
        result = _score_one(item, conn, args.judge_model, args.gamma, args.retrieve_k)
        results.append(result)

        if result["error"]:
            logger.warning("  SKIP: {}", result["error"])
        else:
            logger.info(
                "  facts={} | supported={} | score={:.3f} | penalized={:.3f}",
                result["n_facts"], result["n_supported"],
                result["factscore"] or 0, result["factscore_penalized"] or 0,
            )

        if i < len(valid):
            time.sleep(0.3)

    conn.close()

    # Aggregate
    scored = [r for r in results if r["factscore"] is not None]
    mean_score = sum(r["factscore"] for r in scored) / len(scored) if scored else None
    mean_pen   = sum(r["factscore_penalized"] for r in scored) / len(scored) if scored else None

    from collections import defaultdict  # noqa: PLC0415
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in scored:
        by_cat[r["category"] or "unknown"].append(r["factscore"])
    by_category = {
        cat: {"n": len(scores), "factscore_mean": round(sum(scores)/len(scores), 4)}
        for cat, scores in by_cat.items()
    }

    output = {
        "judge_model": args.judge_model,
        "gamma": args.gamma,
        "retrieve_k": args.retrieve_k,
        "aggregate": {
            "n_evaluated": len(scored),
            "n_skipped": len(results) - len(scored),
            "factscore_mean": round(mean_score, 4) if mean_score is not None else None,
            "factscore_penalized_mean": round(mean_pen, 4) if mean_pen is not None else None,
        },
        "by_category": by_category,
        "per_question": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2))
    logger.info(
        "Done — FActScore={:.3f} (n={}) → {}",
        mean_score or 0, len(scored), args.output,
    )


if __name__ == "__main__":
    main()
