# AniMind — FActScore Evaluation Guide

> **Reference:** Min et al., [FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation](https://arxiv.org/abs/2305.14251), EMNLP 2023.

FActScore measures **factual precision** of RAG-generated answers by:
1. **Decomposing** each answer into independent atomic facts via LLM
2. **Retrieving** supporting passages from a curated knowledge base (BM25/SQLite)
3. **Verifying** each fact against retrieved passages via LLM
4. **Scoring** = `supported_facts / total_facts` (optionally penalized by length)

---

## Environments

| Conda env | Python | Purpose |
|-----------|--------|---------|
| `animind` | 3.11 | Pipeline collection (`collect.py`), KB build, RAGAS |
| `factscore` | 3.9 | FActScore runner — requires `openai<1.0` (old SDK) |

> The two environments are **strictly isolated**. Never mix them.

---

## Eval File Layout

```
backend/eval/
├── test_set.json              # 50 evaluation questions (factual / comparative / recommendation)
├── collect.py                 # Runs RAG pipelines, saves raw_*.json per pipeline
├── factscore_runner.py        # FActScore algorithm — runs in `factscore` env
├── factscore_eval.py          # Subprocess wrapper: calls runner from animind env
├── build_factscore_db.py      # Builds SQLite FTS5 KB from Qdrant (animind env)
├── evaluate.py                # RAGAS metrics + final report (animind env)
│
├── factscore_db/
│   ├── anime_kb.jsonl         # JSONL knowledge source (18 650+ entries)
│   └── anime_kb.db            # SQLite FTS5 — BM25 retrieval for FActScore
│
└── results/
    ├── raw_baseline.json           # Collected pipeline outputs — baseline
    ├── raw_ragv1.json              # Collected pipeline outputs — ragv1
    ├── factscore_baseline_v1.json  # FActScore results — baseline, run v1
    ├── factscore_ragv1_v1.json     # FActScore results — ragv1, run v1
    └── scores_*.json               # RAGAS metric aggregates
```

---

## Pipeline Registry

Pipelines are registered in `collect.py`:

```python
PIPELINE_REGISTRY: dict[str, Any] = {
    "baseline": run_baseline,   # direct retrieve(top-5) → generate
    "ragv1":    run_ragv1,      # rewrite → filter → retrieve(top-20) → rerank(top-5) → generate
    # "ragv2": run_ragv2,       ← add future versions here
}
```

**Adding a new version (e.g. `ragv2`):**
1. Implement `async def run_ragv2(question, oai_client)` in `collect.py`
2. Add `"ragv2": run_ragv2` to `PIPELINE_REGISTRY`
3. Run `--pipeline ragv2` (or `--pipeline all`)

Output files are automatically namespaced: `raw_{pipeline}.json`, `factscore_{pipeline}_{tag}.json`.

---

## Context Format

Each retrieved document is formatted for both answer generation and FActScore verification:

```
DEATH NOTE | Year: 2006 | Format: TV | Score: 8.4/10 | Genres: Mystery, Psychological | Episodes: 37 | Studio: MADHOUSE
Light Yagami is a genius high school student...
```

Key design choices:
- `Year:` is an **explicit label** (not implicit parenthetical) so the LLM can answer "what year did X air?" correctly
- Score is normalized to `X/10` (human-readable) and also stored as `/100` in the KB for FActScore BM25 matching
- Synopsis truncated to 600 chars to stay within context window

---

## Knowledge Base

The SQLite KB is built from all Qdrant points (18 650+ anime entries). Each entry uses structured `field: value` format:

```
title: Fullmetal Alchemist: Brotherhood
english_title: Fullmetal Alchemist: Brotherhood
native_title: 鋼の錬金術師 FULLMETAL ALCHEMIST
year: 2009
format: TV
score: 90/100 (9.0/10)
episodes: 64
genres: Action, Adventure, Drama, Fantasy
studios: bones
status: FINISHED
description: In order for something to be obtained...
```

Rebuild whenever Qdrant data changes:

```bash
cd backend
conda run -n animind python eval/build_factscore_db.py
```

---

## Full Evaluation Run (50 questions)

```bash
cd backend

# Optional: clean previous results for a fresh run
rm -f eval/results/raw_baseline.json eval/results/raw_ragv1.json

# Step 1 — Collect pipeline outputs (~25 min, 50 questions × 2 pipelines)
conda run -n animind python eval/collect.py --pipeline all

# Step 2 — FActScore: baseline
conda run -n factscore python eval/factscore_runner.py \
  --input  eval/results/raw_baseline.json \
  --output eval/results/factscore_baseline_v1.json \
  --db     eval/factscore_db/anime_kb.db \
  --judge-model gpt-4o-mini --gamma 0

# Step 3 — FActScore: ragv1
conda run -n factscore python eval/factscore_runner.py \
  --input  eval/results/raw_ragv1.json \
  --output eval/results/factscore_ragv1_v1.json \
  --db     eval/factscore_db/anime_kb.db \
  --judge-model gpt-4o-mini --gamma 0

# Step 4 — RAGAS metrics + summary report
conda run -n animind python eval/evaluate.py --tag v1
```

---

## Smoke Test (5 questions)

```bash
cd backend

conda run -n animind python eval/collect.py --pipeline all --limit 5

conda run -n factscore python eval/factscore_runner.py \
  --input eval/results/raw_baseline.json \
  --output eval/results/factscore_baseline_smoke.json \
  --db eval/factscore_db/anime_kb.db --judge-model gpt-4o-mini --gamma 0 --limit 5

conda run -n factscore python eval/factscore_runner.py \
  --input eval/results/raw_ragv1.json \
  --output eval/results/factscore_ragv1_smoke.json \
  --db eval/factscore_db/anime_kb.db --judge-model gpt-4o-mini --gamma 0 --limit 5
```

---

## CLI Reference

### `collect.py`

```
usage: collect.py [--pipeline {baseline,ragv1,all}] [--limit N]

--pipeline   Which pipeline(s) to run (default: all)
--limit      Limit to first N questions (smoke testing)

Output: eval/results/raw_{pipeline}.json
```

### `factscore_runner.py`

```
usage: factscore_runner.py --input PATH --output PATH --db PATH [options]

--input        Path to raw_{pipeline}.json
--output       Path to write factscore JSON results
--db           Path to anime_kb.db SQLite file
--openai-key   API key (default: SHOPAIKEY_API_KEY env var)
--openai-base  API base URL (default: SHOPAIKEY_BASE_URL env var)
--judge-model  LLM for decompose + verify (default: gpt-4o-mini)
--gamma        Length penalty, 0 = disabled (default: 0)
--retrieve-k   BM25 passages per atomic fact (default: 5)
--limit        Evaluate first N items only
```

### `build_factscore_db.py`

```
# No CLI args — reads from Qdrant, writes to eval/factscore_db/
conda run -n animind python eval/build_factscore_db.py
```

---

## Output Schema

`factscore_{pipeline}_{tag}.json`:

```json
{
  "judge_model": "gpt-4o-mini",
  "gamma": 0,
  "retrieve_k": 5,
  "aggregate": {
    "n_evaluated": 45,
    "n_skipped": 5,
    "factscore_mean": 0.923,
    "factscore_penalized_mean": 0.923
  },
  "by_category": {
    "factual":        {"n": 20, "factscore_mean": 0.951},
    "comparative":    {"n": 15, "factscore_mean": 0.889},
    "recommendation": {"n": 10, "factscore_mean": 0.912}
  },
  "per_question": [
    {
      "id": "q001",
      "question": "How many episodes does Fullmetal Alchemist: Brotherhood have?",
      "category": "factual",
      "factscore": 1.0,
      "factscore_penalized": 1.0,
      "n_facts": 6,
      "n_supported": 6,
      "facts": [
        {"fact": "FMA Brotherhood has 64 episodes", "supported": true, "n_passages_retrieved": 5}
      ]
    }
  ]
}
```

A question is **skipped** (`factscore: null`) when:
- The answer is empty (pipeline error)
- The LLM decomposer finds no verifiable atomic facts (e.g. refusal answers like "The context does not contain…")

---

## Known Issues & Findings

| Issue | Root Cause | Status |
|-------|-----------|--------|
| `ragv1` q004/q005 retrieval failure | Reranker returns wrong docs for some queries | Known — tracked for ragv2 |
| Event loop warnings on shutdown | `httpx.AsyncClient` closes after asyncio loop ends | Cosmetic — no functional impact |
| Short answers → few facts | LLM given comprehensive system prompt + `max_tokens=1500` | Fixed |
| `Year:` not extracted from `(2006, TV)` | Implicit parenthetical not recognized as air year | Fixed — explicit `Year:` label |
| `Steins;Gate` BM25 miss | `;` was stripped, breaking FTS5 query | Fixed — `re.sub(r'[^\w\s]', ' ')` |

---

## FActScore Citation

```bibtex
@inproceedings{factscore,
  title   = {{FActScore}: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation},
  author  = {Min, Sewon and Krishna, Kalpesh and Lyu, Xinxi and Lewis, Mike and Yih, Wen-tau and Koh, Pang Wei and Iyyer, Mohit and Zettlemoyer, Luke and Hajishirzi, Hannaneh},
  year    = {2023},
  booktitle = {EMNLP},
  url     = {https://arxiv.org/abs/2305.14251}
}
```
