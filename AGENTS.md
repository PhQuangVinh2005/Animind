# AGENTS.md

## Project
AniMind — Anime/Manga RAG Chatbot with LangGraph Agent, Qdrant hybrid search (dense + BM25), Qwen3 reranker, self-correcting retrieval, FActScore + RAGAS evaluation.

## Stack
- Python 3.11 + pip (backend)
- Node.js 20 + npm (frontend)
- LangGraph + LangChain (agent framework)
- FastAPI (backend API)
- Next.js 14 (frontend, self-hosted)
- Qdrant (vector DB, Docker)
- vLLM (Qwen3-Reranker-0.6B, Docker)
- OpenAI API (GPT-4o / GPT-4o-mini, text-embedding-3-small)
- AniList GraphQL API (data source)
- FActScore (factual precision evaluation, custom SQLite BM25 KB)
- RAGAS (retrieval quality evaluation)
- Cloudflare Tunnel (backend + frontend exposure)

## Infrastructure
- Homeserver: RTX 5060Ti 16GB VRAM, R7 5700X, 32GB RAM, Debian 13.4
- VRAM budget: Reranker ~2GB / 16GB
- RAM budget: Qdrant ~200MB + FastAPI ~500MB + vLLM ~2GB + Next.js ~200MB / 32GB

## Commands
- Backend dev: `cd backend && uvicorn app.main:app --reload --port 8000`
- Frontend dev: `cd frontend && npm run dev`
- Docker services: `docker compose up -d` (Qdrant + vLLM reranker)
- Data ingestion: `cd backend && python scripts/ingest.py`  *(dual-vector: dense + BM25)*
- Fetch data: `cd backend && python scripts/fetch_anilist.py`
- Collect eval: `cd backend && conda run -n animind python eval/collect.py --pipeline all`
- FActScore eval: `conda run -n factscore python eval/factscore_runner.py --input eval/results/raw_baseline.json --output eval/results/factscore_baseline_v1.json --db eval/factscore_db/anime_kb.db --judge-model gpt-4o-mini --gamma 0`
- RAGAS eval: `cd backend && conda run -n animind python eval/evaluate.py --tag v1`
- Rebuild KB: `cd backend && conda run -n animind python eval/build_factscore_db.py`
- Tunnel: `cloudflared tunnel run animind`

## Conda Environments
- `animind` — main backend env (Python 3.11, FastAPI, LangGraph, RAGAS, collect.py)
- `factscore` — isolated eval env (Python 3.9, openai<1.0) — for `factscore_runner.py` ONLY

## Conventions
- Backend: Python, type hints everywhere, async where possible
- Frontend: TypeScript strict, single quotes, 2-space indent
- API calls use `httpx.AsyncClient`, NOT `requests`
- LangGraph nodes are pure functions taking `AgentState` and returning partial state
- All config via environment variables in `.env`, NEVER hardcode API keys
- Logging via `loguru`, NOT `print()` or `logging`

## Architecture
- `backend/app/agent/` — LangGraph graph, nodes, tools, state
- `backend/app/rag/` — retriever, reranker client, RAG chain
- `backend/app/api/` — FastAPI routes (REST + SSE)
- `backend/scripts/` — data fetching and ingestion
- `backend/eval/` — FActScore + RAGAS evaluation pipeline (see `backend/eval/README.md`)
- `frontend/` — Next.js chat UI (self-hosted on homeserver)

## Eval Pipeline Registry
```python
# backend/eval/collect.py — PIPELINE_REGISTRY
"baseline" → run_baseline   # direct retrieve(top-5) → generate
"ragv1"    → run_ragv1      # rewrite → filter → retrieve(top-20) → rerank(top-5) → generate
# "ragv2"  → run_ragv2      ← add future RAG versions here
```

## Graph Topology (v3)
```
[START] → router
  ├─ "qa"     → rag → reranker → relevance_gate → (synthesizer | retry rag, max 1)
  ├─ "search" → tool → synthesizer
  └─ "detail" → tool → synthesizer → [END]
```

Relevance gate threshold: 0.4 (top reranker score). On retry, rag_node drops filters and query rewrite for maximum recall.

## Key APIs
- Qdrant: `localhost:6333`
- vLLM Reranker: `localhost:8001/v1/rerank`
- FastAPI Backend: `localhost:8000`
- AniList GraphQL: `https://graphql.anilist.co`

## DON'T
- KHÔNG commit `.env` — dùng `.env.example`
- KHÔNG dùng `requests` library — dùng `httpx`
- KHÔNG dùng `print()` — dùng `loguru`
- KHÔNG sửa `data/raw/` sau khi fetch xong — đây là immutable raw data
- KHÔNG hardcode model names — đặt trong config
- KHÔNG chạy `factscore_runner.py` trong `animind` env — dùng `factscore` env (openai<1.0)
- KHÔNG mix animind/factscore environments trong cùng 1 subprocess call
