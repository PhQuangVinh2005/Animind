# AGENTS.md

## Project
AniMind — Anime/Manga RAG Chatbot with LangGraph Agent, Qdrant vector search, Qwen3 reranker, and RAGAS evaluation.

## Stack
- Python 3.11 + pip (backend)
- Node.js 20 + pnpm (frontend)
- LangGraph + LangChain (agent framework)
- FastAPI (backend API)
- Next.js 14 + Vercel AI SDK (frontend)
- Qdrant (vector DB, Docker)
- vLLM (Qwen3-Reranker-0.6B, Docker)
- OpenAI API (GPT-4o / GPT-4o-mini, text-embedding-3-small)
- AniList GraphQL API (data source)
- RAGAS (retrieval evaluation)
- Cloudflare Tunnel (backend exposure)
- Vercel (frontend deployment)

## Infrastructure
- Homeserver: RTX 5060Ti 16GB VRAM, R7 5700X, 32GB RAM, Debian 13.4
- VRAM budget: Reranker ~2GB / 16GB
- RAM budget: Qdrant ~200MB + FastAPI ~500MB + vLLM ~2GB / 32GB

## Commands
- Backend dev: `cd backend && uvicorn app.main:app --reload --port 8000`
- Frontend dev: `cd frontend && pnpm dev`
- Docker services: `docker compose up -d` (Qdrant + vLLM reranker)
- Data ingestion: `cd backend && python scripts/ingest.py`
- Fetch data: `cd backend && python scripts/fetch_anilist.py`
- Evaluation: `cd backend && python eval/evaluate.py`
- Tunnel: `cloudflared tunnel run animind`

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
- `backend/eval/` — RAGAS evaluation pipeline
- `frontend/` — Next.js chat UI

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
