# AniMind — Anime/Manga RAG Chatbot

An intelligent anime/manga chatbot powered by LangGraph Agent, Qdrant vector search, Qwen3 reranker, and OpenAI GPT-4o.

## Features

- **RAG-powered Q&A** — Ask questions about anime/manga, get answers grounded in real data
- **Tool-calling** — Live search and detail lookup via AniList GraphQL API
- **Reranking** — Qwen3-Reranker-0.6B for improved retrieval quality
- **Streaming** — Real-time token streaming via SSE
- **Evaluation** — RAGAS metrics for retrieval quality assessment

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd animind

# 2. Copy env file
cp .env.example .env
# Edit .env with your OpenAI API key

# 3. Start infrastructure
docker compose up -d

# 4. Ingest data
cd backend
pip install -r requirements.txt
python scripts/fetch_anilist.py
python scripts/ingest.py

# 5. Start backend
uvicorn app.main:app --reload --port 8000

# 6. Start frontend
cd ../frontend
pnpm install
pnpm dev
```

## Architecture

```
Vercel (Next.js) → Cloudflare Tunnel → Homeserver
                                         ├── FastAPI + LangGraph Agent
                                         ├── Qdrant (Docker, :6333)
                                         └── vLLM Reranker (Docker, :8001)
```

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | GPT-4o / GPT-4o-mini |
| Embedding | text-embedding-3-small |
| Vector DB | Qdrant |
| Reranker | Qwen3-Reranker-0.6B (vLLM) |
| Agent | LangGraph + LangChain |
| Backend | FastAPI |
| Frontend | Next.js 14 + Vercel AI SDK |
| Evaluation | RAGAS |

## Documentation

- [Plan](plan_v2.md) — Detailed 7-day implementation plan
- [Decisions](docs/decisions/technical_decisions.md) — Architecture decision records
- [Patterns](docs/patterns/code_patterns.md) — Code patterns and conventions
