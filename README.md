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

## Docker Services

### Start / Stop

```bash
# Start all services (Qdrant + vLLM Reranker)
cd ~/misa/Animind && docker compose up -d

# Stop all services
docker compose down

# View logs
docker compose logs -f
docker compose logs -f reranker   # reranker only
docker compose logs -f qdrant     # qdrant only
```

### Verify Health

```bash
# Qdrant health + collections
curl -s http://localhost:6333/healthz
curl -s http://localhost:6333/collections | python3 -m json.tool

# Reranker health
curl -s http://localhost:8001/health

# Test rerank endpoint
curl -s http://localhost:8001/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Reranker-0.6B",
    "query": "best action anime",
    "documents": ["Naruto is a popular action anime", "Cooking recipes for beginners", "Attack on Titan features intense battles"]
  }' | python3 -m json.tool
```

### Container Details

| Service | Image | Port | Resource |
|---|---|---|---|
| Qdrant | `qdrant/qdrant:v1.17.1` | `:6333` REST, `:6334` gRPC | 512M RAM |
| Reranker | `vllm/vllm-openai:v0.19.1` | `:8001` | ~2GB VRAM, 4G RAM |

### Port Forward (laptop → homeserver)

```bash
ssh -L 6333:localhost:6333 -L 8001:localhost:8001 kaguya@<homeserver-ip>
```

## Architecture

```
Vercel (Next.js) → Cloudflare Tunnel → Homeserver
                                         ├── FastAPI + LangGraph Agent (:8000)
                                         ├── Qdrant (Docker, :6333)
                                         └── vLLM Reranker (Docker, :8001)
```

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | GPT-4o / GPT-4o-mini |
| Embedding | text-embedding-3-small |
| Vector DB | Qdrant v1.17.1 |
| Reranker | Qwen3-Reranker-0.6B (vLLM v0.19.1) |
| Agent | LangGraph + LangChain |
| Backend | FastAPI |
| Frontend | Next.js 14 + Vercel AI SDK |
| Evaluation | RAGAS |

## Documentation

- [Plan](plan_v2.md) — Detailed 7-day implementation plan
- [Decisions](docs/decisions/technical_decisions.md) — Architecture decision records
- [Patterns](docs/patterns/code_patterns.md) — Code patterns and conventions
