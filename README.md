# AniMind — Anime/Manga RAG Chatbot

An intelligent anime/manga chatbot powered by LangGraph Agent, Qdrant vector search, Qwen3 reranker, and GPT-4o. Ask natural language questions about anime and get grounded, accurate answers with real-time streaming.

## Features

- **RAG-powered Q&A** — Answers grounded in AniList data via vector search + reranking
- **Strategy 5 Chunking** — Structured single-chunk per anime: Titles → Genres → Tags → Synopsis → Metadata
- **Metadata Filtering** — Pre-filter by year, genre, score, format before vector search
- **Reranking** — Qwen3-Reranker-0.6B (local, zero cost) for improved retrieval quality
- **Multi-turn Memory** — Persistent conversation history via LangGraph + SQLite checkpointer
- **SSE Streaming** — Real-time token streaming via `POST /chat/stream`
- **Security Headers** — OWASP hardened headers (X-Content-Type-Options, X-Frame-Options, etc.)
- **Request Tracing** — UUID4 `X-Request-ID` on every request/response for log correlation
- **Evaluation** — RAGAS metrics for retrieval quality assessment

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd Animind

# 2. Copy and fill env file
cp .env.example backend/.env
# Edit backend/.env — add SHOPAIKEY_API_KEY

# 3. Start infrastructure (Qdrant + vLLM reranker)
docker compose up -d

# 4. Fetch + ingest data
cd backend
pip install -r requirements.txt
python scripts/fetch_anilist.py        # fetches ~1250 anime records
python scripts/ingest.py --reset       # embeds + upserts to Qdrant

# 5. Start backend
uvicorn app.main:app --reload --port 8000

# 6. (Optional) Start frontend
cd ../frontend
pnpm install
pnpm dev
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness + Qdrant/reranker probe |
| `POST` | `/chat` | Blocking full-response chat |
| `POST` | `/chat/stream` | SSE token-by-token streaming |
| `GET` | `/chat/sessions/{thread_id}` | SQLite persistence probe |

### Chat request format

```bash
# Blocking
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"550e8400-e29b-41d4-a716-446655440000","message":"best action anime 2023"}'

# Streaming (SSE)
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"550e8400-e29b-41d4-a716-446655440000","message":"best action anime 2023"}'
```

`thread_id` must be a valid **UUID4**. Each unique thread_id has its own persistent conversation history.

### Error envelope

All errors use a consistent format:
```json
{
  "error": "validation_error",
  "detail": "thread_id: must be a valid UUID4",
  "request_id": "5f597164-fbf2-4c51-a534-cf51736e82e7"
}
```

## Data Pipeline

```
AniList GraphQL API
      ↓
fetch_anilist.py          → data/raw/anime.json  (39 fields, ~1250 records)
      ↓
ingest.py (process_anime)
  ├── clean_html()        → strip HTML, remove (Source: ...) citations
  ├── chunk text          → Titles. Genres. Tags (rank≥70). Synopsis. Metadata.
  ├── embed               → text-embedding-3-small via ShopAIKey
  └── upsert              → Qdrant collection "anime"
                            payload: title, cover, score, genres, tags, full_data
```

## Ingest Commands

```bash
cd backend

# Test run (10 records)
python scripts/ingest.py --limit 10

# Full clean ingest (drops + recreates collection)
python scripts/ingest.py --reset

# Incremental upsert (update existing, add new)
python scripts/ingest.py

# Verify collection
curl -s http://localhost:6333/collections/anime | python3 -m json.tool
```

## Docker Services

### Start / Stop

```bash
cd ~/misa/Animind

docker compose up -d        # start Qdrant + vLLM reranker
docker compose down         # stop all
docker compose logs -f      # all logs
```

### Verify Health

```bash
# Qdrant
curl -s http://localhost:6333/healthz

# Reranker
curl -s http://localhost:8001/health

# Backend (includes dependency probe)
curl -s http://localhost:8000/health | python3 -m json.tool
```

### Container Details

| Service | Image | Port | Resource |
|---|---|---|---|
| Qdrant | `qdrant/qdrant:v1.17.1` | `:6333` REST, `:6334` gRPC | 512M RAM |
| Reranker | `vllm/vllm-openai:v0.19.1` | `:8001` | ~2GB VRAM, 4G RAM |

### Port Forward (laptop → kaguyaserver)

Run **on your laptop** to access all services locally:

```bash
ssh -N \
  -L 6333:localhost:6333 \
  -L 6334:localhost:6334 \
  -L 8000:localhost:8000 \
  -L 8001:localhost:8001 \
  kaguya@kaguyaserver
```

After connecting, access from your laptop:
- `http://localhost:6333/dashboard` — Qdrant Web UI
- `http://localhost:8000/docs` — FastAPI Swagger UI
- `http://localhost:8001/health` — vLLM Reranker

## Architecture

```
User → Vercel (Next.js)
```

Was the original plan. **Actual architecture (Option B — everything self-hosted):**

```
Browser → https://chat.vinhkaguya.me  (Cloudflare CDN)
               ↓ tunnel
         cloudflared daemon → http://localhost:3000 (Next.js)

Browser → https://api.vinhkaguya.me   (Cloudflare CDN)
               ↓ tunnel
         cloudflared daemon → http://localhost:8000 (FastAPI)

Homeserver (kaguyaserver)
  ├── Next.js frontend    :3000  (animind-frontend.service)
  ├── FastAPI + LangGraph :8000  (animind-backend.service)
  ├── Qdrant (Docker)     :6333
  └── vLLM Reranker       :8001
```

## Backend Structure

```
backend/app/
├── main.py               # FastAPI app, lifespan (AsyncSqliteSaver + agent)
├── config.py             # Pydantic Settings from .env
├── openai_client.py      # make_openai_client() + make_chat_model() factories
├── middleware.py         # RequestID, SecurityHeaders, RequestLogging (pure ASGI)
├── api/
│   ├── routes.py         # /health, /chat, /chat/stream, /chat/sessions/{id}
│   ├── schemas.py        # ChatRequest (UUID4 thread_id), ChatResponse, SessionInfo
│   └── exceptions.py     # Consistent {error, detail, request_id} envelope
├── agent/
│   ├── state.py          # AgentState (TypedDict)
│   ├── nodes.py          # make_nodes() — router, rag, reranker, tool, synthesizer
│   ├── tools.py          # search_anime(), get_anime_details()
│   └── graph.py          # build_graph() — LangGraph StateGraph
└── rag/
    ├── chain.py           # Full RAG pipeline
    ├── retriever.py       # retrieve() — Qdrant similarity search
    └── reranker.py        # rerank() — vLLM /v1/rerank
```

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | GPT-4o-mini (synthesizer via LangChain ChatOpenAI) |
| Embedding | text-embedding-3-small via ShopAIKey |
| Vector DB | Qdrant v1.17.1 |
| Reranker | Qwen3-Reranker-0.6B (vLLM v0.19.1, local) |
| Agent | LangGraph + LangChain |
| Checkpointer | AsyncSqliteSaver (persistent across restarts) |
| Backend | FastAPI + SSE (sse-starlette) |
| Frontend | Next.js 14 (self-hosted, `next start` on homeserver) |
| Tunnel | Cloudflare Tunnel — `chat.vinhkaguya.me` (:3000) + `api.vinhkaguya.me` (:8000) |
| Evaluation | RAGAS |
| Data Source | AniList GraphQL API |

## Environment Variables

See `.env.example` for full reference. Key variables:

| Variable | Description | Default |
|---|---|---|
| `SHOPAIKEY_API_KEY` | API key for ShopAIKey (OpenAI-compatible) | required |
| `SHOPAIKEY_BASE_URL` | Provider base URL | `https://api.shopaikey.com/v1` |
| `OPENAI_MODEL` | Chat model name | `gpt-4o-mini` |
| `QDRANT_URL` | Qdrant REST endpoint | `http://localhost:6333` |
| `QDRANT_COLLECTION` | Collection name | `anime` |
| `RERANKER_URL` | vLLM reranker endpoint | `http://localhost:8001` |
| `AGENT_DB_PATH` | SQLite file for conversation state | `agent_state.db` |
| `FRONTEND_URL` | Allowed CORS origin | `http://localhost:3000` |

## Cloudflare Tunnel (Production)

The backend and frontend are both exposed via the same Cloudflare Tunnel (no port forwarding needed):

```bash
# ~/.cloudflared/config.yml (after Day 5)
tunnel: 80898b88-f6d7-4092-b694-01035e9c2861
credentials-file: /home/kaguya/.cloudflared/80898b88-...json

ingress:
  - hostname: chat.vinhkaguya.me
    service: http://localhost:3000    # Next.js frontend
  - hostname: api.vinhkaguya.me
    service: http://localhost:8000    # FastAPI backend
  - service: http_status:404
```

```bash
# One-time setup (already done for api.vinhkaguya.me):
ANIMIND_HOSTNAME="api.vinhkaguya.me" bash scripts/setup_tunnel.sh

# Service management
sudo systemctl status animind-backend cloudflared-animind
journalctl -u animind-backend -u cloudflared-animind -f
```

## Documentation

- [Plan](plan_v2.md) — Detailed 7-day implementation plan
- [Agent Architecture](docs/architecture/agent.md) — LangGraph graph topology + state schema
- [Decisions](docs/decisions/technical_decisions.md) — Architecture decision records (D1–D22+)
- [Patterns](docs/patterns/code_patterns.md) — Code patterns and conventions
- [Progress](progress.json) — Current implementation status and next steps
