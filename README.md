# AniMind — Anime/Manga RAG Chatbot

An intelligent anime/manga chatbot powered by LangGraph Agent, Qdrant vector search, Qwen3 reranker, and GPT-4o. Ask natural language questions about anime and get grounded, accurate answers.

## Features

- **RAG-powered Q&A** — Answers grounded in AniList data via vector search
- **Strategy 5 Chunking** — Structured single-chunk per anime: Titles → Genres → Tags → Synopsis → Metadata
- **Metadata Filtering** — Pre-filter by year, genre, score, format before vector search
- **Reranking** — Qwen3-Reranker-0.6B (local, zero cost) for improved retrieval quality
- **Streaming** — Real-time token streaming via SSE
- **Evaluation** — RAGAS metrics for retrieval quality assessment

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd Animind

# 2. Copy and fill env file
cp .env.example backend/.env
# Edit backend/.env — add SHOPAIKEY_API_KEY

# 3. Start infrastructure
docker compose up -d

# 4. Fetch + ingest data
cd backend
pip install -r requirements.txt
python scripts/fetch_anilist.py        # fetches ~1250 anime records
python scripts/ingest.py --reset       # embeds + upserts to Qdrant

# 5. Start backend
uvicorn app.main:app --reload --port 8000

# 6. Start frontend
cd ../frontend
pnpm install
pnpm dev
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

# Test run (10 records, no Qdrant write for embedding test)
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
docker compose logs -f qdrant     # Qdrant only
docker compose logs -f reranker   # reranker only
```

### Verify Health

```bash
# Qdrant
curl -s http://localhost:6333/healthz
curl -s http://localhost:6333/collections | python3 -m json.tool

# Reranker
curl -s http://localhost:8001/health

# Test rerank
curl -s http://localhost:8001/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Reranker-0.6B",
    "query": "best action anime",
    "documents": ["Naruto is a popular action anime", "Cooking recipes", "Attack on Titan features intense battles"]
  }' | python3 -m json.tool
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
- `http://localhost:8000/docs` — FastAPI Swagger
- `http://localhost:8001/health` — vLLM Reranker

**"Address already in use" error?** A previous tunnel is still running:
```bash
# Find and kill the old tunnel (on laptop)
sudo lsof -i :6333 -i :6334
sudo kill $(sudo lsof -t -i :6333)
sudo kill $(sudo lsof -t -i :6334)
```

**Persistent config** — add to `~/.ssh/config` on your laptop:
```ssh-config
Host kaguyaserver
    HostName <kaguyaserver-ip>
    User kaguya
    LocalForward 6333 localhost:6333
    LocalForward 6334 localhost:6334
    LocalForward 8000 localhost:8000
    LocalForward 8001 localhost:8001
    ServerAliveInterval 60
    ServerAliveCountMax 3
```
Then just: `ssh -N kaguyaserver`

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
| LLM | GPT-4o / GPT-4o-mini via ShopAIKey |
| Embedding | text-embedding-3-small via ShopAIKey |
| Vector DB | Qdrant v1.17.1 |
| Reranker | Qwen3-Reranker-0.6B (vLLM v0.19.1, local) |
| Agent | LangGraph + LangChain |
| Backend | FastAPI |
| Frontend | Next.js 14 + Vercel AI SDK |
| Evaluation | RAGAS |
| Data Source | AniList GraphQL API |

## Environment Variables

See `.env.example` for full reference. Key variables:

| Variable | Description |
|---|---|
| `SHOPAIKEY_API_KEY` | API key for ShopAIKey (OpenAI-compatible provider) |
| `SHOPAIKEY_BASE_URL` | Default: `https://api.shopaikey.com/v1` |
| `QDRANT_URL` | Default: `http://localhost:6333` |
| `QDRANT_COLLECTION` | Default: `anime` |
| `RERANKER_URL` | Default: `http://localhost:8001` |

## Documentation

- [Plan](plan_v2.md) — Detailed 7-day implementation plan
- [Chunking Strategy](chunking-strategy.md) — Strategy 5: Hybrid Metadata Filter + Single Chunk
- [Decisions](docs/decisions/technical_decisions.md) — Architecture decision records
- [Patterns](docs/patterns/code_patterns.md) — Code patterns and conventions
- [Progress](progress.json) — Current implementation status and next steps
