# AniMind — Developer Guide

> **Scope:** Local development only. The system is not currently exposed to the public internet. This guide covers everything needed to run, test, and extend AniMind on the homeserver.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [First-time Setup](#3-first-time-setup)
4. [Daily Development Workflow](#4-daily-development-workflow)
5. [Project Structure](#5-project-structure)
6. [Environment Variables](#6-environment-variables)
7. [Service Management](#7-service-management)
8. [Key Design Decisions](#8-key-design-decisions)
9. [Code Quality](#9-code-quality)
10. [Evaluation](#10-evaluation)
11. [Smoke Testing](#11-smoke-testing)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Architecture Overview

```
Browser (localhost:3000)
    │
    │  POST /api/chat/stream          (SSE proxy — avoids Cloudflare buffering)
    ▼
Next.js Dev Server (localhost:3000)
    │
    │  POST /chat/stream              (server-to-server, same machine)
    ▼
FastAPI + LangGraph (localhost:8000)
    ├── Router node      — classifies intent: qa | search | detail
    ├── RAG node         — extract_filters → rewrite_query → hybrid retrieve top-20 (dense + BM25 RRF)
    ├── Rerank node      — Qwen3-Reranker-0.6B → top-5 + top_rerank_score
    ├── Relevance Gate   — score ≥ 0.4 → synthesize; score < 0.4 → retry without filters (max 1)
    ├── Tool node        — search_anime / get_anime_details (search/detail intents)
    └── Synthesizer node — GPT-4o-mini streams answer token-by-token via SSE
         │
         ├── Qdrant (localhost:6333)   — vector DB (hybrid: dense + BM25 sparse)
         └── vLLM Reranker (localhost:8001) — local reranker, ~2GB VRAM
```

### Why the Next.js SSE Proxy?

Cloudflare Tunnel buffers entire HTTP responses before delivering them to the browser. This breaks token-by-token SSE streaming. By routing the stream through a same-origin Next.js API route (`/api/chat/stream`), the browser talks directly to the local Next.js server which in turn talks to the FastAPI backend — no Cloudflare in the loop.

Non-streaming calls (health checks, session listing) still go through the Cloudflare tunnel (`api.vinhkaguya.me`) via `NEXT_PUBLIC_API_URL`.

---

## 2. Prerequisites

| Dependency | Version | Purpose |
|------------|---------|---------|
| Conda (Miniforge) | any | Python env management |
| Python | 3.11 | Backend runtime |
| Node.js | 20 | Frontend runtime |
| npm | bundled | Frontend package management |
| Docker + Docker Compose | v2+ | Qdrant + vLLM containers |
| NVIDIA GPU | RTX 5060Ti (16GB VRAM) | vLLM reranker (~2GB VRAM) |
| CUDA drivers | compatible | GPU acceleration |

---

## 3. First-time Setup

### 3.1 Clone and create Python env

```bash
git clone <repo-url>
cd Animind

conda create -n animind python=3.11 -y
conda activate animind
pip install -r backend/requirements.txt
```

### 3.2 Configure environment

```bash
cp .env.example backend/.env
# Edit backend/.env — fill in SHOPAIKEY_API_KEY at minimum
```

Required keys (see [Section 6](#6-environment-variables) for full list):
- `SHOPAIKEY_API_KEY` — LLM + embeddings provider

### 3.3 Configure frontend

```bash
# frontend/.env.local is already in .gitignore — create manually if missing
cat > frontend/.env.local << 'EOF'
NEXT_PUBLIC_API_URL=http://localhost:8000
BACKEND_URL=http://localhost:8000
EOF
```

### 3.4 Install frontend dependencies

```bash
cd frontend && npm install
```

### 3.5 Start all infrastructure

```bash
# From project root
bash scripts/start.sh
```

This starts Docker (Qdrant + vLLM), waits for health checks, then starts the FastAPI backend. Takes ~60s on first run (vLLM model load).

### 3.6 Fetch and ingest data (first time only)

```bash
cd backend
conda activate animind

# Fetch anime data from AniList GraphQL API
python scripts/fetch_anilist.py

# Ingest into Qdrant (embed + upsert)
python scripts/ingest.py
```

> **Note:** Raw data in `data/raw/` is immutable after fetch. Do not re-run `fetch_anilist.py` unless you want fresh data.

### 3.7 Start the frontend

```bash
cd frontend && npm run dev
```

Frontend is now available at **http://localhost:3000**.

---

## 4. Daily Development Workflow

### Start everything

```bash
# Terminal 1 — infrastructure + backend
bash scripts/start.sh

# Terminal 2 — frontend (auto hot-reloads on save)
cd frontend && npm run dev
```

### Stop everything

```bash
bash scripts/stop.sh
```

### Backend hot-reload (during active backend development)

The `start.sh` script uses `--workers 1` without `--reload` for stability. For active backend development, kill the background process and use `--reload` instead:

```bash
# Kill background backend
kill $(cat .logs/backend.pid)

# Start with auto-reload
cd backend
conda activate animind
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### View backend logs

```bash
tail -f .logs/backend.log
```

---

## 5. Project Structure

```
Animind/
├── backend/
│   ├── app/
│   │   ├── agent/              # LangGraph graph, nodes, state, tools
│   │   │   ├── graph.py        # Graph definition + compilation (incl. relevance gate loop)
│   │   │   ├── nodes.py        # router, rag, rerank, relevance_gate, tool, synthesizer nodes
│   │   │   ├── state.py        # AgentState TypedDict (+retry_count, top_rerank_score)
│   │   │   └── tools.py        # search_anime, get_anime_details
│   │   ├── api/
│   │   │   └── routes.py       # POST /chat/stream (SSE), GET /health, /sessions
│   │   ├── rag/
│   │   │   ├── chain.py        # Full RAG pipeline (filter→rewrite→retrieve→rerank→generate) + system prompt
│   │   │   ├── retriever.py    # Qdrant hybrid search (dense + BM25 RRF) + metadata filter builder
│   │   │   └── reranker.py     # vLLM Qwen3 reranker HTTP client
│   │   ├── config.py           # Pydantic settings (reads backend/.env)
│   │   ├── main.py             # FastAPI app + lifespan + CORS + middleware
│   │   ├── middleware.py       # RequestID, SecurityHeaders, Logging middleware
│   │   └── openai_client.py    # ShopAIKey OpenAI-compatible client factory
│   ├── scripts/
│   │   ├── fetch_anilist.py    # AniList GraphQL → data/raw/
│   │   └── ingest.py           # data/raw/ → embeddings → Qdrant
│   ├── eval/
│   │   ├── test_set.json           # 50 evaluation questions
│   │   ├── collect.py              # RAG pipeline runner — saves raw_{pipeline}.json
│   │   ├── factscore_runner.py     # FActScore algorithm (factscore env, openai<1.0)
│   │   ├── factscore_eval.py       # Subprocess wrapper: animind → factscore env
│   │   ├── build_factscore_db.py   # Builds SQLite FTS5 KB from Qdrant
│   │   ├── evaluate.py             # RAGAS metrics + summary report
│   │   ├── factscore_db/           # anime_kb.jsonl + anime_kb.db (FTS5 BM25)
│   │   └── results/                # raw_*.json, factscore_*.json, scores_*.json
│   └── requirements.txt
│
├── frontend/
│   ├── app/
│   │   ├── api/chat/stream/
│   │   │   └── route.ts        # SSE proxy (Next.js → FastAPI, bypasses Cloudflare)
│   │   ├── layout.tsx          # Root layout (font, metadata)
│   │   └── page.tsx            # Main page (session state, sidebar + chat)
│   ├── components/
│   │   ├── AnimeCard.tsx       # Anime result card (cover, score, genres, tags)
│   │   ├── ChatWindow.tsx      # Message list, input bar, streaming state
│   │   ├── MessageBubble.tsx   # Single message (user plain text / assistant markdown)
│   │   └── Sidebar.tsx         # Session list, new chat button
│   ├── lib/
│   │   ├── api.ts              # streamChat() — flat SSE reader, token callbacks
│   │   └── sessions.ts         # localStorage session + per-thread message persistence
│   ├── types/
│   │   └── index.ts            # ChatMessage, AnimePayload, StoredSession
│   └── .env.local              # NEXT_PUBLIC_API_URL, BACKEND_URL (not committed)
│
├── scripts/
│   ├── start.sh                # Start Docker + backend (health-checked)
│   └── stop.sh                 # Stop backend + Docker containers
│
├── docker-compose.yml          # Qdrant (:6333) + vLLM Reranker (:8001)
├── .env.example                # Template — copy to backend/.env
└── AGENTS.md                   # Coding agent rules and conventions
```

---

## 6. Environment Variables

All backend config is in `backend/.env` (never committed).

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SHOPAIKEY_API_KEY` | ✅ | — | LLM + embeddings API key |
| `SHOPAIKEY_BASE_URL` | ✅ | `https://api.shopaikey.com/v1` | OpenAI-compatible endpoint |
| `OPENAI_MODEL` | ✅ | `gpt-4o-mini` | Chat completion model |
| `OPENAI_EMBEDDING_MODEL` | ✅ | `text-embedding-3-small` | Embedding model |
| `QDRANT_URL` | ✅ | `http://localhost:6333` | Qdrant REST endpoint |
| `QDRANT_COLLECTION` | ✅ | `anime` | Collection name |
| `RERANKER_URL` | ✅ | `http://localhost:8001` | vLLM reranker endpoint |
| `RERANKER_MODEL` | ✅ | `Qwen/Qwen3-Reranker-0.6B` | Reranker model name |
| `ANILIST_API_URL` | ✅ | `https://graphql.anilist.co` | AniList GraphQL endpoint |
| `ANILIST_CLIENT_ID` | optional | — | Needed only for user mutations |
| `ANILIST_CLIENT_SECRET` | optional | — | Needed only for user mutations |
| `FRONTEND_URL` | ✅ | `http://localhost:3000` | CORS allowed origin |

Frontend config in `frontend/.env.local`:

| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_API_URL` | Public backend URL (for health/non-streaming) |
| `BACKEND_URL` | Internal backend URL for SSE proxy route (server-side only) |

---

## 7. Service Management

### Ports

| Service | Port | Purpose |
|---------|------|---------|
| Next.js (dev) | `3000` | Frontend + SSE proxy |
| FastAPI | `8000` | REST API + SSE stream |
| Qdrant REST | `6333` | Vector search API + Web UI |
| Qdrant gRPC | `6334` | High-performance vector ops |
| vLLM Reranker | `8001` | OpenAI-compatible reranker |

### Health checks

```bash
# All services at once
curl -s http://localhost:8000/health | python3 -m json.tool
# Returns: {"status":"ok","qdrant":"ok","reranker":"ok"}

# Individual
curl -s http://localhost:3000          # Frontend (200 = running)
curl -s http://localhost:6333          # Qdrant
curl -s http://localhost:8001/health   # vLLM Reranker
```

### Docker containers

```bash
# Status
docker compose ps

# Logs
docker compose logs -f qdrant
docker compose logs -f reranker

# Restart a single service
docker compose restart reranker
```

### Backend process

```bash
# PID (written by start.sh)
cat .logs/backend.pid

# Logs
tail -f .logs/backend.log

# Stop only the backend
kill $(cat .logs/backend.pid)
```

### SSH port-forward (access from your laptop)

```bash
ssh -N \
  -L 3000:localhost:3000 \
  -L 6333:localhost:6333 \
  -L 8000:localhost:8000 \
  -L 8001:localhost:8001 \
  kaguya@kaguyaserver
```

Then open `http://localhost:3000` in your local browser.

---

## 8. Key Design Decisions

### SSE Streaming via Next.js Proxy

**Problem:** Cloudflare Tunnel buffers full HTTP responses, breaking real-time streaming.

**Solution:** Browser posts to `/api/chat/stream` on the Next.js server (same origin, no Cloudflare). The Next.js route handler (`app/api/chat/stream/route.ts`) proxies to `http://localhost:8000/chat/stream` over a direct TCP connection. `cache: 'no-store'` and `dynamic = 'force-dynamic'` prevent Next.js from buffering the response body.

### Flat SSE Reader (no async generator)

The SSE parser in `lib/api.ts` uses a plain `while (true)` loop over `reader.read()` with direct callbacks (`onToken`, `onCards`, `onDone`). An earlier `async function*` generator approach caused silent failures in some browser environments when combined with React 18 batched state updates. The flat loop is simpler and has no generator protocol overhead.

### Conversation Persistence (localStorage)

Messages are stored per-thread in `localStorage` under the key `animind_msgs_<thread_id>`. On thread switch, `loadMessages(threadId)` restores the previous conversation immediately. Messages are saved after `onDone` (via a functional `setMessages` updater to guarantee saving the latest state). Only completed messages (`streaming: false`) are persisted — in-flight streaming messages are never saved.

### Strategy 5 Chunking + Hybrid Search

Each anime is stored as a single Qdrant point with **dual named vectors**:
- **`dense`** — text-embedding-3-small (1536d)
- **`bm25`** — fastembed sparse vectors

Retrieval uses Qdrant's `query_points()` with two `Prefetch` (one per vector) fused via Reciprocal Rank Fusion (RRF). This provides both semantic understanding and exact keyword matching.

Chunk text format:
```
Title (year, format) | Score: X/10 | Genres: A, B, C
Tags: x, y, z
Synopsis...
```

### Self-Correcting Retrieval (Agentic RAG)

After reranking, a **relevance gate** node checks the top reranker score:
- Score ≥ 0.4 → proceed to synthesizer
- Score < 0.4 AND first attempt → retry `rag_node` without filters/rewrite for max recall
- Score < 0.4 AND already retried → proceed with best-effort (future: web search fallback)

### OpenAI Prefix Caching

The `_SYSTEM_PROMPT` in `chain.py` is intentionally long (>1024 tokens). The static content (role, rules, few-shot examples) is always first in the message list, making it a stable prefix that OpenAI auto-caches after the first call. Dynamic content (context passages + user question) is appended as the final user message.

---

## 9. Code Quality

### Frontend

```bash
cd frontend

# Lint (auto-fix)
npm run lint

# Type check
npx tsc --noEmit

# Security audit (high+)
npm audit --audit-level=high
```

### Backend

```bash
cd backend
conda activate animind

# Lint + auto-fix
ruff check app/ --fix

# Type check
mypy app/ --ignore-missing-imports --no-error-summary

# Security scan (medium+ severity)
bandit -r app/ -ll -q
```

### Conventions

| Rule | Detail |
|------|--------|
| No `print()` | Use `loguru` — `from loguru import logger` |
| No `requests` | Use `httpx.AsyncClient` |
| No hardcoded keys | All config via `backend/.env` + `app/config.py` |
| Type hints | Everywhere in Python; TypeScript strict mode |
| Async | All I/O in backend is `async` (FastAPI + httpx) |
| LangGraph nodes | Pure functions: `(AgentState) → partial state dict` |

---

## 10. Evaluation

See [backend/eval/README.md](../backend/eval/README.md) for the full evaluation guide. Quick reference:

### Two metrics

| Metric | Tool | Measures | Env |
|--------|------|---------|-----|
| **FActScore** | `factscore_runner.py` | Factual precision (atomic fact verification) | `factscore` |
| **RAGAS** | `evaluate.py` | Faithfulness, answer relevancy, context recall | `animind` |

### Pipeline registry

```python
# collect.py — PIPELINE_REGISTRY
"baseline"  → run_baseline   # direct retrieve(top-5) → generate
"ragv1"     → run_ragv1      # rewrite → filter → retrieve(top-20) → rerank(top-5) → generate
# "ragv2"  → run_ragv2       ← add future versions here
```

Output files are namespaced automatically: `raw_{pipeline}.json`, `factscore_{pipeline}_{tag}.json`.

### Full eval run (50 questions)

```bash
cd backend

# 1. Collect pipeline outputs (~25 min)
conda run -n animind python eval/collect.py --pipeline all

# 2. FActScore — baseline
conda run -n factscore python eval/factscore_runner.py \
  --input eval/results/raw_baseline.json \
  --output eval/results/factscore_baseline_v1.json \
  --db eval/factscore_db/anime_kb.db --judge-model gpt-4o-mini --gamma 0

# 3. FActScore — ragv1
conda run -n factscore python eval/factscore_runner.py \
  --input eval/results/raw_ragv1.json \
  --output eval/results/factscore_ragv1_v1.json \
  --db eval/factscore_db/anime_kb.db --judge-model gpt-4o-mini --gamma 0

# 4. RAGAS + final report
conda run -n animind python eval/evaluate.py --tag v1
```

### Rebuild knowledge base

```bash
# Run after Qdrant data changes (adds new anime, re-ingest, etc.)
conda run -n animind python eval/build_factscore_db.py
```

### Adding a new RAG version for evaluation

1. Implement `async def run_ragv2(question, oai_client)` in `collect.py`
2. Add `"ragv2": run_ragv2` to `PIPELINE_REGISTRY`
3. Run: `conda run -n animind python eval/collect.py --pipeline ragv2`

---

## 11. Smoke Testing

Run these after any significant change to verify the system is working end-to-end.

### 1. Backend health

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
# Expected: {"status":"ok","qdrant":"ok","reranker":"ok"}
```

### 2. Direct backend SSE stream

```bash
curl -s -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"What is Attack on Titan?","thread_id":"00000000-0000-4000-8000-000000000001"}' \
  --max-time 30 | head -10
# Expected: data: tokens arriving word-by-word
```

### 3. Frontend SSE proxy

```bash
curl -s -N -X POST http://localhost:3000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"best action anime 2023","thread_id":"00000000-0000-4000-8000-000000000002"}' \
  --max-time 30 | head -10
# Expected: same token-by-token stream through the proxy
```

### 4. Frontend page load

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000
# Expected: 200
```

### 5. Eval smoke test (5 questions)

```bash
conda run -n animind python eval/collect.py --pipeline all --limit 5
conda run -n factscore python eval/factscore_runner.py \
  --input eval/results/raw_baseline.json \
  --output eval/results/factscore_baseline_smoke.json \
  --db eval/factscore_db/anime_kb.db --judge-model gpt-4o-mini --gamma 0 --limit 5
# Expected: FActScore logged at end, no SKIP on questions with clear factual answers
```

---

## 12. Troubleshooting

### Backend won't start — `[Errno 98] address already in use`

```bash
# Find and kill whatever owns port 8000
kill $(lsof -ti :8000)
# Then restart
bash scripts/start.sh
```

### Reranker not healthy after 60s

```bash
docker compose logs reranker | tail -30
# If CUDA OOM: check VRAM usage
nvidia-smi
# Restart container
docker compose restart reranker
```

### Frontend shows empty assistant bubble (no text)

1. Check the proxy is running: `curl -s http://localhost:3000/api/chat/stream ...`
2. Check Next.js dev server: `tail /tmp/nextdev.log`
3. Check backend received the request: `tail .logs/backend.log`
4. Verify `BACKEND_URL=http://localhost:8000` in `frontend/.env.local`

### Messages lost when switching sessions

Messages are persisted to `localStorage` under `animind_msgs_<thread_id>`. If they appear missing:
- Open browser DevTools → Application → Local Storage → check keys starting with `animind_msgs_`
- Ensure `onDone` is being called (check backend log for `POST /chat/stream DONE`)

### Qdrant collection empty / no results

```bash
# Check collection stats
curl -s http://localhost:6333/collections/anime | python3 -m json.tool

# Re-ingest if needed
cd backend && conda activate animind
python scripts/ingest.py
```

### vLLM reranker returns 500 / model not loaded

The model (`Qwen3-Reranker-0.6B`) loads from Docker Hub on first run. If the container exits:
```bash
docker compose logs reranker | grep -E "error|Error|FAILED"
# If model download failed, pull manually:
docker compose pull reranker
docker compose up -d reranker
```

---

*Last verified: 2026-04-24 — hybrid search + self-correcting retrieval live, SSE streaming confirmed, lint/types clean.*
