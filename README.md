# AniMind

Anime/Manga RAG chatbot powered by a self-correcting LangGraph agent, Qdrant hybrid search, and a local Qwen3 reranker.

**Live:** [chat.vinhkaguya.me](https://chat.vinhkaguya.me) · **API:** [api.vinhkaguya.me](https://api.vinhkaguya.me)

---

## Stack

| Layer | Technology |
|---|---|
| **LLM** | GPT-4o-mini (via OpenAI-compatible API) |
| **Embeddings** | text-embedding-3-small (1536d) |
| **Vector DB** | Qdrant v1.17.1 (dense + BM25 sparse vectors) |
| **Reranker** | Qwen3-Reranker-0.6B via vLLM v0.19.1 (local, ~2GB VRAM) |
| **Agent** | LangGraph StateGraph with SQLite checkpointer |
| **Backend** | FastAPI + SSE streaming |
| **Frontend** | Next.js 14 (standalone production build) |
| **Infra** | Docker Compose + NGINX reverse proxy + Cloudflare Tunnel |
| **Evaluation** | FActScore (atomic fact verification) + RAGAS (retrieval quality) |
| **Data source** | AniList GraphQL API (~1,250 anime records) |

---

## RAG Pipeline

### Agent Topology

```
[START]
   │
   ▼
 router ─── GPT-4o-mini intent classifier (qa / search / detail)
   │
   ├─ "qa"     → rag → reranker → relevance_gate ──┐
   │              ↑                     │            │
   │              └── retry (max 1) ────┘            ▼
   │                                           synthesizer → [END]
   │
   ├─ "search" → tool → synthesizer → [END]
   │
   └─ "detail" → tool → synthesizer → [END]
```

### QA Pipeline — Node-by-Node I/O

The QA path is the core RAG pipeline. Each node reads from and writes to a shared `AgentState`.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  USER MESSAGE: "What's the score of Clannad: After Story?"             │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─ router_node ────────────────────────────────────────────────────────────┐
│  In:  last user message                                                  │
│  LLM: GPT-4o-mini, 10 few-shot examples, response_format=json_object    │
│  Out: intent = "qa"                                                      │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌─ rag_node ───────────────────────────────────────────────────────────────┐
│  Step 1 — Contextualize (multi-turn only)                                │
│    In:  last message + conversation history (last 4 turn-pairs)          │
│    LLM: GPT-4o-mini, 3 few-shot examples                                │
│    Out: self-contained query (e.g. "score of Clannad: After Story")      │
│                                                                          │
│  Step 2 — Extract Filters  [skipped on retry]                            │
│    In:  contextualized query                                             │
│    LLM: GPT-4o-mini, response_format=json_object                        │
│    Out: FilterParams {genres, year, format, score_min, ...}              │
│                                                                          │
│  Step 3 — Query Rewrite  [skipped on retry]                              │
│    In:  contextualized query                                             │
│    LLM: GPT-4o-mini, 5 few-shot examples                                │
│    Out: keyword-enriched query for embedding                             │
│                                                                          │
│  Step 4 — Hybrid Retrieve (top-20)                                       │
│    In:  rewritten query + FilterParams                                   │
│    Qdrant: dual Prefetch (dense 1536d + BM25 sparse) → RRF fusion        │
│    Out: 20 scored document dicts {title, chunk_text, payload, score}      │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌─ rerank_node ────────────────────────────────────────────────────────────┐
│  In:  20 retrieved docs + retrieval query                                │
│  HTTP: POST vLLM /v1/rerank (Qwen3-Reranker-0.6B, local GPU)            │
│  Out: top-5 reranked docs + top_rerank_score (best relevance score)      │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌─ relevance_gate_node ────────────────────────────────────────────────────┐
│  In:  top_rerank_score + retry_count                                     │
│  Logic:                                                                  │
│    score >= 0.4         → PASS → synthesizer                             │
│    score <  0.4, try=0  → RETRY → rag_node (no filters, no rewrite)      │
│    score <  0.4, try=1  → PASS (exhausted) → synthesizer                 │
│  Out: retry_count++                                                      │
└──────────────────────────────┬───────────────────────────────────────────┘
                               ▼
┌─ synthesizer_node ───────────────────────────────────────────────────────┐
│  In:  reranked_docs (5) + conversation history (5 turn-pairs)            │
│  LLM: ChatOpenAI (GPT-4o-mini), streamed via LangChain runnable         │
│  Prompt:                                                                 │
│    [SYSTEM]  static prompt (>1024 tok, prefix-cached) + format rules     │
│    [HISTORY] last 5 turn-pairs from conversation                         │
│    [USER]    context passages [1]...[5] + user question                   │
│  Out: final_answer (markdown with [1][2] citations) — streamed via SSE   │
└──────────────────────────────────────────────────────────────────────────┘
```

### Search / Detail Paths

| Path | `tool_node` behavior | Synthesizer input |
|---|---|---|
| **search** | `extract_filters()` → `retrieve(top_k=20)` → `rerank(top_k=5)` | Formatted anime list |
| **detail** | Extract title → `scroll()` by ID or `retrieve(top_k=1)` fallback | Full anime profile |

---

## Quick Start

### Prerequisites

- Docker Engine 24+ with Compose v2
- NVIDIA GPU with 4GB+ VRAM (for the reranker)
- An OpenAI-compatible API key (set in `backend/.env`)

### 1. Clone and configure

```bash
git clone https://github.com/PhQuangVinh2005/Animind.git
cd Animind

# Interactive env setup (prompts for API keys, writes to backend/.env)
python scripts/setup_env.py

# Or manually: cp backend/.env.example backend/.env && edit backend/.env
```

### 2. Start everything

```bash
bash scripts/start-all.sh --build
```

This launches 6 containers:

| Container | Purpose | Port |
|---|---|---|
| `animind-qdrant` | Vector database | 6333 |
| `animind-reranker` | Local Qwen3 reranker (GPU) | 8001 |
| `animind-backend` | FastAPI API server | 8000 (via NGINX) |
| `animind-frontend` | Next.js chat UI | 3000 (via NGINX) |
| `animind-nginx` | Reverse proxy | 80 |
| `animind-dozzle` | Log viewer | 127.0.0.1:9999 |

### 3. Ingest data

```bash
# Fetch anime data from AniList (rate-limited, checkpoint-safe)
docker exec animind-backend python scripts/fetch_anilist.py

# Ingest into Qdrant (dense + BM25 dual vectors)
docker exec animind-backend python scripts/ingest.py
```

### 4. Use it

- **Chat UI:** http://localhost:80 (or https://chat.vinhkaguya.me if tunnel is configured)
- **API:** `curl http://localhost:80/health` (via NGINX → backend)
- **Logs:** http://localhost:9999 (Dozzle)

---

## Scripts Reference

| Script | Purpose |
|---|---|
| `scripts/start-all.sh [--build]` | Start all 6 containers |
| `scripts/stop-all.sh` | Stop all containers |
| `scripts/start-backend.sh [--build]` | Backend + infra only |
| `scripts/stop-backend.sh` | Stop backend (infra stays) |
| `scripts/start-frontend.sh [--build]` | Frontend + infra only |
| `scripts/stop-frontend.sh` | Stop frontend (infra stays) |
| `scripts/rebuild.sh [backend\|frontend]` | Force rebuild images (no cache) |
| `scripts/setup_env.py` | Interactive `.env` setup (API keys, model config) |
| `scripts/setup_tunnel.sh` | Cloudflare Tunnel one-time setup (install, login, DNS) |
| `scripts/start.sh` | Alias → `start-all.sh` |
| `scripts/stop.sh` | Alias → `stop-all.sh` |

---

## Architecture

```
Internet
  │
  ├─ chat.vinhkaguya.me ──┐
  ├─ api.vinhkaguya.me  ──┤
  │                        ▼
  │              Cloudflare Tunnel
  │              (cloudflared → localhost:80)
  │                        │
  │                        ▼
  │                ┌──────────────┐
  │                │    NGINX     │ :80
  │                │  (container) │
  │                └──────┬───────┘
  │                       │
  │          ┌────────────┼────────────┐
  │          ▼                         ▼
  │   ┌─────────────┐          ┌────────────┐
  │   │  Frontend   │ :3000    │  Backend   │ :8000
  │   │  (Next.js)  │────SSE──▶│  (FastAPI) │
  │   └─────────────┘          └──────┬─────┘
  │                                   │
  │                    ┌──────────────┼──────────────┐
  │                    ▼              ▼              ▼
  │             ┌──────────┐   ┌──────────┐   ┌──────────┐
  │             │  Qdrant  │   │ Reranker │   │  SQLite  │
  │             │  :6333   │   │  :8001   │   │ (state)  │
  │             └──────────┘   └──────────┘   └──────────┘
```

**SSE streaming path:** Browser → `chat.vinhkaguya.me/api/chat/stream` → Next.js proxy route → `backend:8000/chat/stream` → token-by-token response. The Next.js proxy bypasses Cloudflare's response buffering.

---

## Evaluation Results

**Eval run v2** · 50-question test set · GPT-4o-mini as judge · Reference-free (no ground truth)

| Metric | Baseline (retrieve-only) | RAGv1 (full pipeline) | Delta |
|---|---|---|---|
| **Faithfulness** (RAGAS) | 0.825 | 0.804 | -0.021 |
| **Answer Relevancy** (RAGAS) | 0.619 | 0.571 | -0.047 |
| **FactScore** (custom) | 0.873 | 0.899 | **+0.026** |

> Targets: Faithfulness ≥ 0.80 ✅ · FactScore ≥ 0.75 ✅
>
> Full results: [`backend/eval/results/report_v2.md`](backend/eval/results/report_v2.md) · [`scores_v2.json`](backend/eval/results/scores_v2.json)

### Pipeline Comparison

| Setting | Baseline | RAGv1 |
|---|---|---|
| Query rewrite | ❌ | ✅ GPT-4o-mini |
| Metadata filter | ❌ | ✅ Auto-extracted |
| Retrieval | Top-20 dense | Top-20 hybrid (dense + BM25 RRF) |
| Reranker | ❌ | ✅ Qwen3-Reranker → top-5 |
| Self-correction | ❌ | ✅ Relevance gate (0.4 threshold) |

### Category Breakdown — Faithfulness

| Category | Baseline | RAGv1 | Delta |
|---|---|---|---|
| edge | 0.567 | 0.677 | **+0.110** |
| factual | 0.933 | 0.850 | -0.083 |
| filter | 0.816 | 0.966 | **+0.150** |
| multi_turn | 0.786 | 0.530 | -0.255 |
| semantic | 0.812 | 0.836 | +0.024 |

### Category Breakdown — FactScore

| Category | Baseline | RAGv1 | Delta |
|---|---|---|---|
| edge | 0.875 | 1.000 | **+0.125** |
| factual | 0.933 | 0.950 | +0.017 |
| filter | 0.892 | 0.877 | -0.015 |
| multi_turn | 0.794 | 0.708 | -0.085 |
| semantic | 0.811 | 0.889 | **+0.078** |

---

## Key Design Decisions

| # | Decision | Rationale |
|---|---|---|
| **D1** | AniList over MyAnimeList | 90 req/min, GraphQL, no OAuth for public data |
| **D2** | Qdrant over Milvus | Simpler single-node, native metadata filtering |
| **D3** | Local Qwen3 reranker via vLLM | ~2GB VRAM, zero cost, OpenAI-compatible API |
| **D8** | Hybrid search (dense + BM25) | FActScore audit showed exact-match failures with dense-only |
| **D14** | ShopAIKey as OpenAI proxy | Cost optimization; swap to direct OpenAI by changing base URL |
| **D15** | One vector per anime (no chunking) | Anime records are self-contained semantic units (~300-1500 chars) |
| **D21** | Multi-turn contextualization | Follow-up queries lose context; GPT-4o-mini rewrites them |
| **D28** | SSE proxy through Next.js | Cloudflare Tunnel buffers responses; Next.js proxy bypasses this |
| **D32** | Self-correcting retrieval gate | Retry once without filters when reranker confidence < 0.4 |

> Full decision log (33 entries): previously in `docs/decisions/technical_decisions.md`, now archived.

---

## Project Structure

```
Animind/
├── backend/
│   ├── app/
│   │   ├── agent/          # LangGraph graph, nodes, tools, state
│   │   ├── api/            # FastAPI routes, schemas, exceptions, middleware
│   │   ├── rag/            # retriever, reranker client, RAG chain
│   │   ├── config.py       # pydantic-settings (all env vars)
│   │   ├── main.py         # FastAPI app + lifespan
│   │   └── openai_client.py
│   ├── eval/               # FActScore + RAGAS evaluation pipeline
│   ├── scripts/            # fetch_anilist.py, ingest.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app/                # Next.js pages + API routes
│   ├── components/         # ChatWindow, MessageBubble, AnimeCard, Sidebar
│   ├── lib/                # api.ts, sessions.ts
│   ├── Dockerfile
│   └── package.json
├── nginx/
│   └── nginx.conf          # Reverse proxy config
├── scripts/                # start/stop/rebuild + setup_env.py + setup_tunnel.sh
├── docker-compose.yml      # All 6 services with profiles
├── AGENTS.md               # AI agent configuration
└── README.md               # This file
```

---

## Deployment (Cloudflare Tunnel)

To expose on a custom domain:

1. Install `cloudflared` and create a tunnel
2. Add DNS records: `CNAME chat → <tunnel-id>.cfargotunnel.com`, `CNAME api → <tunnel-id>.cfargotunnel.com`
3. Configure `~/.cloudflared/config.yml`:
   ```yaml
   ingress:
     - hostname: chat.vinhkaguya.me
       service: http://localhost:80
     - hostname: api.vinhkaguya.me
       service: http://localhost:80
     - service: http_status:404
   ```
4. Start the tunnel: `cloudflared tunnel run animind`

---

## License

MIT
