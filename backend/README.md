# AniMind Backend

FastAPI server providing the RAG agent API. Streams responses via SSE, uses Qdrant for hybrid search, and runs a local Qwen3 reranker for document scoring.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check + Qdrant/reranker reachability |
| `POST` | `/chat` | Blocking full-response chat |
| `POST` | `/chat/stream` | SSE token-by-token streaming chat |

### Request/Response

```jsonc
// POST /chat or /chat/stream
{
  "message": "What genres does Fullmetal Alchemist have?",
  "session_id": "uuid-v4"  // thread isolation for multi-turn
}

// POST /chat response
{
  "answer": "Fullmetal Alchemist [1] belongs to...",
  "intent": "qa",
  "sources": [{"title": "...", "score": 0.92}]
}

// POST /chat/stream — SSE events
data: {"type": "token", "content": "Full"}
data: {"type": "token", "content": "metal"}
data: {"type": "cards", "data": [{...anime card...}]}
data: [DONE]
```

---

## Configuration

All settings are loaded from `backend/.env` via [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).

| Variable | Default | Description |
|---|---|---|
| `SHOPAIKEY_API_KEY` | — | OpenAI-compatible API key (required) |
| `SHOPAIKEY_BASE_URL` | `https://api.shopaikey.com/v1` | API base URL (change to `https://api.openai.com/v1` for direct OpenAI) |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model for all LLM calls |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model (1536d) |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant connection URL |
| `QDRANT_COLLECTION` | `anime` | Qdrant collection name |
| `RERANKER_URL` | `http://localhost:8001` | vLLM reranker endpoint |
| `RERANKER_MODEL` | `Qwen/Qwen3-Reranker-0.6B` | Reranker model name |
| `RERANKER_TOP_K` | `5` | Docs returned after reranking |
| `RETRIEVER_TOP_K` | `20` | Docs retrieved before reranking |
| `AGENT_DB_PATH` | `agent_state.db` | SQLite path for LangGraph state |
| `FRONTEND_URL` | `http://localhost:3000` | Allowed CORS origin |

> **Docker note:** When running in Docker, Qdrant and reranker URLs resolve via Docker DNS (e.g. `http://qdrant:6333`, `http://reranker:8001`). These are set in `docker-compose.yml` environment overrides.

---

## Agent Graph

See the [RAG Pipeline section in the root README](../README.md#rag-pipeline) for the full node-by-node diagram.

### File Map

```
app/
├── agent/
│   ├── state.py     # AgentState TypedDict (9 fields)
│   ├── nodes.py     # make_nodes() factory: router, rag, reranker, gate, tool, synthesizer
│   ├── tools.py     # search_anime(), get_anime_details()
│   └── graph.py     # build_graph(): StateGraph + edges + compile
├── rag/
│   ├── retriever.py # retrieve(): Qdrant hybrid search (dense + BM25 RRF)
│   ├── reranker.py  # rerank(): POST to vLLM /v1/rerank
│   └── chain.py     # extract_filters(), rewrite_query(), build_context()
├── api/
│   ├── routes.py    # /health, /chat, /chat/stream
│   ├── schemas.py   # Pydantic request/response models
│   └── middleware.py # RequestID, SecurityHeaders, RequestLogging (pure ASGI)
├── config.py        # Settings (pydantic-settings, loads .env)
├── main.py          # FastAPI app, lifespan, CORS
└── openai_client.py # AsyncOpenAI factory with User-Agent header
```

---

## Data Pipeline

### 1. Fetch from AniList

```bash
python scripts/fetch_anilist.py [--reset]
```

- Queries AniList GraphQL API for all anime (39 fields each)
- Rate-limited: 30 req/min with backoff
- Checkpoint-safe: saves progress every 5 pages to `data/raw/.fetch_checkpoint.json`
- Output: `data/raw/anime.json` (~1,250 records, ~6.2MB)

### 2. Ingest into Qdrant

```bash
python scripts/ingest.py [--reset]
```

- Creates Qdrant collection with named vectors: `dense` (1536d) + `bm25` (sparse)
- Generates structured chunks per anime (titles + genres + tags + synopsis + metadata)
- Embeds with `text-embedding-3-small` + `fastembed` BM25
- Stores full raw JSON in payload for reranker/LLM context
- `--reset` drops and recreates the collection

---

## Development

### Without Docker

```bash
# Prerequisites: Qdrant and vLLM reranker running (via docker compose)
cd backend
pip install -r requirements.txt
cp .env.example .env   # configure API keys

uvicorn app.main:app --reload --port 8000
```

### With Docker

```bash
# From project root
bash scripts/start-backend.sh --build
```

### Testing the agent

```bash
cd backend
python3 -c "
import asyncio
from langchain_core.messages import HumanMessage
from app.openai_client import make_openai_client
from app.agent.graph import build_graph

async def main():
    app = build_graph(make_openai_client())
    cfg = {'configurable': {'thread_id': 'test'}}
    result = await app.ainvoke(
        {'messages': [HumanMessage(content='recommend action anime from 2023')]},
        config=cfg,
    )
    print(f'[{result[\"intent\"]}] {result[\"final_answer\"][:500]}')

asyncio.run(main())
"
```

---

## Evaluation

See [`eval/README.md`](eval/README.md) for the full evaluation pipeline documentation (FActScore + RAGAS).

Quick reference:

```bash
# Collect answers from pipelines
conda run -n animind python eval/collect.py --pipeline all

# Run FActScore (isolated env)
conda run -n factscore python eval/factscore_runner.py \
  --input eval/results/raw_baseline.json \
  --output eval/results/factscore_baseline_v1.json \
  --db eval/factscore_db/anime_kb.db \
  --judge-model gpt-4o-mini --gamma 0

# Run RAGAS evaluation
conda run -n animind python eval/evaluate.py --tag v2
```

---

## Conventions

- **Type hints everywhere**, async where possible
- **`httpx.AsyncClient`** for all HTTP calls (never `requests`)
- **`loguru`** for logging (never `print()` or `logging`)
- **Environment variables** via `.env` (never hardcode keys)
- **LangGraph nodes** are pure functions: `(AgentState) → partial state dict`
