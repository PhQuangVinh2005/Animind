# AniMind Agent Architecture

## Overview

The AniMind agent is a stateful LangGraph `StateGraph` that routes user messages through
three intent-specific pipelines, backed by Qdrant vector search and a Qwen3 reranker.

---

## Graph Topology

```
[START]
   │
   ▼
router_node                  ← GPT-4o-mini intent classification
   │
   ├── "qa"     ──► rag_node ──► rerank_node ──► synthesizer_node ──► [END]
   │
   ├── "search" ──► tool_node ─────────────────► synthesizer_node ──► [END]
   │
   └── "detail" ──► tool_node ─────────────────► synthesizer_node ──► [END]
```

---

## Intent Classification

| Intent | Meaning | Example |
|--------|---------|---------|
| `qa` | Factual question, plot/character discussion, thematic analysis | "What genres does FMA:B have?" |
| `search` | Looking for recommendations / a list matching criteria | "Recommend action anime from 2023" |
| `detail` | Comprehensive info dump for ONE specific title | "Give me full details on Vinland Saga" |

Router falls back to `"qa"` on any LLM or JSON parse failure.

---

## Node Descriptions

### `router_node`
- **Input:** last user message
- **LLM call:** GPT-4o-mini, `response_format=json_object`, 10 few-shot examples, `max_tokens=20`
- **Output:** `{"intent": "qa" | "search" | "detail"}`
- **Writes to state:** `intent`

### `rag_node` *(qa path only)*
Full pipeline sub-step sequence:
1. **Contextualize** — `_contextualize_query()`: GPT-4o-mini rewrites follow-up questions
   into self-contained queries using the last 4 conversation turn-pairs. Skipped on first turn.
2. **Filter extraction** — `extract_filters()` from `chain.py`: GPT-4o-mini parses
   year/genre/format/score from the contextualized query.
3. **Query rewrite** — `rewrite_query()` from `chain.py`: few-shot expansion of vague terms.
4. **Retrieve top-20** — `retrieve()` from `retriever.py`: text-embedding-3-small + Qdrant vector search.
- **Writes to state:** `retrieval_query`, `retrieved_docs` (serialised list of dicts)

### `rerank_node` *(qa path only)*
- **Input:** `state["retrieved_docs"]` (20 docs), `state["retrieval_query"]`
- **Calls:** `rerank()` → POST to vLLM `/v1/rerank` (Qwen3-Reranker-0.6B)
- **Writes to state:** `reranked_docs` (top-5 dicts)

### `tool_node` *(search / detail paths)*
Dispatches based on `state["intent"]`:

- **search** → `extract_filters()` + `search_anime(query, oai_client, **filter_kwargs)`
  - Internally: `retrieve(top_k=20)` + `rerank(top_k=5)` — same quality as qa path
- **detail** → extract title via GPT-4o-mini + `get_anime_details(identifier, oai_client)`
  - Strategy 1: exact `anilist_id` scroll (Qdrant `scroll()`)
  - Strategy 2: fallback to vector similarity `retrieve(top_k=1)`
- **Writes to state:** `tool_output: {"type": "search"|"detail", "data": {...}}`

### `synthesizer_node` *(all paths)*
Builds LLM message list for final generation:

```
[SYSTEM]        _SYSTEM_PROMPT (static, >1024 tokens, prefix-cached)
[USER]          turn 1 user message      ┐
[ASSISTANT]     turn 1 ai answer         │  last 5 turn-pairs
[USER]          turn 2 user message      │  (10 messages max)
[ASSISTANT]     turn 2 ai answer         ┘
[USER]          Context passages: [1]...[N]\n---\nUser question: <current>
```

- **qa path:** context = `_build_context(reranked_docs)` → numbered citation blocks
- **search/detail path:** context = `_build_context_from_tool(tool_output)` → formatted result
- **Writes to state:** `messages` (AIMessage appended), `final_answer`

---

## State Schema

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # full conversation history (add_messages reducer)
    intent: str                              # "qa" | "search" | "detail"
    retrieval_query: str                     # rewritten query used for retrieve + rerank
    retrieved_docs: list[dict]               # top-20 from rag_node (serialised)
    reranked_docs: list[dict]                # top-5 from rerank_node (serialised)
    tool_output: dict                        # {"type": "...", "data": {...}}
    final_answer: str                        # final answer string
```

Each dict in `retrieved_docs` / `reranked_docs`:
```python
{
    "qdrant_id":  int | str,
    "score":      float,       # cosine similarity or reranker score
    "title":      str,         # resolved preferred/english/native title
    "chunk_text": str,         # text used for reranker input
    "payload":    dict,        # full Qdrant payload (year, genres, score, full_data, ...)
}
```

---

## Multi-Turn Memory Design

### Implemented
- **`AsyncSqliteSaver`** checkpointer — state persists across server restarts
- **`thread_id` routing** — each session gets isolated state via LangGraph config
- **Contextual query reformulation** — `_contextualize_query()` converts follow-up
  questions into self-contained retrieval queries using last 4 turn-pairs
- **Conversation history in synthesizer** — last 5 turn-pairs injected into LLM call,
  enabling cross-turn comparisons and references
- **Frontend persistence** — message history stored per-thread in `localStorage`;
  switching sessions restores the previous conversation immediately

### Pipeline (qa turn with history):
```
Last user message
    → _contextualize_query(history[-4 pairs])   # resolve implicit references
    → extract_filters(contextualized)            # parse metadata
    → rewrite_query(contextualized)              # expand for retrieval
    → retrieve(rewritten, top_k=20)              # Qdrant vector search
    → rerank(top_k=5)                            # Qwen3-Reranker
    → _build_context(top5)                       # numbered citation blocks
    → LLM([system] + [history 5 pairs] + [context + question])
    → AIMessage (with [1][2] citations) — streamed token-by-token via SSE
```

### Future
| Feature | Notes |
|---------|-------|
| Message window summarization (>20 turns) | Compress old history |
| Cross-session user memory (Qdrant) | Optional — user preference store |

---

## Dependency Injection

All nodes are created via `make_nodes(oai_client: AsyncOpenAI) -> dict`:

```python
# nodes.py — factory pattern
def make_nodes(oai_client: AsyncOpenAI) -> dict[str, Callable]:
    async def router_node(state: AgentState) -> dict: ...
    async def rag_node(state: AgentState) -> dict: ...
    # ... etc.
    return {"router": router_node, "rag": rag_node, ...}

# graph.py — build once, reuse for all requests
def build_graph(oai_client: AsyncOpenAI, checkpointer=None) -> CompiledStateGraph:
    if checkpointer is None:
        checkpointer = InMemorySaver()
    nodes = make_nodes(oai_client)
    # ... add_node / add_edge / compile
```

**Pattern for Day 4 FastAPI:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncSqliteSaver.from_conn_string(settings.agent_db_path) as saver:
        app.state.agent = build_graph(make_openai_client(), checkpointer=saver)
        yield
```

---

## File Map

```
backend/app/agent/
├── state.py     — AgentState TypedDict definition
├── nodes.py     — make_nodes() factory: all 5 node implementations + helpers
├── tools.py     — search_anime(), get_anime_details() (Qdrant-backed)
└── graph.py     — build_graph(): StateGraph + edges + checkpointer
```

---

## Test Command

```bash
cd ~/misa/Animind/backend
python3 - <<'EOF'
import asyncio
from langchain_core.messages import HumanMessage
from app.openai_client import make_openai_client
from app.agent.graph import build_graph

async def main():
    app = build_graph(make_openai_client())
    cfg = {"configurable": {"thread_id": "test-session"}}
    for query in [
        "give me full details on Vinland Saga",    # detail
        "what about its score and episodes?",       # multi-turn qa
        "recommend action anime from 2023",         # search
    ]:
        result = await app.ainvoke(
            {"messages": [HumanMessage(content=query)]}, config=cfg
        )
        print(f"[{result['intent']}] {result['final_answer'][:300]}\n")

asyncio.run(main())
EOF
```
