# Code Patterns

## ✅ DO: Async HTTP calls with httpx
```python
import httpx

async def fetch_anime(anime_id: int) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://graphql.anilist.co",
            json={"query": QUERY, "variables": {"id": anime_id}},
        )
        resp.raise_for_status()
        return resp.json()
```

## ❌ DON'T: Use requests (blocking)
```python
# KHÔNG viết như này
import requests
resp = requests.get(f"https://api.example.com/{id}")
```

---

## ✅ DO: OpenAI client via factory (ShopAIKey-aware)
```python
# app/openai_client.py
from openai import AsyncOpenAI
from app.config import settings

def make_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.shopaikey_api_key,
        base_url=settings.shopaikey_base_url,
        default_headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        },
    )

# Usage
client = make_openai_client()
resp = await client.embeddings.create(model="text-embedding-3-small", input=["hello"])
```

## ❌ DON'T: Instantiate OpenAI client inline or hardcode keys
```python
# KHÔNG viết như này
client = AsyncOpenAI(api_key="sk-xxxxx")
```

---

## ✅ DO: LangGraph node as pure function
```python
def router_node(state: AgentState) -> dict:
    """Classify intent and return partial state update."""
    intent = classify_intent(state["messages"][-1])
    return {"intent": intent}
```

## ❌ DON'T: Mutate state directly
```python
# KHÔNG viết như này
def router_node(state: AgentState) -> AgentState:
    state["intent"] = "qa"  # Mutating input state!
    return state
```

---

## ✅ DO: Logging with loguru
```python
from loguru import logger

logger.info("Fetched {count} anime records", count=len(records))
logger.error("Reranker failed: {err}", err=str(e))
logger.success("Ingestion complete — {n} vectors.", n=len(vectors))
```

## ❌ DON'T: Use print or stdlib logging
```python
# KHÔNG viết như này
print(f"Fetched {len(records)} records")
```

---

## ✅ DO: Config via pydantic-settings
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    shopaikey_api_key: str = ""
    shopaikey_base_url: str = "https://api.shopaikey.com/v1"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "anime"
    reranker_url: str = "http://localhost:8001"
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    class Config:
        env_file = ".env"

settings = Settings()
```

## ❌ DON'T: Hardcode secrets or URLs
```python
# KHÔNG viết như này
client = OpenAI(api_key="sk-xxxxx")
```

---

## ✅ DO: HTML cleaning for AniList descriptions
```python
import re, html as html_lib

_SOURCE_CITE_RE = re.compile(
    r'\s*\(Source:[^)]*\)|\s*\[Written by[^\]]*\]',
    re.IGNORECASE,
)

def clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'<br\s*/?>', ' ', text)   # <br> → space
    text = re.sub(r'<[^>]+>', '', text)       # strip all tags
    text = html_lib.unescape(text)            # &amp; → & etc.
    text = _SOURCE_CITE_RE.sub('', text)      # drop (Source: ...)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

---

## ✅ DO: process_anime — Strategy 5 chunk + payload
```python
def process_anime(anime_json: dict) -> tuple[str, dict]:
    """
    Returns (chunk_text, qdrant_payload).
    chunk_text  — what gets embedded
    payload     — stored in Qdrant, includes full_data for reranker
    """
    # chunk format: Titles. Genres. Tags(rank≥70). Synopsis. Studio/Year/...
    # payload keys: anilist_id, mal_id, title{}, cover_image, banner_image,
    #               site_url, year, season, genres, tags (ALL), studios,
    #               format, status, episodes, duration, source, is_adult,
    #               score, score_display, full_data
    ...
```

---

## ✅ DO: Reranker call via OpenAI-compatible API
```python
async def rerank(query: str, documents: list[str], top_k: int = 5) -> list[dict]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.reranker_url}/v1/rerank",
            json={
                "model": settings.reranker_model,
                "query": query,
                "documents": documents,
            },
        )
        resp.raise_for_status()
        results = resp.json()["results"]
        return sorted(results, key=lambda x: x["relevance_score"], reverse=True)[:top_k]
```

---

## ✅ DO: Qdrant search with metadata filter
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue, Range

# NOTE: .search() was removed in qdrant-client v1.9+. Use .query_points() instead.
# Example: "action anime 2023 score > 8.0"
qdrant_filter = Filter(
    must=[
        FieldCondition(key="genres", match=MatchAny(any=["Action"])),
        FieldCondition(key="year",   match=MatchValue(value=2023)),
        FieldCondition(key="score",  range=Range(gte=80)),   # 8.0 × 10
    ]
)

client = QdrantClient(url=settings.qdrant_url)
results = client.query_points(
    collection_name=settings.qdrant_collection,
    query=embedding_vector,    # list[float]
    query_filter=qdrant_filter,
    limit=20,                  # top-20 for reranker input
    with_payload=True,
)
for pt in results.points:
    print(pt.id, pt.score, pt.payload.get("title"))
```

## ❌ DON'T: Use the removed .search() method
```python
# REMOVED in qdrant-client >= 1.9 — will raise AttributeError
results = client.search(
    collection_name="anime",
    query_vector=embedding,
    limit=20,
)
```

---

## ✅ DO: Pre-flight service checks before long operations
```python
async def preflight(qdrant: AsyncQdrantClient, openai_client: AsyncOpenAI) -> None:
    """Fail fast with clear messages if services are down."""
    try:
        await qdrant.get_collections()
        logger.success("Qdrant OK ✓")
    except Exception as exc:
        logger.error("Qdrant unreachable — run: docker compose up -d qdrant")
        raise SystemExit(1) from exc

    try:
        resp = await openai_client.embeddings.create(
            model=settings.openai_embedding_model, input=["ping"]
        )
        logger.success("Embedding API OK ✓ (dim={dim})", dim=len(resp.data[0].embedding))
    except Exception as exc:
        logger.error("Embedding API error — check SHOPAIKEY_API_KEY and provider status")
        raise SystemExit(1) from exc
```

---

## ✅ DO: Full RAG pipeline call (rag_answer)
```python
from app.openai_client import make_openai_client
from app.rag.chain import rag_answer

oai = make_openai_client()

# Simple — auto-extracts filters from query, rewrites, retrieves, reranks, generates
answer = await rag_answer("best action anime 2023", oai)

# With explicit filters (bypasses auto-extraction)
answer = await rag_answer(
    "recommend something exciting",
    oai,
    filter_kwargs={"genres": ["Action"], "year": 2023},
)

# Streaming
async for chunk in await rag_answer("top 5 romance anime", oai, stream=True):
    print(chunk, end="", flush=True)
```

---

## ✅ DO: Auto-extract metadata filters from natural language
```python
from app.rag.chain import extract_filters

# Returns a FilterParams dataclass; use .to_dict() for retriever.build_filter()
params = await extract_filters("highly rated sci-fi movies from 2020", oai_client)
# → FilterParams(genres=['Sci-Fi'], format_='MOVIE', year=2020)
print(params.to_dict())
# → {'genres': ['Sci-Fi'], 'format_': 'MOVIE', 'year': 2020}

if not params.is_empty():
    hits = await retrieve(query, oai_client, filter_kwargs=params.to_dict())
```

---

## ✅ DO: QdrantClient singleton (don't re-instantiate per request)
```python
# retriever.py — module-level singleton, not one per call
_qdrant_client: QdrantClient | None = None

def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=settings.qdrant_url)
    return _qdrant_client
```

## ❌ DON'T: Instantiate QdrantClient inside a request handler
```python
# Creates a new TCP connection on every call — wasteful
async def retrieve(query):
    qdrant = QdrantClient(url=settings.qdrant_url)  # KHÔNG làm thế này
    ...
```

---

## ✅ DO: LangGraph node factory pattern (dependency injection)
```python
# nodes.py — bind oai_client via closure, not global state
def make_nodes(oai_client: AsyncOpenAI) -> dict[str, Callable]:
    async def router_node(state: AgentState) -> dict:
        # oai_client available via closure
        resp = await oai_client.chat.completions.create(...)
        return {"intent": intent}

    async def rag_node(state: AgentState) -> dict:
        # all nodes share the same client instance
        ...

    return {"router": router_node, "rag": rag_node, ...}

# graph.py — build once at startup
def build_graph(oai_client: AsyncOpenAI, checkpointer=None) -> CompiledStateGraph:
    if checkpointer is None:
        checkpointer = InMemorySaver()
    nodes = make_nodes(oai_client)
    graph = StateGraph(AgentState)
    graph.add_node("router", nodes["router"])
    # ...
    return graph.compile(checkpointer=checkpointer)
```

## ❌ DON'T: Instantiate AsyncOpenAI inside each node (wastes connections)
```python
# KHÔNG làm thế này
async def router_node(state: AgentState) -> dict:
    oai = AsyncOpenAI(api_key=...)  # new client per call!
    ...
```

---

## ✅ DO: Conditional edges for intent routing
```python
def _route_by_intent(state: AgentState) -> str:
    intent = state.get("intent", "qa")
    if intent in ("search", "detail"):
        return "tool"
    return "rag"

graph.add_conditional_edges(
    "router",
    _route_by_intent,
    {"rag": "rag", "tool": "tool"},
)
```

## ❌ DON'T: Use if/else inside a single mega-node (not inspectable)
```python
# KHÔNG làm thế này — loses LangGraph observability
async def mega_node(state):
    if state["intent"] == "qa":
        ...retrieve + rerank + generate...
    elif state["intent"] == "search":
        ...
```

---

## ✅ DO: Inject conversation history into synthesizer LLM call
```python
def _trim_and_format_history(messages: list, max_turns: int = 5) -> list[dict]:
    """Convert LangGraph messages → OpenAI dicts, last N pairs only."""
    oai_msgs = []
    for msg in messages[:-1]:  # exclude current user message
        if isinstance(msg, HumanMessage):
            oai_msgs.append({"role": "user", "content": msg.content or ""})
        elif isinstance(msg, AIMessage):
            oai_msgs.append({"role": "assistant", "content": msg.content or ""})
    return oai_msgs[-(max_turns * 2):]

# In synthesizer_node:
history = _trim_and_format_history(state["messages"], max_turns=5)
messages_payload = [
    {"role": "system", "content": _SYSTEM_PROMPT},
    *history,                          # last 5 turn-pairs (10 msgs max)
    {"role": "user", "content": user_message_with_rag_context},
]
```

---

## ✅ DO: Contextualize follow-up queries before retrieval
```python
# "what about its score?" + history → "Vinland Saga score episode count"
contextualized = await _contextualize_query(
    current_query=_last_user_text(state),
    messages=state["messages"],
    oai_client=oai_client,
)
# Then use contextualized for filters + rewrite + retrieve
```

## ❌ DON'T: Feed raw follow-up messages directly to the retriever
```python
# KHÔNG làm thế này — loses context, retrieval fails on follow-ups
query = _last_user_text(state)  # "what about its score?" — meaningless standalone
candidates = await retrieve(query, oai_client)
```

---

## ✅ DO: Invoke the agent with thread_id for multi-turn
```python
from langchain_core.messages import HumanMessage
from app.agent.graph import build_graph
from app.openai_client import make_openai_client

agent = build_graph(make_openai_client())

# Same thread_id = same conversation (checkpointer restores state)
cfg = {"configurable": {"thread_id": "user-session-uuid"}}

result = await agent.ainvoke(
    {"messages": [HumanMessage(content="best action anime 2023")]},
    config=cfg,
)
print(result["intent"], result["final_answer"])
```

## ❌ DON'T: Forget thread_id (every message starts a fresh conversation)
```python
# KHÔNG làm thế này — no memory, each call is isolated
result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
```
