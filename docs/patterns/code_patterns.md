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
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# Example: "action anime 2023 score > 8.0"
qdrant_filter = Filter(
    must=[
        FieldCondition(key="genres", match=MatchValue(value="Action")),
        FieldCondition(key="year",   match=MatchValue(value=2023)),
        FieldCondition(key="score",  range=Range(gte=80)),   # 8.0 * 10
    ]
)

results = await qdrant_client.search(
    collection_name=settings.qdrant_collection,
    query_vector=embedding,
    query_filter=qdrant_filter,
    limit=20,       # top-20 for reranker input
    with_payload=True,
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
