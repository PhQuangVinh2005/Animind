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
```

## ❌ DON'T: Use print or stdlib logging
```python
# KHÔNG viết như này
print(f"Fetched {len(records)} records")
```

---

## ✅ DO: Config via environment variables
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    qdrant_url: str = "http://localhost:6333"
    reranker_url: str = "http://localhost:8001"
    model_name: str = "gpt-4o-mini"

    class Config:
        env_file = ".env"
```

## ❌ DON'T: Hardcode secrets or URLs
```python
# KHÔNG viết như này
client = OpenAI(api_key="sk-xxxxx")
```

---

## ✅ DO: Reranker call via OpenAI-compatible API
```python
async def rerank(query: str, documents: list[str], top_k: int = 5) -> list[dict]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.reranker_url}/v1/rerank",
            json={
                "model": "Qwen/Qwen3-Reranker-0.6B",
                "query": query,
                "documents": documents,
            },
        )
        results = resp.json()["results"]
        return sorted(results, key=lambda x: x["relevance_score"], reverse=True)[:top_k]
```
