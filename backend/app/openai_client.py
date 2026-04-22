"""Shared OpenAI client factory for AniMind.

Both the LLM (GPT-4o-mini) and embedding (text-embedding-3-small) use the
ShopAIKey provider — an OpenAI-compatible API at https://api.shopaikey.com/v1.

Usage:
    from app.openai_client import make_openai_client

    client = make_openai_client()
    resp = await client.embeddings.create(model=..., input=...)
    resp = await client.chat.completions.create(model=..., messages=...)
"""

from openai import AsyncOpenAI

from app.config import settings

# Browser-like User-Agent required by the ShopAIKey provider.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def make_openai_client() -> AsyncOpenAI:
    """Return an async OpenAI client pointed at the ShopAIKey endpoint.

    The client is configured with:
    - api_key    — SHOPAIKEY_API_KEY from .env
    - base_url   — SHOPAIKEY_BASE_URL (default: https://api.shopaikey.com/v1)
    - User-Agent — browser-like header required by the provider
    """
    return AsyncOpenAI(
        api_key=settings.shopaikey_api_key,
        base_url=settings.shopaikey_base_url,
        default_headers={"User-Agent": _USER_AGENT},
    )
