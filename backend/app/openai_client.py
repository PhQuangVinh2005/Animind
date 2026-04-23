"""Shared OpenAI client factory for AniMind.

Both the LLM (GPT-4o-mini) and embedding (text-embedding-3-small) use the
ShopAIKey provider — an OpenAI-compatible API at https://api.shopaikey.com/v1.

Usage:
    from app.openai_client import make_openai_client, make_chat_model

    # For embeddings + router/rag/tool nodes (raw SDK):
    client = make_openai_client()
    resp = await client.embeddings.create(model=..., input=...)

    # For synthesizer node (LangChain Runnable — required for astream_events):
    chat_model = make_chat_model()
    response = await chat_model.ainvoke(lc_messages)
"""

from openai import AsyncOpenAI

from app.config import settings

# Browser-like User-Agent required by ShopAIKey's Cloudflare WAF.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def make_openai_client() -> AsyncOpenAI:
    """Return an async OpenAI client pointed at the ShopAIKey endpoint.

    Used by:
    - router_node (intent classification)
    - rag_node (filter extraction, query rewriting, contextualisation)
    - rerank_node (no LLM — calls vLLM directly via httpx)
    - tool_node (title extraction)
    - retriever + reranker (embeddings)

    NOT used by synthesizer_node — that uses make_chat_model() so that
    astream_events() can capture on_chat_model_stream token events.
    """
    return AsyncOpenAI(
        api_key=settings.shopaikey_api_key,
        base_url=settings.shopaikey_base_url,
        default_headers={"User-Agent": _USER_AGENT},
    )


def make_chat_model():
    """Return a LangChain ChatOpenAI Runnable pointed at ShopAIKey.

    WHY:  Raw AsyncOpenAI SDK calls are invisible to LangGraph's
          astream_events() machinery — they bypass LangChain's callback/tracer
          system entirely. Only LangChain Runnables (ChatOpenAI etc.) emit
          on_chat_model_stream events that the SSE endpoint can forward.

    USE:  Call this once in make_nodes() and bind to synthesizer_node.
          All other nodes can keep the raw AsyncOpenAI client — we only
          need streaming from the final synthesis step.

    Returns:
        ChatOpenAI instance with streaming=True, same creds as make_openai_client().
    """
    from langchain_openai import ChatOpenAI  # lazy import — avoids circular dep

    return ChatOpenAI(
        api_key=settings.shopaikey_api_key,
        base_url=settings.shopaikey_base_url,
        model=settings.openai_model,
        temperature=0.3,
        max_tokens=900,
        streaming=True,
        default_headers={"User-Agent": _USER_AGENT},
    )
