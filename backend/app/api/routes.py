"""AniMind API routes — REST + SSE streaming.

Endpoints
---------
GET  /health            — liveness + dependency reachability check
POST /chat              — blocking full-response chat
POST /chat/stream       — SSE token-by-token streaming chat

Design notes
------------
- Agent is retrieved from `request.app.state.agent` (compiled graph stored in
  FastAPI lifespan — zero per-request overhead).
- SSE uses `sse_starlette.sse.EventSourceResponse` wrapping an async generator
  that consumes `astream_events(version="v2")`.
- Only `on_chat_model_stream` events are forwarded; all other internal node
  events (retriever, reranker, tool calls) are silently dropped.
- A `[DONE]` sentinel event is emitted after the stream completes so the
  frontend knows when to stop rendering the typing indicator.
- thread_id comes in the POST body (not a header) — consistent across both
  endpoints.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

import httpx
from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import HumanMessage
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import ChatRequest, ChatResponse, HealthResponse, SessionInfo
from app.config import settings

router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _agent_config(thread_id: str) -> dict:
    """LangGraph config dict for a given session thread_id."""
    return {"configurable": {"thread_id": thread_id}}


async def _check_service(client: httpx.AsyncClient, url: str) -> str:
    """Return 'ok' if the URL responds within 2 s, else 'unreachable'."""
    try:
        resp = await client.get(url, timeout=2.0)
        return "ok" if resp.status_code < 500 else "degraded"
    except Exception:
        return "unreachable"


# ── Health check ──────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Liveness and dependency health check",
)
async def health_check() -> HealthResponse:
    """Check FastAPI liveness and probe Qdrant + reranker reachability."""
    async with httpx.AsyncClient() as client:
        qdrant_status, reranker_status = await asyncio.gather(
            _check_service(client, f"{settings.qdrant_url}/healthz"),
            _check_service(client, f"{settings.reranker_url}/health"),
        )

    return HealthResponse(
        status="ok",
        service="animind",
        qdrant=qdrant_status,
        reranker=reranker_status,
    )


# ── GET /chat/sessions/{thread_id} — checkpointer probe ──────────────────────

@router.get(
    "/chat/sessions/{thread_id}",
    response_model=SessionInfo,
    tags=["chat"],
    summary="Check if a conversation thread has persisted checkpoints",
)
async def get_session(request: Request, thread_id: str) -> SessionInfo:
    """Probe the SQLite checkpointer for an existing thread.

    Returns `exists: true` and a `message_count` if the thread has any
    stored checkpoints. Useful for frontend session restore logic.

    The thread_id must be a valid UUID4 — same constraint as ChatRequest.
    """
    import re
    _UUID4_RE = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )
    if not _UUID4_RE.match(thread_id):
        raise HTTPException(status_code=400, detail="thread_id must be a valid UUID4")

    agent = request.app.state.agent
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")

    config = _agent_config(thread_id.lower())
    state = await agent.aget_state(config)

    if not state or not state.values:
        return SessionInfo(thread_id=thread_id, exists=False, message_count=0)

    messages = state.values.get("messages", [])
    return SessionInfo(
        thread_id=thread_id,
        exists=True,
        message_count=len(messages),
    )


# ── POST /chat — blocking ─────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    summary="Send a message and receive a full response (blocking)",
)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """Invoke the LangGraph agent and return the complete answer.

    Suitable for simple integrations that don't need streaming.
    Uses the same `thread_id` → persistent conversation state as /chat/stream.
    """
    agent = request.app.state.agent
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")

    logger.info(
        "POST /chat | thread={t} | message={m!r}",
        t=body.thread_id,
        m=body.message[:80],
    )

    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=body.message)]},
            config=_agent_config(body.thread_id),
        )
    except Exception as exc:
        logger.exception("Agent invocation failed: {}", exc)
        raise HTTPException(status_code=500, detail="Agent error") from exc

    answer = result.get("final_answer") or ""
    intent = result.get("intent") or "qa"

    logger.info(
        "POST /chat | thread={t} | intent={i} | answer_len={n}",
        t=body.thread_id,
        i=intent,
        n=len(answer),
    )
    return ChatResponse(thread_id=body.thread_id, answer=answer, intent=intent)


# ── POST /chat/stream — SSE ───────────────────────────────────────────────────

@router.post(
    "/chat/stream",
    tags=["chat"],
    summary="Send a message and receive a token-streaming SSE response",
    response_class=EventSourceResponse,
    responses={
        200: {
            "description": "Server-Sent Events stream",
            "content": {"text/event-stream": {}},
        }
    },
)
async def chat_stream(request: Request, body: ChatRequest) -> EventSourceResponse:
    """Stream the agent's response token-by-token via Server-Sent Events.

    SSE event format
    ----------------
    - `data: <token>\\n\\n`   — incremental text chunk from the LLM
    - `data: [DONE]\\n\\n`    — stream completion sentinel
    - `event: error\\ndata: <message>\\n\\n` — on agent exception

    Frontend usage
    --------------
    Use the `EventSource` API (or `@microsoft/fetch-event-source`) pointing to
    this endpoint. Read `event.data` and append to the message buffer until
    `[DONE]` is received.
    """
    agent = request.app.state.agent
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")

    logger.info(
        "POST /chat/stream | thread={t} | message={m!r}",
        t=body.thread_id,
        m=body.message[:80],
    )

    async def token_generator() -> AsyncGenerator[dict, None]:
        """Yield SSE-compatible dicts for EventSourceResponse."""
        tokens_emitted = 0
        try:
            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=body.message)]},
                config=_agent_config(body.thread_id),
                version="v2",
            ):
                kind = event.get("event")

                # ── Only forward LLM token deltas ─────────────────────────────
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk is None:
                        continue
                    # AIMessageChunk carries token text in .content
                    token: str = getattr(chunk, "content", "") or ""
                    if token:
                        tokens_emitted += 1
                        yield {"data": token}

                # ── Optionally log intent from router node ────────────────────
                elif kind == "on_chain_end":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict) and "intent" in output:
                        logger.debug(
                            "stream | thread={t} | intent={i}",
                            t=body.thread_id,
                            i=output["intent"],
                        )

        except asyncio.CancelledError:
            # Client disconnected — normal; just stop
            logger.debug("SSE client disconnected | thread={}", body.thread_id)
            return
        except Exception as exc:
            logger.exception("Agent stream error | thread={} | {}", body.thread_id, exc)
            yield {"event": "error", "data": json.dumps({"detail": str(exc)})}
            return

        # ── Fetch final state (cards + fallback answer) ───────────────────────
        try:
            final_state = await agent.aget_state(_agent_config(body.thread_id))
            if final_state and final_state.values:
                intent = final_state.values.get("intent", "qa")

                # Fallback: if the provider didn't stream tokens (e.g. ShopAIKey),
                # emit the complete final_answer as a single 'answer' event so the
                # frontend can still display the response.
                if tokens_emitted == 0:
                    final_answer: str = final_state.values.get("final_answer") or ""
                    if final_answer:
                        logger.info(
                            "stream | no tokens emitted — sending full answer fallback "
                            "| thread={t} | len={n}",
                            t=body.thread_id,
                            n=len(final_answer),
                        )
                        yield {"event": "answer", "data": final_answer}

                # Emit anime card data
                cards: list[dict] = []

                if intent == "qa":
                    for doc in final_state.values.get("reranked_docs", [])[:5]:
                        payload = doc.get("payload")
                        if payload:
                            cards.append(payload)

                elif intent == "search":
                    tool_out = final_state.values.get("tool_output") or {}
                    for r in (tool_out.get("data") or {}).get("results", []):
                        payload = r.get("payload")
                        if payload:
                            cards.append(payload)

                elif intent == "detail":
                    tool_out = final_state.values.get("tool_output") or {}
                    anime = (tool_out.get("data") or {}).get("anime")
                    if anime:
                        cards.append(anime)

                if cards:
                    yield {"event": "cards", "data": json.dumps(cards)}
        except Exception as exc:
            logger.warning("Could not extract card data from state: {}", exc)

        # Emit completion sentinel
        yield {"data": "[DONE]"}
        logger.info("POST /chat/stream DONE | thread={}", body.thread_id)

    return EventSourceResponse(token_generator())
