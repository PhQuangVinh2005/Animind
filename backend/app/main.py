"""AniMind Backend — FastAPI entrypoint with lifespan-managed agent.

Lifespan sequence
-----------------
startup:
  1. Build AsyncOpenAI client (ShopAIKey)
  2. Open AsyncSqliteSaver (SQLite checkpointer for persistent conversation)
  3. Compile LangGraph StateGraph → store as app.state.agent
shutdown:
  4. AsyncSqliteSaver context manager closes the DB connection cleanly

Middleware stack — effective dispatch order (request path):
  RequestIDMiddleware       — outermost: assigns X-Request-ID first
  SecurityHeadersMiddleware — adds hardened HTTP headers (outside CORS)
  CORSMiddleware            — handles preflight + CORS headers
  RequestLoggingMiddleware  — innermost: logs latency + status

Exception handlers:
  HTTPException            → consistent JSON error envelope
  RequestValidationError   → 422 with field-level detail, no Pydantic internals
  Exception                → 500 with generic message, full trace to loguru

Known gotcha (from progress.json):
    AsyncSqliteSaver.from_conn_string() is an ASYNC context manager.
    Use `async with AsyncSqliteSaver.from_conn_string(...) as saver:`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from loguru import logger
from starlette.exceptions import HTTPException

from app.agent.graph import build_graph
from app.api.exceptions import (
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.api.routes import router
from app.config import settings
from app.middleware import (
    RequestIDMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
)
from app.openai_client import make_openai_client


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup / shutdown of shared resources.

    Resources stored on app.state:
        app.state.agent   — compiled LangGraph CompiledStateGraph
        app.state._saver  — AsyncSqliteSaver (kept for clean-shutdown reference)
    """
    logger.info("AniMind API starting up")
    logger.info("Qdrant:   {}", settings.qdrant_url)
    logger.info("Reranker: {}", settings.reranker_url)
    logger.info("Agent DB: {}", settings.agent_db_path)
    logger.info("CORS origins: frontend={}, +*.vercel.app", settings.frontend_url)

    oai_client = make_openai_client()

    async with AsyncSqliteSaver.from_conn_string(settings.agent_db_path) as saver:
        agent = build_graph(oai_client, checkpointer=saver)
        app.state.agent = agent
        app.state._saver = saver
        logger.info("LangGraph agent compiled and ready (AsyncSqliteSaver)")

        yield   # ← application runs here

    logger.info("AniMind API shut down cleanly")


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AniMind API",
    description=(
        "Anime/Manga RAG Chatbot powered by LangGraph + Qdrant + Qwen3-Reranker.\n\n"
        "## Endpoints\n"
        "- **POST /chat** — blocking full response\n"
        "- **POST /chat/stream** — SSE token streaming\n"
        "- **GET /health** — liveness + Qdrant/reranker probe\n"
        "- **GET /chat/sessions/{thread_id}** — checkpointer persistence probe\n"
    ),
    version="0.4.0",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────
# Starlette add_middleware is LIFO: last registered = outermost = runs first.
# Registration order below (innermost first, outermost last):
#
#   add(Logging)    ← 1st = innermost
#   add(CORS)       ← 2nd
#   add(Security)   ← 3rd — outside CORS so security headers survive
#   add(RequestID)  ← 4th = outermost — runs before everything else

app.add_middleware(RequestLoggingMiddleware)     # innermost (added 1st)
app.add_middleware(
    CORSMiddleware,                              # added 2nd
    allow_origins=[
        settings.frontend_url,        # http://localhost:3000 in dev
        "https://*.vercel.app",        # Vercel preview + prod deployments
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],  # let frontend read the correlation ID
)
app.add_middleware(SecurityHeadersMiddleware)    # added 3rd — outside CORS
app.add_middleware(RequestIDMiddleware)          # added 4th = outermost

# ── Exception handlers ────────────────────────────────────────────────────────

app.add_exception_handler(HTTPException, http_exception_handler)                 # type: ignore[arg-type]
app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
app.add_exception_handler(Exception, unhandled_exception_handler)

# ── Router ────────────────────────────────────────────────────────────────────

app.include_router(router)
