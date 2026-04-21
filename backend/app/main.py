"""AniMind Backend — FastAPI entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import router
from app.config import settings

app = FastAPI(
    title="AniMind API",
    description="Anime/Manga RAG Chatbot powered by LangGraph",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup() -> None:
    logger.info("AniMind API starting up")
    logger.info("Qdrant: {}", settings.qdrant_url)
    logger.info("Reranker: {}", settings.reranker_url)


@app.on_event("shutdown")
async def shutdown() -> None:
    logger.info("AniMind API shutting down")
