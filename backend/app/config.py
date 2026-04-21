"""AniMind Backend — FastAPI application configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "anime"

    # Reranker
    reranker_url: str = "http://localhost:8001"
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    reranker_top_k: int = 5
    retriever_top_k: int = 20

    # AniList
    anilist_api_url: str = "https://graphql.anilist.co"

    # CORS
    frontend_url: str = "http://localhost:3000"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
