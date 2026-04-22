"""AniMind Backend — FastAPI application configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ShopAIKey — OpenAI-compatible provider used for chat + embeddings.
    # Client is initialised with:
    #   OpenAI(api_key=..., base_url=..., default_headers={"User-Agent": ...})
    shopaikey_api_key: str = ""
    shopaikey_base_url: str = "https://api.shopaikey.com/v1"
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

    # AniList OAuth2
    # Public media queries (Page/Media) do not require auth.
    # CLIENT_ID + CLIENT_SECRET → exchange for access_token via Authorization
    # Code Grant (https://docs.anilist.co/guide/auth/authorization-code).
    # ACCESS_TOKEN is a long-lived JWT (1 year). No refresh tokens.
    # Authenticated requests: Authorization: Bearer <access_token>
    anilist_api_url: str = "https://graphql.anilist.co"
    anilist_client_id: str = ""
    anilist_client_secret: str = ""
    anilist_access_token: str = ""   # optional — only for user-specific endpoints

    # LangGraph agent
    agent_db_path: str = "agent_state.db"  # SQLite for conversation checkpoints

    # CORS
    frontend_url: str = "http://localhost:3000"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
