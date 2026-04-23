"""AniMind API — Pydantic V2 request/response schemas."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator


# ── Request ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Payload for POST /chat and POST /chat/stream."""

    thread_id: str = Field(
        ...,
        description=(
            "Session identifier. Must be a UUID4 string. Each unique thread_id "
            "maintains its own conversation history via the LangGraph SQLite checkpointer."
        ),
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's natural-language message to the anime chatbot.",
        examples=["What are the best action anime from 2023?"],
    )

    @field_validator("thread_id")
    @classmethod
    def validate_thread_id(cls, v: str) -> str:
        """Require a valid UUID4 to prevent log injection and SQLite key collisions."""
        _UUID4_RE = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )
        if not _UUID4_RE.match(v):
            raise ValueError(
                "thread_id must be a valid UUID4 "
                "(e.g. 550e8400-e29b-41d4-a716-446655440000)"
            )
        return v.lower()


# ── Response ──────────────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    """Response payload for POST /chat (non-streaming)."""

    thread_id: str = Field(..., description="Echo of the session thread_id.")
    answer: str = Field(..., description="Full generated answer from the agent.")
    intent: str = Field(
        default="qa",
        description="Classified intent: 'qa' | 'search' | 'detail'.",
    )


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    service: str
    qdrant: str = "unknown"
    reranker: str = "unknown"


class SessionInfo(BaseModel):
    """Response for GET /chat/sessions/{thread_id}."""

    thread_id: str
    exists: bool = Field(..., description="True if this thread has persisted checkpoints.")
    message_count: int = Field(
        default=0,
        description="Number of messages (HumanMessage + AIMessage) in the thread history.",
    )
