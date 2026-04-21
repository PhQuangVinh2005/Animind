"""API routes — REST + SSE streaming endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "service": "animind"}


# TODO Day 4: POST /chat — send message, get response
# TODO Day 4: GET /chat/stream — SSE streaming
