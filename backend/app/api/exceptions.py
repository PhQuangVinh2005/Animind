"""AniMind — global exception handlers and error response schema.

All error responses use a single consistent envelope:

    {
        "error":      "<short machine-readable label>",
        "detail":     "<human-readable explanation>",  # may be null
        "request_id": "<UUID4>"                        # for log correlation
    }

Handlers registered here:
    HTTPException          — 4xx / 5xx raised by routes
    RequestValidationError — Pydantic V2 validation failures (422)
    Exception              — unhandled crash fallback (500)

Security principles applied (OWASP API Security Top 10):
    - No stack traces in responses (no internal detail leakage)
    - Generic 500 message; full trace goes to loguru only
    - Validation errors report field path + issue, not raw Pydantic internals
"""

from __future__ import annotations

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger


def _request_id(request: Request) -> str | None:
    """Safely retrieve the request ID set by RequestIDMiddleware."""
    return getattr(request.state, "request_id", None)


def _error_body(error: str, detail: str | None, request_id: str | None) -> dict:
    return {"error": error, "detail": detail, "request_id": request_id}


# ── HTTPException handler ─────────────────────────────────────────────────────

async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Convert FastAPI/Starlette HTTPException to the standard error envelope."""
    rid = _request_id(request)
    logger.warning(
        "HTTP {code} | {method} {path} | {detail} | req={rid}",
        code=exc.status_code,
        method=request.method,
        path=request.url.path,
        detail=exc.detail,
        rid=rid,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(
            error=_status_label(exc.status_code),
            detail=str(exc.detail) if exc.detail else None,
            request_id=rid,
        ),
        headers=exc.headers or {},
    )


# ── RequestValidationError handler ───────────────────────────────────────────

async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return a 422 with a human-readable summary of validation failures.

    Pydantic raw errors are mapped to: "<field>: <message>" pairs.
    We only expose field name and message — not internal Pydantic context.
    """
    rid = _request_id(request)
    issues = []
    for err in exc.errors():
        # loc[-1] is the field name; skip body/query wrapper tokens
        loc = err.get("loc", ())
        field = str(loc[-1]) if loc else "body"
        msg = err.get("msg", "invalid value")
        issues.append(f"{field}: {msg}")

    detail = " | ".join(issues) if issues else "Request body is invalid"
    logger.warning(
        "422 Validation | {method} {path} | {detail} | req={rid}",
        method=request.method,
        path=request.url.path,
        detail=detail,
        rid=rid,
    )
    return JSONResponse(
        status_code=422,
        content=_error_body(
            error="validation_error",
            detail=detail,
            request_id=rid,
        ),
    )


# ── Unhandled exception handler ───────────────────────────────────────────────

async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for any exception that slips through route handlers.

    Security: full traceback is logged server-side only; the client receives
    only a generic 500 message with a request_id for correlation.
    """
    rid = _request_id(request)
    logger.exception(
        "500 Unhandled | {method} {path} | req={rid} | {exc}",
        method=request.method,
        path=request.url.path,
        rid=rid,
        exc=exc,
    )
    return JSONResponse(
        status_code=500,
        content=_error_body(
            error="internal_server_error",
            detail="An unexpected error occurred. Please try again.",
            request_id=rid,
        ),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _status_label(status_code: int) -> str:
    """Short machine-readable label for common HTTP status codes."""
    return {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        409: "conflict",
        422: "validation_error",
        429: "rate_limit_exceeded",
        500: "internal_server_error",
        503: "service_unavailable",
    }.get(status_code, f"http_{status_code}")
