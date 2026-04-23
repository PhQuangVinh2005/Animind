"""AniMind — ASGI middleware stack.

All three middlewares are implemented as pure ASGI (not BaseHTTPMiddleware)
to avoid known Starlette bugs where BaseHTTPMiddleware loses response header
mutations when mixed with CORSMiddleware or other pure-ASGI layers.

Pure ASGI pattern: intercept `http.response.start` message and mutate its
`headers` list before forwarding — this is 100% reliable across all
Starlette / FastAPI versions.

Middleware dispatch order (outermost → innermost):
    1. RequestIDMiddleware       — assigns X-Request-ID first
    2. SecurityHeadersMiddleware — adds hardened HTTP headers
    3. RequestLoggingMiddleware  — logs method/path/status/latency
    4. CORSMiddleware            — (registered in main.py after these)

Security headers applied (OWASP recommendations):
    X-Content-Type-Options: nosniff          — prevent MIME sniffing
    X-Frame-Options: DENY                    — clickjacking protection
    X-XSS-Protection: 1; mode=block         — legacy browser XSS filter
    Referrer-Policy: strict-origin-when-cross-origin
    Permissions-Policy: camera=(), microphone=(), geolocation=()
    X-Request-ID: <uuid>                     — log correlation
"""

from __future__ import annotations

import time
import uuid

from loguru import logger
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send


# ── Request ID ────────────────────────────────────────────────────────────────

class RequestIDMiddleware:
    """Pure-ASGI middleware: attach UUID4 request ID to every request/response.

    - Stored in scope["state"]["request_id"] for downstream access.
    - Echoed as X-Request-ID response header for client-side log correlation.
    - Respects any X-Request-ID forwarded by a proxy (e.g. Cloudflare).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # Generate or propagate an existing request ID
        headers = dict(scope.get("headers", []))
        request_id = (
            headers.get(b"x-request-id", b"").decode() or str(uuid.uuid4())
        )

        # Starlette stores state on scope["state"]
        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                mutable = MutableHeaders(scope=message)
                mutable.append("X-Request-ID", request_id)
            await send(message)

        await self.app(scope, receive, send_with_request_id)


# ── Security headers ──────────────────────────────────────────────────────────

class SecurityHeadersMiddleware:
    """Pure-ASGI middleware: add hardened HTTP response headers on every response.

    Headers are injected into http.response.start before the response is
    sent to the client — guaranteed to appear regardless of response type
    (JSON, streaming, SSE, etc.).
    """

    _HEADERS: list[tuple[str, str]] = [
        ("X-Content-Type-Options", "nosniff"),
        ("X-Frame-Options", "DENY"),
        ("X-XSS-Protection", "1; mode=block"),
        ("Referrer-Policy", "strict-origin-when-cross-origin"),
        ("Permissions-Policy", "camera=(), microphone=(), geolocation=()"),
    ]

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                mutable = MutableHeaders(scope=message)
                for name, value in self._HEADERS:
                    if name not in mutable:
                        mutable.append(name, value)
            await send(message)

        await self.app(scope, receive, send_with_security_headers)


# ── Request logging ───────────────────────────────────────────────────────────

class RequestLoggingMiddleware:
    """Pure-ASGI middleware: structured log per request with latency and status.

    Skips /health to avoid polluting logs with frequent probe noise.
    Format:  METHOD path | status | latency_ms ms | req=<id>
    """

    _SKIP_PATHS = {"/health"}

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in self._SKIP_PATHS:
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "?")
        rid = scope.get("state", {}).get("request_id", "-")
        t0 = time.perf_counter()
        status_code: int = 0

        async def send_with_logging(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
            await send(message)

        await self.app(scope, receive, send_with_logging)

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "{method} {path} | {status} | {lat:.1f}ms | req={rid}",
            method=method,
            path=path,
            status=status_code,
            lat=latency_ms,
            rid=rid,
        )
