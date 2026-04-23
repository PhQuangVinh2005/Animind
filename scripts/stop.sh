#!/bin/bash
# AniMind — Stop all local services
# Usage: bash scripts/stop.sh
#
# Stops (local only):
#   1. FastAPI backend (uvicorn :8000)
#   2. Docker containers (Qdrant + vLLM Reranker)
#
# Does NOT touch: Cloudflare Tunnel, Vercel (cloud services)

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[✔]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/.logs"
UVICORN_PID="$LOG_DIR/backend.pid"

echo ""
echo -e "${YELLOW}╔══════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║      AniMind — Stopping Services     ║${NC}"
echo -e "${YELLOW}╚══════════════════════════════════════╝${NC}"
echo ""

# ── 1. Stop FastAPI backend ───────────────────────────────────────────────────

# Try PID file first (clean shutdown)
if [ -f "$UVICORN_PID" ]; then
    PID=$(cat "$UVICORN_PID")
    if kill -0 "$PID" 2>/dev/null; then
        if kill "$PID" 2>/dev/null; then
            info "Backend stopped (PID $PID)"
        else
            warn "Failed to kill PID $PID"
        fi
    else
        warn "Backend PID $PID not found (already stopped?)"
    fi
    rm -f "$UVICORN_PID"
fi

# Fallback: kill anything on :8000 (covers --reload worker processes too)
if lsof -ti :8000 > /dev/null 2>&1; then
    PIDS=$(lsof -ti :8000)
    # shellcheck disable=SC2086  # intentional word splitting — multiple PIDs
    kill -9 $PIDS 2>/dev/null && info "Killed all processes on :8000 (PIDs: $PIDS)" || true
else
    info "Port 8000 already free"
fi

# ── 2. Stop Docker containers ─────────────────────────────────────────────────
cd "$PROJECT_DIR"
if docker compose ps --services --filter status=running 2>/dev/null | grep -q .; then
    docker compose down
    info "Docker containers stopped (Qdrant + Reranker)"
else
    info "Docker containers already stopped"
fi

# ── 3. Skipped (cloud services — intentionally left running) ──────────────────
echo ""
echo -e "  ${YELLOW}[skipped]${NC} cloudflared-animind  — cloud service, not touched"
echo -e "  ${YELLOW}[skipped]${NC} Vercel               — cloud service, not touched"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║        All Local Services Down       ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════╝${NC}"
echo ""
echo "  To start again: bash $SCRIPT_DIR/start.sh"
echo ""
