#!/bin/bash
# AniMind — Start all local services
# Usage: bash scripts/start.sh [--no-frontend]
#
# Starts:
#   1. Docker (Qdrant :6333 + vLLM Reranker :8001)
#   2. FastAPI backend (uvicorn :8000, conda animind env)
#
# Does NOT touch: Cloudflare Tunnel, Vercel (cloud services)

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${GREEN}[✔]${NC} $1"; }
warn()    { echo -e "${YELLOW}[!]${NC} $1"; }
error()   { echo -e "${RED}[✘]${NC} $1" >&2; exit 1; }
waiting() { echo -e "${BLUE}[…]${NC} $1"; }

# ── Resolve project root (script can be run from anywhere) ────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_DIR/backend"
LOG_DIR="$PROJECT_DIR/.logs"
CONDA_ENV="animind"
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniforge3")"
UVICORN_BIN="$CONDA_BASE/envs/$CONDA_ENV/bin/uvicorn"
UVICORN_LOG="$LOG_DIR/backend.log"
UVICORN_PID="$LOG_DIR/backend.pid"

mkdir -p "$LOG_DIR"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║      AniMind — Starting Services     ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════╝${NC}"
echo ""

# ── 1. Docker services ────────────────────────────────────────────────────────
waiting "Starting Docker containers (Qdrant + vLLM Reranker)..."
cd "$PROJECT_DIR"
docker compose up -d

# ── 2. Wait for Qdrant ────────────────────────────────────────────────────────
waiting "Waiting for Qdrant (:6333)..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
        info "Qdrant healthy"
        break
    fi
    if [ "$i" -eq 30 ]; then
        error "Qdrant did not become healthy within 30s. Check: docker compose logs qdrant"
    fi
    sleep 1
done

# ── 3. Wait for Reranker ──────────────────────────────────────────────────────
waiting "Waiting for vLLM Reranker (:8001) — model load may take ~30s..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
        info "Reranker healthy"
        break
    fi
    if [ "$i" -eq 60 ]; then
        warn "Reranker not healthy after 60s — continuing anyway. Check: docker compose logs reranker"
        break
    fi
    sleep 1
done

# ── 4. Kill any existing uvicorn on :8000 ─────────────────────────────────────
if lsof -ti :8000 > /dev/null 2>&1; then
    warn "Port 8000 already in use — killing existing process..."
    # shellcheck disable=SC2046  # lsof -ti returns space-separated PIDs, intentional splitting
    kill -9 $(lsof -ti :8000) 2>/dev/null || true
    sleep 1
fi

# ── 5. Start FastAPI backend ──────────────────────────────────────────────────
if [ ! -f "$UVICORN_BIN" ]; then
    error "uvicorn not found at $UVICORN_BIN — is conda env '$CONDA_ENV' created?"
fi

if [ ! -f "$BACKEND_DIR/.env" ]; then
    error ".env not found at $BACKEND_DIR/.env — copy .env.example and fill in SHOPAIKEY_API_KEY"
fi

waiting "Starting FastAPI backend (uvicorn :8000)..."
cd "$BACKEND_DIR"
nohup "$UVICORN_BIN" app.main:app \
    --host 127.0.0.1 \
    --port 8000 \
    --workers 1 \
    > "$UVICORN_LOG" 2>&1 &
echo $! > "$UVICORN_PID"
BACKEND_PID=$!

# ── 6. Wait for backend health ────────────────────────────────────────────────
waiting "Waiting for FastAPI backend (:8000)..."
for i in $(seq 1 20); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        info "Backend healthy (PID $BACKEND_PID)"
        break
    fi
    if [ "$i" -eq 20 ]; then
        warn "Backend not healthy after 20s — check logs: tail -f $UVICORN_LOG"
        break
    fi
    sleep 1
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║            All Services Up           ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════╝${NC}"
echo ""
echo "  Qdrant         → http://localhost:6333"
echo "  Qdrant UI      → http://localhost:6333/dashboard"
echo "  Reranker       → http://localhost:8001"
echo "  Backend API    → http://localhost:8000"
echo "  Swagger UI     → http://localhost:8000/docs"
echo ""
echo "  Backend log    → tail -f $UVICORN_LOG"
echo "  Backend PID    → $(cat "$UVICORN_PID" 2>/dev/null || echo '?')"
echo ""
echo "  To stop all:   bash $SCRIPT_DIR/stop.sh"
echo ""
