#!/bin/bash
# AniMind — Start all services (infra + backend + frontend)
# Usage: bash scripts/start-all.sh [--build]
#
# Starts: Qdrant, Reranker, NGINX, Dozzle, Backend, Frontend
# Logs:   http://localhost:9999 (Dozzle)

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${GREEN}[✔]${NC} $1"; }
waiting() { echo -e "${BLUE}[…]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║    AniMind — Starting All Services   ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════╝${NC}"
echo ""

cd "$PROJECT_DIR"

BUILD_FLAG=""
if [[ "${1:-}" == "--build" ]]; then
    BUILD_FLAG="--build"
    info "Build flag detected — rebuilding images"
fi

waiting "Starting all services (infra + backend + frontend)..."
docker compose --profile backend --profile frontend up -d $BUILD_FLAG

# ── Wait for key services ─────────────────────────────────────────────────────
waiting "Waiting for Qdrant..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
        info "Qdrant healthy"; break
    fi
    [ "$i" -eq 30 ] && echo -e "${YELLOW}[!]${NC} Qdrant not ready after 30s"
    sleep 1
done

waiting "Waiting for Backend..."
for i in $(seq 1 30); do
    if docker inspect --format='{{.State.Health.Status}}' animind-backend 2>/dev/null | grep -q healthy; then
        info "Backend healthy"; break
    fi
    [ "$i" -eq 30 ] && echo -e "${YELLOW}[!]${NC} Backend not ready after 30s"
    sleep 1
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          All Services Up             ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════╝${NC}"
echo ""
echo "  Qdrant         → http://localhost:6333"
echo "  Reranker       → http://localhost:8001"
echo "  Backend API    → http://localhost:8000 (via NGINX)"
echo "  Frontend       → http://localhost:3000 (via NGINX)"
echo "  NGINX          → http://localhost:80"
echo "  Dozzle Logs    → http://localhost:9999"
echo ""
echo "  Public:"
echo "    Chat UI      → https://chat.vinhkaguya.me"
echo "    API          → https://api.vinhkaguya.me"
echo ""
echo "  Logs:          docker compose logs -f backend"
echo "  Stop all:      bash $SCRIPT_DIR/stop-all.sh"
echo ""
