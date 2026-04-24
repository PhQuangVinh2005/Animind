#!/bin/bash
# AniMind — Force rebuild all container images (no cache)
# Usage: bash scripts/rebuild.sh [backend|frontend]
#
# Examples:
#   bash scripts/rebuild.sh            # rebuild both
#   bash scripts/rebuild.sh backend    # rebuild backend only
#   bash scripts/rebuild.sh frontend   # rebuild frontend only

set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${GREEN}[✔]${NC} $1"; }
waiting() { echo -e "${BLUE}[…]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

TARGET="${1:-all}"

case "$TARGET" in
    backend)
        waiting "Rebuilding backend image..."
        docker compose --profile backend build --no-cache backend
        info "Backend image rebuilt — run: bash $SCRIPT_DIR/start-backend.sh"
        ;;
    frontend)
        waiting "Rebuilding frontend image..."
        docker compose --profile frontend build --no-cache frontend
        info "Frontend image rebuilt — run: bash $SCRIPT_DIR/start-frontend.sh"
        ;;
    all|*)
        waiting "Rebuilding all images..."
        docker compose --profile backend --profile frontend build --no-cache
        info "All images rebuilt — run: bash $SCRIPT_DIR/start-all.sh"
        ;;
esac
