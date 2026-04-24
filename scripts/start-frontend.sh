#!/bin/bash
# AniMind — Start frontend only (+ infra dependencies)
# Usage: bash scripts/start-frontend.sh [--build]

set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${GREEN}[✔]${NC} $1"; }
waiting() { echo -e "${BLUE}[…]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

BUILD_FLAG=""
[[ "${1:-}" == "--build" ]] && BUILD_FLAG="--build"

waiting "Starting frontend (+ infra if not running)..."
docker compose --profile frontend up -d $BUILD_FLAG

waiting "Waiting for frontend health..."
for i in $(seq 1 30); do
    if docker inspect --format='{{.State.Health.Status}}' animind-frontend 2>/dev/null | grep -q healthy; then
        info "Frontend healthy"; break
    fi
    [ "$i" -eq 30 ] && echo -e "\033[1;33m[!]\033[0m Frontend not ready after 30s"
    sleep 1
done

info "Frontend → http://localhost:3000 | https://chat.vinhkaguya.me"
