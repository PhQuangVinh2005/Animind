#!/bin/bash
# AniMind — Start backend only (+ infra dependencies)
# Usage: bash scripts/start-backend.sh [--build]

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

waiting "Starting backend (+ infra if not running)..."
docker compose --profile backend up -d $BUILD_FLAG

waiting "Waiting for backend health..."
for i in $(seq 1 30); do
    if docker inspect --format='{{.State.Health.Status}}' animind-backend 2>/dev/null | grep -q healthy; then
        info "Backend healthy"; break
    fi
    [ "$i" -eq 30 ] && echo -e "\033[1;33m[!]\033[0m Backend not ready after 30s"
    sleep 1
done

info "Backend API → http://localhost:8000 | https://api.vinhkaguya.me"
