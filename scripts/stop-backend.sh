#!/bin/bash
# AniMind — Stop backend only (infra stays running)
# Usage: bash scripts/stop-backend.sh

set -euo pipefail

GREEN='\033[0;32m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"
docker compose stop backend
echo -e "${GREEN}[✔]${NC} Backend stopped (infra still running)"
