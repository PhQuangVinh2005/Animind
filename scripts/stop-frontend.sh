#!/bin/bash
# AniMind — Stop frontend only (infra stays running)
# Usage: bash scripts/stop-frontend.sh

set -euo pipefail

GREEN='\033[0;32m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"
docker compose stop frontend
echo -e "${GREEN}[✔]${NC} Frontend stopped (infra still running)"
