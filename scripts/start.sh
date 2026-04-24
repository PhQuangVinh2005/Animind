#!/bin/bash
# AniMind — Start all services (wrapper)
# This is a convenience wrapper. See scripts/start-all.sh for the full script.
#
# Old bare-metal version backed up at: scripts/start.sh.bak

exec "$(dirname "${BASH_SOURCE[0]}")/start-all.sh" "$@"
