#!/bin/bash
# AniMind — Stop all services (wrapper)
# This is a convenience wrapper. See scripts/stop-all.sh for the full script.
#
# Old bare-metal version backed up at: scripts/stop.sh.bak

exec "$(dirname "${BASH_SOURCE[0]}")/stop-all.sh" "$@"
