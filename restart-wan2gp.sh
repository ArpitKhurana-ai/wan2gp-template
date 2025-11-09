#!/usr/bin/env bash
set -euo pipefail
pkill -f "wgp.py" || true
sleep 1
exec /opt/start-wan2gp.sh
