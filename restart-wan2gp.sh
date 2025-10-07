#!/usr/bin/env bash
set -euo pipefail
pkill -9 -f "python.*wgp.py" || true
sleep 1
/opt/start-wan2gp.sh
