#!/usr/bin/env bash
set -Eeuo pipefail

LOG="${WAN2GP_LOG:-/workspace/wan2gp.log}"
WAN2GP_DIR="${WAN2GP_DIR:-/opt/Wan2GP}"
PY="/opt/conda/bin/python"
PORT="${WAN2GP_PORT:-7862}"

mkdir -p /workspace/outputs /workspace/.cache "$(dirname "$LOG")" \
         /workspace/hf-home /workspace/hf-cache /workspace/.torchinductor
touch "$LOG"

# 16G swap (best-effort)
if ! grep -q "/workspace/wan2gp.swap" /proc/swaps 2>/dev/null; then
  (fallocate -l 16G /workspace/wan2gp.swap || dd if=/dev/zero of=/workspace/wan2gp.swap bs=1G count=16) || true
  chmod 600 /workspace/wan2gp.swap || true
  mkswap /workspace/wan2gp.swap >/dev/null 2>&1 || true
  swapon /workspace/wan2gp.swap  >/dev/null 2>&1 || true
fi

# memory-friendly defaults
export GRADIO_NUM_WORKERS=1
export GRADIO_CONCURRENCY_COUNT=1
export UV_THREADPOOL_SIZE=8
export MMGP_RESERVED_RAM_GB=10

# ensure gradio server vars in runtime too (no-op if already set by Dockerfile)
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-$PORT}"
export GRADIO_ROOT_PATH="${GRADIO_ROOT_PATH:-/}"
export GRADIO_ALLOW_FLAGGING="${GRADIO_ALLOW_FLAGGING:-never}"
export GRADIO_SHARE="${GRADIO_SHARE:-False}"
export GRADIO_USE_CDN="${GRADIO_USE_CDN:-False}"

# ffmpeg defaults (override by env if needed)
export WANGP_FFMPEG_VIDEO="${WANGP_FFMPEG_VIDEO:-"-c:v h264_nvenc -preset p1 -rc vbr -cq 22 -pix_fmt yuv420p -vf scale=trunc(iw/16)*16:trunc(ih/16)*16"}"
export WANGP_FFMPEG_AUDIO="${WANGP_FFMPEG_AUDIO:-"-c:a aac -b:a 128k"}"

echo "[$(date -u +%H:%M:%S)] === BOOT $(date -u) | Wan2GP Docker Optimized ===" | tee -a "$LOG"
echo "[$(date -u +%H:%M:%S)] 🚀 Starting Wan2GP on :$PORT" | tee -a "$LOG"
$PY - <<'PY' 2>/dev/null | sed 's/^/[gradio] /' | tee -a "$LOG" || true
import gradio, sys
print("version", gradio.__version__)
PY

cd "$WAN2GP_DIR"
"$PY" wgp.py --listen --server-port "$PORT" >>"$LOG" 2>&1 &

echo "[$(date -u +%H:%M:%S)] ⌛ Waiting for UI ..." | tee -a "$LOG"
for i in $(seq 1 180); do
  if curl -fs "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
    echo "[$(date -u +%H:%M:%S)] ✅ Wan2GP UI READY on port ${PORT}" | tee -a "$LOG"
    break
  fi
  sleep 2
done

tail -f "$LOG"
