#!/usr/bin/env bash
set -euo pipefail

export WORKSPACE_DIR="/workspace"
export HF_HOME="$WORKSPACE_DIR/hf-home"
export HUGGINGFACE_HUB_CACHE="$WORKSPACE_DIR/hf-cache"
export TRANSFORMERS_CACHE="$WORKSPACE_DIR/hf-cache"
export XDG_CACHE_HOME="$WORKSPACE_DIR/.cache"
export WAN2GP_DIR="/opt/Wan2GP"
export PATH="/opt/conda/bin:$PATH"

mkdir -p "$WORKSPACE_DIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" \
         "$WORKSPACE_DIR/models" "$WORKSPACE_DIR/outputs" "$WORKSPACE_DIR/loras"

echo "[BOOT] Wan2GP — ProbeAI (CUDA) :: $(date)"
nvidia-smi || true

# 1) Pull mirrored LoRAs (GitHub Releases) via manifest
if [ -f /opt/assets/manifest.json ]; then
  echo "[BOOT] Bootstrapping mirrored assets via manifest…"
  python /opt/scripts/bootstrap_assets.py --manifest /opt/assets/manifest.json
else
  echo "[WARN] /opt/assets/manifest.json not found; skipping bootstrap."
fi

# 2) Optional: pull any private HF assets if a token is provided (no token == skip)
if [ "${HF_TOKEN:-}" != "" ]; then
  echo "[BOOT] HF_TOKEN present; private HF assets will be accessible if your code requests them."
else
  echo "[INFO] No HF_TOKEN set. Public mirrors should suffice."
fi

# 3) Start the app if present
APP_PY="$WAN2GP_DIR/app.py"
if [ -f "$APP_PY" ]; then
  echo "[RUN] Starting app: $APP_PY (port 7862)"
  exec python "$APP_PY" --port 7862
fi

# Fallback: try the project’s own launcher if it exists
if [ -f "$WAN2GP_DIR/webui.sh" ]; then
  echo "[RUN] Starting webui.sh (port 7862)"
  exec bash "$WAN2GP_DIR/webui.sh"
fi

echo "[HOLD] No app.py or webui.sh found in $WAN2GP_DIR."
echo "       Container is up; attach a shell and run what you need."
tail -f /dev/null
