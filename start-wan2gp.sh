#!/usr/bin/env bash
set -Eeuo pipefail

# ========== Config ==========
WAN2GP_PORT="${WAN2GP_PORT:-7862}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
WAN2GP_DIR="${WAN2GP_DIR:-/opt/Wan2GP}"
LOG="${WAN2GP_LOG:-/workspace/wan2gp.log}"

# Optional auth (off by default)
WAN2GP_USERNAME="${WAN2GP_USERNAME:-}"
WAN2GP_PASSWORD="${WAN2GP_PASSWORD:-}"

# Use public GH release by default (no HF token)
LORA_DIR="${LORA_DIR:-/workspace/loras/wan2.2-animate-lightning}"
LORA1_NAME="${LORA1_NAME:-lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors}"
LORA1_URL="${LORA1_URL:-https://github.com/ArpitKhurana-ai/wan2gp-assets/releases/download/v1.0/${LORA1_NAME}}"
LORA2_NAME="${LORA2_NAME:-Wan2.1_I2V_14B_FusionX_LoRA.safetensors}"
LORA2_URL="${LORA2_URL:-https://github.com/ArpitKhurana-ai/wan2gp-assets/releases/download/v1.0/${LORA2_NAME}}"

# Optional manifest bootstrap (raw URL to assets/manifest.json)
BOOTSTRAP_ASSETS="${BOOTSTRAP_ASSETS:-1}"
MANIFEST_URL="${MANIFEST_URL:-https://raw.githubusercontent.com/ArpitKhurana-ai/wan2gp-template/main/assets/manifest.json}"

HEALTH_TRIES="${HEALTH_TRIES:-180}"   # 6min @ 2s
HEALTH_INTERVAL=2

# ========== Helpers ==========
PY="/opt/conda/bin/python"
JUPY="/opt/conda/bin/jupyter"

log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
ensure_dirs(){
  mkdir -p /workspace/models /workspace/outputs /workspace/.cache "$LORA_DIR"
  touch "$LOG"
}
gpu_name(){ nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo unknown; }

fetch_one(){ # name url dest_dir
  local name="$1"; local url="$2"; local dest_dir="$3"
  local path="${dest_dir%/}/$name"
  mkdir -p "$dest_dir"
  if [ -s "$path" ]; then
    log "Asset present: $name → $dest_dir"
    return 0
  fi
  log "Downloading: $name"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x16 -s16 -k1M --file-allocation=none -o "$name" -d "$dest_dir" "$url" \
      || { log "WARN: aria2c failed for $name, trying curl"; curl -fL --retry 3 -o "$path" "$url" || true; }
  else
    curl -fL --retry 3 -o "$path" "$url" || true
  fi
  if [ -s "$path" ]; then
    log "Ready: $name"
  else
    log "WARN: Missing after download: $name ($url)"
  fi
}

prefetch_loras(){
  fetch_one "$LORA1_NAME" "$LORA1_URL" "$LORA_DIR"
  fetch_one "$LORA2_NAME" "$LORA2_URL" "$LORA_DIR"
}

bootstrap_assets(){
  [ "$BOOTSTRAP_ASSETS" = "1" ] || { log "Bootstrap disabled (BOOTSTRAP_ASSETS=0)"; return 0; }
  log "Bootstrapping assets from manifest: $MANIFEST_URL"
  local mf="/tmp/manifest.json"
  if ! curl -fsSL "$MANIFEST_URL" -o "$mf"; then
    log "WARN: failed to fetch manifest"
    return 0
  fi

  # Parse manifest with Python and feed lines into while loop
  while IFS='|' read -r name url dest; do
    [ -n "${name:-}" ] || continue
    fetch_one "$name" "$url" "$dest"
  done < <(
    "$PY" - "$mf" <<'PYCODE'
import json, sys
mf = sys.argv[1]
data = json.load(open(mf))
for it in data.get("items", []):
    name = (it.get("name") or "").strip()
    url  = (it.get("url")  or "").strip()
    dest = (it.get("dest") or "/workspace/models").strip()
    if name and url:
        print(f"{name}|{url}|{dest}")
PYCODE
  )
}

disable_sage_on_blackwell(){
  local g="$(gpu_name)"
  if echo "$g" | grep -Eiq '5090|blackwell|gb2|gb|50'; then
    export WAN2GP_ATTENTION="sdpa"
    log "Blackwell/5090 detected: forcing attention = SDPA"
  fi
}

log "Ensuring JupyterLab is installed..."
pip install --no-cache-dir jupyterlab==4.1.8 notebook==7.1.2 ipykernel --upgrade

start_jupyter(){
  if [ "${ENABLE_JUPYTER:-1}" = "1" ]; then
    log "Ensuring JupyterLab is installed..."
    export PATH="/opt/conda/bin:$PATH"
    pip install --quiet --no-cache-dir jupyterlab==4.1.8 notebook==7.1.2 ipykernel --upgrade

    log "Launching JupyterLab safely as root on :${JUPYTER_PORT}"
    nohup /opt/conda/bin/jupyter lab \
      --ip=0.0.0.0 \
      --port="${JUPYTER_PORT}" \
      --allow-root \
      --ServerApp.allow_origin='*' \
      --ServerApp.allow_remote_access=True \
      --ServerApp.disable_check_xsrf=True \
      --NotebookApp.token='' \
      --NotebookApp.password='' \
      --no-browser >>"$LOG" 2>&1 &
    sleep 6

    if lsof -i :"${JUPYTER_PORT}" >/dev/null 2>&1; then
      log "✅ JupyterLab successfully bound to port ${JUPYTER_PORT}"
    else
      log "❌ JupyterLab failed to start or bind on port ${JUPYTER_PORT}"
      tail -n 20 "$LOG" | grep -i "jupyter" || log "(no jupyter error found in log tail)"
    fi
  fi
}


start_wan2gp(){
  # Optional basic auth via env
  if [ -n "$WAN2GP_USERNAME" ] && [ -n "$WAN2GP_PASSWORD" ]; then
    export WAN2GP_AUTH_USER="$WAN2GP_USERNAME"
    export WAN2GP_AUTH_PASS="$WAN2GP_PASSWORD"
    log "Auth enabled via env (user set)"
  fi

# Enable hf_transfer if installed (safe inline check)
  if $PY -c "import importlib, sys; sys.exit(0 if getattr(importlib, 'util', None) and importlib.util.find_spec('hf_transfer') else 1)"; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
    log 'hf_transfer detected and enabled'
  else
    log 'hf_transfer not found (skipping)'
  fi

  log "Starting Wan2GP on :${WAN2GP_PORT} (logs → $LOG)"
  "$PY" "${WAN2GP_DIR}/wgp.py" --listen --server-port "${WAN2GP_PORT}" >>"$LOG" 2>&1 &
}

health_wait(){
  log "Waiting for HTTP 200 on / (port ${WAN2GP_PORT})…"
  for _ in $(seq 1 "$HEALTH_TRIES"); do
    if curl -fsS "http://127.0.0.1:${WAN2GP_PORT}/" >/dev/null 2>&1; then
      log "Wan2GP UI is READY on port ${WAN2GP_PORT}"
      return 0
    fi
    sleep "$HEALTH_INTERVAL"
  done
  log "WARN: UI not healthy within the wait window. Tail logs for details."
}

# ========== Run ==========
ensure_dirs
log "=== BOOT $(date -u) | Wan2GP — ProbeAI (CUDA 12.8) ==="
log "GPU: $(gpu_name)"

disable_sage_on_blackwell
prefetch_loras
bootstrap_assets
start_jupyter
start_wan2gp
health_wait

# Keep container alive indefinitely
log "Wan2GP services launched. Container will stay alive."
tail -f /dev/null

