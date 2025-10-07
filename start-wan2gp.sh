#!/usr/bin/env bash
set -Eeuo pipefail

# ========== Config ==========
WAN2GP_PORT="${WAN2GP_PORT:-7862}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
WAN2GP_DIR="${WAN2GP_DIR:-/opt/Wan2GP}"
VENV_DIR="${VENV_DIR:-/opt/venv}"               # will be used only if it actually exists
LOG="${WAN2GP_LOG:-/workspace/wan2gp.log}"

# Prefer Conda python from base image; fall back to system python
export PATH="/opt/conda/bin:$PATH"
PYTHON_BIN="${PYTHON_BIN:-$(command -v /opt/conda/bin/python || command -v python)}"

# Optional auth (disabled by default for zero-friction)
WAN2GP_USERNAME="${WAN2GP_USERNAME:-}"
WAN2GP_PASSWORD="${WAN2GP_PASSWORD:-}"

# Mirrored LoRAs (defaults now point to your GitHub Release = no HF token)
LORA_DIR="${LORA_DIR:-/workspace/loras/wan2.2-animate-lightning}"
LORA1_NAME="${LORA1_NAME:-lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors}"
LORA1_URL="${LORA1_URL:-https://github.com/ArpitKhurana-ai/wan2gp-assets/releases/download/v1.0/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors}"
LORA2_NAME="${LORA2_NAME:-Wan2.1_I2V_14B_FusionX_LoRA.safetensors}"
LORA2_URL="${LORA2_URL:-https://github.com/ArpitKhurana-ai/wan2gp-assets/releases/download/v1.0/Wan2.1_I2V_14B_FusionX_LoRA.safetensors}"

# Optional manifest bootstrap (public mirrors via GH Releases)
MANIFEST_PATH="${MANIFEST_PATH:-/opt/assets/manifest.json}"
BOOTSTRAP_PY="${BOOTSTRAP_PY:-/opt/scripts/bootstrap_assets.py}"

# Health waits
HEALTH_TRIES="${HEALTH_TRIES:-180}"   # 6 minutes @ 2s
HEALTH_INTERVAL=2

# ========== Helpers ==========
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG" ; }
ensure_dirs(){
  mkdir -p /workspace/models /workspace/outputs /workspace/.cache "$LORA_DIR"
  mkdir -p /workspace/hf-home /workspace/hf-cache
  export HF_HOME="/workspace/hf-home"
  export HUGGINGFACE_HUB_CACHE="/workspace/hf-cache"
  export TRANSFORMERS_CACHE="/workspace/hf-cache"
  export XDG_CACHE_HOME="/workspace/.cache"
  touch "$LOG"
}
gpu_name(){ nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo unknown; }

prefetch_lora(){
  local name="$1" url="$2"
  local path="$LORA_DIR/$name"
  if [ -s "$path" ]; then
    log "LoRA present: $name"
    return 0
  fi
  log "Fetching LoRA: $name"
  aria2c -x16 -s16 -k1M --file-allocation=none -o "$name" -d "$LORA_DIR" "$url" \
    || (log "WARN: aria2c failed for $name, trying curl"; curl -L --retry 3 --retry-delay 2 -o "$path" "$url" || true)
  if [ -s "$path" ]; then
    log "LoRA ready: $name"
  else
    log "WARN: LoRA missing after download attempt: $name"
  fi
}

bootstrap_from_manifest(){
  if [ -f "$MANIFEST_PATH" ] && [ -f "$BOOTSTRAP_PY" ]; then
    log "Bootstrapping assets from manifest: $MANIFEST_PATH"
    "$PYTHON_BIN" "$BOOTSTRAP_PY" --manifest "$MANIFEST_PATH" || log "WARN: manifest bootstrap failed (continuing)"
  else
    log "INFO: Manifest/bootstrap script not found; skipping bootstrap."
  fi
}

disable_sage_on_blackwell(){
  local g="$(gpu_name)"
  if echo "$g" | grep -Eiq '5090|blackwell|gb2|gb|50'; then
    export WAN2GP_ATTENTION="sdpa"
    log "Blackwell/5090 detected: forcing attention = SDPA"
  fi
}

maybe_activate(){
  # Activate venv only if it exists; otherwise we rely on Conda python directly
  if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1090
    . "${VENV_DIR}/bin/activate"
  fi
}

start_jupyter(){
  if [ "${ENABLE_JUPYTER:-1}" = "1" ]; then
    maybe_activate
    "$PYTHON_BIN" -m jupyterlab --ip=0.0.0.0 --port="${JUPYTER_PORT}" --NotebookApp.token='' --NotebookApp.password='' \
      >>"$LOG" 2>&1 &
    log "JupyterLab starting on :${JUPYTER_PORT}"
  fi
}

start_wan2gp(){
  maybe_activate

  # Optional basic auth via env (off by default)
  if [ -n "$WAN2GP_USERNAME" ] && [ -n "$WAN2GP_PASSWORD" ]; then
    export WAN2GP_AUTH_USER="$WAN2GP_USERNAME"
    export WAN2GP_AUTH_PASS="$WAN2GP_PASSWORD"
    log "Auth enabled via env (user set)"
  fi

  # Honor hf_transfer if present
  if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1; then
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec('hf_transfer') else 1)
PY
  then
    export HF_HUB_ENABLE_HF_TRANSFER=1
  fi

  log "Starting Wan2GP on :${WAN2GP_PORT} (logs → $LOG)"
  "$PYTHON_BIN" "${WAN2GP_DIR}/wgp.py" \
    --listen \
    --server-port "${WAN2GP_PORT}" \
    >>"$LOG" 2>&1 &
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
log "=== BOOT $(date -u) | Wan2GP — ProbeAI (CUDA) ==="
log "GPU: $(gpu_name)"

disable_sage_on_blackwell

# Pull mirrored assets first (idempotent), then ensure the two key LoRAs exist
bootstrap_from_manifest
prefetch_lora "$LORA1_NAME" "$LORA1_URL"
prefetch_lora "$LORA2_NAME" "$LORA2_URL"

start_jupyter
start_wan2gp
health_wait

# Keep container in foreground
wait -n || true
