#!/usr/bin/env bash
set -Eeuo pipefail

# ========== Config ==========
WAN2GP_PORT="${WAN2GP_PORT:-7862}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
WAN2GP_DIR="${WAN2GP_DIR:-/opt/Wan2GP}"
VENV_DIR="${VENV_DIR:-/opt/venv}"
LOG="${WAN2GP_LOG:-/workspace/wan2gp.log}"

# Optional auth (disabled by default for zero-friction)
WAN2GP_USERNAME="${WAN2GP_USERNAME:-}"
WAN2GP_PASSWORD="${WAN2GP_PASSWORD:-}"

# Lightning LoRAs we prefetch quickly if missing (tiny; keeps boot fast)
# Provide working URLs if you want to override; defaults assume public HF paths.
LORA_DIR="${LORA_DIR:-/workspace/loras/wan2.2-animate-lightning}"
LORA1_NAME="${LORA1_NAME:-lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors}"
LORA1_URL="${LORA1_URL:-https://huggingface.co/Wan-AI/LightX2V-LoRA/resolve/main/${LORA1_NAME}}"
LORA2_NAME="${LORA2_NAME:-Wan2.1_I2V_14B_FusionX_LoRA.safetensors}"
LORA2_URL="${LORA2_URL:-https://huggingface.co/Wan-AI/LightX2V-LoRA/resolve/main/${LORA2_NAME}}"

# Health waits
HEALTH_TRIES="${HEALTH_TRIES:-180}"   # 6 minutes @ 2s
HEALTH_INTERVAL=2

# ========== Helpers ==========
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG" ; }
ensure_dirs(){
  mkdir -p /workspace/models /workspace/outputs /workspace/.cache "$LORA_DIR"
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
  # Parallel + resume
  aria2c -x16 -s16 -k1M --file-allocation=none -o "$name" -d "$LORA_DIR" "$url" \
    || (log "WARN: aria2c failed for $name, trying curl"; curl -L --retry 3 -o "$path" "$url" || true)
  if [ -s "$path" ]; then
    log "LoRA ready: $name"
  else
    log "WARN: LoRA missing after download attempt: $name"
  fi
}

disable_sage_on_blackwell(){
  local g="$(gpu_name)"
  if echo "$g" | grep -Eiq '5090|blackwell|gb2|gb|50'; then
    # Many templates require SDPA when SageAttention causes CUDA asserts on Blackwell.
    export WAN2GP_ATTENTION="sdpa"
    log "Blackwell/5090 detected: forcing attention = SDPA"
  fi
}

start_jupyter(){
  if [ "${ENABLE_JUPYTER:-1}" = "1" ]; then
    . "${VENV_DIR}/bin/activate"
    jupyter lab --ip=0.0.0.0 --port="${JUPYTER_PORT}" --NotebookApp.token='' --NotebookApp.password='' \
      >>"$LOG" 2>&1 &
    log "JupyterLab starting on :${JUPYTER_PORT}"
  fi
}

start_wan2gp(){
  . "${VENV_DIR}/bin/activate"

  # Optional basic auth via env (off by default)
  if [ -n "$WAN2GP_USERNAME" ] && [ -n "$WAN2GP_PASSWORD" ]; then
    export WAN2GP_AUTH_USER="$WAN2GP_USERNAME"
    export WAN2GP_AUTH_PASS="$WAN2GP_PASSWORD"
    log "Auth enabled via env (user set)"
  fi

  # Honor hf_transfer if present
  if python -c "import importlib; import sys; sys.exit(0 if importlib.util.find_spec('hf_transfer') else 1)"; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
  fi

  log "Starting Wan2GP on :${WAN2GP_PORT} (logs → $LOG)"
  python "${WAN2GP_DIR}/wgp.py" \
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
log "=== BOOT $(date -u) | Wan2GP — ProbeAI (CUDA 12.8) ==="
log "GPU: $(gpu_name)"

disable_sage_on_blackwell

# Pre-place Lightning LoRAs (small, quick)
prefetch_lora "$LORA1_NAME" "$LORA1_URL"
prefetch_lora "$LORA2_NAME" "$LORA2_URL"

start_jupyter
start_wan2gp
health_wait

# Keep container in foreground
wait -n || true
