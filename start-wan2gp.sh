#!/usr/bin/env bash
set -Eeuo pipefail

# ========== Config ==========
WAN2GP_PORT="${WAN2GP_PORT:-7862}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
WAN2GP_DIR="${WAN2GP_DIR:-/opt/Wan2GP}"
LOG="${WAN2GP_LOG:-/workspace/wan2gp.log}"

# Optional auth
WAN2GP_USERNAME="${WAN2GP_USERNAME:-}"
WAN2GP_PASSWORD="${WAN2GP_PASSWORD:-}"

# Use your public GH release by default (no HF token needed)
LORA_DIR="${LORA_DIR:-/workspace/loras/wan2.2-animate-lightning}"
LORA1_NAME="${LORA1_NAME:-lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors}"
LORA1_URL="${LORA1_URL:-https://github.com/ArpitKhurana-ai/wan2gp-assets/releases/download/v1.0/${LORA1_NAME}}"
LORA2_NAME="${LORA2_NAME:-Wan2.1_I2V_14B_FusionX_LoRA.safetensors}"
LORA2_URL="${LORA2_URL:-https://github.com/ArpitKhurana-ai/wan2gp-assets/releases/download/v1.0/${LORA2_NAME}}"

# Optional manifest bootstrap (raw GitHub URL to your manifest.json)
BOOTSTRAP_ASSETS="${BOOTSTRAP_ASSETS:-1}"
MANIFEST_URL="${MANIFEST_URL:-https://raw.githubusercontent.com/ArpitKhurana-ai/wan2gp-template/main/assets/manifest.json}"

HEALTH_TRIES="${HEALTH_TRIES:-180}"
HEALTH_INTERVAL=2

# ========== Helpers ==========
PY="/opt/conda/bin/python"
JUPY="/opt/conda/bin/jupyter"

log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG" ; }
ensure_dirs(){
  mkdir -p /workspace/models /workspace/outputs /workspace/.cache "$LORA_DIR"
  touch "$LOG"
}
gpu_name(){ nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo unknown; }

fetch_one(){
  local name="$1" url="$2" dest_dir="$3"
  local path="${dest_dir%/}/$name"
  if [ -s "$path" ]; then
    log "Asset present: $name"
    return 0
  fi
  log "Downloading: $name"
  aria2c -x16 -s16 -k1M --file-allocation=none -o "$name" -d "$dest_dir" "$url" \
    || (log "WARN aria2c failed → curl fallback: $name"; curl -L --retry 3 -o "$path" "$url" || true)
  [ -s "$path" ] && log "Ready: $name" || log "WARN: missing after download: $name"
}

bootstrap_from_manifest(){
  # downloads all items in manifest to their "dest"
  if [ "$BOOTSTRAP_ASSETS" != "1" ]; then
    log "Manifest bootstrap disabled (BOOTSTRAP_ASSETS=0)"
    return 0
  fi
  log "Fetching manifest: $MANIFEST_URL"
  tmp="$(mktemp)"
  if ! curl -fsSL "$MANIFEST_URL" -o "$tmp"; then
    log "WARN: could not fetch manifest (skipping)"
    rm -f "$tmp"; return 0
  fi
  "$PY" - <<PY 2>/dev/null | while IFS=$'\t' read -r dest name url; do
import json,sys,os
p=json.load(open("$tmp","r"))
for it in p.get("items",[]):
    dest=it.get("dest",".")
    os.makedirs(dest,exist_ok=True)
    print(f"{dest}\t{it.get('name','')}\t{it.get('url','')}")
PY
  do
    fetch_one "$name" "$url" "$dest"
  done
  rm -f "$tmp"
}

prefetch_loras_quick(){
  fetch_one "$LORA1_NAME" "$LORA1_URL" "$LORA_DIR"
  fetch_one "$LORA2_NAME" "$LORA2_URL" "$LORA_DIR"
}

disable_sage_on_blackwell(){
  local g="$(gpu_name)"
  if echo "$g" | grep -Eiq '5090|blackwell|gb2|gb|50'; then
    export WAN2GP_ATTENTION="sdpa"
    log "Blackwell/5090 detected → forcing attention=SDPA"
  fi
}

start_jupyter(){
  if [ "${ENABLE_JUPYTER:-1}" = "1" ]; then
    "$JUPY" lab --ip=0.0.0.0 --port="${JUPYTER_PORT}" --NotebookApp.token='' --NotebookApp.password='' \
      >>"$LOG" 2>&1 &
    log "JupyterLab starting on :${JUPYTER_PORT}"
  fi
}

start_wan2gp(){
  # Optional basic auth via env (off by default)
  if [ -n "$WAN2GP_USERNAME" ] && [ -n "$WAN2GP_PASSWORD" ]; then
    export WAN2GP_AUTH_USER="$WAN2GP_USERNAME"
    export WAN2GP_AUTH_PASS="$WAN2GP_PASSWORD"
    log "Auth enabled via env (user set)"
  fi

  # Enable hf_transfer if installed
  if "$PY" - <<'PY'; then
import importlib,sys
sys.exit(0 if importlib.util.find_spec('hf_transfer') else 1)
PY
  then export HF_HUB_ENABLE_HF_TRANSFER=1; fi

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
  log "WARN: UI not healthy in time. Tail $LOG"
}

# ========== Run ==========
ensure_dirs
log "=== BOOT $(date -u) | Wan2GP — ProbeAI (CUDA 12.8) ==="
log "GPU: $(gpu_name)"

disable_sage_on_blackwell
bootstrap_from_manifest         # pull everything in assets/manifest.json
prefetch_loras_quick            # plus the two headline LoRAs

start_jupyter
start_wan2gp
health_wait
wait -n || true
