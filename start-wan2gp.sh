#!/usr/bin/env bash
set -Eeuo pipefail

# ================= CONFIG =================
WAN2GP_PORT="${WAN2GP_PORT:-7862}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
WAN2GP_DIR="${WAN2GP_DIR:-/opt/Wan2GP}"
LOG="${WAN2GP_LOG:-/workspace/wan2gp.log}"

WAN2GP_USERNAME="${WAN2GP_USERNAME:-}"
WAN2GP_PASSWORD="${WAN2GP_PASSWORD:-}"

# ---- LoRA CONFIG ----
LORA_CACHE_DIR="${LORA_CACHE_DIR:-/workspace/loras_cache}"   # temp cache
WAN2GP_LORA_DIR="${WAN2GP_LORA_DIR:-/opt/Wan2GP/loras}"     # ✅ official path for Wan2GP
LORA1_NAME="lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
LORA1_URL="https://github.com/ArpitKhurana-ai/wan2gp-assets/releases/download/v1.0/${LORA1_NAME}"
LORA2_NAME="Wan2.1_I2V_14B_FusionX_LoRA.safetensors"
LORA2_URL="https://github.com/ArpitKhurana-ai/wan2gp-assets/releases/download/v1.0/${LORA2_NAME}"

# ---- CACHE / GPU / WATCHDOG ----
HF_HOME="/workspace/hf-home"
HF_HUB_CACHE="/workspace/hf-cache"
MEMORY_WATCHDOG=1
CLEANUP_LORAS=1
DISK_THRESHOLD_GB=80
HEALTH_TRIES=180
HEALTH_INTERVAL=2

PY="/opt/conda/bin/python"

# ================= HELPERS =================
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

ensure_dirs(){
  mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$WAN2GP_LORA_DIR" "$LORA_CACHE_DIR" /workspace/outputs /workspace/.cache
  touch "$LOG"
}

gpu_name(){ nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo unknown; }
gpu_mem_gb(){ nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk '{print int($1/1024)}' || echo 0; }

# ================= CORE OPS =================
fetch_one(){
  local name="$1"; local url="$2"; local dest="$3"; local path="${dest%/}/$name"
  mkdir -p "$dest"
  [ -s "$path" ] && { log "✅ Cached: $name"; return; }
  log "⬇️ Downloading $name ..."
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x16 -s16 -k1M --file-allocation=none -d "$dest" -o "$name" "$url" \
      || curl -fL --retry 3 -o "$path" "$url"
  else
    curl -fL --retry 3 -o "$path" "$url"
  fi
  [ -s "$path" ] && log "✅ Ready: $name" || log "⚠️ Failed: $name"
}

prefetch_loras(){
  log "📦 Prefetching LoRAs..."
  fetch_one "$LORA1_NAME" "$LORA1_URL" "$LORA_CACHE_DIR" &
  fetch_one "$LORA2_NAME" "$LORA2_URL" "$LORA_CACHE_DIR" &
  wait
}

sync_loras(){
  log "🔄 Syncing LoRAs to official Wan2GP path..."
  mkdir -p "$WAN2GP_LORA_DIR"
  find "$LORA_CACHE_DIR" -maxdepth 1 -type f -name "*.safetensors" -exec cp -u {} "$WAN2GP_LORA_DIR"/ \; || true
  ln -sf "$WAN2GP_LORA_DIR" /workspace/loras        # visible for user
  log "✅ LoRAs placed in: $WAN2GP_LORA_DIR"
  ls -lh "$WAN2GP_LORA_DIR" | tee -a "$LOG" || true
}

cleanup_old_loras(){
  if [ "$CLEANUP_LORAS" = "1" ]; then
    local used_gb=$(df -BG /workspace | awk 'NR==2{gsub("G","",$3);print $3}')
    if (( used_gb > DISK_THRESHOLD_GB )); then
      log "🧹 Disk usage ${used_gb}GB > ${DISK_THRESHOLD_GB}GB — cleaning old LoRAs..."
      find "$WAN2GP_LORA_DIR" -type f -name "*.safetensors" -mtime +7 -delete || true
      log "✅ Cleanup done."
    fi
  fi
}

oom_prevention(){
  if [ "$MEMORY_WATCHDOG" = "1" ]; then
    local mem=$(gpu_mem_gb)
    if (( mem < 20 )); then
      export WAN2GP_PRECISION="bf16"; export WAN2GP_BATCH="1"
      log "⚙️ Low-VRAM (${mem} GB) → bf16 / batch 1"
    elif (( mem <= 40 )); then
      export WAN2GP_PRECISION="bf16"; export WAN2GP_BATCH="2"
      log "⚙️ Mid-VRAM (${mem} GB) → bf16 / batch 2"
    else
      export WAN2GP_PRECISION="fp16"; export WAN2GP_BATCH="4"
      log "⚙️ High-VRAM (${mem} GB) → fp16 / batch 4"
    fi
  fi
}

hf_transfer_cache(){
  if $PY -c "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('hf_transfer') else 1)" >/dev/null 2>&1; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
    export HF_HOME="$HF_HOME"
    export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
    log "⚡ Using hf_transfer cache ($HF_HUB_CACHE)"
  else
    log "ℹ️ hf_transfer not found; skipping acceleration."
  fi
}

# ================= JUPYTER =================
start_jupyter(){
  if [ "${ENABLE_JUPYTER:-1}" = "1" ]; then
    export PATH="/opt/conda/bin:$PATH"
    if ! command -v jupyter >/dev/null 2>&1; then
      log "📦 Installing Jupyter (first-time only)..."
      pip install --no-cache-dir jupyterlab==4.1.8 notebook==7.1.2 ipykernel jupyter_core==5.7.2
    fi
    mkdir -p /workspace/.jupyter
    export JUPYTER_CONFIG_DIR=/workspace/.jupyter
    log "🚀 Launching JupyterLab on port ${JUPYTER_PORT}"
    nohup jupyter lab --ip=0.0.0.0 --port="${JUPYTER_PORT}" --no-browser \
      --allow-root --ServerApp.allow_origin='*' --ServerApp.disable_check_xsrf=True \
      --ServerApp.allow_remote_access=True --ServerApp.root_dir=/workspace \
      --NotebookApp.token='' --NotebookApp.password='' \
      --ServerApp.default_url=/lab >>"$LOG" 2>&1 &
    sleep 5
    lsof -i :"${JUPYTER_PORT}" >/dev/null 2>&1 && log "✅ Jupyter ready" || log "❌ Jupyter failed to bind"
  fi
}

# ================= WAN2GP =================
start_wan2gp(){
  if [ -n "$WAN2GP_USERNAME" ] && [ -n "$WAN2GP_PASSWORD" ]; then
    export WAN2GP_AUTH_USER="$WAN2GP_USERNAME"
    export WAN2GP_AUTH_PASS="$WAN2GP_PASSWORD"
  fi
  log "🚀 Starting Wan2GP on :${WAN2GP_PORT}"
  "$PY" "${WAN2GP_DIR}/wgp.py" --listen --server-port "${WAN2GP_PORT}" >>"$LOG" 2>&1 &
}

health_wait(){
  log "⌛ Waiting for Wan2GP..."
  for _ in $(seq 1 "$HEALTH_TRIES"); do
    if curl -fs "http://127.0.0.1:${WAN2GP_PORT}/" >/dev/null 2>&1; then
      log "✅ Wan2GP UI READY on port ${WAN2GP_PORT}"
      return
    fi
    sleep "$HEALTH_INTERVAL"
  done
  log "⚠️ Timed out waiting for UI."
}

# ================= EXECUTION =================
ensure_dirs
log "=== BOOT $(date -u) | Wan2GP Optimized v4 ==="
log "GPU: $(gpu_name) | VRAM: $(gpu_mem_gb) GB"

oom_prevention
hf_transfer_cache
prefetch_loras
sync_loras
cleanup_old_loras
start_jupyter &
start_wan2gp
health_wait

log "✅ Wan2GP + Jupyter launched — Optimized Boot Complete."
tail -f /dev/null
