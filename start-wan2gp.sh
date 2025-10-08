#!/usr/bin/env bash
set -Eeuo pipefail

# ========== CONFIG ==========
WAN2GP_PORT="${WAN2GP_PORT:-7862}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
WAN2GP_DIR="${WAN2GP_DIR:-/opt/Wan2GP}"
LOG="${WAN2GP_LOG:-/workspace/wan2gp.log}"

WAN2GP_USERNAME="${WAN2GP_USERNAME:-}"
WAN2GP_PASSWORD="${WAN2GP_PASSWORD:-}"

# ---- LoRA Config ----
LORA_CACHE_DIR="${LORA_CACHE_DIR:-/workspace/loras}"              # temporary download cache
WAN2GP_LORA_DIR="${WAN2GP_LORA_DIR:-/opt/Wan2GP/models/loras}"   # Wan2GP reads from here
LORA1_NAME="${LORA1_NAME:-lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors}"
LORA1_URL="${LORA1_URL:-https://github.com/ArpitKhurana-ai/wan2gp-assets/releases/download/v1.0/${LORA1_NAME}}"
LORA2_NAME="${LORA2_NAME:-Wan2.1_I2V_14B_FusionX_LoRA.safetensors}"
LORA2_URL="${LORA2_URL:-https://github.com/ArpitKhurana-ai/wan2gp-assets/releases/download/v1.0/${LORA2_NAME}}"

BOOTSTRAP_ASSETS="${BOOTSTRAP_ASSETS:-1}"
MANIFEST_URL="${MANIFEST_URL:-https://raw.githubusercontent.com/ArpitKhurana-ai/wan2gp-template/main/assets/manifest.json}"

# ---- GPU / Cache / OOM settings ----
HF_HOME="/workspace/hf-home"
HF_HUB_CACHE="/workspace/hf-cache"
MEMORY_WATCHDOG="${MEMORY_WATCHDOG:-1}"
CLEANUP_LORAS="${CLEANUP_LORAS:-1}"
DISK_THRESHOLD_GB="${DISK_THRESHOLD_GB:-80}"

HEALTH_TRIES="${HEALTH_TRIES:-180}"
HEALTH_INTERVAL=2

# ========== HELPERS ==========
PY="/opt/conda/bin/python"
JUPY="/opt/conda/bin/jupyter"

log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

ensure_dirs(){
  mkdir -p /workspace/models /workspace/outputs /workspace/.cache "$HF_HOME" "$HF_HUB_CACHE" "$LORA_CACHE_DIR" "$WAN2GP_LORA_DIR"
  touch "$LOG"
}

gpu_name(){ nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo unknown; }
gpu_mem_gb(){ nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk '{print int($1/1024)}' || echo 0; }

fetch_one(){
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
  [ -s "$path" ] && log "Ready: $name" || log "WARN: Missing after download: $name ($url)"
}

prefetch_loras(){
  log "Prefetching LoRAs..."
  fetch_one "$LORA1_NAME" "$LORA1_URL" "$LORA_CACHE_DIR"
  fetch_one "$LORA2_NAME" "$LORA2_URL" "$LORA_CACHE_DIR"
}

sync_loras(){
  log "Syncing LoRAs into Wan2GP (correct internal path)..."
  mkdir -p "$WAN2GP_DIR/wan2gp/models/loras"

  # Copy .safetensors into the internal models path
  find "$LORA_CACHE_DIR" -type f -name "*.safetensors" -exec cp -u {} "$WAN2GP_DIR/wan2gp/models/loras"/ \; || true

  # Create convenient symlinks for Jupyter and UI visibility
  ln -sf "$WAN2GP_DIR/wan2gp/models/loras" /workspace/loras
  ln -sf "$WAN2GP_DIR/wan2gp/models/loras" /workspace/models/loras
  ln -sf "$WAN2GP_DIR/wan2gp/models/loras" "$WAN2GP_DIR/models/loras"

  log "✅ LoRAs synced to: $WAN2GP_DIR/wan2gp/models/loras"
  log "🧾 Listing current LoRAs:"
  ls -lh "$WAN2GP_DIR/wan2gp/models/loras" | tee -a "$LOG" || true
}

cleanup_old_loras(){
  if [ "$CLEANUP_LORAS" = "1" ]; then
    local used=$(df -h /workspace | awk 'NR==2{print +$3}')
    if [ "${used%.*}" -gt "$DISK_THRESHOLD_GB" ]; then
      log "🧹 Disk usage high (${used}GB) → Cleaning old LoRAs..."
      find "$WAN2GP_LORA_DIR" -type f -name "*.safetensors" -mtime +7 -delete || true
      log "✅ Old LoRAs cleaned."
    fi
  fi
}

bootstrap_assets(){
  [ "$BOOTSTRAP_ASSETS" = "1" ] || { log "Bootstrap disabled"; return 0; }
  local mf="/tmp/manifest.json"
  if ! curl -fsSL "$MANIFEST_URL" -o "$mf"; then
    log "WARN: failed to fetch manifest"; return 0;
  fi
  while IFS='|' read -r name url dest; do
    [ -n "${name:-}" ] || continue
    fetch_one "$name" "$url" "$dest"
  done < <(
    "$PY" - "$mf" <<'PYCODE'
import json, sys
mf=sys.argv[1]; data=json.load(open(mf))
for it in data.get("items", []):
    n,u,d=(it.get("name") or "").strip(),(it.get("url") or "").strip(),(it.get("dest") or "/workspace/models").strip()
    if n and u: print(f"{n}|{u}|{d}")
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

oom_prevention(){
  if [ "$MEMORY_WATCHDOG" = "1" ]; then
    local mem=$(gpu_mem_gb)
    if (( mem > 0 && mem < 20 )); then
      export WAN2GP_PRECISION="bf16"
      export WAN2GP_BATCH="1"
      log "⚙️ Low-VRAM GPU detected (${mem} GB) → reduced batch size + bf16 precision."
    elif (( mem <= 40 )); then
      export WAN2GP_PRECISION="bf16"
      export WAN2GP_BATCH="2"
      log "⚙️ Mid-range GPU (${mem} GB) → optimized settings."
    else
      export WAN2GP_PRECISION="fp16"
      export WAN2GP_BATCH="4"
      log "⚙️ High-VRAM GPU (${mem} GB) → full precision."
    fi
  fi
}

hf_transfer_cache(){
  if python -c "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('hf_transfer') else 1)" >/dev/null 2>&1; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
    export HF_HOME="$HF_HOME"
    export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
    log "⚡ hf_transfer active — caching via $HF_HUB_CACHE"
  else
    log "hf_transfer not installed; skipping cache acceleration"
  fi
}

start_jupyter(){
  if [ "${ENABLE_JUPYTER:-1}" = "1" ]; then
    export PATH="/opt/conda/bin:$PATH"
    pip install --quiet --no-cache-dir jupyterlab==4.1.8 notebook==7.1.2 ipykernel jupyter_core==5.7.2 --upgrade
    mkdir -p /workspace/.jupyter
    export JUPYTER_CONFIG_DIR=/workspace/.jupyter
    log "Launching JupyterLab on :${JUPYTER_PORT}"
    nohup /opt/conda/bin/python -m jupyterlab \
      --ip=0.0.0.0 --port="${JUPYTER_PORT}" --no-browser \
      --allow-root --ServerApp.allow_origin='*' --ServerApp.disable_check_xsrf=True \
      --ServerApp.allow_remote_access=True --ServerApp.root_dir=/workspace \
      --NotebookApp.token='' --NotebookApp.password='' \
      --ServerApp.default_url=/lab >>"$LOG" 2>&1 &
    sleep 6
    lsof -i :"${JUPYTER_PORT}" >/dev/null 2>&1 && log "✅ Jupyter ready on port ${JUPYTER_PORT}" || log "❌ Jupyter failed to bind"
  fi
}

start_wan2gp(){
  if [ -n "$WAN2GP_USERNAME" ] && [ -n "$WAN2GP_PASSWORD" ]; then
    export WAN2GP_AUTH_USER="$WAN2GP_USERNAME"
    export WAN2GP_AUTH_PASS="$WAN2GP_PASSWORD"
  fi
  log "Starting Wan2GP on :${WAN2GP_PORT}"
  "$PY" "${WAN2GP_DIR}/wgp.py" --listen --server-port "${WAN2GP_PORT}" >>"$LOG" 2>&1 &
}

health_wait(){
  log "Waiting for Wan2GP to respond..."
  for _ in $(seq 1 "$HEALTH_TRIES"); do
    if curl -fsS "http://127.0.0.1:${WAN2GP_PORT}/" >/dev/null 2>&1; then
      log "✅ Wan2GP UI READY on port ${WAN2GP_PORT}"
      return 0
    fi
    sleep "$HEALTH_INTERVAL"
  done
  log "⚠️ UI not responding within timeout."
}

# ========== EXECUTION ==========
ensure_dirs
log "=== BOOT $(date -u) | Wan2GP (CUDA 12.8) ==="
log "GPU: $(gpu_name) | VRAM: $(gpu_mem_gb) GB"

disable_sage_on_blackwell
oom_prevention
hf_transfer_cache
prefetch_loras
sync_loras
cleanup_old_loras
bootstrap_assets
start_jupyter
start_wan2gp
health_wait

log "✅ Wan2GP & Jupyter launched — container will stay alive."
tail -f /dev/null
