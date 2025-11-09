#!/usr/bin/env bash
set -Eeuo pipefail

# ================= CONFIG =================
WAN2GP_PORT="${WAN2GP_PORT:-7862}"
WAN2GP_DIR="${WAN2GP_DIR:-/opt/Wan2GP}"
LOG="${WAN2GP_LOG:-/workspace/wan2gp.log}"

WAN2GP_USERNAME="${WAN2GP_USERNAME:-}"
WAN2GP_PASSWORD="${WAN2GP_PASSWORD:-}"

# ---- LoRA CONFIG ----
LORA_CACHE_DIR="${LORA_CACHE_DIR:-/workspace/loras_cache}"       # temp cache
WAN2GP_LORA_DIR="${WAN2GP_LORA_DIR:-/opt/Wan2GP/loras_i2v}"     # ✅ correct for Animate/I2V
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

# ================= WAN2GP AUTO-UPDATE (one-time sync + lock) =================
WAN2GP_REPO="${WAN2GP_REPO:-https://github.com/deepbeepmeep/Wan2GP.git}"
# Pinned tested commit (Nov 8, 2025). Change this only when you want to refresh.
WAN2GP_REF="${WAN2GP_REF:-bdfa6b272409b35a4ac8369502ba3cd7e7f72f90}"
# Locked by default (no nightlies). Set to 0 to re-sync to WAN2GP_REF again on next boot.
WAN2GP_LOCK="${WAN2GP_LOCK:-1}"

sync_wan2gp_code() {
  log "🔁 Sync Wan2GP code → $WAN2GP_REF (lock=$WAN2GP_LOCK)"
  if [ "${WAN2GP_LOCK}" = "1" ] && [ -d "${WAN2GP_DIR}/.git" ]; then
    local cur
    cur="$(git -C "${WAN2GP_DIR}" rev-parse HEAD 2>/dev/null || echo 'unknown')"
    if [ "$cur" = "$WAN2GP_REF" ]; then
      log "🔒 Already at pinned ref; skipping update."
      return
    fi
  fi

  if command -v git >/dev/null 2>&1; then
    if [ ! -d "${WAN2GP_DIR}/.git" ]; then
      rm -rf "${WAN2GP_DIR}"
      git clone --depth=1 "$WAN2GP_REPO" "${WAN2GP_DIR}" >>"$LOG" 2>&1
    fi
    pushd "${WAN2GP_DIR}" >/dev/null
    git fetch --depth=1 origin >>"$LOG" 2>&1 || true
    # checkout the exact ref (commit, tag, or branch)
    git checkout -q "$WAN2GP_REF" >>"$LOG" 2>&1 || {
      log "⚠️ git checkout failed for $WAN2GP_REF; trying to fetch specific ref"
      git fetch origin "$WAN2GP_REF" >>"$LOG" 2>&1 || true
      git checkout -q "$WAN2GP_REF" >>"$LOG" 2>&1 || { log "❌ could not checkout $WAN2GP_REF"; exit 1; }
    }
    popd >/dev/null
  else
    # Fallback: download zip for the commit
    log "ℹ️ git not found; downloading zip for $WAN2GP_REF"
    rm -rf "${WAN2GP_DIR}" /workspace/_wan2gp_zip
    mkdir -p /workspace/_wan2gp_zip
    curl -fL "https://github.com/deepbeepmeep/Wan2GP/archive/${WAN2GP_REF}.zip" -o /workspace/_wan2gp_zip/Wan2GP.zip
    unzip -q /workspace/_wan2gp_zip/Wan2GP.zip -d /workspace/_wan2gp_zip
    mv "/workspace/_wan2gp_zip/Wan2GP-${WAN2GP_REF}" "${WAN2GP_DIR}"
    rm -rf /workspace/_wan2gp_zip
  fi

  # (Re)install python deps if requirements changed
  if [ -f "${WAN2GP_DIR}/requirements.txt" ]; then
    pip install --no-cache-dir -r "${WAN2GP_DIR}/requirements.txt" >>"$LOG" 2>&1 || true
  fi

  local rev; rev=$(git -C "${WAN2GP_DIR}" rev-parse --short HEAD 2>/dev/null || echo "$WAN2GP_REF")
  log "✅ Wan2GP synced @ $rev"
}

# ================= HOTFIX: MatAnyOne SAM guard =================
apply_matanyone_hotfix() {
  local target="${WAN2GP_DIR}/preprocessing/matanyone/app.py"
  if [ ! -f "$target" ]; then
    log "ℹ️ MatAnyOne file not found yet; skipping hotfix"
    return 0
  fi

  # Only patch if not already present
  if ! grep -q "HOTFIX: ensure a default SAM if none" "$target" 2>/dev/null; then
    log "🩹 Applying MatAnyOne hotfix (default SAM if matanyone_model is None)"
    $PY - "$target" <<'PY' || { echo "[hotfix] failed"; exit 1; }
import io,sys,re,os
p=sys.argv[1]
s=open(p,'r',encoding='utf-8').read()

pattern=r"(def\s+select_SAM\(state\):\s*\n\s*global\s+matanyone_model\s*\n)"
inject=(
    r"\1"
    r"    # HOTFIX: ensure a default SAM if none\n"
    r"    if matanyone_model is None:\n"
    r"        try:\n"
    r"            from .sam import build_sam_model\n"
    r"            matanyone_model = build_sam_model('sam2_t')  # fast, safe default\n"
    r"        except Exception as e:\n"
    r"            raise RuntimeError(f'MatAnyOne SAM init failed: {e}')\n"
)

new, n = re.subn(pattern, "".join(inject), s, count=1, flags=re.M)
if n==0:
    # Fallback: try adding right after function start even if formatting differs
    s = s.replace("def select_SAM(state):\n    global matanyone_model\n",
                  "def select_SAM(state):\n    global matanyone_model\n"
                  "    # HOTFIX: ensure a default SAM if none\n"
                  "    if matanyone_model is None:\n"
                  "        try:\n"
                  "            from .sam import build_sam_model\n"
                  "            matanyone_model = build_sam_model('sam2_t')  # fast, safe default\n"
                  "        except Exception as e:\n"
                  "            raise RuntimeError(f'MatAnyOne SAM init failed: {e}')\n")
else:
    s = new

open(p,'w',encoding='utf-8').write(s)
print("[hotfix] MatAnyOne patched")
PY
  else
    log "✔️ MatAnyOne hotfix already present"
  fi
}

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
  log "📦 Placing LoRAs in correct folder for Wan2.2 Animate (loras_i2v)..."
  mkdir -p "$WAN2GP_LORA_DIR"
  find "$LORA_CACHE_DIR" -maxdepth 1 -type f -name "*.safetensors" -exec cp -u {} "$WAN2GP_LORA_DIR"/ \; || true
  ln -sf "$WAN2GP_LORA_DIR" /workspace/loras
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

# ---- RAM hardening (add before start_wan2gp) ----
enable_swap() {
  # 16G swap to avoid container OOM; idempotent
  if ! grep -q "/workspace/wan2gp.swap" /proc/swaps 2>/dev/null; then
    fallocate -l 16G /workspace/wan2gp.swap || dd if=/dev/zero of=/workspace/wan2gp.swap bs=1G count=16
    chmod 600 /workspace/wan2gp.swap
    mkswap /workspace/wan2gp.swap >/dev/null 2>&1 || true
    swapon /workspace/wan2gp.swap  >/dev/null 2>&1 || true
    log "🛟 Swap enabled (16G) to prevent RAM OOM."
  fi
}

tune_runtime() {
  # keep queues small (avoid multiple in-flight jobs holding frames in RAM)
  export GRADIO_NUM_WORKERS=1
  export GRADIO_CONCURRENCY_COUNT=1
  export UV_THREADPOOL_SIZE=8
  # hint mmgp to reserve less RAM if supported (safe if ignored)
  export MMGP_RESERVED_RAM_GB=10
  # avoid huge Python object caches
  export PYTHONUNBUFFERED=1
}

enable_swap
tune_runtime

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
log "=== BOOT $(date -u) | Wan2GP Optimized v5 (loras_i2v) ==="
log "GPU: $(gpu_name) | VRAM: $(gpu_mem_gb) GB"

oom_prevention
hf_transfer_cache
# ⬇️ one-time update to the pinned commit, then locked
sync_wan2gp_code
# ⬇️ patch MatAnyOne so Load Video never fails when SAM isn't preloaded
apply_matanyone_hotfix
prefetch_loras
sync_loras
cleanup_old_loras
start_wan2gp
health_wait

log "✅ Wan2GP launched — Optimized Boot Complete."
tail -f /dev/null
