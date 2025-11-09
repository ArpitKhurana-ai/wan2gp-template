# Wan2GP — ProbeAI (A40 / RTX 5090)
# Base: CUDA 12.8 / cuDNN runtime with Python + PyTorch preinstalled (conda @ /opt/conda)
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# ---- Environment ----
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_NO_BUILD_ISOLATION=1 \
    WAN2GP_DIR=/opt/Wan2GP \
    WAN2GP_PORT=7862 \
    JUPYTER_PORT=8888 \
    WAN2GP_LOG=/workspace/wan2gp.log \
    HF_HOME=/workspace/hf-home \
    HUGGINGFACE_HUB_CACHE=/workspace/hf-cache \
    XDG_CACHE_HOME=/workspace/.cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TORCHINDUCTOR_CACHE_DIR=/workspace/.torchinductor \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256" \
    # helps when MKL/NumPy/OpenMP mismatch happens on some hosts
    MKL_THREADING_LAYER=GNU \
    # ffmpeg defaults (NVENC + pad to /16 to avoid macro-block warnings)
    WANGP_FFMPEG_VIDEO="-c:v h264_nvenc -preset p1 -rc vbr -cq 22 -pix_fmt yuv420p -vf scale=trunc(iw/16)*16:trunc(ih/16)*16" \
    WANGP_FFMPEG_AUDIO="-c:a aac -b:a 128k" \
    # skip auto-install of default plugins at runtime — we bake them
    WAN2GP_SKIP_DEFAULT_PLUGINS=1

# Use the base image’s conda Python (Torch already present here)
ENV PATH="/opt/conda/bin:${PATH}"

# ---- System deps (toolchain + minimal X/GL for OpenCV/insightface) ----
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git git-lfs curl ca-certificates ffmpeg aria2 tini jq unzip \
    build-essential python3-dev pkg-config \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
 && git lfs install \
 && rm -rf /var/lib/apt/lists/*

# ---- Clone Wan2GP (pin a commit via build arg; default to main) ----
ARG WAN2GP_REPO="https://github.com/deepbeepmeep/Wan2GP.git"
ARG WAN2GP_COMMIT="bdfa6b272409b35a4ac8369502ba3cd7e7f72f90"
RUN git clone --depth=1 ${WAN2GP_REPO} ${WAN2GP_DIR} && \
    ( [ "${WAN2GP_COMMIT}" = "main" ] || (cd ${WAN2GP_DIR} && git fetch --depth=1 origin ${WAN2GP_COMMIT} && git checkout ${WAN2GP_COMMIT}) )

# ---- Python deps (compile-safe order) ----
# NOTE: Torch already exists in /opt/conda from the base image; do NOT reinstall it.
RUN python -V && \
    # 1) Tooling needed to build any C/C++/CUDA wheels cleanly
    python -m pip install --upgrade pip wheel && \
    python -m pip install --no-deps "numpy<2.1" "cython<3.2" "setuptools<75" && \
    # 2) Wan2GP requirements WITH dependencies
    if [ -f "${WAN2GP_DIR}/requirements.txt" ]; then \
      python -m pip install -r ${WAN2GP_DIR}/requirements.txt ; \
    fi && \
    # 3) Common extras + huggingface_hub (often missed by some envs)
    python -m pip install \
      accelerate transformers diffusers gradio timm einops safetensors pillow \
      pydantic psutil uvicorn fastapi hf_transfer huggingface_hub && \
    # 4) Pre-install two Wan2GP default UI plugins so boot skips auto-install
    python -m pip install wan2gp-gallery wan2gp-lora-multipliers-ui && \
    # 5) Sanity check at build time
    python - <<'PY'
import torch, transformers, diffusers, numpy, huggingface_hub
print("Sanity:", torch.__version__, transformers.__version__, diffusers.__version__, numpy.__version__, huggingface_hub.__version__)
PY

# ---- Prefetch LoRAs into the correct folder for Animate/I2V ----
RUN mkdir -p "${WAN2GP_DIR}/loras_i2v" /workspace/hf-home /workspace/hf-cache /workspace/outputs /workspace/.torchinductor
WORKDIR ${WAN2GP_DIR}/loras_i2v
# LightX2V LoRA (Rank64)
ADD https://github.com/ArpitKhurana-ai/wan2gp-assets/releases/download/v1.0/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors \
    ${WAN2GP_DIR}/loras_i2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors
# Wan2.1 FusionX LoRA
ADD https://github.com/ArpitKhurana-ai/wan2gp-assets/releases/download/v1.0/Wan2.1_I2V_14B_FusionX_LoRA.safetensors \
    ${WAN2GP_DIR}/loras_i2v/Wan2.1_I2V_14B_FusionX_LoRA.safetensors

# ---- Modernize deprecated autocast -> torch.amp.autocast('cuda', ...) ----
RUN sed -i "s/torch.cuda.amp.autocast(/torch.amp.autocast('cuda', /g" \
    ${WAN2GP_DIR}/models/wan/animate/motion_encoder.py || true

# ---- Runtime entry assets ----
COPY start-wan2gp.sh /opt/start-wan2gp.sh
COPY restart-wan2gp.sh /usr/local/bin/restart-wan2gp.sh
RUN chmod +x /opt/start-wan2gp.sh /usr/local/bin/restart-wan2gp.sh

# ---- Prepare persistent caches (avoid permission issues at runtime) ----
RUN mkdir -p /workspace /workspace/outputs /workspace/models \
           ${HF_HOME} ${HUGGINGFACE_HUB_CACHE} ${XDG_CACHE_HOME} ${TORCHINDUCTOR_CACHE_DIR}

# ---- Container defaults ----
WORKDIR ${WAN2GP_DIR}
EXPOSE 7862 8888

ENTRYPOINT ["/usr/bin/tini","-g","--"]
CMD ["/opt/start-wan2gp.sh"]
