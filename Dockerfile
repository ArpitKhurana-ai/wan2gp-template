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
    # helps when MKL/NumPy/OpenMP mismatch happens in some hosts
    MKL_THREADING_LAYER=GNU

# Use the base image’s conda Python (Torch already present here)
ENV PATH="/opt/conda/bin:${PATH}"

# ---- System deps (toolchain + minimal X/GL for OpenCV/insightface) ----
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git git-lfs curl ca-certificates ffmpeg aria2 tini jq \
    build-essential python3-dev pkg-config \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
 && git lfs install \
 && rm -rf /var/lib/apt/lists/*

# ---- Clone Wan2GP (pin a commit via build arg; default to main) ----
ARG WAN2GP_REPO="https://github.com/deepbeepmeep/Wan2GP.git"
ARG WAN2GP_COMMIT="main"
RUN git clone --depth=1 ${WAN2GP_REPO} ${WAN2GP_DIR} && \
    ( [ "${WAN2GP_COMMIT}" = "main" ] || (cd ${WAN2GP_DIR} && git fetch --depth=1 origin ${WAN2GP_COMMIT} && git checkout ${WAN2GP_COMMIT}) )

# ---- Python deps (compile-safe order) ----
# NOTE: Torch already exists in /opt/conda from the base image; do NOT reinstall it.
RUN python -V && \
    # 1) Tooling needed to build any C/C++/CUDA wheels cleanly
    python -m pip install --upgrade pip wheel && \
    python -m pip install --no-deps "numpy<2.1" "cython<3.2" "setuptools<75" && \
    # 2) Wan2GP requirements WITH dependencies (fixes missing huggingface_hub etc.)
    if [ -f "${WAN2GP_DIR}/requirements.txt" ]; then \
      python -m pip install -r ${WAN2GP_DIR}/requirements.txt ; \
    fi && \
    # 3) Common extras WITH dependencies + explicitly add huggingface_hub
    python -m pip install \
      accelerate transformers diffusers gradio timm einops safetensors pillow \
      pydantic numpy psutil uvicorn fastapi jupyterlab hf_transfer huggingface_hub && \
    # 4) Sanity check: import key libs so we fail at build-time if anything is off
    python - <<'PY'
import torch, transformers, diffusers, numpy, huggingface_hub
print("Sanity:", torch.__version__, transformers.__version__, diffusers.__version__, numpy.__version__, huggingface_hub.__version__)
PY

# ---- Runtime entry assets ----
COPY start-wan2gp.sh /opt/start-wan2gp.sh
COPY restart-wan2gp.sh /usr/local/bin/restart-wan2gp.sh
RUN chmod +x /opt/start-wan2gp.sh /usr/local/bin/restart-wan2gp.sh

# ---- Prepare persistent caches (avoid permission issues at runtime) ----
RUN mkdir -p /workspace /workspace/outputs /workspace/models \
           ${HF_HOME} ${HUGGINGFACE_HUB_CACHE} ${XDG_CACHE_HOME}

# ---- Container defaults ----
WORKDIR ${WAN2GP_DIR}
EXPOSE 7862 8888

ENTRYPOINT ["/usr/bin/tini","-g","--"]
CMD ["/opt/start-wan2gp.sh"]
