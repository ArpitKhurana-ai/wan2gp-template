# Wan2GP — ProbeAI (A40 / RTX 5090)
# Base: CUDA 12.8 / cuDNN runtime with Python + (typically) PyTorch preinstalled
# Force amd64 so builds on M1/arm64 target the same arch RunPod uses.
FROM --platform=linux/amd64 pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_NO_BUILD_ISOLATION=1 \
    WAN2GP_DIR=/opt/Wan2GP \
    VENV_DIR=/opt/venv \
    WAN2GP_PORT=7862 \
    JUPYTER_PORT=8888 \
    WAN2GP_LOG=/workspace/wan2gp.log

# ---- System deps (add toolchain + minimal X/GL for insightface/opencv) ----
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git git-lfs curl ca-certificates ffmpeg aria2 tini jq \
    build-essential python3-dev pkg-config \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ---- Create venv and upgrade base tools ----
RUN python -m venv ${VENV_DIR} && . ${VENV_DIR}/bin/activate && \
    python -m pip install --upgrade pip wheel setuptools

# ---- Clone Wan2GP (pin a commit via build arg; default to main) ----
ARG WAN2GP_REPO="https://github.com/deepbeepmeep/Wan2GP.git"
ARG WAN2GP_COMMIT="main"
RUN git clone --depth=1 ${WAN2GP_REPO} ${WAN2GP_DIR} && \
    ( [ "${WAN2GP_COMMIT}" = "main" ] || (cd ${WAN2GP_DIR} && git fetch --depth=1 origin ${WAN2GP_COMMIT} && git checkout ${WAN2GP_COMMIT}) )

# ---- Python deps from Wan2GP (no heavy work at runtime) ----
# NOTE: Torch is already in the base image (cu128). We avoid re-installing here.
RUN . ${VENV_DIR}/bin/activate && \
    python -V && \
    pip install --no-deps "numpy<2.1" "cython<3.2" "setuptools<75" wheel && \
    if [ -f "${WAN2GP_DIR}/requirements.txt" ]; then \
      pip install --no-deps -r ${WAN2GP_DIR}/requirements.txt ; \
    fi && \
    pip install --no-deps \
      accelerate transformers diffusers gradio timm einops safetensors pillow \
      pydantic numpy psutil uvicorn fastapi jupyterlab hf_transfer
      accelerate \
      transformers \
      diffusers \
      gradio \
      timm \
      einops \
      safetensors \
      pillow \
      pydantic \
      numpy \
      psutil \
      uvicorn \
      fastapi \
      jupyterlab \
      hf_transfer

# ---- Runtime entry assets ----
COPY start-wan2gp.sh /opt/start-wan2gp.sh
COPY restart-wan2gp.sh /usr/local/bin/restart-wan2gp.sh
RUN chmod +x /opt/start-wan2gp.sh /usr/local/bin/restart-wan2gp.sh

WORKDIR ${WAN2GP_DIR}
EXPOSE 7862 8888

ENTRYPOINT ["/usr/bin/tini","-g","--"]
CMD ["/opt/start-wan2gp.sh"]
