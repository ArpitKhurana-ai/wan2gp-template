# Wan2GP — ProbeAI (A40 / RTX 5090)
# Base: CUDA 12.8 / cuDNN runtime with Python + (typically) PyTorch preinstalled
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # Where Wan2GP will run from (code lives here)
    WAN2GP_DIR=/opt/Wan2GP \
    # Prebuilt venv lives here
    VENV_DIR=/opt/venv \
    # Default run ports
    WAN2GP_PORT=7862 \
    JUPYTER_PORT=8888 \
    # Runtime logs
    WAN2GP_LOG=/workspace/wan2gp.log

# ---- System deps (lean) ----
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git git-lfs curl ca-certificates ffmpeg aria2 tini jq \
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
# NOTE: We intentionally DO NOT install torch here; base image already matches cu128.
RUN . ${VENV_DIR}/bin/activate && \
    python -V && \
    # Requirements file may live at root; if not, fall back to setup/install
    if [ -f "${WAN2GP_DIR}/requirements.txt" ]; then \
      pip install --no-deps -r ${WAN2GP_DIR}/requirements.txt ; \
    fi && \
    # Common extras frequently used by Wan2GP modules
    pip install --no-deps \
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
# Start script (entrypoint) + tiny convenience restart script
COPY start-wan2gp.sh /opt/start-wan2gp.sh
COPY restart-wan2gp.sh /usr/local/bin/restart-wan2gp.sh
RUN chmod +x /opt/start-wan2gp.sh /usr/local/bin/restart-wan2gp.sh

# A neutral working directory
WORKDIR ${WAN2GP_DIR}

# Expose default ports
EXPOSE 7862 8888

# Tini handles proper signal forwarding
ENTRYPOINT ["/usr/bin/tini","-g","--"]
CMD ["/opt/start-wan2gp.sh"]
