🚀 Wan2GP — ProbeAI Optimized (RTX 5090 / A40 / CUDA 12.8)

Instantly run Wan 2.2 Animate 14B — the world’s best open-source AI video generation model — with pre-baked LoRAs, full CUDA GPU optimization, and Dockerized stability.
Designed for RunPod GPU users (A40 / RTX 5090), this template delivers fast boot times, 1-click setup, and studio-quality video replacement and animation.

⚙️ Overview

This is the fastest, most stable Wan2GP setup ever built, fully Dockerized and production-ready — no manual installs, no crashes, no guesswork.
It’s engineered for creators, researchers, and developers who want plug-and-play AI video generation using Wan 2.2 Animate, with optimized performance across RunPod GPUs.

🧠 Why This Template Stands Out

✅ Fully Dockerized — everything is built, pinned, and ready.
✅ No pip installs or git clones on boot — instant startup.
✅ Pinned Wan2GP commit (Nov 2025) — no breaking nightly updates.
✅ MatAnyOne SAM Hotfix + Autocast Patch — eliminates known runtime crashes.
✅ Lightning LoRAs preloaded for Wan 2.2 Animate / I2V.
✅ 16 GB Swap + Dynamic Precision (bf16/fp16) — prevents GPU OOMs.
✅ NVENC accelerated encoding — faster video processing and exports.
✅ Persistent /workspace volume keeps models, outputs, and cache safe.
✅ Auto-optimized for RTX 5090, A40, and A100 GPUs.

🧩 Built-In LoRAs
Type	File	Purpose
Animate 14B (I2V)	lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors	Lightning-fast motion generation
FusionX	Wan2.1_I2V_14B_FusionX_LoRA.safetensors	Higher realism, color consistency

Both are pre-baked into the image — first job starts instantly, no waiting for LoRA downloads.

🧰 Tech Stack
Component	Version / Details
Base Image	PyTorch 2.8 • CUDA 12.8 • cuDNN 9
Framework	Wan2GP v9.4 (pinned commit)
OS	Ubuntu 22.04 Slim
Key Libraries	Hugging Face Hub, Diffusers, Transformers, Accelerate, Gradio
Encoding	NVIDIA NVENC (h264 + AAC audio)
⚡ Quick Start (RunPod)

Launch Template → choose GPU (A40 or RTX 5090).

Wait until logs show:

✅ Wan2GP UI READY on port 7862


Open HTTP Port 7862 → Wan2GP UI.

Select Wan 2.2 Animate 14B, upload control video + reference image.

Hit Generate — model downloads once, then stays cached.

Subsequent runs are near-instant.

🧱 Persistent Layout

All your models, LoRAs, outputs, and logs live inside a single persistent mount (/workspace):

/workspace/outputs     → rendered videos & images  
/workspace/hf-home     → Hugging Face credentials  
/workspace/hf-cache    → model weights cache  
/workspace/wan2gp.log  → live logs  


No need for multiple mounts — RunPod keeps /workspace attached across sessions.

⏱ Performance Benchmarks
Stage	RTX 5090	A40
Cold boot (first time)	3–5 min	5–7 min
Warm boot (restart)	≤ 2 min	≤ 3 min
First Animate generation	~10 min (30 GB model DL)	~12 min
Subsequent runs (5 s 720p)	1–2 min	~3 min
🧩 Optional Environment Variables
Variable	Purpose
WAN2GP_PORT=7862	Change UI port
WAN2GP_USERNAME / WAN2GP_PASSWORD	Enable basic auth
WAN2GP_PRECISION=fp16/bf16	Manual precision override
HF_HUB_ENABLE_HF_TRANSFER=1	Accelerated Hugging Face downloads
🧠 Challenges Solved

✅ Dependency chaos → pre-installed, pre-compiled
✅ Runtime patching → MatAnyOne hotfix applied at build-time
✅ GPU OOMs → swap + precision tuner
✅ Long boot delays → pip-free instant boot
✅ 5090 SDPA issues → auto Scaled Dot-Product Attention fallback
✅ Model persistence → unified /workspace structure
✅ Frequent source changes → locked, stable commit

🧾 Logs & Debug

Follow logs in real-time:

tail -f /workspace/wan2gp.log


If the UI doesn’t load within 5 minutes, check GPU utilization or model download progress inside the log.

🔧 System Optimizations Inside the Image

TORCHINDUCTOR_CACHE_DIR=/workspace/.torchinductor (warm kernels)

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

CUBLAS_WORKSPACE_CONFIG=:16:8

hf_transfer acceleration enabled

swap enabled (16 GB)

GRADIO_NUM_WORKERS=1 for lower memory footprint

🧭 Maintained by ProbeAI

Built and tested by ProbeAI
 — the creators of high-performance AI toolchains and content workflows.
Updated regularly as new stable Wan2GP releases are verified.

GitHub: ArpitKhurana-ai/wan2gp-template

🔍 Tags
Wan 2.2 Animate, Wan2GP, AI Video Generator, RunPod Template, RTX 5090, A40, Docker, LoRA, Image-to-Video, DeepBeepMeep, Open-Source Animate, AI Animation Model, Video Diffusion Model, SDPA, ProbeAI, CUDA 12.8, Docker GPU Template.
