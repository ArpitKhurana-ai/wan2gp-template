# Wan2GP — ProbeAI (A40 / RTX 5090)

**One-click RunPod template** for the Wan2GP web app — a fast, no-nonsense UI to run modern open video/image models.  
This template targets **A40** and **RTX 5090** GPUs on **CUDA 12.8**, with **lazy model download** (fast boot) and **Lightning LoRAs** prepped for Wan 2.2 Animate.

---

## What you get

- ✅ **Wan2GP** UI on **port 7862** (no ComfyUI)
- ✅ **JupyterLab** on **port 8888** (no token)
- ✅ **CUDA 12.8** + pinned Python deps (prebuilt venv)
- ✅ **Lightning LoRAs** for Wan 2.2 Animate (downloaded quickly on first boot)
- ✅ **Persistent storage** at **/workspace** (models, outputs, logs)
- ✅ **5090 attention fix**: auto-fallback to **Scaled Dot-Product Attention** (avoid SageAttention issues)
- ✅ Clean logging to `/workspace/wan2gp.log`

> ⚠️ **Lazy Base Models**: the large base checkpoints (e.g., Wan 2.2 Animate ~30 GB) are **not** downloaded at boot.  
> They download **only when you first press Generate**, keeping first boot under ~5 minutes. Subsequent runs are fast.

---

## Quick Start

1. **Launch the template** (A40 or RTX 5090).
2. Wait until logs show: `Wan2GP UI is READY on port 7862`.
3. Click **Connect** → **7862** to open the UI.
4. Pick **Wan 2.2 Animate**, set your prompt/media, and hit **Generate**.  
   - On the **first** Animate run, the model will download (~30 GB). You’ll see progress in the logs.  
   - Next runs reuse the cached model from `/workspace/models`.

**JupyterLab** (optional): connect to port **8888** (no token).

---

## Expected Timings

- **Cold boot (first ever)**: ~3–5 min (RTX 5090) / ~5–7 min (A40)  
- **Restarted boot**: ≤ 2 min  
- **First Animate generation**: includes one-time model download (larger wait)  
- **Subsequent Animate generations**: fast (Lightning LoRA 4-step path)

> Tip: keep your **RunPod Volume** attached so models persist between sessions.

---

## Logs & Troubleshooting

- Tail logs:
  ```bash
  tail -f /workspace/wan2gp.log
