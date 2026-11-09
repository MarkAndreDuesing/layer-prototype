# Plan: moving the prototype's heavy steps to Kaggle

Written 2026-11-07. Plan only, nothing on Kaggle is built yet. **[est]** = estimate that hasn't been tested. Facts come from [Free hostable layered models.md](Free%20hostable%20layered%20models.md) and [Layer based architecture models and hosting.md](Layer%20based%20architecture%20models%20and%20hosting.md), unless marked otherwise.

## 1. What free Kaggle gives us

| Resource | Free tier | What it means for us |
|---|---|---|
| GPU | **2× T4 (16 GB each)** or 1× P100 (16 GB) | No native bf16 or fp8 on T4, so models run in fp16, or quantized and upcast |
| Quota | About 30 GPU-hours/week, sessions up to about 12 h (secondary sources) | Fine for play sessions and tests; not an always-on server |
| RAM | About 29 GB | Enough to hold SDXL plus deltas, or to offload part of a large model |
| Disk | Writable working dir (about 20 GB) plus read-only attached Datasets/Models | Upload model weights once as a private Kaggle Dataset/Model and attach it, instead of re-downloading every session |
| Internet | Must be switched on (needs phone verification) | Needed for pip installs, Hugging Face downloads and the tunnel |
| Terms (read 2026-11-07, kaggle.com/terms and kaggle.com/aup) | Terms of Use: "only for your own internal, personal, non-commercial use"; no cryptomining. The Acceptable Use Policy forbids using resources for "server farming", "tools for circumvention/obfuscation" and "activity unrelated to ML data science". No explicit rule on tunnels | **No explicit ban on tunnels.** Since this is a personal, non-commercial project, a tunnelled backend for your own UI is treated as allowed. Modal stays as the alternative with built-in web endpoints |

## 2. Models used or planned: can free Kaggle run them?

| Model | Role in the pipeline | Used so far? | Kaggle T4 verdict | Notes |
|---|---|---|---|---|
| SDXL-Turbo | Fast text-to-image and img2img | Yes (local, about 80 s/image on CPU) | ✅ Yes | fp16, about 7 GB VRAM, a few seconds per image [est] |
| **SDXL base 1.0 + LayerDiffuse** (transparent-object LoRA, `bg2ble`, `bgble2fg`) | Scene-to-layer chain: re-prompt a layer or add a story event | Yes (local test, this session) | ✅ **Yes; the main reason to move** | fp16 at the released settings (20 steps, CFG 7, ~1024 px). With **2× T4**, the `bg2ble` UNet can sit on GPU 0 and the `bgble2fg` UNet on GPU 1, so no delta swapping is needed. Under 1 min per element [est] |
| SD1.5 + LayerDiffuse joint (fg + bg + blend in one pass) | Joint coherent two-layer generation (SD1.5 only) | No | ✅ Yes | Small. Worth trying because SDXL never got a joint release |
| Qwen-Image-Layered (20B) | Splits one image into coherent RGBA layers | No (only your Comfy Cloud workflow) | ⚠️ **Unproven** | BF16 needs about 40–45 GB. Would need GGUF/4-bit (about 11–12 GB) plus its Qwen2.5-VL-7B text encoder, split across the two T4s or offloaded to RAM. Minutes per image if it works at all [est]. Fallbacks: Modal with a bigger GPU (paid beyond the credit), or fal at about $0.05/image (unverified) |
| Qwen-Image-Edit (20B) | Content edits of one layer | No | ⚠️ Same as Qwen-Image-Layered | Same size class, same caveats |
| Qwen-Image-2.1 (7B) | Stronger text-to-image, native RGBA output | No | ✅ Likely | Unsloth says the GGUF needs about 11 GB. **Non-commercial licence** |
| FLUX.2 Klein 4B | Stronger text-to-image / reference edit | No | ✅ Likely | Card: "as little as 13 GB"; the model card links a Kaggle notebook. Apache-2.0 |
| Z-Image Turbo (6B) + scribble ControlNet | Stronger text-to-image, sketch-to-image | No | ✅ Likely | The developers say it fits within 16 GB |
| FLUX 3 Image Klein (5.6B) | Text-to-image | No | ⚠️ Borderline | Card peak is 25–31 GB; would need the INT8 build plus offload on a T4 |
| Krea 2 Turbo (from your Comfy workflow) | Text-to-image | Only on Comfy Cloud | ❓ Unknown | Size not researched |
| Ollama `qwen3:4b` (LLM) | Prompt refining, word breakdown | Yes (local) | ✅ Yes, but **keep local** | Small and already works on the laptop. Moving it gains little |
| Ollama `qwen2.5vl:3b` (vision model) | Image-to-text per layer | Yes (local, about 55 s/layer) | ✅ Yes | About 5 s on a T4 [est]; worth moving if captions stay in the loop |
| Florence-2 / SAM 2.1 / BiRefNet | Possible add-ons | No | ✅ Yes | All small |

**Summary:** everything in the LayerDiffuse chain, SD1.5 joint generation and the stronger 4–7B text-to-image models fit a free T4. The 20B Qwen-Image models (Layered, Edit) are the only ones that may not fit, and need testing.

## 3. Target architecture (updated after the terms check)

| Platform | Use | Why |
|---|---|---|
| **Laptop** | Gradio UI, scene files, small LLM / vision model (optional) | Already works |
| **Modal** (live backend) | `prototype/remote/modal_app.py`: the same `remote_server.py` (`/chain/background`, `/chain/blend`, `/chain/extract`, `/health`) as a Modal web endpoint on an L4 (or T4), weights cached in a Modal Volume, shut down after 5 idle minutes | Web endpoints are a supported feature (no tunnel). Starter plan: $30/month credit; T4 about $0.59/h, L4 about $0.80/h (pricing page, 2026-11-07) |
| **Kaggle** (free; experiments **and** live backend) | `prototype/kaggle/layerproto_chain.ipynb`: the whole engine runs inside the notebook on a T4 at 1024 px. Section 6 can also serve the loaded chain to the local UI through a Cloudflare quick tunnel, with a per-session token | Free. Personal, non-commercial project, and nothing in the terms explicitly bans a tunnel, so this is treated as allowed (your decision, 2026-11-07). Keep it to your own use; no cryptomining |
| **fal.ai** (Qwen-Image-Layered) | Hosted `fal-ai/qwen-image-layered` endpoint | **$0.05 per image**, page tagged "Commercial use"; no free credits; prepaid billing |
| Colab free | Not used | The FAQ forbids "bypassing the notebook UI to interact primarily via a web UI" and "connecting to remote proxies" |

Local `prototype/config.json` for the Modal backend: `"image_backend": "remote"`, `"remote_url": "https://<workspace>--layerproto-chain-serve.modal.run"`, `"remote_token": "…"`, `"width": 1024`, `"height": 1024`.

## 4. Steps

1. **Settle the local test** of `bg2ble` → `bgble2fg` on SDXL base (running now). Confirm the method itself works before moving it.
2. **Make the chain the only generation path** in the engine. Remove the "solo, centered" wrapper, independent per-layer generation and box fitting as a generation step.
3. **Extend `remote_server.py`** with `/background`, `/blend`, `/extract`, `/health`, and add a matching remote client to `backends/`.
4. **Write a Kaggle notebook** (`prototype/kaggle/layerproto_server.ipynb`):
   1. pip install the requirements, then copy or clone `prototype/`;
   2. load SDXL base fp16, with the two delta UNets on the two T4s;
   3. start the server, then the tunnel, and print the URL and token.
5. **Cache the weights** as a private Kaggle Dataset (SDXL base fp16 and the LayerDiffuse files), so sessions start in minutes.
6. **Measure** time per element on the T4 at 1024 px and 20 steps, and compare quality with the local test.
7. **Then test Qwen-Image-Layered (4-bit) on the 2× T4.** If it doesn't fit, use the Modal or fal fallback.
8. **Later:** a stronger text-to-image model (FLUX.2 Klein 4B or Z-Image Turbo) as `/generate`, following [Qwen layered multi model pipeline plan.md](Qwen%20layered%20multi%20model%20pipeline%20plan.md).

## 5. Open points

- Whether Kaggle's current terms allow tunnelling an API out of a notebook.
- Whether 4-bit Qwen-Image-Layered fits and how fast it runs.
- Whether Krea 2 Turbo's size fits a T4.
