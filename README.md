# Layer prototype: coherent layered image generation for a storytelling game

The prototype's core loop is:

**text prompt → coherent layered image** (layers that share style and placement, like a PSD) **→ edit single layers** (re-prompt, move, resize) **→ recomposite**

This repo gathers the code, experiments, research and decisions so far. Personal, non-commercial team prototype.

## What's where

| Folder / file | Contents |
|---|---|
| [`prototype/`](prototype/) | The app (Gradio UI, LLM scene split, LayerDiffuse chain) plus the Kaggle notebooks and tools. Start with [prototype/README.md](prototype/README.md). |
| [`prototype/docs/qwen-layered-on-kaggle.md`](prototype/docs/qwen-layered-on-kaggle.md) | Qwen-Image-Layered on free Kaggle T4s: how to run it, why it needs a patch, gotchas |
| [`prototype/layerproto/fp16_mixed.py`](prototype/layerproto/fp16_mixed.py) | Makes bf16-only Qwen-Image models run on fp16 GPUs (T4) without NaNs |
| [`docs/findings-log.md`](docs/findings-log.md) | **All findings and decisions in date order**: what worked, what failed, measured numbers |
| [`reports/`](reports/) | Research reports: layered-generation models, hosting, the coherent pipeline, Kaggle plans, TPU route, speed and API costs |
| [`research_notes/`](research_notes/) | Raw source notes behind the reports |
| [`SETUP_AND_USAGE_GUIDE.md`](SETUP_AND_USAGE_GUIDE.md) | Earlier phase: sketch-to-image on a CPU laptop with FastSD CPU (3 tested paths) |
| [`experiments/`](experiments/) | Sketch-to-image test plan, results, ranking sheet, input sketches, comparison sheets and kept images. Raw image folders are not included. |
| [`fastsdcpu_local/`](fastsdcpu_local/) | Our patch to FastSD CPU and an offline launcher |

## Current status (2026-11-09)

| Part | Status |
|---|---|
| Local app (CPU laptop) | Works. LLM scene split with Ollama `qwen3:4b`, LayerDiffuse SDXL layers, per-layer re-prompt / refine / sketch, captions, recomposite with layer order. About 80 s per 512 px layer. |
| LayerDiffuse scene-to-layer chain (each element generated on the existing scene) | Works on SDXL base (about 11 min per element on CPU) and on a Kaggle T4 (about 18 s per pass at 1024 px) |
| Qwen-Image-Layered (image → RGBA layers) on Kaggle 2× T4 | Works with `fp16_mixed`: 640 px, 4 layers, 50 steps take 35 min. **Too slow**: the target is ≤ 40 s. A speed sweep at 512 / 384 px, with fewer steps and no CFG, is running. |
| Hosted alternatives | Comfy Cloud (RTX PRO 6000) does it in seconds. fal costs about $0.05 per run. Kaggle TPU v5e-8 could plausibly reach about 40 s, but needs a port (2–5 days, unproven). See [reports/research_speed_and_apis.md](reports/research_speed_and_apis.md) and [reports/research_tpu_route.md](reports/research_tpu_route.md). |

## Quick start

1. Clone this repo, then follow [prototype/README.md](prototype/README.md) for the local app. It needs Ollama, and the LayerDiffuse code from [prototype/vendor/README.md](prototype/vendor/README.md).
2. For Qwen-Image-Layered on Kaggle, follow [prototype/docs/qwen-layered-on-kaggle.md](prototype/docs/qwen-layered-on-kaggle.md). The batch notebook and cached weights are public on Kaggle.
3. For the earlier sketch-to-image tests, follow [SETUP_AND_USAGE_GUIDE.md](SETUP_AND_USAGE_GUIDE.md) and [fastsdcpu_local/README.md](fastsdcpu_local/README.md).
