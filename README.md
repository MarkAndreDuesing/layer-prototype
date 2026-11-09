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
| [`prototype/layerproto/`](prototype/layerproto/) loop modules | `qwen_layered.py` (split, Heun wrapper, prompt cache), `backends/layerdiffuse_chain.py` (`relocate()` move/resize + re-harmonise), `relight.py` (IC-Light relight of a layer), `shadow.py` (procedural shadow layer), `events.py` (story-event router + instant previews), `drift_eval.py` (drift over long edit sequences) |
| [`docs/investigation-checklist.md`](docs/investigation-checklist.md) | **Everything asked to investigate, with status** |
| [`reports/Layered loop open paths.md`](reports/Layered%20loop%20open%20paths.md) | Latest research: Qwen-Image-2.1 RGBA, FLUX layer models on T4, harmonisation, story events, drift evaluation |
| [`docs/summary-2026-11-09-research-block.md`](docs/summary-2026-11-09-research-block.md) | **Latest summary**: 27–32 s Qwen split on free T4, full split → move → re-harmonise loop, paper audit, research gap, LMU GPUs |
| [`docs/findings-log.md`](docs/findings-log.md) | **All findings and decisions in date order**: what worked, what failed, measured numbers |
| [`reports/Layered methods - how to get them working.md`](reports/Layered%20methods%20-%20how%20to%20get%20them%20working.md) | Every layered-generation paper: what runs, what could run, what to borrow; LayerFusion / DreamLayer move-resize analysis |
| [`reports/Research gap and questions.md`](reports/Research%20gap%20and%20questions.md) | Research gaps the prototype should fill (persistent layers + story events + re-harmonised moves, measured technically) |
| [`reports/University GPU access LMU.md`](reports/University%20GPU%20access%20LMU.md) | LMU Informatics / LRZ GPU options versus paid APIs, with an access guide |
| [`docs/open-paths.md`](docs/open-paths.md) | **Open routes and ideas still to research**: where strong text-to-image models fit, segment + inpaint, story events, budget |
| [`reports/`](reports/) | Research reports: layered-generation models, hosting, the coherent pipeline, Kaggle plans, TPU route, speed and API costs |
| [`reports/Image generation models hosting and costs.md`](reports/Image%20generation%20models%20hosting%20and%20costs.md) | The research before Phase 1: free and cheap places to run image models (Kaggle, LMU GPUs, Spaces, APIs, OpenRouter), a price-to-performance metric, sketch-input methods, CPU-only options, and links to test the candidate models |
| [`reports/research_cpu_speedups.md`](reports/research_cpu_speedups.md) | Ways to speed up runs on the CPU laptop (OpenVINO, tiny decoder, fewer steps, iGPU, pix2pix-turbo, DeepCache, few-step LoRAs), with a test order |
| [`research_notes/`](research_notes/) | Raw source notes behind the reports |
| [`docs/fastsd-cpu/`](docs/fastsd-cpu/) | Earlier phase: offline image generation on a CPU laptop with FastSD CPU. [INSTALL.md](docs/fastsd-cpu/INSTALL.md) and [USAGE.md](docs/fastsd-cpu/USAGE.md), two paths (SD-Turbo, SDXL-Turbo) |
| [`docs/session-summary-sketch-to-image.md`](docs/session-summary-sketch-to-image.md) | How the sketch-to-image phase went, and why each decision was made |
| [`docs/archive/`](docs/archive/) | Superseded documents (the first 3-path guide) |
| [`experiments/`](experiments/) | Sketch-to-image test plan, results, ranking sheet, input sketches, comparison sheets, the user's favourites and offline-test images. Raw image folders are not included. |
| [`fastsdcpu_local/`](fastsdcpu_local/) | Our patch to FastSD CPU and an offline launcher |

## Current status (2026-11-09)

| Part | Status |
|---|---|
| Local app (CPU laptop) | Works. LLM scene split with Ollama `qwen3:4b`, LayerDiffuse SDXL layers, per-layer re-prompt / refine / sketch, captions, recomposite with layer order. About 80 s per 512 px layer. |
| LayerDiffuse scene-to-layer chain (each element generated on the existing scene) | Works on SDXL base (about 11 min per element on CPU) and on a Kaggle T4 (about 18 s per pass at 1024 px) |
| Qwen-Image-Layered (image → RGBA layers) on Kaggle 2× T4 | **Works within the time target:** 384 px, 3–4 layers, 4 steps, no CFG take **27–30 s** per split on the free T4s, with `fp16_mixed`. The defaults (640 px, 50 steps) took 35 min. Tested on 3 art styles; see [docs/findings-log.md](docs/findings-log.md), Phase 4. |
| Move / resize a layer with re-harmonisation | `relocate()` v5 on a T4: about 4 s per edit, identity largely kept (pinning + paste-back) |
| Story events, shadows, relighting | Router, procedural shadow layer and IC-Light relight work on CPU; GPU runs pending |
| FLUX layer models (RevealLayer, LayerDiffuse-Flux, ART+), Qwen-Image-2.1 | Notebooks ready with fp16 fixes; RevealLayer runs end to end on 2× T4 (69 s), output still noise. Kaggle outage on 2026-11-09 blocked further runs |
| Hosted alternatives | Comfy Cloud (RTX PRO 6000) does it in seconds. fal costs about $0.05 per run. Kaggle TPU v5e-8 could plausibly reach about 40 s, but needs a port (2–5 days, unproven). See [reports/research_speed_and_apis.md](reports/research_speed_and_apis.md) and [reports/research_tpu_route.md](reports/research_tpu_route.md). |

## Quick start

1. Clone this repo, then follow [prototype/README.md](prototype/README.md) for the local app. It needs Ollama, and the LayerDiffuse code from [prototype/vendor/README.md](prototype/vendor/README.md).
2. For Qwen-Image-Layered on Kaggle, follow [prototype/docs/qwen-layered-on-kaggle.md](prototype/docs/qwen-layered-on-kaggle.md). The batch notebook and cached weights are public on Kaggle.
3. For offline CPU image generation with FastSD CPU (the earlier sketch-to-image phase), follow [docs/fastsd-cpu/INSTALL.md](docs/fastsd-cpu/INSTALL.md), then [docs/fastsd-cpu/USAGE.md](docs/fastsd-cpu/USAGE.md).
