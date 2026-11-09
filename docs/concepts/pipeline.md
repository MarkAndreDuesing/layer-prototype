# The pipeline: roles, flow and data model

> **Role: concept (stable explanation).** How it works and why. Current numbers live in [../status.md](../status.md) and the factsheets in [../models/](../models/README.md); history in [../log.md](../log.md).

## Goal

**text prompt → coherent layered scene → modify single layers (re-prompt, move, resize) → recompose → story events change the scene.** Each step uses the model that is best at it; no single model does everything.

## Why "generate one image, then split it"

Generating each object separately and pasting them together gives a **sticker collage**: objects don't share lighting, perspective or style (the first prototype did exactly this: an LLM split the prompt into nouns, LayerDiffuse made one transparent layer per noun, boxes placed them). Coherence comes from either
1. **one image first, then split** (Qwen-Image-Layered, RevealLayer, SAM2 + LaMa), or
2. **joint or conditioned generation** (ART+ from text + boxes; the LayerDiffuse chain drawing each new element *into* the current scene).

Analysis and the five candidate pipelines: [../archive/2026-11-07 Coherent layered generation pipeline.md](../archive/2026-11-07%20Coherent%20layered%20generation%20pipeline.md).

## Roles (one model per role, each behind its own interface)

| # | Role | Default now | Alternatives | Where it runs |
|---|---|---|---|---|
| 1 | Prompt optimiser + scene plan (nouns, boxes) | Ollama `qwen3:4b` | API LLM | laptop |
| 2 | Text → whole scene | SDXL-Turbo (CPU) | Qwen-Image-2.1, FLUX.1-dev GGUF, ART+ (text + boxes → layers directly) | laptop / Kaggle |
| 3 | Sketch + prompt → scene | SDXL-Turbo img2img | — | laptop |
| 4 | **Decompose into layers** | **Qwen-Image-Layered** | RevealLayer (with boxes, fills hidden parts), Stable-Layers, CLD, SAM2 + LaMa, LayerD | Kaggle 2× T4 |
| 5a | Re-prompt one layer through the scene | LayerDiffuse `bg2ble → bgble2fg` on the composite of the *other* layers | Qwen-Image-2.1 edit of the layer itself | Kaggle T4 |
| 5b | Content edit of one layer | Qwen-Image-2.1 RGBA edit | Qwen-Image-Edit, FLUX Kontext | Kaggle |
| 6 | Add an element / story event | LayerDiffuse chain on the composite; event router ([story-events.md](story-events.md)) | edit composite then re-split | Kaggle / laptop |
| 7 | Move / resize + re-harmonise | `relocate()` ([relocate.md](relocate.md)) + relight + shadow ([harmonisation.md](harmonisation.md)) | plain canvas move (preview) | Kaggle T4 |
| 8 | Recompose | alpha compositing in code | — | laptop |
| 9 | Captions / word links (add-on) | `qwen2.5vl:3b` captions per layer | API vision model | laptop |

## Flow
1. Players describe the scene (text, optionally a sketch); role 1 refines it.
2. Role 2/3 makes candidate whole images; the group picks one.
3. Role 4 splits it into N aligned, full-canvas RGBA layers.
4. Players arrange layers (move, resize, reorder); role 7 re-harmonises on demand, role 8 recomposes.
5. Per-layer fixes: re-prompt (5a), content edit (5b), or re-split one layer.
6. Story events (role 6) add or change layers; players accept or edit.

## Data model (`scene.json`)
- **Scene:** prompt, base image, canvas size, ordered layers, edit history.
- **Layer:** full-canvas RGBA PNG; transform `{x, y, scale, rotation}`; visibility; provenance (decomposed / re-prompted / edited / added by event); prompt, seed; optional caption.

Earlier plans (kept as snapshots): [../archive/2026-11-07 Qwen layered multi model pipeline plan.md](../archive/2026-11-07%20Qwen%20layered%20multi%20model%20pipeline%20plan.md), [../archive/2026-11-07 Layer based architecture models and hosting.md](../archive/2026-11-07%20Layer%20based%20architecture%20models%20and%20hosting.md).
