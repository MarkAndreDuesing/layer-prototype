# The pipeline: loop, routes, roles, data model

> **Role: concept (high).** How the prototype's loop is built and why. Current numbers: [../status.md](../status.md); models: [../research/papers-survey.md](../research/papers-survey.md). Mechanics of move/resize, relighting and story events: [editing.md](editing.md).

## 1. The loop

```mermaid
flowchart LR
    P[Players' text / sketch] --> L[LLM: scene plan<br/>layer names + boxes]
    L --> G[Whole scene image<br/>SDXL-Turbo · FLUX · Qwen-Image-2.1]
    G --> D[Split into RGBA layers<br/>Qwen-Image-Layered · Stable-Layers · RevealLayer]
    L -. text + boxes .-> J[Joint layer generation<br/>ART+ · LayerDiffuse chain]
    J --> S
    D --> S[(Layer stack<br/>scene.json)]
    S --> E1[Re-prompt a layer<br/>LayerDiffuse chain · Qwen-Image-2.1 edit]
    S --> E2[Move / resize<br/>relocate + relight + shadow]
    S --> E3[Story event<br/>router → per-layer actions]
    E1 --> S
    E2 --> S
    E3 --> S
    S --> R[Recomposite → display]
```

**Why "generate one image, then split it" (or generate jointly):** generating every object on its own and pasting them gives a **sticker collage**: objects don't share lighting, perspective or style. The first prototype did exactly that (LLM splits the prompt into nouns → one LayerDiffuse layer per noun → boxes), so the LLM split is now only a helper (layer names, boxes, captions).

## 2. Candidate routes (from the 11-07 analysis, updated with measurements)

| | Route | Coherence | Re-prompt one layer | Move / resize | On free hardware (measured) |
|---|---|---|---|---|---|
| **A** | text-to-image → **Qwen-Image-Layered** (or Stable-Layers) | highest: all layers from one image | inpaint the composite in the layer mask, or edit the RGBA layer (Qwen-Image-2.1) | 2D + `relocate()` | ✅ 27–35 s per split on 2× T4 |
| **B** | background → per object LayerDiffuse `bg2ble → bgble2fg` on the composite so far | high: each object drawn into the real scene | re-run both steps with the other layers as background | `relocate()`; lighting baked in | ✅ 18 s per element (T4) |
| **C** | text-to-image → segmenter (SAM2, optionally grounded by the LLM's nouns) → LaMa / SDXL fill | as good as the source; occluded parts missing | inpaint the composite, re-segment | 2D; a moved object shows a "bite" unless completed | ✅ 3–7 s (T4); grouping poor so far |
| **D** | native joint models (ART+, PSDiffusion, DreamLayer, LASAGNA) | high in the papers | not described | DreamLayer only | ART+ runs (output still wrong); others unreleased |
| **E** | layout first (LLM / user boxes) → whole-canvas img2img ≈ 0.6 → split with A or C | high and respects placement | as A or C | placement fixed up front | untested |
| **F** | image + boxes → **RevealLayer** (or CLD) | high; hidden parts and background filled | as A | moving reveals no hole | ✅ 134 s (2× T4) |

**Re-prompting a layer while staying coherent:** never regenerate the layer alone. Background = composite of all other layers → mask = the layer's region → draw the new content into that scene (LayerDiffuse `bg2ble`, SDXL inpaint) → extract it as a layer (`bgble2fg`, a segmenter, or re-split). Qwen-Image-2.1 can edit the RGBA layer directly but doesn't see the scene.

### Which text-to-image generator feeds route A? (run v11, 384 px, 4–5 layers, 2× T4)

Same splitter, three generators. Tiles: image | recomposite | background | layers.

| Generator → scene | Splitter | Time | Recomposite diff | What came out |
|---|---|---|---|---|
| FLUX.1-dev (campfire in a pine forest, painting) | Stable-Layers, Heun | 57 s | 11.0 | two tree groups, campfire as its own layer, empty clearing as background; the cleanest split of the three |
| FLUX.1-dev | base, Euler | 32 s | 6.3 | 3 frames mostly empty (92–100 % transparent) |
| Qwen-Image-2.1 (fox & cottage from the caption) | Stable-Layers, Heun | 66 s | 21.2 | arch + path, cottage, tree, fox; mist-like background |
| Qwen-Image-2.1 | base, Euler | 36 s | 17.4 | 4 object frames, uneven |
| SDXL-Turbo (the shared fox & cottage) | Stable-Layers, Heun | 66 s | 20.7 | sky, meadow, cottage + tree together, path, fox |
| SDXL-Turbo | base, Euler | 36 s | 16.7 | 4 frames, fox and cottage separated |

FLUX.1-dev:

![FLUX split](../img/gen_flux_split.png)

Qwen-Image-2.1:

![Qwen-Image-2.1 split](../img/gen_q21_split.png)

SDXL-Turbo:

![SDXL-Turbo split](../img/gen_sdxlturbo_split.png)

Reading: every generator gives a separable picture; the scenes with clearly spaced objects and visible depth (FLUX campfire, Qwen-2.1 cottage) split most cleanly, while SDXL-Turbo's overlapping cottage and tree stay together. Stable-Layers gives cleaner layers but shifts colour (diff 11–21 vs 6–17 for the base model) and takes about 30 s longer. One scene and one seed per generator, so this compares examples, not generators.

## 3. Roles (one model per role, each behind its own interface)

| # | Role | Default now | Alternatives | Runs on |
|---|---|---|---|---|
| 1 | prompt refine + scene plan (nouns, boxes) | Ollama `qwen3:4b` | API LLM | laptop |
| 2 | text → whole scene | SDXL-Turbo | Qwen-Image-2.1, FLUX.1-dev GGUF | laptop / Kaggle |
| 3 | sketch + prompt → scene | SDXL-Turbo img2img | — | laptop |
| 4 | split into layers | Qwen-Image-Layered | Stable-Layers, RevealLayer, CLD, SAM2 + LaMa, LayerD | Kaggle 2× T4 |
| 5 | add / re-prompt a layer in context | LayerDiffuse chain | Qwen-Image-2.1 RGBA edit | Kaggle T4 |
| 6 | move / resize + re-harmonise | `relocate()` + IC-Light + shadow layer | plain 2D move (preview) | Kaggle T4 |
| 7 | story event | event router + previews | edit composite, re-split | laptop + Kaggle |
| 8 | recompose | alpha compositing | — | laptop |
| 9 | captions / word links (add-on) | `qwen2.5vl:3b` | API vision model | laptop |

## 4. Data model (`scene.json`)

| Object | Fields |
|---|---|
| Scene | prompt, chosen base image, canvas size, ordered layers, edit history |
| Layer | full-canvas RGBA PNG; transform `{x, y, scale, rotation}`; visibility; provenance (decomposed / re-prompted / edited / added by event); prompt, seed; optional caption |

## 5. Build status of the original plan

| Phase (from the 11-07 plan) | Status |
|---|---|
| 1. Remote decompose (Qwen-Image-Layered on Kaggle) | ✅ done (batch notebook; no live endpoint yet) |
| 2. Stronger text-to-image | ⚠️ tested (FLUX, Qwen-Image-2.1), not wired into the app |
| 3. "Generate → choose → decompose" in the UI + canvas | ⚠️ "Decompose an image (GPU)" tab + `Engine.decompose_image` + server route `/decompose` + `RemoteDecomposer` built and tested with a fake decomposer (`tools/test_decompose_roundtrip.py`); the route is also proven with a **real model**: `Sam2Decomposer` (Grounding DINO + SAM2 + LaMa, `backends/sam2_decomposer.py`) behind `/decompose` on a T4, called through the real client, by noun 12 s and by boxes 5 s after a 43 s load, five layers + background stored as a scene (`tools/sam2_server_demo.py`, `results_sam2_server/`); a tunnel from Kaggle to the laptop and Qwen-Image-Layered / CLD behind the same route are open |
| 4. Re-prompt / story-event layers via the chain | ✅ on Kaggle; app integration open |
| 5. Content edit of one layer (Qwen-Image-2.1 / Qwen-Image-Edit) | ✅ Qwen-Image-2.1 tested |
| 6. Captions and word links | ✅ in the local app |

Snapshots with the original reasoning: 2026-11-07 Coherent layered generation pipeline (removed 2026-11-10; in git history before commit 35dc02a), 2026-11-07 Qwen layered multi model pipeline plan (removed 2026-11-10; in git history before commit 35dc02a).
