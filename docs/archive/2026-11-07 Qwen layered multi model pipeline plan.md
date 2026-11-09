# Plan: Qwen-Image-Layered multi-model pipeline (hosted GPU / Kaggle)

Status (updated 2026-11-09): **partly built.** Role 4 (decompose) runs on Kaggle 2× T4 in 27–35 s; roles 5a/6 (LayerDiffuse chain) run on a T4; move/resize with re-harmonisation (`relocate()`) works in about 4 s; roles 2 (stronger text-to-image) and 5b (Qwen-Image-Edit) are not built yet; Qwen-Image-2.1 is being tested as an option for 2 and 5b. Written 2026-11-07. Builds on [Coherent layered generation pipeline.md](2026-11-07%20Coherent%20layered%20generation%20pipeline.md) and [Layer based architecture models and hosting.md](2026-11-07%20Layer%20based%20architecture%20models%20and%20hosting.md). **[est]** marks estimates that haven't been tested.

## 1. Goal

The core loop, from the prototype plan:

**text prompt → coherent layered image → modify single layers (prompt, move, resize) → recomposed image.**

Each step uses the model that is best at it. No single model is asked to do everything.

| Model | Specific strength used here |
|---|---|
| Qwen-Image-Layered | Splits one finished image into N RGBA layers that composite back to exactly that image, with hidden areas filled in. Coherent because every layer comes from the same picture |
| LayerDiffuse SDXL (scene-to-layer: `bg2ble` → `bgble2fg`) | Generates one new element *into* an existing scene, matching its lighting and perspective, and returns that element as its own aligned RGBA layer |

## 2. Pipeline modules (one model per role)

Each role sits behind its own interface, the way the image and LLM backends already do in `prototype/layerproto/`. Any role can therefore run locally, on Kaggle, or hosted.

| # | Role | Default model | Alternatives | Runs on | Core? |
|---|---|---|---|---|---|
| 1 | **Prompt optimiser**: turns players' text into one clean scene prompt | Ollama `qwen3:4b` (local) | API LLM (OpenRouter), or vLLM on Kaggle | Laptop | Optional |
| 2 | **Text-to-image**: N whole-scene candidates (plan step 3) | Remote: Qwen-Image-2.1, FLUX.2 Klein 4B, Z-Image Turbo or FLUX 3 Image Klein | Local SDXL-Turbo (CPU, about 80 s per image) | Kaggle T4 / hosted, or laptop | **Core** |
| 3 | **Image-to-image**: sketch plus prompt to a scene (plan step 1, sketch input) | SDXL-Turbo img2img (local) | FLUX.2 Klein 4B reference edit, Z-Image Turbo + scribble ControlNet | Laptop or Kaggle | Core when sketching |
| 4 | **Decompose into layers** | **Qwen-Image-Layered** (layer count chosen by the user, e.g. 4–6) | None runnable with equal quality. CLD needs boxes and FLUX.1-dev (non-commercial) | Hosted, or Kaggle with quantization [est] | **Core** |
| 5a | **Re-prompt one layer, keeping it a separate layer** | **LayerDiffuse `bg2ble` → `bgble2fg`**, conditioned on the composite of all *other* layers | none | Kaggle T4 (fp16); laptop works but slowly | **Core** |
| 5b | **Content edit of one layer** (recolour, replace, change pose) | Qwen-Image-Edit on that layer's RGBA, as in the Qwen-Image-Layered repo flow | FLUX.2 Klein 4B edit | Hosted / Kaggle | Core (alternative to 5a) |
| 6 | **Add a new element from a story event** ("a forest fire starts") | **LayerDiffuse `bg2ble` → `bgble2fg`** on the current composite, giving a new aligned layer | Qwen-Image-Edit on the composite, then decompose again (step 4) | Kaggle / laptop | Core for plan steps 6–8 |
| 7 | **Recompose** | Plain alpha compositing in code, following the layer order and each layer's move/resize transform | — | Laptop | **Core** |
| 7b | **Harmonize** (optional, preview only) | Low-strength img2img on the flattened image | Qwen-Image-Edit "unify lighting" | Any | Optional |
| 8 | **Word breakdown / image-to-text** (add-on) | Vision model (`qwen2.5vl:3b`) captions each decomposed layer. The LLM maps captions to the nouns in the prompt (which word lives in which layer) | API vision model | Laptop | Optional add-on |

## 3. Flow, mapped to the prototype plan

1. **Describe (plan step 1).** Players enter text, and optionally a sketch. Optionally, role 1 refines the text.
2. **Generate interpretations (plan step 3).** Role 2 makes N whole-scene images from the one prompt, or role 3 makes them from a sketch.
3. **Choose (plan step 4).** The group picks one image.
4. **Split into layers (plan step 2).** Role 4 decomposes it into N RGBA layers. Every layer stays full-canvas and aligned, with hidden parts filled in. Each layer is stored as a PNG with transform `{x, y, scale, rotation}` set to the identity.
5. **Arrange (plan step 2).** Users move, resize and rotate layers and change their order. These are canvas operations only, with no model call. Role 7 recomposes.
6. **Fix details per layer (plan step 5).**
   - **Re-prompt, staying coherent (5a):**
     1. Composite every layer except the target one.
     2. Run `bg2ble` with the new prompt to draw the new content into the scene.
     3. Run `bgble2fg` to extract it as the replacement layer.
     4. The other layers stay untouched.
   - **Edit the content in place (5b):** apply Qwen-Image-Edit to the layer itself.
   - **Re-split a layer:** run role 4 on that one layer (the repo supports recursive decomposition).
7. **Story events (plan steps 6–8).** Role 6 adds the event as new layer(s). The players accept, re-prompt (5a/5b) or move them.
8. **Optional add-ons.** Role 8 captions the layers and links them to words in the prompt. Role 7b produces a harmonized preview.

Why this is coherent: the first set of layers comes from one image (role 4). Every later change goes through the current scene (role 5a/6), not through isolated generations.

## 4. Hosting

| Option | What runs there | Notes |
|---|---|---|
| **Kaggle T4 16 GB (free, about 30 GPU-h/week)** | Roles 2, 3, 5a, 6 (SDXL fp16 fits) and role 4 quantized | **Measured:** Qwen-Image-Layered GGUF Q4 split over both T4s with `fp16_mixed`: 27–35 s per split at 384 px. Run `layerproto/remote_server.py` (or ComfyUI's API) behind a tunnel |
| **Modal ($30/month free credit)** | Same as Kaggle, plus an L4/A10 for role 4 at a stable HTTPS endpoint | Doesn't break any terms; the cleanest live backend |
| **Hosted per-image API (fal / Replicate)** | Role 4 only | About $0.05 per image on fal (unverified). Needs a card |
| **Laptop (CPU)** | Roles 1, 7, 8, and slowly roles 2, 3 and 5a/6 (SDXL-Turbo) | Role 4 is not feasible |
| Comfy Cloud | Your existing reference workflow | Used for reference and quick checks only, since you prefer not to rely on a proprietary paid UI |

## 5. Data model (extends the current `scene.json`)

- **Scene:** prompt, chosen base image, canvas size, ordered layers, history of edits.
- **Layer:**
  - full-canvas RGBA PNG
  - transform `{x, y, scale, rotation}`
  - visibility
  - provenance: decomposed, re-prompted (`bg2ble`/`bgble2fg`), edited (Qwen-Image-Edit) or added (story event)
  - prompt used, seed, and an optional caption

## 6. Build phases

1. **Remote role 4.** Write a Kaggle notebook that loads Qwen-Image-Layered (diffusers `QwenImageLayeredPipeline` or ComfyUI) and exposes `POST /decompose {image, n_layers} → [RGBA PNGs]`. Measure memory use and time on a T4. If it doesn't fit, use Modal or fal.
2. **Remote roles 2/3.** Add `/generate` (one stronger text-to-image model) to the same server. Keep local SDXL-Turbo as the fallback.
3. **Prototype integration.** Add a "Generate → choose → decompose" path to the UI, plus a canvas for move, resize and rotate.
4. **Role 5a/6.** Reuse the LayerDiffuse chain backend from the local draft, pointed at the remote fp16 server.
5. **Role 5b.** Add Qwen-Image-Edit or FLUX.2 Klein edit as a second per-layer edit option.
6. **Add-ons.** Role 8 captions and word links; role 7b harmonize preview.

## 7. Open questions to test first

1. Does Qwen-Image-Layered (quantized) fit on a T4, and how long does it take per image? **Answered:** yes on 2× T4, 27–35 s (findings log, Phase 4).
2. Do the LayerDiffuse SDXL conditional models keep quality with SDXL-Turbo (few steps, CFG 1), or do they need SDXL base at 20 steps and CFG 7? **Answered:** the extract step fails on Turbo; SDXL base at 20 steps works (findings log, Phase 2).
3. Does a layer re-prompted with `bg2ble` → `bgble2fg` keep the lighting of the layers decomposed by Qwen? Lighting gets baked into the new layer, so moving it later may look slightly off (LayerDiffuse paper, Sec. 4.12).
4. When the decomposed layers include the background as one big layer, does re-prompting the background need its own path, such as `fg2ble` → `fgble2bg` with the foreground layers as the condition?
