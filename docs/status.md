# Current status

> **Role: current state.** The one place for "where are we now". Update this file whenever something changes; other files link here instead of repeating numbers. New to the project: [handoff.md](handoff.md). History is in [log.md](log.md); open work in [todo.md](todo.md). Last update: 2026-11-11 (first-draft hand-off).

**Goal (the loop):** one text prompt → a coherent layered scene → edit single layers (re-prompt, move, resize) → recomposite → story events change the scene. Free hardware: Kaggle 2× T4 (fp16 only) and a CPU laptop. No user study is planned; the prototype and its technical measurements are the contribution ([research/research-gap.md](research/research-gap.md)).

**Shared test scene:** fox & cottage (storybook style, **made by SDXL-Turbo**, seed 13), chosen 2026-11-09: [../prototype/test_scene/fox_cottage.png](../prototype/test_scene/fox_cottage.png), with names, hand-drawn boxes and caption in [fox_cottage.json](../prototype/test_scene/fox_cottage.json). Which model made which part of every picture in the model pages: [models/provenance.md](models/provenance.md).

**Recommended core path (proposal, the user decides):** SDXL-Turbo makes the first picture → Qwen-Image-2.1 splits and modifies it → Stable-Layers / CLD as optional splitters → IC-Light, shadow layers and `relocate()` on single layers; RevealLayer for inpainting. Reasoning and keep / explore / drop per model: [models/pipeline-plan.md](models/pipeline-plan.md).

## Each step of the loop

| Loop step | What works now | Speed (free hardware, warm model) | Details |
|---|---|---|---|
| Prompt → scene plan (LLM) | Ollama `qwen3:4b` refine + word / layer breakdown | about 30-110 s on CPU | [concepts/pipeline.md](concepts/pipeline.md) |
| Prompt → whole image | **SDXL-Turbo** (fast first step); Qwen-Image-2.1 text to image | SDXL-Turbo 1.1-1.5 s on a T4 (about a minute on the CPU); Qwen-2.1 17-23 s | [models/sdxl-turbo.md](models/sdxl-turbo.md) |
| Image → layers | **Qwen-Image-2.1 extraction** (clean on the daylight scene, weaker on night / rain edits); **Stable-Layers at 640 px** (one clean layer per object on the scene and on the edits); CLD and SAM2 + Qwen-2.1 as options | Qwen-2.1 118 s (four objects + background); Stable-Layers about 100 s (Euler) / 178 s (Heun); CLD 116-459 s; SAM2 3 s | [models/qwen-image-2.1.md](models/qwen-image-2.1.md), [models/stable-layers.md](models/stable-layers.md) |
| Hidden parts / background without the objects | Qwen-2.1 ("remove X", "extract only X without Y"); RevealLayer's filled backgrounds (inpainting role) | about 24 s per call; RevealLayer 85-358 s | [models/amodal-fill.md](models/amodal-fill.md), [models/revealayer.md](models/revealayer.md) |
| Re-prompt / edit one layer | **Qwen-Image-2.1 RGBA edit** with an alpha check-and-retry (10 of 10 edits accepted) | 22-29 s per edit | [models/qwen-image-2.1.md](models/qwen-image-2.1.md) |
| Move / resize a layer | direct manipulation (instant) and `relocate()` (warp + SDXL re-harmonise, LayerDiffuse SDXL engine); a Qwen-2.1 harmonise gives a convincing shadow but redraws the picture | `relocate()` 3.4 s; Qwen-2.1 22-29 s | [models/relocate-harmonise.md](models/relocate-harmonise.md) |
| Relight / shadows | **Qwen-2.1 per-layer relight** (best on night and rain objects); **IC-Light** kept pending more tests; procedural shadow layer (multiply option) | 22-26 s per layer; IC-Light about 45 s per layer on the CPU; shadow 0.5 s for five layers | [models/relight-ic-light.md](models/relight-ic-light.md), [models/shadow-layers.md](models/shadow-layers.md) |
| Story events | Qwen-2.1 whole-scene edit (night, heavy rain keep the composition) then a split; per-layer relight; LLM router + instant CPU previews (router not wired into the app) | edit about 22 s + split about 100-180 s | [concepts/editing.md](concepts/editing.md) |
| Recomposite | alpha compositing in code | instant | — |
| Measure drift over many edits | harness + first real run (6 fox moves): persistent layers 60 dB outside the edit, re-decompose every turn 59 → 28 dB, flat Qwen-2.1 20 → 14 dB with the fox duplicated at turn 5; LPIPS / DINO agree | one scene, arms A, B and C | [research/drift-evaluation.md](research/drift-evaluation.md) |

## Each model: state and role

The full measured table is generated in [models/README.md](models/README.md); the method overview is [research/papers-survey.md](research/papers-survey.md). Verdict notes by the user are under the role line of each factsheet.

| Model | Role in the plan | Best measured result | Factsheet |
|---|---|---|---|
| SDXL-Turbo | first image (text, image to image) | 1.1-1.5 s per 512 px image on a T4; image to image replaces the scene | [→](models/sdxl-turbo.md) |
| Qwen-Image-2.1 | **core**: edit, extract, remove, relight, amodal, text to image | see the table above | [→](models/qwen-image-2.1.md) |
| Stable-Layers | optional splitter (clean on the scene and on story edits) | 640 px, 4 steps: sky / ground / cottage / tree / fox as layers, about 100 s | [→](models/stable-layers.md) |
| CLD | optional splitter | decent layers at 4-16 steps (116-459 s), invents a different background | [→](models/cld.md) |
| Qwen-Image-Layered (base) | in contention (inconsistent) | good on the forest scene; no object layers on the fox scene at 384-640 px | [→](models/qwen-image-layered.md) |
| RevealLayer | inpainting (filled backgrounds) | clean filled backgrounds, layers follow boxes; 85-358 s | [→](models/revealayer.md) |
| SAM2 + Grounding DINO (+ LaMa) | fast masks | 3 s by name; blurry fill, hidden parts missing | [→](models/sam2-lama.md), [→](models/grounding-dino.md) |
| IC-Light | relight, kept pending more tests | night grade + IC-Light gives a cool cast; poor on rain objects | [→](models/relight-ic-light.md) |
| `relocate()` (LayerDiffuse SDXL) | optional re-harmonise step | 3.4 s per move | [→](models/relocate-harmonise.md) |
| Shadow layers | helper | 0.5 s for five layers | [→](models/shadow-layers.md) |
| SDXL-inpaint | maybe usable (user) | fills of the cottage's hidden part | [→](models/amodal-fill.md) |
| Qwen Lightning | candidate to drop | no object layers at 384, 512 or 640 px; story composites not improved | [→](models/qwen-lightning.md) |
| LayerD | limited; look a little more | peels the fox with generic matting, rest one layer | [→](models/layerd.md) |
| ART+, LayerDiffuse SDXL (generator and effect layers), LayerDiffuse-Flux | not pursued (user) | — | [→](models/art-plus.md), [→](models/layerdiffuse-sdxl.md), [→](models/layerdiffuse-flux.md) |
| PSDiffusion, DreamLayer, OmniPSD, LASAGNA, LayerFusion … | no weights / code | — | [papers-survey §2](research/papers-survey.md#2-methods-we-cannot-run-no-weights-or-code) |

## Infrastructure
- **Kaggle:** 2× T4, 30 GPU-h/week, **max 2 batch GPU sessions at once**; benchmark notebooks are generated by `prototype/tools/make_profile_notebook.py` / `make_*.py` and pushed with the CLI; serve notebooks (tunnel) are imported in the editor ([guides/kaggle.md](guides/kaggle.md)). Quota: check the editor (it showed about 15 of 30 hours used at one point in this block; not tracked here).
- **TPU:** CLI batch runs got no TPU; the interactive TPU session sat at queue position 1510 and no probe ran ([research/hosting.md](research/hosting.md#tpu)).
- **Repo:** github.com/MarkAndreDuesing/layer-prototype.
