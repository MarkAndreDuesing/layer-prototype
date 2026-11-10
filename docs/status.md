# Current status

> **Role: current state.** The one place for "where are we now". Update this file whenever something changes; other files link here instead of repeating numbers. History is in [log.md](log.md); open work in [todo.md](todo.md). Last update: 2026-11-10.

**Goal (the loop):** one text prompt → a coherent layered scene → edit single layers (re-prompt, move, resize) → recomposite → story events change the scene. Free hardware: Kaggle 2× T4 (fp16 only) and a CPU laptop. No user study is planned; the prototype and its technical measurements are the contribution ([research/research-gap.md](research/research-gap.md)).

**Shared test scene:** fox & cottage (storybook style, SDXL-Turbo seed 13), chosen 2026-11-09: [../prototype/test_scene/fox_cottage.png](../prototype/test_scene/fox_cottage.png). Layer names, hand-drawn boxes (village houses, cottage, stone gate arch, big tree, red fox) and caption: [fox_cottage.json](../prototype/test_scene/fox_cottage.json); preview with boxes: [fox_cottage_boxes.jpg](../prototype/test_scene/fox_cottage_boxes.jpg). Candidates: [../prototype/kaggle/test_scene_candidates/](../prototype/kaggle/test_scene_candidates/). Older test scenes (impressionist forest, pale sketch, pixel-art camp) stay as style checks.

## Each step of the loop

| Loop step | What works now | Speed (free hardware) | Details |
|---|---|---|---|
| Prompt → scene plan (LLM) | Ollama `qwen3:4b` refine + word/layer breakdown | about 30–110 s on CPU | [concepts/pipeline.md](concepts/pipeline.md) |
| Prompt → whole image | SDXL-Turbo (CPU), FLUX.1-dev GGUF (T4), Qwen-Image-2.1 (T4) | CPU ~60–160 s; T4 18–41 s | [models/](models/README.md) |
| Image → layers | **Qwen-Image-Layered** (main), **RevealLayer** (image + boxes, fills hidden parts), Stable-Layers | 27–35 s / 134 s / 34 s | [models/qwen-image-layered.md](models/qwen-image-layered.md), [models/revealayer.md](models/revealayer.md) |
| Text → layers directly | LayerDiffuse SDXL chain (one element at a time); ART+ (text + boxes → background + transparent layers, ≈ 5 min) | 18 s per pass at 1024 px (T4) | [models/layerdiffuse-sdxl.md](models/layerdiffuse-sdxl.md), [models/art-plus.md](models/art-plus.md) |
| Re-prompt one layer | LayerDiffuse `bg2ble → bgble2fg` through the composite; **Qwen-Image-2.1 edits a transparent layer directly** | 18 s / 23 s | [models/qwen-image-2.1.md](models/qwen-image-2.1.md) |
| Extract one object as a layer | Qwen-Image-2.1 extraction (clean, in place); Grounding DINO + SAM2 by name | 22 s / 7 s | [models/sam2-lama.md](models/sam2-lama.md) |
| Move / resize a layer | `relocate()` v5: warp + SDXL re-harmonise + interior pinning + paste-back | about 4 s per edit (T4) | [editing.md §1](concepts/editing.md#1-move--resize-with-re-harmonisation-relocate) |
| Relight / shadows | IC-Light relight + procedural shadow layer, tested with `relocate()` on the fox (T4) | IC-Light 5 s, shadow 0.04 s (T4) | [editing.md §2](concepts/editing.md#2-relighting-and-shadows) |
| Story events | Event router (LLM → per-layer actions) + instant procedural previews (CPU); **Qwen-Image-2.1 scene edits (night, heavy rain) keep the composition** | 30 s routing (CPU) + 22 s edit (T4) | [editing.md §3](concepts/editing.md#3-story-events-across-layers) |
| Recomposite | Alpha compositing in code | instant | — |
| Measure drift over many edits | Harness built (3 arms), self-test only | — | [research/drift-evaluation.md](research/drift-evaluation.md) |

## Each model: best result so far

One line per model; the full measured table is generated in [models/README.md](models/README.md), the method overview is [research/papers-survey.md](research/papers-survey.md).

| Model | Best result on our hardware | State | Factsheet |
|---|---|---|---|
| Qwen-Image-Layered | 27–35 s per split (2× T4); fox & cottage: bg / roof / cottage + tree / fox in 33 s | ✅ main decomposer | [→](models/qwen-image-layered.md) |
| Stable-Layers | fox & cottage: sky / meadow / cottage + tree / path / fox (39 s Euler, 67 s Heun); colour shift | ✅ | [→](models/stable-layers.md) |
| RevealLayer | 134 s, background with hidden areas filled + clean layers (forest) | ✅ forest and fox & cottage (140 s, 5 boxes: filled background + cottage / gate / tree / fox) | [→](models/revealayer.md) |
| Qwen-Image-2.1 | about 20 s: fox extracted cleanly, fox → wolf keeping alpha, **strong night / heavy-rain scene edits keeping the composition** | ✅ | [→](models/qwen-image-2.1.md) |
| LayerDiffuse-Flux | 58 s per transparent object (one T4), semi-transparent glass | ✅ | [→](models/layerdiffuse-flux.md) |
| LayerDiffuse SDXL + chain | 18 s per element (T4); `relocate()` 4 s | ✅ prototype base | [→](models/layerdiffuse-sdxl.md) |
| ART+ | text + boxes → clean background + transparent object layers (≈ 5 min, 8.5 GB); night scene very dark | ✅ since v9 (stacked-LoRA fix) | [→](models/art-plus.md) |
| SAM2 + LaMa | **by noun (Grounding DINO + SAM2): house / cottage / gate / tree / fox in 7 s**, fully automatic; blurry background fill | ✅ by noun | [→](models/sam2-lama.md) |
| LayerD | 3 clean layers on a flat graphic design (206 s CPU); finds no foreground on painted scenes (matting empty) ; with generic BiRefNet it peels the fox cleanly, then takes the rest of the scene as one layer | ⚠️ partial on painted scenes | [→](models/layerd.md) |
| Qwen Lightning LoRA | no speed-up at 4 steps | ⚠️ retry at 2 steps | [→](models/qwen-lightning.md) |
| CLD | — | ⏳ not run yet | [→](models/cld.md) |
| Plain FLUX.1-dev (control) | 41 s for 16 steps, no overflow | ✅ check only | — |
| PSDiffusion, DreamLayer, OmniPSD, LASAGNA, LayerFusion … | — | ❌ no weights / code | [papers-survey §2](research/papers-survey.md#2-methods-we-cannot-run-no-weights-or-code) |

## Infrastructure
- **Kaggle:** 2× T4, 30 GPU-h/week, **max 2 batch GPU sessions at once**; notebooks are generated by `prototype/tools/make_*.py` and pushed with the CLI ([guides/kaggle.md](guides/kaggle.md)). Quota used this week: about 7 h of 30 (check on kaggle.com/settings).
- **TPU:** CLI batch runs got no TPU (standard CPU image); next try is an interactive TPU session ([research/hosting.md](research/hosting.md#tpu)).
- **Repo:** github.com/MarkAndreDuesing/layer-prototype.
