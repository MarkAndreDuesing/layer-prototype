# To do and open questions

> **Role: current.** Tasks and open ideas in one place (merges the old investigation checklist and open-paths page, both kept in [archive/](archive/MOVED.md)). Tick items here; record what happened in [log.md](log.md) and the resulting state in [status.md](status.md). The user decides verdicts on models; nothing is "rejected" here.

## 1. Get every model working, then benchmark it
Goal per model: a happy path, best features, and numbers on the shared fox & cottage scene (time, VRAM, layer count, recomposite diff, visual sheet), filled into its factsheet in [models/](models/README.md).

- [x] Fox & cottage test scene chosen (seed 13): `prototype/test_scene/fox_cottage.png` + `fox_cottage.json` (names, boxes, caption)
- [x] Fox & cottage: Qwen-Image-Layered and Stable-Layers done (Stable-Layers cleanest: sky / meadow / cottage + tree / path / fox)
- [ ] Run every working model on it: Qwen-Image-Layered, Stable-Layers, RevealLayer, Qwen-Image-2.1, LayerDiffuse chain, SAM2 + LaMa, LayerD
- [ ] `tools/build_benchmarks.py` → comparison table in `models/README.md`
- [x] **RevealLayer:** works on forest and fox & cottage (140 s at 12 steps; filled background + clean object layers)
- [x] **ART+:** works (v9): size fix (`vae_scale_factor` 16) + stacked LoRAs (our loader kept only the last one)
- [x] ART+ on fox & cottage from its JSON boxes (v10): clean styled objects per box, white background, gate arch → car
- [x] ART+: "painting" prefix + background sentence (v11): flat blue background, still vignettes
- [x] ART+ objects pasted on RevealLayer's background: style / scale clash (sticker look), needs harmonisation
- [x] Harmonise ART+ objects (`relocate()` 0.4 / 0.6 + IC-Light): does not close the style gap
- [ ] ART+ at 1024 px; use ART+'s own whole-image frame as the background instead of another model's
- [x] **CLD:** works on fox & cottage (v3): clean layers + inpainted background
- [x] CLD on Qwen-2.1 night / rain composites (v4): works with the same boxes
- [x] CLD with 7 boxes (v5): fits (13.3 GB, 434 s); hedge good, ground path not extractable as a layer
- [ ] CLD at 1024 px; compare moving layers over CLD vs RevealLayer background
- [x] **LayerDiffuse-Flux:** works (run 4), 58 s per object
- [x] **Stable-Layers:** Heun works (v9); fox & cottage: cleanest split so far. Next: colour-shift correction
- [x] **Qwen Lightning LoRA:** 2-step runs + scale sweep done (v10): half the time, worse split
- [x] **LayerD:** weights load fine; 3 layers on a flat design, none on painted scenes (matting predicts no foreground)
- [x] LayerD loop with generic BiRefNet on fox & cottage: fox peeled cleanly, rest of scene one layer (v11)
- [x] **SAM2 + LaMa by noun:** Grounding DINO + SAM2 gives clean object layers fully automatically (7 s)
- [x] SAM2 route: SDXL inpaint for the background hole (v3): sharp, in style (28 s)
- [ ] SAM2 route: amodal completion of occluded objects (pix2gestalt or SDXL-inpaint per object)
- [x] **Qwen-Image-2.1 on fox & cottage:** extraction, fox → wolf, night and heavy-rain edits all work (about 20 s each)
- [x] Qwen-Image-2.1: night / rain composites re-split (v10): works with Stable-Layers
- [ ] Qwen-Image-2.1: cottage extraction without the fox, halo fix on generated objects
- [x] **OmniPSD / PSDiffusion / DreamLayer:** re-checked 2026-11-10: OmniPSD code + dataset only (no weights; RGBA-VAE weights still a placeholder path), PSDiffusion repo not found, nothing for DreamLayer/LayerFusion on HF. Re-check occasionally ([papers-survey.md §2](research/papers-survey.md#2-methods-we-cannot-run-no-weights-or-code))

## 2. The loop on GPU
- [x] `relocate()` + IC-Light relight + shadow layer together on a T4: fox moved on RevealLayer's fox & cottage split, no hole, 3.4 s + 0.04 s + 5 s ([editing.md §4](concepts/editing.md#4-full-move-loop-on-fox--cottage))
- [ ] attention sharing for thin parts; IC-Light keeps original hue (it reddens the fox)
- [x] Story events, generative pass: `bg_relight` (background edit + IC-Light) and `recompose` (edit + re-split) both measured: recompose looks better, IC-Light v1 gives colour casts ([editing.md §5](concepts/editing.md#5-story-event-measured-bg_relight-vs-recompose-fox--cottage-night-and-heavy-rain))
- [ ] Story events: LayerDiffuse effect layers (fog / fire / glow); per-layer Qwen-2.1 relight with alpha check-and-retry done: all 10 edits accepted; the retry logic is `layerproto/layer_edit.py` (self-test passes); next: add shadow layers and call it from the `per_layer` event strategy once a Qwen-2.1 backend exists in the app
- [x] Drift experiment with real editors, arms A (relocate chain) and B (Qwen-2.1 flat chain), 6 fox moves ([research/drift-evaluation.md](research/drift-evaluation.md))
- [x] Drift arm C building block: re-decompose cycles lose 20 dB at once and 6–11 dB more by cycle 5 ([drift-evaluation.md](research/drift-evaluation.md))
- [x] Drift arm C with real edits (SAM2 re-decomposition each turn): 35 dB after the first re-split, 28 dB at turn 6 (between A 60 dB and B 14 dB)
- [ ] Drift: more scenes, re-prompt and relight edits, LPIPS / DINO
- [~] Wire the GPU modules into the app: image → layers path done and proven with a real model (SAM2 decomposer on a T4 through the real route, 5–12 s); open: Kaggle → laptop tunnel (you run it), Qwen-Image-Layered / CLD behind `/decompose`, `/relocate` + relight + shadow routes

## 3. Open questions from the team voice note (2026-11-08)
- [x] **Where do strong text-to-image models fit?** As the base-image generator before decomposition; tested FLUX.1-dev, Qwen-Image-2.1 and SDXL-Turbo scenes through Qwen-Image-Layered / Stable-Layers: all split, FLUX and Qwen-2.1 most cleanly ([pipeline.md §2](concepts/pipeline.md)). Licences: FLUX.1-dev and Qwen Research are non-commercial.
- [x] **Image-to-image front end** (sketch + prompt → image) → decomposer: the sketch-to-image experiment outputs (`pixelart_camp` = canny-conditioned img2img, `sketch_skeleton` = raw-sketch img2img at strength 0.8) went through Qwen-Image-Layered / Stable-Layers in sweeps v9 and v10 and split like any other image (e.g. pale skeleton separated by Lightning). So the front end is just "any image → route A"; the LayerDiffuse chain builds its own layers and does not take an image. SDXL-Turbo img2img is about 58 s on the laptop.
- [x] **LayerDiffuse on complex scenes** (7 elements): works at 1024 px (48 s per element, clean layers), garbled at the 512 px default; layout and scale uncontrolled ([layerdiffuse-sdxl.md](models/layerdiffuse-sdxl.md))
- [ ] Chain with a box-aware layout (draw each element into its box via inpainting mask) or set the engine default to 1024 px
- [~] **Continuous map development / story events:** router + previews built; whole-scene effects on GPU open.
- [~] **Prompt breakdown with AI placement + harmonisation:** the "sticker collage" problem; IC-Light + shadow modules built and tested together on a T4 (fox move).
- [~] **Generate → segment → inpaint:** first attempt done (see SAM2 + LaMa above).

## 4. Speed and hosting
- [~] **TPU:** probe notebook ready (`prototype/tools/make_tpu_probe.py`); CLI batch runs got no TPU; next: interactive TPU session in the Kaggle editor; torch must be pinned to 2.9 for torch_xla 2.9.
- [x] Qwen levers: prompt embeddings encoded once per image in the batch notebook (`cached_encode`), 2-layer splits 25.7 s (v9b); Lightning 2 steps 22 s (v10)
- [ ] API budget: about $0.05 per Qwen-Image-Layered run on fal; €100 ≈ 2,000 runs, better for final demos ([research/hosting.md](research/hosting.md)).

## 5. Documentation
- [ ] Keep factsheets and [status.md](status.md) in sync after each run; run `prototype/tools/check_doc_links.py` before pushing.
