# To do and open questions

> **Role: current.** Tasks and open ideas in one place (merges the old investigation checklist and open-paths page, both kept in [archive/](archive/MOVED.md)). Tick items here; record what happened in [log.md](log.md) and the resulting state in [status.md](status.md). The user decides verdicts on models; nothing is "rejected" here.

## 1. Get every model working, then benchmark it
Goal per model: a happy path, best features, and numbers on the shared fox & cottage scene (time, VRAM, layer count, recomposite diff, visual sheet), filled into its factsheet in [models/](models/README.md).

- [x] Fox & cottage test scene chosen (seed 13): `prototype/test_scene/fox_cottage.png` + `fox_cottage.json` (names, boxes, caption)
- [x] Fox & cottage: Qwen-Image-Layered and Stable-Layers done (Stable-Layers cleanest: sky / meadow / cottage + tree / path / fox)
- [x] Run every working model on it: Qwen-Image-Layered, Stable-Layers, RevealLayer, CLD, ART+ (from the scene's caption + boxes), Qwen-Image-2.1, SAM2 + LaMa, LayerD (LayerDiffuse chain builds its own scene: 7-element village scene instead)
- [x] `tools/build_benchmarks.py` → comparison table in `models/README.md` (rerun after every result)
- [x] **RevealLayer:** works on forest and fox & cottage (140 s at 12 steps; filled background + clean object layers)
- [x] **ART+:** works (v9): size fix (`vae_scale_factor` 16) + stacked LoRAs (our loader kept only the last one)
- [x] ART+ on fox & cottage from its JSON boxes (v10): clean styled objects per box, white background, gate arch → car
- [x] ART+: "painting" prefix + background sentence (v11): flat blue background, still vignettes
- [x] ART+ objects pasted on RevealLayer's background: style / scale clash (sticker look), needs harmonisation
- [x] Harmonise ART+ objects (`relocate()` 0.4 / 0.6 + IC-Light): does not close the style gap
- [~] ART+ at 1024 px (probe: fits, 11.0 GiB, about 190 s per step, 16 steps about 50 min; full run open, needs a 55-min cap); use ART+'s own whole-image frame as the background instead of another model's
- [x] **CLD:** works on fox & cottage (v3): clean layers + inpainted background
- [x] CLD on Qwen-2.1 night / rain composites (v4): works with the same boxes
- [x] CLD with 7 boxes (v5): fits (13.3 GB, 434 s); hedge good, ground path not extractable as a layer
- [ ] CLD at 1024 px; background comparison CLD vs RevealLayer done from existing runs (CLD's background is a separate generation, see models/cld.md)
- [x] **LayerDiffuse-Flux:** works (run 4), 58 s per object
- [x] **Stable-Layers:** Heun works (v9); fox & cottage: cleanest split so far. Next: colour-shift correction
- [x] **Qwen Lightning LoRA:** 2-step runs + scale sweep done (v10): half the time, worse split
- [x] **LayerD:** weights load fine; 3 layers on a flat design, none on painted scenes (matting predicts no foreground)
- [x] LayerD loop with generic BiRefNet on fox & cottage: fox peeled cleanly, rest of scene one layer (v11)
- [x] **SAM2 + LaMa by noun:** Grounding DINO + SAM2 gives clean object layers fully automatically (7 s)
- [x] SAM2 route: SDXL inpaint for the background hole (v3): sharp, in style (28 s)
- [x] SAM2 route: amodal completion of occluded objects: SDXL-inpaint per object done (31 s, cottage whole, [sam2-lama](models/sam2-lama.md)); pix2gestalt not tried; wired into the SAM2 `/decompose` route as the "Complete hidden parts" option (run on the live server 2026-11-10: works, 80 s with load, cottage fill patchy; fill prompt and mask to improve); Qwen-2.1 "extract only the cottage without X" gives a whole cottage, see [qwen-image-2.1](models/qwen-image-2.1.md)
- [x] **Qwen-Image-2.1 on fox & cottage:** extraction, fox → wolf, night and heavy-rain edits all work (about 20 s each)
- [x] Qwen-Image-2.1: night / rain composites re-split (v10): works with Stable-Layers
- [x] Halo fix on generated objects: `matte_fix.defringe` removes the rim but softens the outline ([qwen-image-2.1.md](models/qwen-image-2.1.md))
- [x] Qwen-Image-2.1: cottage extraction without the fox (done 2026-11-10, whole cottage, see models/qwen-image-2.1.md)
- [x] **OmniPSD / PSDiffusion / DreamLayer:** re-checked 2026-11-10: OmniPSD code + dataset only (no weights; RGBA-VAE weights still a placeholder path), PSDiffusion repo not found, nothing for DreamLayer/LayerFusion on HF. Re-check occasionally ([papers-survey.md §2](research/papers-survey.md#2-methods-we-cannot-run-no-weights-or-code))

## 2. The loop on GPU
- [x] `relocate()` + IC-Light relight + shadow layer together on a T4: fox moved on RevealLayer's fox & cottage split, no hole, 3.4 s + 0.04 s + 5 s ([editing.md §4](concepts/editing.md#4-full-move-loop-on-fox--cottage))
- [x] IC-Light reddens the fox: `restore_chroma` (keep 0.8 gives a* 16.4 vs 20.3 relit vs 15.2 original)
- [ ] attention sharing for thin parts
- [x] Story events, generative pass: `bg_relight` (background edit + IC-Light) and `recompose` (edit + re-split) both measured: recompose looks better, IC-Light v1 gives colour casts ([editing.md §5](concepts/editing.md#5-story-event-measured-bg_relight-vs-recompose-fox--cottage-night-and-heavy-rain))
- [ ] Story events: LayerDiffuse effect layers tried (fog / glow / snow give objects or the scene itself, see [layerdiffuse-sdxl](models/layerdiffuse-sdxl.md); smaller-scale prompts next); per-layer Qwen-2.1 relight with alpha check-and-retry done: all 10 edits accepted; the retry logic is `layerproto/layer_edit.py` (self-test passes); shadow layers and the `per_layer` strategy are now wired in `layerproto/event_apply.py` (CPU-tested with a stub editor); open: a Qwen-2.1 editor backend behind the server, (10 edits re-run with the stricter visible-area check: still 10 of 10 accepted)
- [x] Drift experiment with real editors, arms A (relocate chain) and B (Qwen-2.1 flat chain), 6 fox moves ([research/drift-evaluation.md](research/drift-evaluation.md))
- [x] Drift arm C building block: re-decompose cycles lose 20 dB at once and 6–11 dB more by cycle 5 ([drift-evaluation.md](research/drift-evaluation.md))
- [x] Drift arm C with real edits (SAM2 re-decomposition each turn): 35 dB after the first re-split, 28 dB at turn 6 (between A 60 dB and B 14 dB)
- [ ] Drift: more scenes, re-prompt and relight edits (LPIPS / DINO done on the fox move: same ordering as PSNR, see research/drift-evaluation.md)
- [~] Wire the GPU modules into the app: image → layers path done and proven with a real model (SAM2 decomposer on a T4 through the real route, 5–12 s); the Kaggle → laptop tunnel notebook is ready ([guides/kaggle.md §3b](guides/kaggle.md), 6 steps, you run it), Qwen-Image-2.1 extraction behind `/decompose` (works through the real route in a T4 x2 session, 114 s, clean background; open: arch scale / box use, a tunnel notebook for it), Qwen-Image-Layered / CLD behind `/decompose`, `/chain/relocate` route + `RemoteChain.relocate` + `Engine.relocate_layer` + button **Fit + re-harmonise (GPU)** (done, tested with a fake chain: `tools/test_relocate_route.py`; real run on a server open), relight + shadow routes (`layerproto/event_apply.py` has the CPU glue)

## 3. Open questions from the team voice note (2026-11-08)
- [x] **Where do strong text-to-image models fit?** As the base-image generator before decomposition; tested FLUX.1-dev, Qwen-Image-2.1 and SDXL-Turbo scenes through Qwen-Image-Layered / Stable-Layers: all split, FLUX and Qwen-2.1 most cleanly ([pipeline.md §2](concepts/pipeline.md)). Licences: FLUX.1-dev and Qwen Research are non-commercial.
- [x] **Image-to-image front end** (sketch + prompt → image) → decomposer: the sketch-to-image experiment outputs (`pixelart_camp` = canny-conditioned img2img, `sketch_skeleton` = raw-sketch img2img at strength 0.8) went through Qwen-Image-Layered / Stable-Layers in sweeps v9 and v10 and split like any other image (e.g. pale skeleton separated by Lightning). So the front end is just "any image → route A"; the LayerDiffuse chain builds its own layers and does not take an image. SDXL-Turbo img2img is about 58 s on the laptop.
- [x] **LayerDiffuse on complex scenes** (7 elements): works at 1024 px (48 s per element, clean layers), garbled at the 512 px default; layout and scale uncontrolled ([layerdiffuse-sdxl.md](models/layerdiffuse-sdxl.md))
- [~] Chain with a box-aware layout: CPU part done, `Engine.fit_layer_to_box` / `fit_all_to_boxes` and a **Fit to planned box** button in the layer-order panel move and resize a generated element into the planner's box (aspect kept, centred; test `tools/test_fit_to_box.py`); open: re-harmonising it with `relocate()` after the fit, and 1024 px as the engine default on a GPU
- [~] **Continuous map development / story events:** router + previews built; whole-scene effects on GPU open.
- [~] **Prompt breakdown with AI placement + harmonisation:** the "sticker collage" problem; IC-Light + shadow modules built and tested together on a T4 (fox move).
- [~] **Generate → segment → inpaint:** first attempt done (see SAM2 + LaMa above).

## 4. Speed and hosting
- [~] **TPU:** probe notebook ready (`prototype/tools/make_tpu_probe.py`); CLI batch runs got no TPU; next: interactive TPU session in the Kaggle editor (2026-11-10: the editor offers "TPU v5e-8" in Session options; importing `tpu-probe.ipynb` through Claude in Chrome froze the editor tab twice, so import it by hand: File > Import Notebook); the user then started it and Kaggle queued it at #1510, so no TPU ran yet; torch must be pinned to 2.9 for torch_xla 2.9.
- [x] Qwen levers: prompt embeddings encoded once per image in the batch notebook (`cached_encode`), 2-layer splits 25.7 s (v9b); Lightning 2 steps 22 s (v10)
- [x] API budget (already in research/hosting.md §2): about $0.05 per Qwen-Image-Layered run on fal; €100 ≈ 2,000 runs, better for final demos ([research/hosting.md](research/hosting.md)).

## 5. Documentation
- [ ] Keep factsheets and [status.md](status.md) in sync after each run; run `prototype/tools/check_doc_links.py` before pushing.
