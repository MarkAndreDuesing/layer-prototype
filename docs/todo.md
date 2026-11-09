# To do and open questions

> **Role: current.** Tasks and open ideas in one place (merges the old investigation checklist and open-paths page, both kept in [archive/](archive/MOVED.md)). Tick items here; record what happened in [log.md](log.md) and the resulting state in [status.md](status.md). The user decides verdicts on models; nothing is "rejected" here.

## 1. Get every model working, then benchmark it
Goal per model: a happy path, best features, and numbers on the shared fox & cottage scene (time, VRAM, layer count, recomposite diff, visual sheet), filled into its factsheet in [models/](models/README.md).

- [x] Fox & cottage test scene chosen (seed 13): `prototype/test_scene/fox_cottage.png` + `fox_cottage.json` (names, boxes, caption)
- [~] Fox & cottage runs queued: Qwen-Image-Layered (4/5/6 layers, 384/512 px), Stable-Layers (Euler/Heun), RevealLayer (true_gs 1 / 12 steps and 3.5 / 16 steps)
- [ ] Run every working model on it: Qwen-Image-Layered, Stable-Layers, RevealLayer, Qwen-Image-2.1, LayerDiffuse chain, SAM2 + LaMa, LayerD
- [ ] `tools/build_benchmarks.py` → comparison table in `models/README.md`
- [~] **RevealLayer:** works on the forest (run 11); fox & cottage run 14 (5 boxes, true_gs 1/12 and 3.5/16) queued
- [~] **ART+:** runs end to end (run 5); output wrong (small image in a corner, opaque rectangles) → debug size handling and the transparent decode
- [ ] **CLD:** notebook (same code family as RevealLayer; image + boxes + ControlNet adapter)
- [x] **LayerDiffuse-Flux:** works (run 4), 58 s per object
- [x] **Stable-Layers:** Heun works (v9); fox & cottage: cleanest split so far. Next: colour-shift correction
- [ ] **Qwen Lightning LoRA:** 2-step runs, scale sweep 0.25–0.75
- [ ] **LayerD:** threshold / recursion sweep; fox & cottage; a graphic-style image
- [ ] **SAM2 + LaMa:** segment by noun (points/boxes from the LLM breakdown), mask merging, LaMa vs SDXL inpaint, amodal completion (pix2gestalt)
- [ ] **Qwen-Image-2.1:** stronger scene-wide edits (the rain edit was subtle), fix the dark halo on RGBA objects, use it as base-image generator
- [ ] **OmniPSD / PSDiffusion / DreamLayer:** re-check releases occasionally ([papers-survey.md §2](research/papers-survey.md#2-methods-we-cannot-run-no-weights-or-code))

## 2. The loop on GPU
- [ ] `relocate()` + IC-Light relight + shadow layer together on a T4 ([editing.md §2](concepts/editing.md#2-relighting-and-shadows)); attention sharing for thin parts
- [ ] Story events, generative pass: background edit + relight layers, LayerDiffuse effect layers, or edit-and-re-split ([editing.md §3](concepts/editing.md#3-story-events-across-layers))
- [ ] Drift experiment with real editors (layered / flat / re-decompose arms) ([research/drift-evaluation.md](research/drift-evaluation.md))
- [ ] Wire the GPU modules into the app (`prototype/app.py`) as a selectable backend

## 3. Open questions from the team voice note (2026-11-08)
- [ ] **Where do strong text-to-image models fit?** Most likely as the base-image generator before decomposition (Comfy Cloud's reference: Krea2-Turbo → Qwen-Image-Layered, 6 layers). Which shortlisted models give the most separable images, which run on Kaggle, which licences? Planned test: Qwen-Image-2.1 / FLUX / SDXL-Turbo scenes → Qwen-Image-Layered.
- [ ] **Image-to-image front end** (sketch + prompt → image) → decomposer: untested; how it fits the LayerDiffuse route (which builds its own layers) is unclear. SDXL-Turbo img2img is about 58 s on the laptop.
- [ ] **LayerDiffuse on complex scenes** (5+ or overlapping objects): not evaluated; the paper shows better results than the team saw, possibly settings.
- [~] **Continuous map development / story events:** router + previews built; whole-scene effects on GPU open.
- [~] **Prompt breakdown with AI placement + harmonisation:** the "sticker collage" problem; IC-Light + shadow modules built, GPU test open.
- [~] **Generate → segment → inpaint:** first attempt done (see SAM2 + LaMa above).

## 4. Speed and hosting
- [~] **TPU:** probe notebook ready (`prototype/tools/make_tpu_probe.py`); CLI batch runs got no TPU; next: interactive TPU session in the Kaggle editor; torch must be pinned to 2.9 for torch_xla 2.9.
- [ ] Remaining Qwen levers: text-encoder cache in the batch notebook (code exists: `cached_encode`), 2-layer speed.
- [ ] API budget: about $0.05 per Qwen-Image-Layered run on fal; €100 ≈ 2,000 runs, better for final demos ([research/hosting.md](research/hosting.md)).

## 5. Documentation
- [ ] Keep factsheets and [status.md](status.md) in sync after each run; run `prototype/tools/check_doc_links.py` before pushing.
