# To do and open questions

> **Role: current.** Tasks and open ideas in one place (merges the old investigation checklist and open-paths page, both kept in [archive/](archive/MOVED.md)). Tick items here; record what happened in [log.md](log.md) and the resulting state in [status.md](status.md). The user decides verdicts on models; nothing is "rejected" here.

## 1. Get every model working, then benchmark it
Goal per model: a happy path, best features, and numbers on the shared fox & cottage scene (time, VRAM, layer count, recomposite diff, visual sheet), filled into its factsheet in [models/](models/README.md).

- [x] Fox & cottage test scene chosen (seed 13): `prototype/test_scene/fox_cottage.png` + `fox_cottage.json` (names, boxes, caption)
- [~] Fox & cottage runs queued: Qwen-Image-Layered (4/5/6 layers, 384/512 px), Stable-Layers (Euler/Heun), RevealLayer (true_gs 1 / 12 steps and 3.5 / 16 steps)
- [ ] Run every working model on it: Qwen-Image-Layered, Stable-Layers, RevealLayer, Qwen-Image-2.1, LayerDiffuse chain, SAM2 + LaMa, LayerD
- [ ] `tools/build_benchmarks.py` → comparison table in `models/README.md`
- [~] **RevealLayer:** works (run 11). Next: the second scene and `true_gs` 3.5 / 16 steps, fox & cottage with boxes
- [~] **ART+:** run 3 (decoder checkpoint fix) running
- [ ] **CLD:** notebook (same code family as RevealLayer; image + boxes + ControlNet adapter)
- [~] **LayerDiffuse-Flux:** run 4 (LoRA keys converted from kohya/BFL format) queued
- [~] **Stable-Layers:** v9 sweep (Heun via `HeunVelocity`, 2 layers) re-queued; then colour-shift correction
- [ ] **Qwen Lightning LoRA:** 2-step runs, scale sweep 0.25–0.75
- [ ] **LayerD:** threshold / recursion sweep; fox & cottage; a graphic-style image
- [ ] **SAM2 + LaMa:** segment by noun (points/boxes from the LLM breakdown), mask merging, LaMa vs SDXL inpaint, amodal completion (pix2gestalt)
- [ ] **Qwen-Image-2.1:** stronger scene-wide edits (the rain edit was subtle), fix the dark halo on RGBA objects, use it as base-image generator
- [ ] **OmniPSD / PSDiffusion / DreamLayer:** re-check releases occasionally ([models/unreleased.md](models/unreleased.md))

## 2. The loop on GPU
- [ ] `relocate()` + IC-Light relight + shadow layer together on a T4 ([concepts/harmonisation.md](concepts/harmonisation.md)); attention sharing for thin parts
- [ ] Story events, generative pass: background edit + relight layers, LayerDiffuse effect layers, or edit-and-re-split ([concepts/story-events.md](concepts/story-events.md))
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
