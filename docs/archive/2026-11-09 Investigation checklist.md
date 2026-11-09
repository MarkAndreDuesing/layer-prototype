# Investigation checklist

Everything the user has asked to investigate across the recent sessions, with status. **Done** means verified and written up, with the location given. **Open** means not finished yet. Updated as work progresses.

## A. Qwen-Image-Layered on Kaggle

- [x] Run on free Kaggle T4s despite having no bf16: `fp16_mixed` ([findings-log](../log.md), Phase 3)
- [x] Explore every Kaggle option, including the TPU ([research_tpu_route.md](2026-11-09%20TPU%20route.md)). The TPU port was not attempted.
- [x] Reduce generation time towards ≤ 40–60 s by downscaling: 27–35 s ([findings-log](../log.md), Phase 4)
- [x] Keep running configuration adjustments in the background: sweeps v3–v7, the Lightning LoRA
- [ ] Remaining levers from the plan: fp16 attention everywhere (partly done in `fp16_mixed`), caching the text-encoder outputs, 2 layers
- [ ] TPU v5e-8 port (20 h/week of free quota): researched, not attempted

## B. Reusability for the team

- [x] Git repo, public on GitHub ([layer-prototype](https://github.com/MarkAndreDuesing/layer-prototype))
- [x] Public Kaggle notebooks and weight caches (the 3 originals; new trial notebooks are private because publishing them was blocked)
- [x] `fp16_mixed` as a module, plus a team guide ([qwen-layered-on-kaggle.md](../guides/kaggle.md))
- [x] Consolidate all session material into the repo (README, findings log, experiments, research notes)
- [x] Voice-note ideas that are new or unconfirmed ([open-paths.md](2026-11-09%20Open%20paths.md))

## C. Hosting and cost

- [x] Alternative hosts for Qwen-Image-Layered: Lightning AI, Modal, HF ZeroGPU, WaveSpeed, fal
- [x] Comfy Cloud vs paid APIs ([research_speed_and_apis.md](2026-11-09%20Speed%20and%20APIs.md))
- [x] LMU Informatics / LRZ GPUs vs the paid API ([University GPU access LMU.md](2026-11-09%20University%20GPU%20access%20LMU.md)). Many points are marked unverified; questions for the RBG are listed.

## D. Layered-generation papers (`other_layered_methods.md` and newer)

- [x] First audit of every method's code and weights ([joint_text_to_layers.md](../../research_notes/Layered%20methods%20deep%20dive/joint_text_to_layers.md), [other_psd_methods.md](../../research_notes/Layered%20methods%20deep%20dive/other_psd_methods.md))
- [x] **Double-check every "not released" claim** ([release_status_recheck.md](../../research_notes/Layered%20methods%20deep%20dive/release_status_recheck.md)): still unreleased except OmniPSD code (no weights), LayerCraft code + LoRA, and an unofficial ART+ mirror with a fresh web search: PSDiffusion, DreamLayer, LASAGNA, LayerFusion, OmniPSD weights, LayeringDiff, LayerDecomp, DiffDecompose, Text2Layer, RefLayer, LayerCraft, LiWi, PixelART
- [ ] **Actually try the methods called infeasible on Kaggle**, at reduced settings as with Qwen:
  - [x] Stable-Layers: runs at 34 s (4 steps); cleaner split but colour shift; Heun still open
  - [~] RevealLayer (FLUX.1-dev GGUF + LoRA): run 5 runs end to end on 2× T4 (69 s / 12 steps, peak 10.5/11.2 GB) but gives noise (fp16 overflow); runs 6/7 add `fp16_mixed`, blocked by the Kaggle outage of 2026-11-09
  - [ ] ART+ / PrismLayers (Flux-class): its LoRAs appear to be on HF inside CLD's checkpoint (`pre_trained_LoRA`, `prism_ft_LoRA`), which avoids Google Drive ([re-check](../../research_notes/Layered%20methods%20deep%20dive/heavy_models_recheck_2026-11-09.md))
  - [ ] CLD (FLUX + inpainting ControlNet): code and file sizes read, adaptation planned ([re-check](../../research_notes/Layered%20methods%20deep%20dive/heavy_models_recheck_2026-11-09.md)); waits for the RevealLayer port (same code base)
  - [ ] Qwen-Image-2.1 RGBA (7B): **most promising**, with native RGBA VAE, RGBA editing, subject extraction, Turbo at 8 steps and GGUF Q4 at 3.9 GB; notebook ready (`tools/make_qwen21_rgba_trial.py`), waiting for Kaggle
  - [ ] LayerDiffuse-Flux: notebook pushed (`layerdiffuse-flux-trial`, `fp16_mixed` + unmerged LoRA), queued during the Kaggle outage
  - [ ] OmniPSD: weights still not released (re-checked 2026-11-09)
- [~] LayerD: first attempt returned 1 layer per scene. Next: other thresholds and recursion depth, larger input, and graphic-style inputs (its training domain)
- [~] SAM2 + LaMa (segment + inpaint): first attempt fast, poor automatic grouping. Next: prompted/box segmentation from the LLM layer names (Grounded-SAM style), mask merging, a better inpainter
- [x] **Spatial editing in every paper that has it** (parts 1–3 in `research_notes/Layered methods deep dive/spatial_edit_survey_verified_part*.md`): only DreamLayer has a move/resize algorithm. The decomposition papers just re-composite. The object-move papers keep identity by latent pinning plus attention sharing. (move / resize / re-harmonise): how each does it and what we can borrow. So far only LayerFusion, DreamLayer and PSDiffusion were read ([spatial_edit_mechanics.md](../../research_notes/Layered%20methods%20deep%20dive/spatial_edit_mechanics.md)).
- [x] LayerFusion spatial edit: the paper has none; DreamLayer has one
- [x] Our training-free move/resize re-harmonisation (`relocate()`): works, about 4 s per edit; identity drift is open
- [~] Fix identity drift in `relocate()`: v5 (fixed pin mask + paste-back) largely fixes it; attention sharing not yet tried
- [x] Research gap and research questions ([Research gap and questions.md](../research/research-gap.md))
- [x] Re-check the research gap ([Research gap and questions.md](../research/research-gap.md), section 5): G2 confirmed, G3 narrowed, G5 partly answered by our measurements; Qwen-Image-2.1 moves the novelty further towards evaluation and HCI

## E. Voice-note paths still open ([open-paths.md](2026-11-09%20Open%20paths.md))

- [ ] Where the strong text-to-image models fit (base-image generator → decomposer): untested
- [ ] Image-to-image front end → Qwen-Image-Layered: untested
- [ ] LayerDiffuse on complex scenes (many or overlapping objects): not evaluated
- [~] Scene-wide story events (fire, rain, night): researched; no layered method does it (gap G2). Three training-free routes listed ([re-check](../../research_notes/Layered%20methods%20deep%20dive/heavy_models_recheck_2026-11-09.md), section 4); a 'rainy night' test is in the Qwen-Image-2.1 notebook
- [~] Prompt breakdown with AI placement plus harmonisation (IC-Light): IC-Light v1 (SD1.5, background-conditioned) fits a T4 and could relight moved layers; v2 (FLUX) has no released weights. Not tried yet

## F. Exhaustive open-paths research (2026-11-09 evening)

Report: [Layered loop open paths.md](2026-11-09%20Layered%20loop%20open%20paths.md). Notes: `research_notes/Layered loop open paths/`.

- [x] Qwen-Image-2.1 RGBA in depth. The stock VAE produces NaNs in fp16, so the notebook now uses the texture-fix VAE, plus an NF4 fallback and a NaN check
- [x] FLUX layered models on T4, plus where their weights come from
  - ART base LoRA = CLD `pre_trained_LoRA`, confirmed by hash; it is Apache-2.0 on HF
  - RevealLayer fp16 hazards patched: the -1e9 mask, the refiner residual, and the VAE decode
- [x] Harmonisation: IC-Light v1 fbc plus a procedural shadow layer; IC-Light v2 never released
- [x] Story events across layers: no method exists (G2 confirmed); four strategies compared
- [x] Drift evaluation: EdiVal-Agent harness, 3-arm protocol; the re-decomposition arm would be new
- [x] HCI literature checked (ImageDirector, UIST '26, via the LMU ACM proxy): context only; no user study is planned
- [ ] Kaggle trials once Kaggle recovers: Qwen-Image-2.1 (auto-push), plain-FLUX control, RevealLayer v8, LayerDiffuse-Flux

## G. Non-Kaggle follow-ups (2026-11-09 night)

- [x] Procedural shadow layer `prototype/layerproto/shadow.py`: EasyOmnimatte-style recipe, with a contact line per object and a cap on shadow length; about 0.6 s per layer on CPU. Results in `prototype/kaggle/results_local_shadow/`
- [x] Drift-evaluation harness `prototype/layerproto/drift_eval.py`: 3 arms, metrics measured outside the edited regions, per-layer alpha IoU/L1, AUC. Self-test with stand-in editors (`tools/drift_harness_selftest.py`; not a result about real editors)
- [x] Story-event router `layerproto/events.py`: Ollama qwen3:4b routes 5 test events sensibly (about 30 s each on CPU); instant CPU preview with grading and procedural rain/snow/fog/smoke/glow overlays (`results_local_events/`). Weak spot: prompts from the 4B model are sometimes poor ('solid gold campfire')
- [x] IC-Light fbc relight module `layerproto/relight.py` (with detail transfer and original alpha): CPU smoke test relights a layer to a night background (8 steps, 384 px, 44 s on CPU). Not yet wired into `relocate()`; fire loses brightness (no notion of light sources)
- [x] Qwen levers: `cached_encode()` (disk cache of prompt embeddings) and `HeunVelocity` (exact Heun step through the pipeline's Euler step; toy-ODE check: error 0.05 vs 0.37 for Euler at 8 steps). Qwen batch v9 sweep (Stable-Layers with Heun, 2-layer splits) generated, waiting for Kaggle
- [x] Notebooks ready: `flux-control-trial` (plain FLUX fp16 vs `fp16_mixed`), `artplus-trial` (text + boxes → layers; base LoRA + transparent VAE from official `thuteam/CLD`, the rest from Drive, with the HF mirror as fallback), RevealLayer with 3 fp16 patches, Qwen-Image-2.1 with the texture-fix VAE
- [ ] CLD notebook: waits for RevealLayer to work (same code base)
- [ ] Voice-note E items: feed the Qwen-Image-2.1 scene images into the Qwen batch as new inputs (base-image generator → decomposer); LayerDiffuse complex scenes via `layerproto-chain-batch`. Both need Kaggle

## H. Keep exploring (user decides verdicts; benchmarks + happy path per model)

- [ ] Qwen Lightning LoRA: retry with matched steps and CFG, other LoRA scales, and 2-step/4-step variants; report speed and fidelity per setting
- [ ] LayerD: parameter sweep (see above); local CPU run if the install is allowed, else Kaggle
- [ ] SAM2 + LaMa: text/box-prompted segmentation from the LLM layer names, mask merging, LaMa vs SDXL inpaint for the hidden parts
- [ ] Stable-Layers: Heun v9 sweep, more steps at 384 px, colour-shift correction (match the recomposite to the input)
- [~] TPU probe: notebook ready; 3 CLI runs got no TPU (CPU image), torch 2.11 vs torch_xla ≤ 2.9 found. Next: interactive TPU session in the editor
- [ ] Benchmark table + happy path per model on the shared fox & cottage test scene (plus the old 3 scenes as style checks): time, VRAM, layers, recomposite error, visual sheet
- [~] Shared test scene: fox & cottage chosen by the user; variants generated (`prototype/tools/make_test_scene.py`, `prototype/kaggle/test_scene_candidates/`)
