# Layered generation methods: what runs, what could be made to run, what to borrow

Written 2026-11-09. This is a synthesis of four research passes. The detailed notes, with sources and [U] marks for unverified claims, are in:
- [research_notes/Layered methods deep dive/joint_text_to_layers.md](../research_notes/Layered%20methods%20deep%20dive/joint_text_to_layers.md)
- [other_psd_methods.md](../research_notes/Layered%20methods%20deep%20dive/other_psd_methods.md)
- [spatial_edit_mechanics.md](../research_notes/Layered%20methods%20deep%20dive/spatial_edit_mechanics.md)
- [Research gap and questions.md](Research%20gap%20and%20questions.md)

Several facts here come from pages read through a summarising fetch tool, not word for word. Check the claims marked [U] in the notes before building on them.

## 1. Status of every method

| Method | What it does | Code / weights | Runs on our hardware? | How we could get it working |
|---|---|---|---|---|
| **Qwen-Image-Layered** | image → N RGBA layers | Yes, Apache-2.0 | **Yes, 27–35 s on 2× T4** (384 px, 4 layers, 4 steps, CFG 1, our `fp16_mixed`) | Running; main decomposer so far. Heun and 2-layer variants in the v9 sweep. |
| **Stable-Layers** (Stability AI) | Qwen-Image-Layered plus an RL-tuned LoRA (330 MB, PEFT) | Yes: `StabilityLabs/Stable-Layers`; Stability Community licence | **Yes at reduced settings:** 4 Euler steps 34 s, 8 steps 71 s (384 px). README asks for Heun, 50 steps, 640 px. | Tried (v8): cleaner semantic split, but a colour shift (recomposite diff 15–18 vs 7). Next: Heun via `HeunVelocity` (ready), more steps, colour correction. |
| **LayerD** (ICCV 2025) | image → layers: cut out the top layer, fill in beneath, repeat | Yes, Apache-2.0 (repo archived 2026-10-30) | Yes: BiRefNet plus LaMa, CPU | Tried once on a Kaggle CPU (needs `transformers<5`): about 60 s per image, but returned 1 layer on our 3 painted scenes (it was trained on graphic designs). Next: thresholds, recursion depth, the fox & cottage scene, a graphic-style input. |
| **RevealLayer** (ICML 2026) | image + boxes → RGBA layers, including hidden parts | Code Apache-2.0, LoRA on FLUX.1-dev | **Runs end to end on 2× T4** (GGUF Q4, 512 px, 69 s for 12 steps), output still noise from fp16 overflow | Run 8 patches three fp16 hazards in its code; a plain-FLUX control run checks the base. Waiting for Kaggle GPUs. |
| **ART+ / PrismLayers** | global prompt + boxes → variable number of RGBA layers (joint generation) | Code MIT; weights on Google Drive; **ART base LoRA hash-identical to `thuteam/CLD` `pre_trained_LoRA` (Apache-2.0)** | Notebook ready (GGUF Q4, unmerged LoRAs, `fp16_mixed`) | Only FLUX model that goes **text + boxes → layers** without an input image. Waiting for Kaggle GPUs. |
| **CLD** | image + one box per layer → layers | Code MIT; weights on HF (Apache-2.0) on FLUX.1-dev (non-commercial) | Plan ready (same code base as RevealLayer) | Notebook after RevealLayer works. |
| **LayerDiffuse-Flux** | text → single transparent image (FLUX) | LoRA + transparent VAE on HF (AFL-3.0) | Notebook ready (`fp16_mixed`, unmerged LoRA) | Queued during the Kaggle outage. |
| **LayerFusion** | joint fg + bg + blend generation sharing attention | **No code**, but **training-free** on SDXL + LayerDiffuse | Yes, it's our stack | **Re-implement**: the mask and attention-blending equations are in the paper (see §2). Not started. |
| **PSDiffusion** | joint multi-layer generation on SDXL | "Code coming soon" (re-checked 2026-11-09) | Would fit | Borrow its partial joint self-attention without training. |
| **DreamLayer** | joint 2–4 layers on SD1.5, with move/resize and shadows | Dataset on HF (`LL3RD/DreamLayer`); code "coming soon" | Fits | Its harmonisation step is the basis of our `relocate()`. A small fine-tune on its dataset is possible [U]. |
| **OmniPSD** | text/image → PSD with a fixed fg/mid/bg grid | Code yes, weights not released (re-checked) | FLUX.1-dev | Blocked until the weights appear |
| **LASAGNA** (Adobe) | text → composite + background + foreground with shadows | Not released | — | Ideas only. |
| **LayerCraft** | LLM agents plan and edit layered images (has a narrative demo) | Code + LoRA released (re-checked) | — | Design reference for our Ollama planner and event router |
| **Qwen-Image-2.1** | text → RGBA; **edits a transparent layer; extracts a subject**; masked local edits | Yes; Qwen Research licence (non-commercial) | 7B; Turbo 8 steps; GGUF Q4 3.9 GB; stock VAE NaNs in fp16, texture-fix VAE is fp16-safe | Notebook ready (re-prompt a layer, extract, scene-wide edit). Waiting for Kaggle GPUs. No multi-layer split mode. |
| **Alfie** | training-free single transparent object | Code yes | Yes [U] | Alternative to LayerDiffuse for sprites; not tried |
| **See-through** | anime image → up to 23 PSD layers | Code yes | [U] | Only if the game's art style is anime; not tried |
| **Generate → SAM2 → (amodal) → LaMa** | build-your-own decomposition | All parts public | **Yes: 3–7 s on a T4** | Tried with "largest K masks": fast, near-exact recomposite, poor grouping. Next: segment by noun from the LLM breakdown, amodal completion. |

## 2. The LayerFusion question: can its edit phase rescue LayerDiffuse?

**Answer: not directly.** LayerFusion is training-free, but **its paper describes no spatial (move/resize) editing**. It only says that layers make editing "straightforward". What it contributes is **harmonised joint generation**:
- It runs a LayerDiffuse foreground branch next to a plain-SDXL background and blend branch (RealVisXL by default).
- It builds soft and hard masks from the foreground branch's self-attention and its `<EOS>` cross-attention at t = 0.8T.
- It blends attention between the branches. Eq. 1: `a'_Blended = a_FG·m_soft + a_Blended·(1 − m_soft)`. Eq. 2: `a'_FG = a'_Blended·m_hard + a_FG·(1 − m_hard)`.

The move/resize mechanism we actually want comes from **DreamLayer's harmonisation step**:
- Each layer is denoised on its own down to about t = 600/1000.
- The edit (move, flip or resize) is applied to the foreground latents and alpha.
- The latents are alpha-composited and denoised together from there, which is how shadows follow the object.

DreamLayer trained its model, but this step itself needs no training.

**Status 2026-11-09:** built as `LayerDiffuseChain.relocate()` (v5: Euler sampler, mask extraction, fixed pin mask, optional paste-back): **about 4 s per edit at 512 px on a T4**, identity largely kept. Details in [findings-log.md](../docs/findings-log.md), Phase 5. The original proposal follows.

**Proposed design** (our extension, not from any paper; full pseudocode in `spatial_edit_mechanics.md`):
1. Warp the layer's RGBA to its new box in pixel space and paste it over the background.
2. VAE-encode, then re-noise to strength 0.4–0.6.
3. Denoise with our `bg2ble` blend pipeline. For the first ~40% of steps, pin the object's interior to its re-noised original, so the rim can grow shadows and contact.
4. Optional: add LayerFusion-style attention blending inside the alpha mask.
5. Re-extract a clean RGBA layer with `bgble2fg`.

[est] About 65–75 s per edit on a T4 at 1024 px, and less at 512 px.

**Known risks:**
- **Strength trade-off:** a low strength gives weak new shadows and leaves a stale one at the old spot; a high strength makes the object drift.
- **Old position:** it needs inpainting if the background still contains the object.
- **Up-scaling** blurs the object.
- **Multiple objects** occluding each other still just stack without trained joint attention.

## 3. Order of trials (the user decides verdicts)

Every method above stays in play until tested well enough to compare. Current order, by what is ready:
1. **Qwen-Image-Layered** at the fast settings, plus Stable-Layers with Heun and 2-layer splits (v9 sweep ready).
2. **Qwen-Image-2.1** for re-prompting and extracting single layers (notebook ready).
3. **FLUX models** on 2× T4: FLUX control run, RevealLayer run 8, ART+, LayerDiffuse-Flux, then CLD.
4. **Training-free move/resize** (`relocate()` v5) combined with IC-Light relighting and the shadow layer.
5. **Re-tries:** LayerD and SAM2 + LaMa with the next attempts listed above, Lightning LoRA at 2 steps.
6. All of them on the shared **fox & cottage test scene**, with the same benchmark table (time, VRAM, layer count, recomposite error, visual sheet) and a happy path per model.

## 4. Research gap (summary of [Research gap and questions.md](Research%20gap%20and%20questions.md))

No released method covers our whole loop. More importantly, **the editing mechanics alone are not new**: Qwen-Image-Layered plus single-layer editing already decomposes, edits a layer and recomposites.

What nobody has measured or built:
- **G1:** whether layer stacks stay consistent over long chains of story edits. Multi-turn editing benchmarks show flat images drift after a few turns, but they never test layer stacks.
- **G2:** a story event selecting and regenerating only the affected layers.
- **G3:** re-harmonising a layer after a move or resize with released tools.
- **G5:** the quality-versus-latency trade-off on free GPUs.

Candidate research questions, answered through the prototype and technical measurements (details in the report):
- **RQ1:** do layer stacks drift less than flat editing over 10–20 story events?
- **RQ2:** can an LLM pick which layers a story event should change?
- **RQ4:** does training-free move/resize re-harmonisation look plausible at interactive speed on a T4? (`relocate()` v5: about 4 s.)
- **RQ5:** where is the latency/quality frontier on free hardware? Our sweeps already give the first data points.
