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
| **Qwen-Image-Layered** | image → N RGBA layers | Yes, Apache-2.0 | **Yes, about 33–41 s on 2× T4** (384 px, 4–5 steps, our `fp16_mixed`) | Running now. This is our main route. |
| **Stable-Layers** (Stability AI) | Qwen-Image-Layered plus an RL-tuned LoRA (330 MB, PEFT; trained with Qwen 3.5 9B as the teacher) | Yes: `StabilityLabs/Stable-Layers`; Stability Community licence (free below $1M revenue; attribution required) | Its README says a non-Heun sampler, fewer steps or a higher resolution "will garble the results". It is fixed at Heun, 50 steps (about 100 model evaluations), 640 px, CFG 1. On our T4 that is about 100 × 21 s, i.e. 35 min. | **Not suited to our speed target.** Only worth trying for quality comparisons (e.g. on LRZ). |
| **LayerD** (ICCV 2025) | image → layers: cut out the top layer, fill in beneath, repeat | Yes, Apache-2.0 (repo archived 2026-10-30) | Yes: BiRefNet plus LaMa, runs on CPU | **Tested 2026-11-09** on a Kaggle CPU (needs `transformers<5`): about 60 s per image, but on all 3 of our scenes (impressionist forest, pixel-art camp, pale sketch) it returned **1 layer, the unchanged image**. It was trained on graphic designs, so it finds no "top layer" to peel off in painted scenes. **Not usable without retraining.** Results in `prototype/kaggle/results_layerd_trial/`. |
| **RevealLayer** (ICML 2026) | image + boxes → RGBA layers, including hidden parts | Code Apache-2.0, LoRA on FLUX.1-dev | FLUX on a T4 is hard (fp16 overflow, 12B) [U] | LRZ, or FLUX GGUF with an fp16 workaround like ours. Its amodal completion (filling in hidden parts) fits "move an object, the background stays complete". |
| **ART+ / PrismLayers** | global prompt + boxes → variable number of RGBA layers (joint generation) | Code MIT; weights on Google Drive | Flux-class; hard on a T4 | LRZ. Before downloading, check whether it bundles the ART base that Microsoft withdrew (data-provenance issue). |
| **CLD** | image → layers, one box per layer | Code MIT; weights on HF (FLUX.1-dev, non-commercial) | Hard on a T4 | Import tool only |
| **LayerFusion** | joint fg + bg + blend generation sharing attention | **No code**, but **training-free** on SDXL + LayerDiffuse | Yes, it's our stack | **Re-implement**: the mask and attention-blending equations are in the paper (see §2). |
| **PSDiffusion** | joint multi-layer generation on SDXL | "Code coming soon" | Would fit | Borrow its partial joint self-attention without training; its own LoRA needs about 60 A800-hours to train. |
| **DreamLayer** | joint 2–4 layers on SD1.5, with move/resize and shadows | Dataset on HF (`LL3RD/DreamLayer`); code "coming soon" | Fits | Borrow its harmonisation step without training. A small fine-tune on its dataset is possible [U]. |
| **OmniPSD** | text/image → PSD with a fixed fg/mid/bg grid | Code yes, weights not linked | FLUX.1-dev | Blocked until the weights appear |
| **LASAGNA** (Adobe) | text → composite + background + foreground with shadows | Not released | — | Ideas only. Its fg-given-bg mode is what LayerDiffuse SD1.5 already does. |
| **LayerCraft** | LLM agents plan and edit layered images (has a narrative demo) | Code unreleased | — | Use as a design reference for our Ollama planner |
| **Qwen-Image-2.1** | text → single RGBA object | Yes; Qwen Research licence (non-commercial) | 7B, with our fp16 workaround | Per-object sprites only |
| **Alfie** | training-free single transparent object | Code yes | Yes [U] | Alternative to LayerDiffuse for sprites |
| **See-through** | anime image → up to 23 PSD layers | Code yes | [U] | Only if the game's art style is anime |
| **Generate → SAM2 → amodal completion (pix2gestalt) → LaMa** | build-your-own decomposition | All parts public | Yes | Our "segment + inpaint" route, with LayerD's peel loop as the template |

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

## 3. Ranked recommendations

1. **Keep Qwen-Image-Layered as the decomposer.** It is the only route that runs end to end on free hardware within the time limit.
2. **Stable-Layers** needs 50 Heun steps at 640 px by design, so it doesn't fit our time limit. Keep it for offline quality comparisons only.
3. **Prototype the training-free move/resize re-harmonisation** (§2) on the LayerDiffuse chain. This is our own extension, and it ties to research question RQ4.
4. **Generate → SAM2 → amodal → LaMa** as a cheap second decomposer to compare against Qwen-Image-Layered.
5. **LRZ only:** RevealLayer and ART+ as stronger joint and amodal models, if a supervisor's LRZ AI project becomes available (see [University GPU access LMU.md](University%20GPU%20access%20LMU.md)).

## 4. Research gap (summary of [Research gap and questions.md](Research%20gap%20and%20questions.md))

No released method covers our whole loop. More importantly, **the editing mechanics alone are not new**: Qwen-Image-Layered plus single-layer editing already decomposes, edits a layer and recomposites.

What nobody has measured or built:
- **G1:** whether layer stacks stay consistent over long chains of story edits. Multi-turn editing benchmarks show flat images drift after a few turns, but they never test layer stacks.
- **G2:** a story event selecting and regenerating only the affected layers.
- **G3:** re-harmonising a layer after a move or resize with released tools.
- **G4:** any user study of layer-based generative scene editing.
- **G5:** the quality-versus-latency trade-off on free GPUs.

Candidate research questions (details, metrics and study designs are in the report):
- **RQ1:** do layer stacks drift less than flat editing over 10–20 story events?
- **RQ2:** can an LLM pick which layers a story event should change?
- **RQ3 (HCI):** do layer controls raise players' sense of agency and creativity support compared with prompt-only editing?
- **RQ4:** does training-free move/resize re-harmonisation look plausible within about one minute on a T4?
- **RQ5:** where is the latency/quality frontier, and how long will players wait? Our sweeps already give the first data points.
