# Research gap and research questions: layered generation for an interactive story game

Checked 2026-11-09. [U] = not verified in this pass; it comes from earlier notes or a search snippet only.
Inputs:
- the local notes in `research_notes/Layered methods deep dive/` and `research_notes/Coherent layered generation pipeline/`
- the Qwen-Image-Layered (QIL) paper, arXiv HTML https://arxiv.org/html/2512.15603
- GitHub pages for Stable-Layers and RevealLayer, and the LayerCraft abstract
- web searches for game/story and HCI work

## 0. What our loop is
1. A text prompt goes in.
2. The system makes a coherent RGBA layer stack.
3. The user edits single layers: re-prompt, move or resize.
4. The layers are recomposited.
5. A story event changes the scene, for example "the forest is now on fire".
6. Steps 3–5 repeat many times in one play session.

Measured practical facts:
- QIL decomposes in about 33–41 s on a free Kaggle T4 (384 px, 3–4 layers, 4–5 steps).
- The LayerDiffuse SDXL chain (bg → blend → fg) works.

## 1. Coverage matrix

Legend: Y = yes, shown in the paper or repo. P = partial, qualitative only, or only possible with extra glue. N = no or not described. ? = unknown [U].
Column key: "Latency <1 min free GPU" means under one minute per operation on a T4-class or free GPU.

| Method | text→layers | image→layers | per-layer regen in context | move/resize + re-harmonise | temporal/story consistency over many edits | Latency <1 min free GPU | User study / HCI eval | Open weights |
|---|---|---|---|---|---|---|---|---|
| **Qwen-Image-Layered** ([arXiv 2512.15603](https://arxiv.org/html/2512.15603), [HF](https://huggingface.co/Qwen/Qwen-Image-Layered)) | P: a T2L variant is trained, but results are qualitative only (Fig. 8) and the README calls them "limited" | Y: Crello benchmark, RGB L1 and alpha IoU | P: the repo tool edits one RGBA layer with Qwen-Image-Edit; nothing ties that edit to the other layers | P: move/resize are plain pixel operations with no re-harmonisation; Fig. 7 is qualitative | N: not discussed | Y (our measurement): about 35 s, but only at 384 px and 4–5 steps (the card recommends 50 steps at 640 px) | N | Y, Apache-2.0 |
| **Stable-Layers** ([GitHub](https://github.com/Stability-AI/Stable-Layers), [arXiv 2605.30257](https://arxiv.org/abs/2605.30257)) | N | Y: a LoRA on QIL, RL-tuned with a VLM teacher | N | N | N | N: the README recommends an 80 GB-class GPU | N (README) | Y: LoRA on HF, Stability Community licence |
| **LayerDiffuse** ([arXiv 2402.17113](https://arxiv.org/abs/2402.17113)) | Y: fg; SD1.5 joint fg+bg; iterative multi-layer | N | Y: bg→fg and fg→bg conditional | N: no placement control | N | Y: about 4–20 s per step on a 4090; our SDXL chain runs on a T4 | P: the paper has a user-preference study [U] | Y |
| **LayerFusion** ([arXiv 2412.04460](https://arxiv.org/abs/2412.04460)) | Y: fg/bg/blend triplet | N | N | N: editing called "straight-forward", no algorithm | N | Probably: training-free on SDXL [U] | ? [U] | No code needed (training-free); no code released |
| **PSDiffusion** ([arXiv 2505.11468](https://arxiv.org/abs/2505.11468)) | Y | N | P: "layer-wise editing" claimed, no procedure | N | N | Would fit, but cannot run | ? [U] | N: "code coming soon" |
| **DreamLayer** ([arXiv 2503.12838](https://arxiv.org/abs/2503.12838)) | Y: 2–4 layers | P: training-free inversion | N | **Y**: latent move/flip/resize, and shadows follow (IRH) | N | Would fit (SD1.5) | ? [U] | N; dataset only |
| **OmniPSD** ([arXiv 2512.09247](https://arxiv.org/abs/2512.09247)) | Y: fixed fg/mid/bg | Y: Kontext | N | N | N | N: FLUX 12B | ? [U] | N: code yes, weights no |
| **ART+ / PrismLayers** ([arXiv 2505.22523](https://arxiv.org/abs/2505.22523), [GitHub](https://github.com/redredsheep/PrismLayers)) | Y: global caption plus boxes, variable N | N | N | N | N | N: FLUX-class | P: pairwise preference against ART (60% win) | Y: Google Drive (base ART provenance concern) |
| **CLD** ([arXiv 2511.16249](https://arxiv.org/abs/2511.16249)) | N | Y: image plus one box per layer | N | N | N | N: FLUX + ControlNet | ? | Y |
| **RevealLayer** ([GitHub](https://github.com/360CVGroup/RevealLayer), [arXiv 2605.11818](https://arxiv.org/abs/2605.11818)) | N | Y: image plus boxes, recovers occluded content | N | P: a complete background makes moves clean, but there is no re-harmonisation | N | N: FLUX.1-dev LoRA, VRAM not stated | N (README) | Y: Apache-2.0 repo; needs FLUX.1-dev |
| **LayerD** ([GitHub](https://github.com/CyberAgentAILab/LayerD), [arXiv 2509.25134](https://arxiv.org/abs/2509.25134)) | N | Y: graphic design, iterative peel + inpaint | N | N | N | Y: small models | N | Y, Apache-2.0 |
| **LASAGNA** ([arXiv 2601.15507](https://arxiv.org/abs/2601.15507)) | Y | N | **Y**: FG_Gen/BG_Gen with shadows and reflections | P | N | Would fit (2B) | Benchmark; its authors note that prior work relies on bespoke user studies | N |
| **LayerCraft** ([arXiv 2504.00010](https://arxiv.org/abs/2504.00010)) | Y: LLM agent plans the layout, layer-by-layer | N | P: object integration | P | P: "narrative scene generation" demo and "iterative refine"; no consistency eval | ? | N (abstract) | N: "code will be released" |

Adjacent work, outside the matrix:
- Layered Diffusion Brushes, ICCV 2025 ([paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Gholami_Streamlining_Image_Editing_with_Layered_Diffusion_Brushes_ICCV_2025_paper.pdf)). It treats a "layer" as an edit-parameter layer on one image, not an RGBA scene stack. It **has a user study with artists**, so it is the closest HCI precedent.
- Multi-turn editing benchmarks:
  - EdiVal-Agent, ICLR 2026 ([arXiv 2509.13399](https://arxiv.org/abs/2509.13399)). It reports that Qwen-Image-Edit degrades after the first turn and that flow models get progressively brighter with each edit.
  - Omni IIE Bench, 2026 ([arXiv 2603.16944](https://arxiv.org/abs/2603.16944)).
  - Multi-turn Consistent Image Editing ([arXiv 2505.04320](https://arxiv.org/abs/2505.04320)): drift appears by about the 4th turn.

  All of these work on **flat images**, not layer stacks.
- Game and story work:
  - Narrative-to-Scene Generation ([arXiv 2509.04481](https://arxiv.org/html/2509.04481v1)): tile-based, and it lists temporal consistency as future work.
  - Unbounded ([arXiv 2410.18975](https://arxiv.org/html/2410.18975v2)): flat images with regional IP-Adapter.
  - MagicScroll ([page](https://cislab.hkust-gz.edu.cn/publications/magicscroll-enhancing-immersive-storytelling-with-controllable-scroll-image-generation/)): multi-layer scroll images for VR storytelling [U, details not read].

  None of them uses RGBA layer stacks that persist across story events.

## 2. Real gaps

**G1. No method evaluates layered scenes over long edit sequences.**
- Evidence: every method in the matrix evaluates a single generation or decomposition.
- The QIL paper asserts that consistency is "fundamentally ensured" by the layered representation, but it has no metric for this, no limitations section and no repeated-edit experiment (read from the arXiv HTML).
- The multi-turn benchmarks (EdiVal, Omni IIE) show drift in flat editors but never test a layer stack.
- So the claim "layers prevent drift" is **untested**. This is the strongest gap and QIL does not close it.

**G2. Story-event edits (global state changes) on a layer stack are unaddressed.**
- An event like "the forest is on fire" must change several layers coherently: background, tree layers, lighting and smoke. Other layers must stay fixed, such as characters whose identity must persist.
- No method does selective multi-layer re-generation driven by a semantic event.
- LASAGNA does one conditional layer, LayerCraft does object insertion, and QIL edits a layer in isolation.

**G3. Re-harmonisation after move/resize is not available in any released model.**
- DreamLayer's IRH is the only trained mechanism, and it has no code.
- QIL treats moves as pixel operations, so shadows and lighting stay stale.

**G4. No HCI evaluation of layer-based generative scene editing for players or novices.**
- No user study found in QIL, Stable-Layers, RevealLayer or LayerCraft (pages read).
- ART+ only has preference votes.
- The only study with real users is Layered Diffusion Brushes, which uses a different notion of "layer" and has an artist audience.
- A search for CHI/UIST layered generative editing papers found none [the search was not exhaustive; check the ACM DL].

**G5. Interactive latency on free hardware is unstudied.**
- Strong decomposers need 40–80 GB (Stable-Layers, FLUX-based CLD, RevealLayer, ART+).
- QIL quality at low steps and 384 px, which is our setting, is unreported. The trade-off between quality and latency in a game loop is unknown.

## 3. Candidate research questions

### RQ1 (technical + eval): Do RGBA layer stacks reduce drift over long story-edit sequences compared with flat-image editing?
- **Gap:** G1. The QIL claim is unmeasured, and the drift benchmarks are flat-only.
- **How we answer it:** run the same scripted story of 10–20 events through three conditions:
  - (a) flat Qwen-Image-Edit on the composite
  - (b) QIL decompose once, then edit only the affected layers, then recomposite
  - (c) LayerDiffuse chain regeneration of the affected layers
- **Metrics:**
  - Unchanged-layer preservation: LPIPS/SSIM of untouched regions or layers against turn 0
  - Character identity: DINO/CLIP similarity of the character crop
  - Edit success: VLM judge, EdiVal-style
  - Global luminance drift
  - Alpha IoU stability
- **Feasibility:** high. All three conditions run on a T4. Use about 30 stories × 15 turns.
- **Risk already solved:** low to medium. Watch EdiVal follow-ups and the Stable-Layers paper (its full text was not read [U]).

### RQ2 (technical): How should a story event be routed to layers?
Sub-question: can an LLM planner choose which layers to regenerate, and with what conditioning, so that event fidelity is high and the rest of the scene stays stable?
- **Gap:** G2.
- **How we answer it:**
  1. Ollama/LLM maps the event to a per-layer action set (keep / re-prompt / regenerate conditioned on the composite / add a new layer such as smoke).
  2. LayerDiffuse bg→fg and fg→bg run the regeneration.
- **Comparison:** against "regenerate everything" and "edit the flat composite".
- **Metrics:** the RQ1 metrics plus planner accuracy against hand labels.
- **Feasibility:** high.
- **Risk:** LayerCraft is the nearest work: agentic and layered, with a narrative demo. But it has no code and no temporal evaluation. Cite it and differentiate.

### RQ3 (HCI): Do layer-level controls improve players' sense of agency, predictability and creative satisfaction compared with prompt-only editing?
- **Gap:** G4.
- **How we answer it:** a within-subjects study, n ≈ 16–24 LMU students, with two interfaces: (A) a layer panel with re-prompt, drag and resize per layer; (B) a chat-only prompt edit of the whole image. Each participant does the same story tasks in both, in counterbalanced order.
- **Measures:**
  - Task success and time
  - Number of undo/retry actions
  - Questionnaires: NASA-TLX, the Creativity Support Index (CSI), and an agency/predictability Likert scale (e.g. a sense-of-agency scale)
  - Semi-structured interview with thematic analysis
- **Feasibility:** high. Latency of 35–70 s per operation is a confound, so report it, or use a pre-generated cache or Wizard-of-Oz for part of the study.
- **Risk:** low. No CHI/UIST study was found [search not exhaustive].

### RQ4 (technical): Can training-free re-harmonisation (DreamLayer-IRH-style latent re-noise) make moved/resized layers physically plausible within an interactive budget?
- **Gap:** G3.
- **How we answer it:** the design in `spatial_edit_mechanics.md`. Sweep the strength s and compare against a naive paste.
- **Metrics:**
  - Human two-alternative forced choice (2AFC) on plausibility
  - Shadow and contact presence, judged by a VLM
  - Identity preservation of the moved layer
  - Latency, measured
- **Feasibility:** medium. The estimate is about 65–75 s on a T4, so it is at the 1-minute edge.
- **Risk:** medium. LASAGNA and DreamLayer may release code.

### RQ5 (systems/eval): What is the latency–quality frontier of open layered models on free GPUs, and where do players stop tolerating waits?
- **Gap:** G5.
- **How we answer it:** sweep QIL steps (4–50) and resolution (384–640), and the LayerDiffuse settings.
- **Metrics:** decomposition quality (Crello protocol from LayerD), plus a small perceptual study of acceptable wait time inside the game.
- **Feasibility:** high, but less novel. Use it as a supporting contribution to RQ1 or RQ3.

**Recommendation:** combine RQ1 + RQ2 as the technical core ("Do persistent layer stacks keep a story scene stable across many events?"), and use RQ3 as the user evaluation.

## 4. Honest note: does anything already do our whole loop?
No. As of 2026-11-09, no released method combines all of these:
- text → coherent layers
- in-context per-layer regeneration
- re-harmonised moves
- event-driven, multi-turn consistency
- free-GPU latency

The individual parts do exist:
- QIL (plus Qwen-Image-Edit) covers decompose → edit layer → recomposite, with open weights. With low settings it runs on a T4. So the *editing mechanics* alone are **not** a novel contribution.
- LayerCraft is conceptually closest to the whole loop (LLM agent, layers, narrative demo). Its code is unreleased and it has no consistency or user evaluation.

Our novelty must therefore lie in two things: the **evaluation over time / story events** and the **HCI study**, not in a new generator.

Unverified items:
- the full text of Stable-Layers (2605.30257) and RevealLayer papers
- user studies inside the PSDiffusion, DreamLayer and LayerDiffuse papers
- MagicScroll details
- an exhaustive ACM DL search for CHI/UIST work
