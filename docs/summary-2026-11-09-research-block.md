# Research block summary (2026-11-09)

This page summarises a few hours of unsupervised research and Kaggle runs on three tracks:
- **A.** Layered-generation papers and the research gap
- **B.** LMU / LRZ GPUs versus paid APIs
- **C.** Making Qwen-Image-Layered fast enough

Details are in the linked reports and in [findings-log.md](findings-log.md), Phases 4 and 5.

## Headline results

1. **Qwen-Image-Layered now splits a scene in 27–32 s on Kaggle's free 2× T4.** Settings: 384 px, 3–4 layers, 4 steps, no CFG, transformer kept on the GPUs, with our `fp16_mixed` patch. The defaults (640 px, 50 steps) took 35 min. It was tested on 3 art styles (impressionist, pixel art, pale sketch).
2. **The full loop works on free hardware:**
   1. Qwen split, about 32 s.
   2. Move or resize a layer, then re-harmonise it into the scene: about 3.5 s per edit with our training-free `relocate()` on SDXL + LayerDiffuse.
   3. Recomposite.

   Weak spot: the moved object gets partly redrawn (identity drift).
3. **No released method covers our loop, and the editing mechanics alone are not new.** The defensible research contribution is measuring **consistency across many story edits** (layered vs flat), choosing **which layers a story event should change**, and a **user study** of layer-based editing. See [Research gap and questions.md](../reports/Research%20gap%20and%20questions.md).
4. **Tested and rejected:**
   - LayerD returns 1 layer on our scenes.
   - The Lightning LoRA hurts fidelity and doesn't speed anything up.
   - SAM2 + LaMa is fast but gives meaningless groupings.
   - Stable-Layers needs 50 Heun steps by design.
5. **University GPUs:** students can use the institute's Slurm cluster without registering. The stronger "abakus" GPU servers need an emailed request to the RBG (Rechnerbetriebsgruppe, the institute's IT group) with a good reason. LRZ AI Systems (A100 / H100) only through a supervisor's AI project. For about 2,000 runs, paid APIs cost about $100–200. See [University GPU access LMU.md](../reports/University%20GPU%20access%20LMU.md).

## Track C: Qwen-Image-Layered speed (Kaggle 2× T4, GGUF Q4_K_S)

| Run | What changed | Best result |
|---|---|---|
| v2 | `fp16_mixed` fixes the NaNs | 640 px, 50 steps: 35 min |
| v3 | 512 / 384 px, fewer steps, CFG 1 | CFG 1 halves the time without visible loss; 384 px, 8 steps: 63 s |
| v4 | 4–5 steps | 384 px, 4 steps: 33 s |
| v5 | transformer stays on the GPUs; 3 images | **384 px, 3 layers, 4 steps: 27 s**; 3 steps: 20 s |
| v6 | Lightning 4-step LoRA (untested combination) | Separates thin subjects better, but colours drift, fidelity halves and it adds 5 s. **Not adopted.** |
| v7 | recommended settings; RGBA layers saved | 384 px, 4 layers, 4 steps: 32–35 s. All three scenes split usefully, including the pale skeleton. |

**Recommended default: 384 px, 4 layers, 4 steps, CFG 1, `offload=False`.** Use 512 px with 3 steps (about 31–42 s) when sharpness matters more than speed.

GPU quota: the account shows 6.2 h used this week out of 30 h, including the earlier editor sessions. This block used about 3 h of it.

## Track A: papers (see [Layered methods - how to get them working.md](../reports/Layered%20methods%20-%20how%20to%20get%20them%20working.md))

- **Usable today on free GPUs:** Qwen-Image-Layered (our main route) and LayerDiffuse.
- **Released but too heavy for a T4:** ART+ / PrismLayers, RevealLayer, CLD and OmniPSD (code only) are Flux-based, so LRZ only. Stable-Layers' design rules out short runs.
- **Unreleased but borrowable without training:** LayerFusion's attention-mask blending, DreamLayer's harmonisation step and PSDiffusion's partial joint attention.
- **LayerFusion has no spatial-edit phase.** It does harmonised joint generation. The move/resize idea comes from DreamLayer. We implemented a training-free version (`relocate()`, below).
- **New finds:** LayerD, RevealLayer, Stable-Layers, See-through and LayerCraft. Also the DreamLayer dataset on HF, usable for a small fine-tune later.

## Track A, part 2: our own move/resize re-harmonisation

[`LayerDiffuseChain.relocate()`](../prototype/layerproto/backends/layerdiffuse_chain.py) works in four steps:
1. Warp the layer to its new box.
2. SDXL img2img conditioned on the rest of the scene (LayerDiffuse `bg2ble`) at strength 0.3, with the object's interior pinned early on.
3. Take the moved object's alpha, plus wherever the image changed around it (light, glow, shadow).
4. Output a new RGBA layer.

| Version | Result |
|---|---|
| v1 | SDE scheduler plus model extraction: artifacts and a translucent layer |
| **v2** | **Euler scheduler plus mask extraction: clean, about 4 s per edit** |
| On Qwen layers | Works on all 3 scenes. The object takes on the scene's style and light, but is partly redrawn. A longer pin doesn't help. |

**Next step:** identity conditioning, e.g. IP-Adapter on the original object.

## Track B: GPUs

| Option | Access | GPUs | For us |
|---|---|---|---|
| Institute Slurm cluster (RBG) | all students, no registration | not published | try now; check the GPU types and bf16 support |
| "abakus" GPU servers | email rbg@ifi.lmu.de, with a reason confirmed by a supervisor | "more powerful" (not specified) | ask |
| LRZ AI Systems | through a supervisor's AI-enabled LRZ project | V100, A100, H100 | best hardware; slowest to get |
| fal.ai / WaveSpeed | card, minutes to set up | — | about $100–200 per 2,000 runs |
| Kaggle 2× T4 | free, 30 h/week | T4 | **already works at about 30 s per split** |

## Proposed next steps (need the team's decision)

1. Pick the research question. Suggested: RQ1, layered vs flat drift over 10–20 story events, plus RQ3, a user study. Both are feasible on the T4 setup.
2. Wire the fast Qwen split and `relocate()` into the prototype app as a Kaggle-hosted backend.
3. Fix identity drift in `relocate()` (IP-Adapter / reference attention), then evaluate it properly (RQ4).
4. Email the RBG for abakus access, and ask a supervisor about an LRZ AI project if larger models (ART+, RevealLayer) become interesting.
