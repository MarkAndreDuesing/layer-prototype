# Papers and methods survey

> **Role: research synthesis (current).** Summarises the sources; details and citations stay in the linked snapshots and research notes.

One table of every layered-generation / decomposition method we found, with release status and our use. Per-model run details are in the factsheets ([../models/](../models/README.md)). Sources: [../../research_notes/Layered methods deep dive/joint_text_to_layers.md](../../research_notes/Layered%20methods%20deep%20dive/joint_text_to_layers.md), [other_psd_methods.md](../../research_notes/Layered%20methods%20deep%20dive/other_psd_methods.md), [../../research_notes/Coherent layered generation pipeline/other_layered_methods.md](../../research_notes/Coherent%20layered%20generation%20pipeline/other_layered_methods.md), [release_status_recheck.md](../../research_notes/Layered%20methods%20deep%20dive/release_status_recheck.md), [../../research_notes/Layered image generation models/](../../research_notes/Layered%20image%20generation%20models/), and the snapshot [../archive/2026-11-09 Layered methods - how to get them working.md](../archive/2026-11-09%20Layered%20methods%20-%20how%20to%20get%20them%20working.md).

## 1. All methods

| Method | Type | Code / weights | Our status |
|---|---|---|---|
| Qwen-Image-Layered (arXiv 2512.15603) | image → N layers | yes, Apache-2.0 | runs, main ([factsheet](../models/qwen-image-layered.md)) |
| Stable-Layers (Stability AI) | LoRA on Qwen-Image-Layered | yes, community licence | runs at reduced settings ([factsheet](../models/stable-layers.md)) |
| RevealLayer (ICML 2026) | image + boxes → layers incl. hidden parts | yes (FLUX.1-dev) | runs ([factsheet](../models/revealayer.md)) |
| ART / ART+ (PrismLayers, arXiv 2505.22523) | text + boxes → layers (joint) | yes (Drive; base also in CLD) | in testing ([factsheet](../models/art-plus.md)) |
| CLD (arXiv 2511.16249) | image + boxes → layers | yes (HF) | next ([factsheet](../models/cld.md)) |
| LayerDiffuse (arXiv 2402.17113) | transparent objects; conditional fg/bg | yes | base of the prototype ([factsheet](../models/layerdiffuse-sdxl.md)) |
| LayerDiffuse-Flux | transparent objects on FLUX | yes | in testing ([factsheet](../models/layerdiffuse-flux.md)) |
| Qwen-Image-2.1 | RGBA generation, layer edit, extraction | yes, research licence | runs ([factsheet](../models/qwen-image-2.1.md)) |
| LayerD (ICCV 2025) | image → layers by peeling | yes | tried once ([factsheet](../models/layerd.md)) |
| SAM2 (+ LaMa, pix2gestalt) | segment + inpaint (+ amodal) | yes | tried once ([factsheet](../models/sam2-lama.md)) |
| LayerFusion | joint fg/bg/blend, training-free | no code | idea ([factsheet](../models/layerfusion.md)) |
| PSDiffusion, DreamLayer, OmniPSD (weights), LASAGNA, LayeringDiff, LayerDecomp, DiffDecompose, Text2Layer, RefLayer, LiWi, PixelART | various | not released | [unreleased.md](../models/unreleased.md) |
| LayerCraft | LLM agents for layered editing | code + LoRA | design reference |
| Alfie, See-through, Generative Omnimatte, TransPixar | sprites / anime PSD / video layers / RGBA video | yes | not tried |

## 2. Spatial editing (move / resize) in the papers
- **Only DreamLayer** has a move/resize mechanism: denoise each layer separately to about t = 600/1000, apply the edit to the foreground latents and alpha, then composite the latents and denoise together, so shadows follow the object. Training-free in itself; our `relocate()` is built on this idea ([../concepts/relocate.md](../concepts/relocate.md)).
- **LayerFusion** describes no spatial editing (layers make editing "straightforward"); it contributes attention blending for joint generation.
- **Decomposition papers** (Qwen-Image-Layered, Stable-Layers, CLD, RevealLayer) only re-composite after a move: lighting and shadows stay stale.
- **Object-move papers** (DragonDiffusion, DiffEditor, RegionDrag, InstantDrag, DesignEdit, GeoDiffuser, Magic Fixup, DiffUHaul) keep identity with latent pinning and self-attention K/V sharing.
- Survey with sources: [../../research_notes/Layered methods deep dive/spatial_edit_survey_verified_part1.md](../../research_notes/Layered%20methods%20deep%20dive/spatial_edit_survey_verified_part1.md), [part 2](../../research_notes/Layered%20methods%20deep%20dive/spatial_edit_survey_verified_part2.md), [part 3](../../research_notes/Layered%20methods%20deep%20dive/spatial_edit_survey_verified_part3.md), [spatial_edit_mechanics.md](../../research_notes/Layered%20methods%20deep%20dive/spatial_edit_mechanics.md).

## 3. Related topics
- Harmonisation and relighting: [../concepts/harmonisation.md](../concepts/harmonisation.md).
- Scene-wide story events: [../concepts/story-events.md](../concepts/story-events.md).
- Drift over long edit sequences: [drift-evaluation.md](drift-evaluation.md).
- HCI literature (context only, no study planned): [../../research_notes/Layered loop open paths/hci_layered_generative_editing.md](../../research_notes/Layered%20loop%20open%20paths/hci_layered_generative_editing.md).
