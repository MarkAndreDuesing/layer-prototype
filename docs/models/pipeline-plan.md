# Which models go where: pipeline plan after the first-look assessment

> **Role: proposal for the first prototype (2026-11-11, revised after the attribution audit [provenance.md](provenance.md)).** Built from the user's assessments in the factsheets ([README](README.md)) and the measurements in them. Everything under "Proposed verdict" is a proposal; the user decides. Numbers are measured on the fox & cottage scene unless stated; the combined run that tests these combinations is not done yet (see section 7).

## 1. What the evidence says, in one table

| Need in the pipeline | What worked on the fox & cottage scene | Cost (compute, warm model) | What did not |
|---|---|---|---|
| **Scene edit / story event** (night, rain, new element) | **Qwen-Image-2.1**: whole-scene night and rain edits keep the composition ([qwen-image-2.1](qwen-image-2.1.md)) | about 22 s per edit | — |
| **Split a scene into object layers** | **Stable-Layers at the native 640 px**: sky, ground, cottage, tree, fox as separate clean layers, on the scene and on the night and rain composites ([stable-layers](stable-layers.md)) | 102 s (4 Euler steps); 178 s with Heun gives the same layers | the same model at 384 / 512 px gives bands; the base weights, and Lightning, do not split this scene ([qwen-image-layered](qwen-image-layered.md), [qwen-lightning](qwen-lightning.md)) |
| **Split with control by boxes**, filled background | **RevealLayer** (clean removal, filled background, layers follow the boxes), **CLD** (decent layers, invented replacement background) | 85-358 s and 116-459 s | both slow and FLUX.1-dev non-commercial; CLD's background is a different scene |
| **Fast masks** | **Grounding DINO + SAM2**: object cut-outs by name | 3 s | blurry LaMa fill; hidden parts missing ([sam2-lama](sam2-lama.md), [grounding-dino](grounding-dino.md)) |
| **Background without objects / hidden parts** | **Qwen-2.1** "remove X and fill" (one call) and "extract only the cottage without X"; RevealLayer's filled background | about 24 s per call | SDXL-inpaint: good for the first three fills, cannot add elements convincingly ([amodal-fill](amodal-fill.md)) |
| **Add / remove / modify one layer** | **Qwen-2.1** RGBA edit with the alpha retry (10 of 10 edits accepted); text to transparent object | 20-26 s per edit | LayerDiffuse-Flux and ART+ objects do not match the scene's style ([layerdiffuse-flux](layerdiffuse-flux.md), [art-plus](art-plus.md)) |
| **Relight a layer** | **IC-Light** (with the chroma fix), **Qwen-2.1** relight prompts ([relight-ic-light](relight-ic-light.md)) | 44 s on the laptop CPU / seconds on a GPU [est]; 20-26 s | IC-Light shifts hue (partly fixable) |
| **Shadows** | procedural shadow layer ([shadow-layers](shadow-layers.md)) | 0.5 s for five layers, CPU | — |
| **Move / resize** | direct manipulation of the layer in the frontend (instant); `relocate()` to re-harmonise ([relocate-harmonise](relocate-harmonise.md)) | 0 s / 3.4 s | `relocate()` runs on LayerDiffuse SDXL (see 4) |
| **Image to image** | **Qwen-2.1** only, so far | about 22 s | not compared with other image-to-image models yet |
| **Effects (fog, glow, snow) as layers** | procedural overlays | CPU, instant | LayerDiffuse SDXL effect layers gave objects or the scene itself ([layerdiffuse-sdxl](layerdiffuse-sdxl.md)) |

## 1b. Core path after the user's update (2026-11-11)

**SDXL-Turbo makes the first picture (text to image, or image to image from a sketch or an edit prompt); Qwen-Image-2.1 splits it and modifies it.** This is what `results_qwen21_fox/sheet.jpg` already is (SDXL-Turbo scene + Qwen-2.1 operations; only the two text-to-image tiles are Qwen-2.1 alone, [provenance](provenance.md)). SDXL-Turbo is not a layer model and is not in the layer analysis, but it stays in the pipeline as the simple fast first step; measured: about 1.1-1.5 s per 512 px image on a T4 against 17-23 s for Qwen-2.1 text to image, but its image to image regenerates the scene (the fox became a horse), so story edits stay with Qwen-2.1 ([sdxl-turbo.md](sdxl-turbo.md)). Stable-Layers and CLD remain **options for splitting** when they have an advantage over Qwen-2.1. Measured so far: on the daylight scene Qwen-2.1 extraction gives a clean background and whole objects (118 s for four objects, arch mis-scaled); on the night and rain edits Qwen-2.1 extraction left the cottage and tree in the background, cut a block for the cottage and invented a misplaced arch (139-184 s) while Stable-Layers split both cleanly (178 s) ([qwen-image-2.1.md](qwen-image-2.1.md), [stable-layers.md](stable-layers.md)). Runtime is of the same order. RevealLayer is kept for **inpainting** (filled backgrounds, hidden parts), not for splitting. SDXL-inpaint may be usable in some capacity; LayerDiffuse effect layers are dropped; LayerDiffuse SDXL stays only as the engine of `relocate()`.

## 2. Proposed pipeline, as roles (one model per role)

```mermaid
flowchart LR
    T[Text or uploaded image] --> G[Scene image<br/>SDXL-Turbo text to image / image to image]
    G --> S[Split once, offline<br/>Stable-Layers 640 px]
    S --> L[(Object layers + background<br/>edited by the user directly)]
    L --> E[Story event or edit]
    E --> E1[Qwen-2.1 whole-scene edit<br/>then re-split with Stable-Layers]
    E --> E2[Qwen-2.1 per-layer edit<br/>with the alpha retry]
    E --> E3[IC-Light relight,<br/>CPU grade as instant preview]
    L --> SH[Shadow layers, CPU]
    L --> R[Recomposite]
    E1 --> R
    E2 --> R
    E3 --> R
    SH --> R
```

The key timing consequence: **splitting a scene takes about 100 s, so it is done once per scene** (when the scene is generated or uploaded), not per request. The live loop works on the existing layers: move and resize are instant, a relight preview is a CPU grade, and the slow steps are one Qwen-2.1 call (about 24 s) for one layer or one background. A story event that changes the whole scene costs one edit (22 s) plus a re-split (about 100 s), so it runs in the background while the CPU preview is shown ([concepts/editing.md §3](../concepts/editing.md)).

## 3. Proposed combinations

| Combination | Models and their roles | Why | Open point |
|---|---|---|---|
| **A. Story editor (core)** | Qwen-2.1 edits the scene → Stable-Layers (640 px) splits the edit → Qwen-2.1 per-layer edit / IC-Light relight for single layers → shadow layers → recomposite | each model does what the first look found it best at; every step has a measured result | the whole chain has not been run end to end; the split step is about 100 s |
| **B. Fast object extraction** | Grounding DINO + SAM2 for the masks (3 s) → Qwen-2.1 for the background without the object and for hidden parts (about 24 s per call) instead of LaMa | keeps SAM2's speed and fixes its weak points (blurry fill, missing hidden parts) with the model that did those well | not tested as a combination; the pieces were measured separately |
| **C. Box-controlled / quality path** | RevealLayer (boxes, filled background) or CLD (boxes, decent layers) | the only methods where the user places the groups by boxes; RevealLayer's background removal is the strongest | slow (85-460 s) and FLUX.1-dev non-commercial; worth keeping only if the box control is wanted in the frontend |
| **D. Placement and harmonising** | user moves the layer directly; `relocate()` re-harmonises on request; shadow layer follows | movement is best done by the user; harmonising is the optional model step | `relocate()` needs LayerDiffuse SDXL (see 4) |

## 4. Interactions that constrain the verdicts

1. **Dropping LayerDiffuse SDXL has two consequences.** (a) `relocate()` (harmonise after a move) runs on it; without it the "re-harmonise" button has no engine until a replacement exists. (b) The current scene generator (the layer-by-layer chain, `Engine.generate_all`) is built on it; the replacement is "text → scene with Qwen-2.1, then split once" (the generate-then-decompose route in [concepts/pipeline.md](../concepts/pipeline.md)). Proposal: stop developing it as a generator, **keep it only as the engine of `relocate()` until a Qwen-2.1 based harmonise has been tested**, then decide.
2. **Who gets credit for what:** the story edit (night, rain, a new element) is Qwen-2.1's; Stable-Layers (or another splitter) only splits the result; IC-Light, shadows and `relocate()` act on layers made by the splitter. Attribution per image: [provenance.md](provenance.md).
3. **Stable-Layers needs the Qwen-Image-Layered weights** (it is a LoRA on them), so those stay on the server even though the base model alone splits this scene poorly. The base model is the comparison for Stable-Layers, not a separate pipeline step.
4. **Qwen-2.1 is the hub:** edit, image to image, add / remove / modify, background removal, amodal, relight and its output is what Stable-Layers splits. Its weaknesses (about 24 s per call, non-commercial licence, object position and scale not controlled in extraction) therefore matter most. A second model for the same roles (an alternative editor) is the main redundancy to consider.
5. **IC-Light and the shadow layer are independent** of the other models (SD1.5, and CPU), which makes them cheap to keep.
6. **Servers:** one profile per session for now (`layerproto/serve/profiles.py`). Combination A needs two profiles (`qwen21` and `qwen-layered`); the Qwen-2.1 profile alone loads about 21 GiB across the two T4s (8.0 + 12.7 GiB) and the Qwen-Image-Layered transformer is another 12 GB or more, so keeping both loaded probably does not fit (not measured together); a live demo would swap profiles or use a larger card. That is the "rethink" the user expects for later.

## 5. Proposed verdicts

| Model / approach | Proposed verdict | Reason (from the factsheets) |
|---|---|---|
| Qwen-Image-2.1 | **Core**; keep developing | scene edit, story composites, image to image, add / remove / modify, background inpainting, amodal, relight; slow per call |
| Stable-Layers (on Qwen-Image-Layered weights) | **Optional splitter next to Qwen-2.1 extraction** (the user), the one that gave clean layers in one pass (not a story-composite model: the night and rain edits are Qwen-2.1's) | the only splitter that gave clean object layers at 640 px on the scene and on Qwen-2.1's edits of it (base and Lightning did not; RevealLayer was not run on the edits); colour shift in the recomposite; about 100 s |
| Shadow layers | **Core** (CPU helper) | instant, any layer |
| IC-Light + chroma fix | **Keep, tune further** | relights a daylight fox plausibly (darker, redder); on the objects of a night or rain scene it left them pale (the good night look in the `bg_relight` strips is Qwen-2.1's edit of the background, [provenance](provenance.md)); the Qwen-2.1 per-layer relight is the working one there. Tuning (2026-11-11): a night pre-grade fixes the not-dark failure, but on night and rain objects Qwen-2.1's per-layer relight is the better result and IC-Light's rain objects stay pale; IC-Light's niche (matching light inside one scene) is not yet tested ([relight-ic-light.md](relight-ic-light.md)) |
| Grounding DINO + SAM2 | **Keep** (fast masks); combine with Qwen-2.1 fill (combination B) | 3 s; the fill and hidden parts are the weak part, not the masks |
| `relocate()` / harmonise | **Keep as an optional step**; move/resize itself is direct user interaction | 3.4 s; depends on LayerDiffuse SDXL today |
| RevealLayer | **Keep for inpainting (filled backgrounds, hidden parts), not for splitting** (the user, 2026-11-11); box control is a side benefit | its own results: clean filled backgrounds (the strongest background removal seen) and layers that follow boxes; it also supplied the input layers of the move and per-layer relight strips, whose other effects are other models' work; slow, FLUX non-commercial |
| CLD | **Keep as an optional splitter** (the user: unless Qwen-2.1 has an absolute advantage; runtime matters) | its own results: decent layers at 4-16 steps, an invented replacement background (a different scene), slow. The style variation on the night and rain strip is the Qwen-2.1 edit's, not CLD's |
| Qwen-Image-Layered (base) | **Keep in contention** (the user, 2026-11-11): it is the base of Stable-Layers and may still have advantages over Qwen-2.1 | inconsistent: good on the forest, no object layers on the fox scene at any setting tried; to test: more scenes, whether it keeps what Qwen-2.1 extraction loses (all objects in one pass, no per-object call), the full-precision question |
| Qwen Lightning | **Candidate to drop** (the user decides) | no object layers on the fox scene at 384, 512 or 640 px; story composites not improved at 640 px |
| LayerD | **Park, look at it a little more** (the user's note) | limited on painted scenes; CPU only |
| SDXL-inpaint amodal fill | **Park**; Qwen-2.1 replaces it for amodal | its own result is the two cottage fills (the rest of those panels is SAM2 / LaMa / the input); the other panels of the strip the first look judged were LayerDiffuse effect layers, a different model; redundant with Qwen-2.1 extraction |
| ART+ | **Drop** | all bad (the user); generates its own scene, does not split ours |
| LayerDiffuse SDXL | **Keep exclusively as the engine of `relocate()`** (the user) while a replacement is explored, Qwen-2.1 first; no further development as a generator | all bad as a generator (the user); see 4.1 and the harmonise test in section 7 |
| LayerDiffuse-Flux | **Drop** | all bad (the user); objects do not match the scene's style |

## 6. Feature coverage if these verdicts hold

| Feature | Provided by | Gap |
|---|---|---|
| Split | Stable-Layers (640 px); SAM2 + Qwen-2.1 (fast, one object per call); RevealLayer / CLD (boxes) | 30 s per split is not reached by any model that gives object layers (SAM2 is 3 s but needs Qwen-2.1 for a good fill) |
| Harmonise | `relocate()` (LayerDiffuse SDXL) | a Qwen-2.1 based harmonise is untested |
| Story composite | Qwen-2.1 edit + Stable-Layers split; per-layer relight | the end-to-end run |
| Spatial edit | Qwen-2.1 text edits on a layer; boxes (RevealLayer, CLD); masks (SAM2) | an edit restricted to a drawn region is untested |
| Image to image | Qwen-2.1 | no other image-to-image model compared yet |
| Add / remove / modify | Qwen-2.1 | none measured against another model |
| Relight | IC-Light, Qwen-2.1 | further tuning |
| Effects | procedural overlays | a Qwen-2.1 transparent effect layer (fog, glow) is untried |
| Amodal | Qwen-2.1, RevealLayer | none |

## 7. Next steps toward the combined run

1. **The combined run (the user's plan):** combination A end to end on the fox & cottage scene and on two more scenes, with the model and role named for every output.
2. **Check that Stable-Layers' split holds on other scenes** (the forest, the pixel-art and the sketch scenes; Qwen-Image-Layered was inconsistent), and measure 640 px with 2-3 steps for a shorter split.
3. **Combination B:** SAM2 masks plus Qwen-2.1 fill, against Qwen-2.1 extraction alone.
4. **A Qwen-2.1 harmonise** (first test done 2026-11-11, [relocate-harmonise.md](relocate-harmonise.md): a convincing shadow but whole-picture drift, one unstable seed and a re-posed fox with references, 22-29 s; a lighting-mismatch case is still to test) ("re-light and blend this moved object into the scene") against `relocate()`, so LayerDiffuse SDXL can be dropped fully.
5. **Image to image:** compare Qwen-2.1 with other image-to-image models.
6. **Effects:** a Qwen-2.1 transparent fog / glow / snow layer.
7. **Decide Lightning** after the 640 px result (listed as a candidate to drop above).
8. **Server design for live use:** both profiles resident, weights pre-downloaded (the Qwen-Image-Layered weights already are, as Kaggle notebook outputs).

## 8. Not yet looked at (for after the first prototype)

From [../research/papers-survey.md](../research/papers-survey.md): **LayerFusion** (training-free attention blending, re-implementable, a route to harmonised joint generation), the **DreamLayer** dataset (a possible fine-tune for move / resize), **OmniPSD** (no weights), **LASAGNA** (shadows and reflections as layers, not released). Directions with no entry there yet and not verified [U]: further instruction-based image-editing models as alternatives to Qwen-2.1 (for story edits and image to image), newer segmentation models than SAM2, other relighting models than IC-Light v1, and layered models released after 2026-11-09. A short literature check on these is the next research task once the prototype loop is stable.

Licences: FLUX.1-dev (RevealLayer, CLD) and Qwen Research (Qwen-2.1) are non-commercial, Stable-Layers is under the Stability community licence; this is a personal non-commercial project, so all remain usable now ([TEMPLATE.md](TEMPLATE.md) lists the rest).
