# Qwen-Image-Layered
> **Role: current factsheet.** Template: What it does · Licence & weights · How we run it (happy path) · Benchmarks · Best features · Weak spots · Status · Next attempts · Sources. Verdicts are the user's.

> **User assessment (final intermediate, 2026-11-11):** Surprisingly bad layer extraction in some cases, in others pretty decent: quite inconsistent. It showed promise before (`results_qwen_v7_layers/layers_forest_384px_4L_4st`, `results_qwen_sweep_v5/v5_forest_384px_4L_4st.png`). Find out what is going wrong; remove from consideration if unfixable.

## What it does

Splits one finished image into N full-canvas RGBA layers (N chosen per run, e.g. 3–8) that composite back to the input; hidden areas behind objects are filled. 20B MMDiT, image → layers (not text → layers).

## Licence & weights

Apache-2.0, [Qwen/Qwen-Image-Layered](https://huggingface.co/Qwen/Qwen-Image-Layered). We use a GGUF Q4_K_S transformer (12.4 GB) plus the original text encoder and VAE, cached as public Kaggle notebook outputs (`markandreduesing/qwen-layered-weights-transformer`, `-base`).

## How we run it (happy path)

1. Kaggle 2× T4, transformer split over both GPUs (blocks 0–29 / 30–59), `fp16_mixed` applied ([../concepts/fp16-on-t4.md](../concepts/fp16-on-t4.md)).
2. **384 px, 4 layers, 4 steps, `true_cfg=1.0`, `offload=False`** → 27–35 s per split.
3. Code: [`../../prototype/layerproto/qwen_layered.py`](../../prototype/layerproto/qwen_layered.py) (`decompose`, `recompose`, `cached_encode`, `HeunVelocity`); notebook generator [`../../prototype/tools/make_qwen_layered_batch.py`](../../prototype/tools/make_qwen_layered_batch.py); step-by-step guide [../guides/kaggle.md](../guides/kaggle.md).

## Benchmarks

| Setting | Time per split | Recomposite diff | Notes |
|---|---|---|---|
| 640 px, 4 layers, 50 steps, CFG 4 (defaults) | 35 min (42 s/step) | 4.1 | reference quality |
| 512 px, 3 layers, 12 steps, CFG 1 | 139 s | — | same split as CFG 4 |
| 384 px, 3 layers, 3 steps, CFG 1 | 20 s | — | softer edges |
| **384 px, 4 layers, 4 steps, CFG 1** | **27–35 s** | about 7 | recommended |
| 512 px, 4 layers, 4 steps, CFG 1 | 48 s | — | sharper |
| 384 px, 4 layers, 4 **Heun** steps | 58 s | 7.0 | 2 calls per step |
| 384 px, **2 layers**, 4 steps | 25.7 s | 5.9–6.5 | background + one foreground layer |

Peak memory 8.0 / 7.6 GB (fast settings). Measured on the forest, pixel-art and sketch scenes (sweeps v3–v7, [`../../prototype/kaggle/results_qwen_sweep_v5/`](../../prototype/kaggle/results_qwen_sweep_v5/), [`results_qwen_v7_layers/`](../../prototype/kaggle/results_qwen_v7_layers/)). Fox & cottage (shared test scene): 4 layers in 33 s → background / house roof / cottage + tree / fox (diff 9.4); 6 layers 41 s; 5 layers at 512 px 62 s (tree and fox separate). Results: [`../../prototype/kaggle/results_qwen_fox_v11/`](../../prototype/kaggle/results_qwen_fox_v11/).

## Fox & cottage benchmark

Input: the shared scene ([TEMPLATE.md](TEMPLATE.md)); the model takes the image and a layer count (no boxes). Strips: input | recomposite | layers (green = transparent).

| | Settings | Compute (2× T4) | Recomposite diff | What the layers show |
|---|---|---|---|---|
| **A. Best case measured on this scene** | 512 px, 5 layers, 4 steps, CFG 1 | 62 s | 13.4 | background, tree and fox separate, cottage faint |
| **B. Live-demo budget (about 30 s)** | 384 px, 4 layers, 4 steps, CFG 1 | 33 s | 9.4 | background / roof / cottage + tree / fox |

A, 512 px, 5 layers:

![Qwen-Image-Layered 512 px 5 layers](../img/fox_qwenlayered_512_5L.png)

B, 384 px, 4 layers:

![Qwen-Image-Layered 384 px 4 layers](../img/fox_qwenlayered_384_4L.png)

A has not yet been run with more steps (12–16, still under 8 minutes); the 640 px, 50-step reference setting takes 35 min and was measured on the forest scene only. Results: `prototype/kaggle/results_qwen_fox_v11/`.

### Profile benchmark (warm model in one Kaggle session, 2026-11-11)

Run through the new `qwen-layered` server profile ([`layerproto/serve/qwen_layered.py`](../../prototype/layerproto/serve/qwen_layered.py)); the model was loaded once (354 s), each case was run twice and the second run is reported. Compute is the decompose call only. Results: `prototype/kaggle/results_profile_qwen-layered/`.

| | Settings | Compute | Layers | Recomposite diff |
|---|---|---|---|---|
| **A. Best case (up to 8 min)** | 512 px, 5 layers, 16 Euler steps, CFG 1 | 254.8 s | 5 | 8.54 |
| **B. Live-demo budget (about 30 s)** | 384 px, 4 layers, 4 Euler steps, CFG 1 | 34.5 s | 4 | 8.68 |

A (512 px, 16 steps), measured: the background layer is 67.5 % opaque (the top 114 px are transparent) and has the objects partly removed (colour difference 26 / 41 / 66 for cottage / tree / fox); layer 1 is a top band (24.5 %, sky and tree parts), layer 2 covers half the canvas and mixes pieces of the cottage, tree and fox (47-74 % of their boxes), layers 3 and 4 are fox fragments (21 % and 14 % of its box). The fox is cut across several layers; the objects are mixed, not separated:

![A, 512 px 16 steps](../img/bench_qwenlayered_A.jpg)

B (384 px, 4 layers, 4 steps) is the same case as row 2 above: nothing is taken out of the background, the extra layers are a ghost, a top slab and the fox:

![B, 384 px 4 steps](../img/bench_qwenlayered_B.jpg)

16 steps cost 7× the 4-step run and improved the recomposite difference only from 8.68 to 8.54, but the recomposite difference does not measure separation (see the diagnosis above): the 16-step split is mixed, the 4-step split extracts almost nothing.

### Diagnosis: what the model returns on the fox & cottage scene (2026-11-11, replaces an earlier wrong reading)

Question from the first look: the forest runs of the past were good, the fox & cottage splits are not. Code regression, or the scene? One session (`qwen-layered-diag` profile, current code and weights) ran the forest as a control and the fox scene at other settings. Rows top to bottom: forest 384 px (control), fox 384 px, fox 384 px with an empty caption, fox 640 px 4 steps, 640 px 20 steps, 640 px 12 steps CFG 4. Strip: input | recomposite | layer 0 (background) | layers 1-3. Magenta = transparent. Numbers from [`layer_report.py`](../../prototype/tools/layer_report.py); results in `prototype/kaggle/results_profile_qwen-layered-diag/`.

![diagnosis](../img/qwenlayered_diagnosis.jpg)

"Objects left in the background" below is the colour difference between layer 0 and the input inside the SAM2 masks of the cottage, the tree and the fox: about 10 means the object is still in the background layer (nothing was taken out), large means it was removed and the hole filled. The recomposite difference cannot show this, because duplicated content recomposites perfectly.

| Row | Compute | Cottage / tree / fox still in the background (colour diff) | What the layers actually are |
|---|---|---|---|
| 1 forest, 384 px, 4 steps (control) | 35.1 s | n/a | **a real split**: background without objects, the two bare trees, the pines, the campfire |
| 2 fox scene, 384 px, 4 steps | 32.6 s | 8.8 / 10.5 / 11.5: **all three still in the background** | layer 1: a 0.7 % ghost of the roof; layer 2: a rectangular slab (top 70 % of the canvas, full width) holding sky, tree and cottage, a duplicate of what the background already shows; layer 3: the fox (32 % of its box = the whole fox) with a few path pixels. Only the fox is extracted, and it also stays in the background |
| 3 fox scene, 384 px, **empty caption** | 33.3 s | 9.5 / 8.9 / 9.7: all three still in the background | **nothing extracted**: layers 1-3 hold 2.7 %, 0.6 % and 0.5 % of the canvas (ghost fragments); the background is the whole picture. **Not acceptable.** |
| 4 fox scene, **640 px**, 4 steps | 88.2 s | 70.8 / 111.8 / **9.8** | the background is a blank cream wash **with the fox still in it**; layer 1: a ground slab (bottom 36 %) with a fox-shaped hole; layer 2: one slab (top 64 %, full width) holding sky, tree, cottage and hedge together; layer 3: a 0.6 % door fragment |
| 5 640 px, 20 steps | 431.9 s | 32.7 / 78.4 / **5.9** | green wash background with the fox in it; **layer 1 empty (0.0 %)**; layer 2 the top slab; layer 3 two door fragments |
| 6 640 px, 12 steps, CFG 4 | 516.9 s | 117 / 156 / **25.1** | white background with the fox in it; layer 1 the ground slab with a fox-shaped hole; layer 2 the top slab; layer 3 empty |

**What this shows**
- It is **not a code regression**: the forest control reproduces the good old result with the current code and weights.
- On the fox & cottage scene the base model does **not return object layers** at any setting we tried (Stable-Layers = the same weights plus a LoRA does at 640 px: [stable-layers.md](stable-layers.md)). At 384 px nothing is taken out of the background and the extra layers are duplicates or ghosts (row 3: nothing at all). At 640 px it cuts the picture into two horizontal blocks, a top block with sky + tree + cottage and a ground slab, and the fox stays on the background layer. The cottage and the tree are never separated from each other or from the sky.
- More steps (20) and CFG 4 do not change that; the empty caption makes 384 px worse; the caption itself is not the cause.
- Why it works on the forest and not here is **not established**. Candidates we have not separated: the scene (flat cartoon colours, dense overlapping objects, no depth cues, unlike the forest painting), our quantised Q4_K_S GGUF with fp16_mixed numerics on a T4 (the forest works under the same pipeline, so it is not simply broken), or the model's own limit. A run of the same image through a full-precision host (the hosted API costs about $0.05 per run, see [../research/hosting.md](../research/hosting.md)) would separate "our numerics" from "the model"; not run yet, the user decides.

### Native 640 px session, 5 layers (base model, no LoRA; 2026-11-11)

Same session as in [stable-layers.md](stable-layers.md#native-640-px-session-warm-model-2026-11-11-the-split-that-works), strips `docs/img/qwenlayered_640_fox.jpg` (row 1) and `docs/img/qwenlayered_640_story.jpg` (rows 1 and 4).

| Case | Compute | Layers, checked |
|---|---|---|
| fox & cottage, 4 Euler steps | 97.0 s | background: a blank cream wash with the fox still in it (colour difference 77 / 117 / 10 for cottage / tree / fox); layer 1 the ground slab with a fox-shaped hole; layer 2 a top slab (64 %) with sky and hedge; layer 3 cottage and tree together (73 % / 80 % of their boxes) with holes; layer 4 a 0.7 % fragment |
| night composite, 4 Euler steps | 98.8 s | background: a blank white image; layer 1 a dark smear; layer 2 a ground smear; **layer 3 holds the complete night scene** (100 % opaque: cottage, tree and fox together); layer 4 empty |
| rain composite, 4 Euler steps | 98.4 s | background white; layer 1 the scene with holes (94 %); layer 2 a ground smear; layer 3 the scene again with pink holes (71 %); layer 4 the fox |

The base model does not give object layers on these scenes at 640 px either; the LoRA of [Stable-Layers](stable-layers.md) does on the same weights and settings. **Wording fix for the diagnosis above:** "does not return object layers at any setting we tried" holds for the base model only.

## Special features

| Feature | Status | Evidence |
|---|---|---|
| Split an image into layers | **supported** | strips above |
| Move / resize a layer | **partial**: layers are full-canvas RGBA, so a canvas move works; no harmonising | [relocate-harmonise.md](relocate-harmonise.md) (the move loop used another model's split) |
| Harmonise after a move | not tried on these layers | |
| Story composite | the edit is Qwen-2.1's; this model only **splits** a night / rain edit (poor at 384 and 640 px) | below, [provenance](provenance.md) |
| Spatial edit | n/a (no text or box control) | |
| Image to image | n/a | |
| Add / remove / modify a layer | **partial**: choose N; edit a layer with another model | [qwen-image-2.1.md](qwen-image-2.1.md) |
| Relight a layer | n/a | [relight-ic-light.md](relight-ic-light.md) |
| Effects as layers | n/a | |
| Amodal completion | **supported** (hidden areas behind objects are filled; faint ghosts of old positions remain) | strips above |

**Story composite: re-split after Qwen-Image-2.1 edited the scene** (run v10, 384 px, 5 layers, 4 steps, base Euler; night then heavy rain). The base model splits the night edit worse than Stable-Layers (3 of 4 object frames 99–100 % transparent) but the rain edit with more layers filled (diff 10.8): [qwen-image-2.1.md](qwen-image-2.1.md).

![night composite re-split](../img/fox_qwenlayered_night_resplit.png)

![rain composite re-split](../img/fox_qwenlayered_rain_resplit.png)

**Re-decomposing every turn** (drift arm C): [../research/drift-evaluation.md](../research/drift-evaluation.md), strip `redecompose_cycles.jpg`.

## Best features

- Most meaningful semantic layers of all routes tried so far (background, tree groups, campfire group, characters).
- Layers recomposite to the input almost exactly; background filled behind objects.
- Works across painterly, pixel-art and sketch styles.

## Weak spots

- Sometimes groups unrelated objects into one layer (skeleton + tree); thin or low-contrast subjects may stay in the background at 3 layers.
- At 384 px detail is limited; old object positions may leave faint ghosts in the background layer.

## Status

Working; main decomposer. Heun, 2-layer and fox & cottage variants measured (v9, v10); Lightning 2-step in [qwen-lightning.md](qwen-lightning.md).

## Next attempts

Heun at 8 steps; TPU route ([../research/hosting.md](../research/hosting.md#tpu)).

## Sources

Paper arXiv 2512.15603; history in [../log.md](../log.md) Phases 3–5; notes [`../../research_notes/Coherent layered generation pipeline/qwen_layered_and_layerdiffuse_papers.md`](../../research_notes/Coherent%20layered%20generation%20pipeline/qwen_layered_and_layerdiffuse_papers.md).
