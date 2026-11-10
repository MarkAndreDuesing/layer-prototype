# ART+ / PrismLayers
> **Role: current factsheet.** Template: What it does · Licence & weights · How we run it (happy path) · Benchmarks · Best features · Weak spots · Status · Next attempts · Sources. Verdicts are the user's.

## What it does

Global caption + layout boxes → a variable number of RGBA layers generated jointly (Anonymous Region Transformer). **Text + boxes → layers without an input image.**

## Licence & weights

Code MIT ([redredsheep/PrismLayers](https://github.com/redredsheep/PrismLayers)); weights on Google Drive. The ART base LoRA is byte-identical (SHA-256) to `pre_trained_LoRA` in the official `thuteam/CLD` repo (Apache-2.0); the transparent decoder is CLD's `trans_vae`. An unofficial HF mirror `24yearsold/PrismLayers_ckpt` exists (fallback). FLUX.1-dev non-commercial.

## How we run it (happy path)

GGUF Q4_K_S FLUX.1-dev re-classed into `CustomFluxTransformer2DModel` (max 52 layers); three LoRAs attached unmerged (pre_fuse, ART base, ART+); `fp16_mixed`; 512 px, 16 steps, CFG 4; diffusers 0.32.2. Generator: [`../../prototype/tools/make_artplus_trial.py`](../../prototype/tools/make_artplus_trial.py).

## Benchmarks

| Run | Result |
|---|---|
| 2 | all LoRAs matched (190 / 304 / 304), weights from the official Drive folders; decoder checkpoint load failed (`weights_only`) |
| 3 | loaded 295 s (7.0 / 11.0 GB); fp16 inputs to fp32 VAEs |
| 4 | denoising finished; OOM in the fp32 VAE decode |
| 5 | end to end: 345 s / 285 s at 16 steps, 10 GB; image small in one corner, opaque rectangles |
| **6** | **`vae_scale_factor` 16 restored → full-size coherent scenes** (330 s / 282 s, 8.2 GB); layers still opaque crops of the whole image |
| 7 | ART+'s own transparent decoder (Drive, mirror fallback): same opaque crops |
| 8 | diagnostics: `layer_pe` distinct per layer, final latents differ per frame → conditioning reaches the model |
| **9** | **bug found in our `fp16_mixed.add_lora`: each LoRA replaced the previous one on the same module, so only the ART+ LoRA was active (pre_fuse and ART base dropped). Stacked → real transparent layers** (356 s / 294 s at 16 steps, 8.5 GB) |

Run 9, case 1 (whole | background | skeleton | fire | log | merged):

![ART+ v9 case 1](../img/artplus_v9_case1.jpg)

**Run 10, fox & cottage** (caption + the 5 JSON boxes, "cartoon" style prefix; 353 s, 8.6 GB). Whole | background | village houses | cottage | gate arch | tree | fox | merged:

![ART+ v10 fox & cottage](../img/artplus_v10_fox.jpg)

Every box gets its own clean, consistently styled transparent object at the right place; but the background is plain white (sticker look), objects are small "island" vignettes, and the gate-arch box became a red car.

Case 0 of run 9 (night forest): trees, campfire + bench and moon come out as separate layers, but the whole/background frames are almost black.

![ART+ v9 case 0](../img/artplus_v9_case0.jpg)

## Facts from the paper and repo

| Topic | Fact |
|---|---|
| Lineage | ART trained from FLUX.1-dev; ART+ fine-tuned from it (20K iterations at 512², then 10K at 1024²) on PrismLayersPro (Apache-2.0 dataset) |
| Input | JSON: caption with a style prefix ("This is a [STYLE] style image. …", 21 styles) + layout boxes; the first two boxes are the full image; no input image needed |
| Weights | Drive folders: `pre_fuse_lora`, `base_ckpt` (ART), `artplus_PrismLayersPro(100k)_ckpt` (`layer_pe` + LoRA), `transp_vae`; unofficial HF mirror `24yearsold/PrismLayers_ckpt` (5.56 GB, 100k variant only) |
| Original pipeline | the two LoRAs are fused into FLUX, the ART+ LoRA loaded as adapter "layer"; diffusers 0.31, torch 2.4, bf16 |
| diffusers ≥ 0.32 | FLUX `vae_scale_factor` changed from 16 to 8 → set `pipe.vae_scale_factor = 16`, else latents for twice the size and boxes in one corner |
| LoRA files | diffusers format, no alpha keys (scale 1.0); ART / ART+ rank 64, pre-fuse rank 16 |

## Best features

The only released FLUX model that makes a layered scene straight from text + boxes (boxes can come from our LLM breakdown).

## Weak spots

Trained at 1024 px (we run 512); heavy; Drive downloads can hit quotas.

## Status

**Works since run 9:** text + boxes → background + clean transparent object layers on 1× T4 in about 5 min (16 steps, 512 px, 8.5 GB). Night scene very dark. Results: `prototype/kaggle/results_artplus_v9/`.

## Next attempts

Fox scene with a style prefix that implies a full background ("painting"/"illustration" instead of "cartoon") and the sky/lawn as an explicit box; 1024 px; per-box captions if the pipeline accepts them.

## Sources

[`../../research_notes/Layered methods deep dive/joint_text_to_layers.md`](../../research_notes/Layered%20methods%20deep%20dive/joint_text_to_layers.md); [`../../research_notes/Layered loop open paths/flux_layered_t4_and_weights.md`](../../research_notes/Layered%20loop%20open%20paths/flux_layered_t4_and_weights.md).
