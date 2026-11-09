# ART+ / PrismLayers
> **Role: current factsheet.** Template: What it does · Licence & weights · How we run it (happy path) · Benchmarks · Best features · Weak spots · Status · Next attempts · Sources. Verdicts are the user's.

## What it does

Global caption + layout boxes → a variable number of RGBA layers generated jointly (Anonymous Region Transformer). **Text + boxes → layers without an input image.**

## Licence & weights

Code MIT ([redredsheep/PrismLayers](https://github.com/redredsheep/PrismLayers)); weights on Google Drive. The ART base LoRA is byte-identical (SHA-256) to `pre_trained_LoRA` in the official `thuteam/CLD` repo (Apache-2.0); the transparent decoder is CLD's `trans_vae`. An unofficial HF mirror `24yearsold/PrismLayers_ckpt` exists (fallback). FLUX.1-dev non-commercial.

## How we run it (happy path)

GGUF Q4_K_S FLUX.1-dev re-classed into `CustomFluxTransformer2DModel` (max 52 layers); three LoRAs attached unmerged (pre_fuse, ART base, ART+); `fp16_mixed`; 512 px, 16 steps, CFG 4; diffusers 0.32.2. Generator: [`../../prototype/tools/make_artplus_trial.py`](../../prototype/tools/make_artplus_trial.py).

## Benchmarks

Run 2: all LoRAs matched (190 / 304 / 304), weights from the official Drive folders; stopped loading the decoder checkpoint. Run 3 running.

## Best features

The only released FLUX model that makes a layered scene straight from text + boxes (boxes can come from our LLM breakdown).

## Weak spots

Trained at 1024 px (we run 512); heavy; Drive downloads can hit quotas.

## Status

Run 3 running.

## Next attempts

Generate the fox & cottage scene from text + LLM boxes; compare with LayerDiffuse chain.

## Sources

[`../../research_notes/Layered methods deep dive/joint_text_to_layers.md`](../../research_notes/Layered%20methods%20deep%20dive/joint_text_to_layers.md); [`../../research_notes/Layered loop open paths/flux_layered_t4_and_weights.md`](../../research_notes/Layered%20loop%20open%20paths/flux_layered_t4_and_weights.md).
