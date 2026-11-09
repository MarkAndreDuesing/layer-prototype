# Stable-Layers (Stability AI)
> **Role: current factsheet.** Template: What it does · Licence & weights · How we run it (happy path) · Benchmarks · Best features · Weak spots · Status · Next attempts · Sources. Verdicts are the user's.

## What it does

An RL-tuned LoRA (330 MB, PEFT, 420 modules) on top of Qwen-Image-Layered, trained with a VLM teacher for cleaner semantic splits.

## Licence & weights

Stability AI Community licence (free below $1M revenue, attribution). `StabilityLabs/Stable-Layers`.

## How we run it (happy path)

Same stack as Qwen-Image-Layered; LoRA attached unmerged with `fp16_mixed.add_lora` (all 420 modules match). 384 px, 4 layers, 4 Euler steps, CFG 1. Heun: `decompose(..., sampler="heun")`. Generator: [`../../prototype/tools/make_qwen_layered_batch.py`](../../prototype/tools/make_qwen_layered_batch.py) (SWEEP cell).

## Benchmarks

| Setting | Time | Recomposite diff |
|---|---|---|
| 4 Euler steps, 384 px | 34 s | 15–18 (base model ≈ 7) |
| 8 Euler steps, 384 px | 71 s | — |
| **4 Heun steps, 384 px** (forest / sketch / pixel art) | 58–61 s | 15.5–17.9 |
| 8 Heun steps, 384 px | 130 s | 14.8 |
| 2 layers, 4 Euler steps | 26.5 s | 14.2 |

Heun results: [`../../prototype/kaggle/results_qwen_v9b/`](../../prototype/kaggle/results_qwen_v9b/). Euler results: [`../../prototype/kaggle/results_stable_layers_v8/`](../../prototype/kaggle/results_stable_layers_v8/). The README asks for Heun, 50 steps, 640 px.

## Best features

Cleaner semantic separation than the base model: on the sketch the background is truly empty paper and the fire gets its own layer.

## Weak spots

Colour and contrast shift, so the recomposite is about 2× further from the input.

## Status

Runs with Euler (v8) and with exact Heun (v9, via `HeunVelocity`). With Heun the forest background comes out very clean (trees and campfire fully removed, filled), trees split into clean layers; on the sketch the skeleton, fire and logs stay in one layer. Colour shift remains (diff 15–18).

## Next attempts

Heun 4–8 steps; colour correction (match the recomposite's colour statistics to the input); fox & cottage.

## Sources

[../log.md](../log.md) Phase 5; [`../../research_notes/Layered methods deep dive/heavy_models_inference_plans.md`](../../research_notes/Layered%20methods%20deep%20dive/heavy_models_inference_plans.md) §4.
