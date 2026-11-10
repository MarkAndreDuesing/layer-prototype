# Stable-Layers (Stability AI)
> **Role: current factsheet.** Template: What it does · Licence & weights · How we run it (happy path) · Benchmarks · Best features · Weak spots · Status · Next attempts · Sources. Verdicts are the user's.

> **User assessment (first look, 2026-11-11):** Pretty decent layer extraction, particularly useful for story composites (with a Qwen-Image-2.1 edit).

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
| **fox & cottage, 5 layers, 4 Euler steps** | 39 s | 21.0 |
| fox & cottage, 5 layers, 4 Heun steps | 67 s | 20.6 |

On fox & cottage both give the cleanest split so far: **sky / meadow / cottage + tree / path / fox** ([`results_qwen_fox_v11/`](../../prototype/kaggle/results_qwen_fox_v11/)).

Heun results: [`../../prototype/kaggle/results_qwen_v9b/`](../../prototype/kaggle/results_qwen_v9b/). Euler results: [`../../prototype/kaggle/results_stable_layers_v8/`](../../prototype/kaggle/results_stable_layers_v8/). The README asks for Heun, 50 steps, 640 px.

## Fox & cottage benchmark

Same input and strip layout as [qwen-image-layered.md](qwen-image-layered.md). Stable-Layers is the base model plus the RL-tuned LoRA, so the cost per split is the same.

| | Settings | Compute (2× T4) | Recomposite diff | What the layers show |
|---|---|---|---|---|
| **A. Best case measured** | 384 px, 5 layers, 4 **Heun** steps, CFG 1 | 67 s | 20.6 | sky / meadow / cottage + tree / path / fox, the same clean split |
| **B. Live-demo budget (about 30 s)** | 384 px, 5 layers, 4 **Euler** steps, CFG 1 | 39 s | 21.0 | sky / meadow / cottage + tree / path / fox |

A (Heun):

![Stable-Layers Heun](../img/fox_stablelayers_heun.png)

B (Euler):

![Stable-Layers Euler](../img/fox_stablelayers_euler.png)

The colour shift (diff about 21 against 9–13 for the base model) is visible in the recomposite. Results: `prototype/kaggle/results_qwen_fox_v11/`.

### Profile benchmark (warm model in one Kaggle session, 2026-11-11)

Same `qwen-layered` profile as the base model (the LoRA is swapped in a few seconds); second run reported. Results: `prototype/kaggle/results_profile_qwen-layered/`.

| | Settings | Compute | Layers | Recomposite diff |
|---|---|---|---|---|
| **A. Best case (up to 8 min)** | 512 px, 5 layers, 8 Heun steps, CFG 1 | 248.4 s | 5 | 16.09 |
| **B. Live-demo budget (about 30 s)** | 384 px, 5 layers, 3 Euler steps, CFG 1 | 30.9 s | 5 | 23.2 |

A (8 Heun steps at 512 px): a clean filled meadow as the background, and separate layers for the cottage with its gate arch, the path, the tree and the fox (plus a layer with the left-hand sky and tree):

![A, 512 px Heun](../img/bench_stablelayers_A.jpg)

B (3 Euler steps at 384 px):

![B, 384 px 3 steps](../img/bench_stablelayers_B.jpg)

## Special features

| Feature | Status | Evidence |
|---|---|---|
| Split an image into layers | **supported** | strips above |
| Move / resize a layer | **partial** (canvas move; no harmonising) | [relocate-harmonise.md](relocate-harmonise.md) |
| Story composite | **supported**: re-split of night and rain edits | below |
| Spatial edit / image to image | n/a | |
| Add / remove / modify a layer | **partial** (N chosen per run) | |
| Relight / effects | n/a | |
| Amodal completion | **supported** (background filled behind objects; very clean with Heun) | strips above |

**Story composites** (Qwen-Image-2.1 edit, then Stable-Layers Heun, 5 layers, 384 px, 67–70 s). Night: moon + sky bits / cottage with lit windows / tree / fox, diff 4.2. Heavy rain: cottage / tree / fox plus a wet-path layer, diff 11.2:

![night composite re-split](../img/fox_stablelayers_night_resplit.png)

![rain composite re-split](../img/fox_stablelayers_rain_resplit.png)

## Best features

Cleaner semantic separation than the base model: on the sketch the background is truly empty paper and the fire gets its own layer.

## Weak spots

Colour and contrast shift, so the recomposite is about 2× further from the input.

## Status

Runs with Euler (v8) and with exact Heun (v9, via `HeunVelocity`). With Heun the forest background comes out very clean (trees and campfire fully removed, filled), trees split into clean layers; on the sketch the skeleton, fire and logs stay in one layer. Colour shift remains (diff 15–18).

## Next attempts

Heun 4–8 steps; colour correction (match the recomposite's colour statistics to the input); fox & cottage.

## Sources

[../log.md](../log.md) Phase 5; `../../research_notes/Layered methods deep dive/heavy_models_inference_plans` (removed 2026-11-10; in git history before commit 35dc02a) §4.
