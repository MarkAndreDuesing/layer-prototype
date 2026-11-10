# Qwen-Image-Lightning LoRA (on Qwen-Image-Layered)
> **Role: current factsheet.** Template: What it does · Licence & weights · How we run it (happy path) · Benchmarks · Best features · Weak spots · Status · Next attempts · Sources. Verdicts are the user's.

## What it does

lightx2v's distillation LoRA for plain Qwen-Image (4–8 steps). Untested on Layered before us.

## Licence & weights

`lightx2v/Qwen-Image-Lightning` (Apache-2.0 per card).

## How we run it (happy path)

Attach unmerged with `fp16_mixed.add_lora` (all 720 modules match Layered); Lightning scheduler config from the card; 384 px, 4 layers, 4 steps.

## Benchmarks

| Variant | Total | Recomposite diff |
|---|---|---|
| base model | 30 s | 7.1 |
| scheduler only | 35 s | 8.6 |
| LoRA 0.5 | 39 s | 10.3 |
| LoRA 1.0 | 37–39 s | 11–13.5 |

Results: [`../../prototype/kaggle/results_qwen_sweep_v6_lightning/`](../../prototype/kaggle/results_qwen_sweep_v6_lightning/).

## Run v10: 2 and 4 steps, scale sweep (fox & cottage, 384 px, 5 layers, 2× T4)

| Scale | Steps | Total | Recomposite diff | Layers look |
|---|---|---|---|---|
| 0.25 | 2 | 22.5 s | 32.3 | not split (frames mostly overlapping) |
| 0.5 | 2 | 22.5 s | 22.1 | partial |
| 0.75 | 2 | 22.5 s | 15.2 | partial |
| 1.0 | 2 | 22.4 s | 15.0 | tree + fox separated, cottage/houses almost empty |
| 0.5 | 4 | 43.6 s | 11.1 | cottage + tree + fox, cleanest of the sweep |
| 1.0 | 4 | 43.9 s | 13.8 | tree and fox, cottage nearly empty |
| *base, Euler, 4 steps (for reference)* | 4 | 39 s | 6–8 | all five objects |

2 steps halve the time (22 s vs 39–44 s) but the split quality is clearly worse; no speed-up at 4 steps. Fox & cottage at 1.0 / 2 steps (input | recomposite | background | layers):

![Lightning 1.0, 2 steps](../img/lightning_2step.png)

Results: `prototype/kaggle/results_qwen_v10/`.

## Best features

Separates thin, low-contrast subjects better (the pale skeleton came out cleanly).

## Weak spots

At 4 steps no speed-up (the base already works at 4), about 5 s LoRA overhead, colour/texture drift.

## Status

Tried in sweep v6 and run v10 (2-step and scale sweep).

## Next attempts

Only for hard thin/low-contrast scenes (the pale skeleton) where the base fails; for speed prefer the base model at 2 layers.

## Sources

[../log.md](../log.md), Sweep v6.
