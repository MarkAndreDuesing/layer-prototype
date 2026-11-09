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

## Best features

Separates thin, low-contrast subjects better (the pale skeleton came out cleanly).

## Weak spots

At 4 steps no speed-up (the base already works at 4), about 5 s LoRA overhead, colour/texture drift.

## Status

Tried once (sweep v6).

## Next attempts

2-step runs, scale sweep 0.25–0.75, only for hard scenes.

## Sources

[../log.md](../log.md), Sweep v6.
