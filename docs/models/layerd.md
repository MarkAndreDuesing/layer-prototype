# LayerD (ICCV 2025)
> **Role: current factsheet.** Template: What it does · Licence & weights · How we run it (happy path) · Benchmarks · Best features · Weak spots · Status · Next attempts · Sources. Verdicts are the user's.

## What it does

Image → layers by repeated peeling: matte out the top layer (BiRefNet), inpaint beneath (LaMa), repeat.

## Licence & weights

Apache-2.0, `cyberagent/layerd-birefnet` (repo archived 2026-10-30).

## How we run it (happy path)

Kaggle CPU (no GPU quota), `transformers<5`. Generator: [`../../prototype/tools/make_layerd_trial.py`](../../prototype/tools/make_layerd_trial.py).

## Benchmarks

About 60 s per image on CPU; returned **1 layer** (the unchanged image) on all 3 painted scenes. Results: [`../../prototype/kaggle/results_layerd_trial/`](../../prototype/kaggle/results_layerd_trial/).

## Best features

Small, fast, CPU-only; trained for graphic-design images.

## Weak spots

Found nothing to peel in painterly scenes at default settings.

## Status

Tried once.

## Next attempts

Lower thresholds, more recursion depth; fox & cottage; a flatter graphic-style image.

## Sources

[../log.md](../log.md) Phase 5.
