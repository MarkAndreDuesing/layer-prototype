# CLD (Controllable Layer Decomposition)
> **Role: current factsheet.** Template: What it does · Licence & weights · How we run it (happy path) · Benchmarks · Best features · Weak spots · Status · Next attempts · Sources. Verdicts are the user's.

## What it does

Image + caption + one box per layer → RGBA whole image, background and one layer per box. Fine-tuned FLUX.1-dev (ART base + PrismLayers fine-tune + decouple LoRA) with an inpainting ControlNet adapter.

## Licence & weights

Code MIT ([monkek123King/CLD](https://github.com/monkek123King/CLD)); weights `thuteam/CLD` (Apache-2.0; LoRAs 598 MB each, transparent VAE 1.02 GB) + `alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha` (4.28 GB); FLUX.1-dev non-commercial.

## How we run it (happy path)

Planned: same pattern as RevealLayer/ART+ (GGUF re-class, unmerged LoRAs, `fp16_mixed`), ControlNet on GPU 1; 512 px, about 16 steps, ≤ 3 boxes.

## Benchmarks

Not run yet.

## Facts from the paper and repo

| Topic | Fact |
|---|---|
| Training | rank-64 LoRA on FLUX.1-dev, 25K iterations at 1024², on the PrismLayersPro training split; reuses ART's transparent decoder |
| Weights lineage | `pre_trained_LoRA` = ART base (SHA-256 identical); `prism_ft_LoRA` = ART+ PrismLayersPro or CLD's own fine-tune (unclear) |
| Inputs | image (to the ControlNet adapter) + caption + boxes (snapped to 16 px); adapter conditioning scale 0.9, CFG 4 |
| Outputs | RGBA whole image, RGBA background, one RGBA layer per box |

## Best features

Controllable decomposition by boxes, like RevealLayer, with a different training recipe.

## Weak spots

Needs an input image (not text → layers); three LoRAs + ControlNet → memory-tight.

## Status

Notebook not written yet; next after ART+ (shares its code family).

## Next attempts

Write `make_cld_trial.py` once ART+ works.

## Sources

[`../../research_notes/Layered methods deep dive/heavy_models_recheck_2026-11-09.md`](../../research_notes/Layered%20methods%20deep%20dive/heavy_models_recheck_2026-11-09.md) §2.
