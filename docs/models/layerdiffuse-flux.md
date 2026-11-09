# LayerDiffuse-Flux
> **Role: current factsheet.** Template: What it does · Licence & weights · How we run it (happy path) · Benchmarks · Best features · Weak spots · Status · Next attempts · Sources. Verdicts are the user's.

## What it does

LayerDiffuse's transparency for FLUX.1-dev: a LoRA plus a transparent VAE decoder; text → one transparent image.

## Licence & weights

`RedAIGC/Flux-version-LayerDiffuse` (`layerlora.safetensors`, `TransparentVAE.pth`), AFL-3.0 on the HF card; FLUX.1-dev non-commercial.

## How we run it (happy path)

GGUF Q4_K_S FLUX.1-dev + encoders/VAE from ungated `ostris/Flex.1-alpha`; LoRA keys converted from BFL/kohya format to diffusers, attached unmerged; 512 px, 16 steps. Generator: [`../../prototype/tools/make_layerdiffuse_flux_trial.py`](../../prototype/tools/make_layerdiffuse_flux_trial.py).

## Benchmarks

Run 2: 55 s per image at 512 px / 16 steps, 7.6 GB, but **without** the LoRA (0 keys matched). Run 4 pending.

## Best features

FLUX-quality single transparent objects (expected; not yet confirmed).

## Weak spots

Single images only, no multi-layer scenes; heavy model.

## Status

Run 4 (LoRA key conversion) queued.

## Next attempts

Confirm transparency; compare with LayerDiffuse SDXL and Qwen-Image-2.1 RGBA on the same objects.

## Sources

[`../../research_notes/Layered loop open paths/flux_layered_t4_and_weights.md`](../../research_notes/Layered%20loop%20open%20paths/flux_layered_t4_and_weights.md); [../log.md](../log.md) Phase 7.
