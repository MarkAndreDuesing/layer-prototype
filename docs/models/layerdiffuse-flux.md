# LayerDiffuse-Flux
> **Role: current factsheet.** Template: What it does · Licence & weights · How we run it (happy path) · Benchmarks · Best features · Weak spots · Status · Next attempts · Sources. Verdicts are the user's.

## What it does

LayerDiffuse's transparency for FLUX.1-dev: a LoRA plus a transparent VAE decoder; text → one transparent image.

## Licence & weights

`RedAIGC/Flux-version-LayerDiffuse` (`layerlora.safetensors`, `TransparentVAE.pth`), AFL-3.0 on the HF card; FLUX.1-dev non-commercial.

## How we run it (happy path)

GGUF Q4_K_S FLUX.1-dev + encoders/VAE from ungated `ostris/Flex.1-alpha`; LoRA keys converted from BFL/kohya format to diffusers, attached unmerged; 512 px, 16 steps. Generator: [`../../prototype/tools/make_layerdiffuse_flux_trial.py`](../../prototype/tools/make_layerdiffuse_flux_trial.py).

## Benchmarks

| Object (512 px, 16 steps) | Time | Transparent | Result |
|---|---|---|---|
| campfire | 63 s | 75% | clean cut-out, stylised fire |
| deer | 57 s | 82% | clean, fine legs and antlers |
| glass bottle | 58 s | 77% | **semi-transparent glass** |

Peak 8.9 GB (one T4). Run 2 (55 s, 0 LoRA keys matched) was plain FLUX. Results: [`../../prototype/kaggle/results_ldflux_v4/`](../../prototype/kaggle/results_ldflux_v4/).

## Best features

Clean FLUX-quality transparent objects, including real semi-transparency (glass); one T4 is enough.

## Weak spots

Single images only, no multi-layer scenes; heavy model.

## Status

Works since run 4 (LoRA keys converted from BFL/kohya format; all 494 modules matched).

## Next attempts

Confirm transparency; compare with LayerDiffuse SDXL and Qwen-Image-2.1 RGBA on the same objects.

## Sources

[`../../research_notes/Layered loop open paths/flux_layered_t4_and_weights.md`](../../research_notes/Layered%20loop%20open%20paths/flux_layered_t4_and_weights.md); [../log.md](../log.md) Phase 7.
