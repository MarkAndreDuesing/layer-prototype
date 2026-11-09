# LayerDiffuse SDXL (and our scene-to-layer chain)
> **Role: current factsheet.** Template: What it does · Licence & weights · How we run it (happy path) · Benchmarks · Best features · Weak spots · Status · Next attempts · Sources. Verdicts are the user's.

## What it does

Native transparency for Stable Diffusion: generates a transparent object, or (conditional modes) draws an object into a given background and extracts it as an aligned RGBA layer. Our chain uses `bg2ble → bgble2fg` per element on the current composite ([../concepts/layerdiffuse-modes.md](../concepts/layerdiffuse-modes.md)).

## Licence & weights

Code Apache-2.0, weights OpenRAIL-M ([LayerDiffusion/layerdiffusion-v1](https://huggingface.co/LayerDiffusion/layerdiffusion-v1)); MIT diffusers port vendored in `prototype/vendor/diffuser_layerdiffuse`.

## How we run it (happy path)

SDXL **base** (not Turbo), 20 steps, CFG 7, Euler; 1024 px on a T4 or 512 px on CPU. Code: [`../../prototype/layerproto/backends/layerdiffuse_chain.py`](../../prototype/layerproto/backends/layerdiffuse_chain.py). Object prompts need "a single X, solo, centered"; "no background" makes it paint an opaque backdrop.

## Benchmarks

| Setting | Time |
|---|---|
| one chain pass, 1024 px, T4 | about 18 s |
| full chain notebook incl. downloads | 291 s |
| per element, 512 px, laptop CPU | about 11 min (base) / 80 s per plain layer (Turbo) |
| `relocate()` move/resize, 512 px, T4 | about 4 s |

## Best features

- Adds an element *into* the existing scene with matching light and perspective, as its own aligned layer.
- Basis of `relocate()` re-harmonisation and story-event layers.
- Runs on CPU (slowly) and comfortably on a T4.

## Weak spots

- The extract step fails on SDXL-Turbo (0–5% transparent).
- One element per pass; complex scenes with many overlapping objects not evaluated.

## Status

Working base of the prototype.

## Next attempts

Complex scenes (5+ objects); fox & cottage built element by element; combine with IC-Light and shadow layers.

## Sources

LayerDiffuse paper arXiv 2402.17113; [../log.md](../log.md) Phase 2.
