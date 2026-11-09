# LayerDiffuse SDXL (and our scene-to-layer chain)
> **Role: current factsheet.** Template: What it does · Licence & weights · How we run it (happy path) · Benchmarks · Best features · Weak spots · Status · Next attempts · Sources. Verdicts are the user's.

## What it does

Native transparency for Stable Diffusion: generates a transparent object, or (conditional modes) draws an object into a given background and extracts it as an aligned RGBA layer. Our chain uses `bg2ble → bgble2fg` per element on the current composite (modes below).

## Modes (how LayerDiffuse works)

LayerDiffuse teaches the model a **latent transparency** signal that a special decoder turns into alpha.

| Mode | Input | Output | Released for |
|---|---|---|---|
| `fg` | prompt | one transparent object | SD1.5, SDXL |
| joint fg + bg | prompt(s) | foreground + background together | SD1.5 only |
| `bg2ble` | background + prompt | the new object drawn into that background | SD1.5, SDXL |
| `bgble2fg` | background + blended image | that object as a transparent layer | SD1.5, SDXL |
| `fg2ble` / `fgble2bg` | foreground (+ blend) | a background around a given object | SD1.5, SDXL |

**Our chain:** composite the current scene → `bg2ble` draws the new element into it → `bgble2fg` extracts it as its own aligned layer → stack it.

## Licence & weights

Code Apache-2.0, weights OpenRAIL-M ([LayerDiffusion/layerdiffusion-v1](https://huggingface.co/LayerDiffusion/layerdiffusion-v1)); MIT diffusers port vendored in `prototype/vendor/diffuser_layerdiffuse`.

## How we run it (happy path)

SDXL **base** (not Turbo), 20 steps, CFG 7, Euler; 1024 px on a T4 or 512 px on CPU. Code: [`../../prototype/layerproto/backends/layerdiffuse_chain.py`](../../prototype/layerproto/backends/layerdiffuse_chain.py). Object prompts need "a single X, solo, centered"; "no background" / "isolated object" makes it paint an opaque backdrop. Lighting is baked into a layer, so moving it later needs `relocate()`. The vendored diffusers port needed keyword arguments for diffusers 0.37 (`shift_factor`).

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
