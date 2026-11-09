# LayerDiffuse modes and our scene-to-layer chain

> **Role: concept (stable explanation).** How it works and why. Current numbers live in [../status.md](../status.md) and the factsheets in [../models/](../models/README.md); history in [../log.md](../log.md).

LayerDiffuse (Zhang & Agrawala 2024) teaches a latent diffusion model to produce **latent transparency**: an extra signal in the latent that a special decoder turns into an alpha channel.

## Modes
| Mode | Input | Output | Released for |
|---|---|---|---|
| `fg` | prompt | one transparent object | SD1.5, SDXL |
| joint fg + bg | prompt(s) | foreground + background together | SD1.5 only |
| `bg2ble` | background image + prompt | blended image: the new object drawn into that background | SD1.5, SDXL |
| `bgble2fg` | background + blended image | the new object as a transparent layer | SD1.5, SDXL |
| `fg2ble`, `fgble2bg` | foreground (+ blend) | background around a given object | SD1.5, SDXL |

## Our chain
For each new element: composite the current scene → `bg2ble` draws the element into it (matching light and perspective) → `bgble2fg` extracts it as its own aligned RGBA layer → stack it. Code: [`../../prototype/layerproto/backends/layerdiffuse_chain.py`](../../prototype/layerproto/backends/layerdiffuse_chain.py).

## Practical rules (measured)
- **Use SDXL base, 20 steps, CFG 7.** On SDXL-Turbo the extract step fails (0–5% transparent).
- Object prompts need "a single X, solo, centered"; "no background" / "isolated object" makes it paint an opaque backdrop.
- Lighting is baked into a generated layer; moving it later needs re-harmonisation ([relocate.md](relocate.md)).
- The vendored diffusers port needed keyword arguments for diffusers 0.37 (`shift_factor`).

Factsheet: [../models/layerdiffuse-sdxl.md](../models/layerdiffuse-sdxl.md). Papers: [../../research_notes/Coherent layered generation pipeline/qwen_layered_and_layerdiffuse_papers.md](../../research_notes/Coherent%20layered%20generation%20pipeline/qwen_layered_and_layerdiffuse_papers.md).
