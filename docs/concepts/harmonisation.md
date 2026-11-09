# Harmonising placed and moved layers

> **Role: concept (stable explanation).** How it works and why. Current numbers live in [../status.md](../status.md) and the factsheets in [../models/](../models/README.md); history in [../log.md](../log.md).

**Problem:** a moved, added or re-prompted layer must match its surroundings: lighting, colour, contact, shadow, style, while keeping its identity. Literature survey: [../../research_notes/Layered loop open paths/object_harmonisation.md](../../research_notes/Layered%20loop%20open%20paths/object_harmonisation.md).

## Our components
1. **`relocate()`** ([relocate.md](relocate.md)): SDXL + LayerDiffuse re-harmonisation with interior pinning; about 4 s on a T4.
2. **IC-Light relight** ([`../../prototype/layerproto/relight.py`](../../prototype/layerproto/relight.py)): IC-Light v1 "fbc" (SD1.5, Apache-2.0) relights a foreground against a given background. Realistic Vision 5.1 UNet widened from 4 to 12 input channels; foreground (transparent pixels filled grey) and background latents are concatenated at every step. We put back the layer's own alpha and transfer the original's high-frequency detail, so only the light changes. CPU test: 44 s (8 steps, 384 px); a T4 should take seconds. IC-Light v2 (FLUX) never released weights.
3. **Shadow layer** ([`../../prototype/layerproto/shadow.py`](../../prototype/layerproto/shadow.py)): procedural cast shadow as its own black RGBA layer (EasyOmnimatte recipe): squash and shear the object's alpha along the light direction, blur more with distance, fade, plus a contact ambient-occlusion blob; one contact line per object in the layer; length capped so tall objects' shadows stay on the ground. About 0.6 s per layer on CPU.

## Order in the loop
warp → `relocate()` → IC-Light relight (optional) → paste back detail and alpha → shadow layer below the object.

## Identity tricks
Interior latent pinning, paste-back, detail transfer; attention/KV sharing (RegionDrag-style) not tried yet. No paper compares these head-to-head.

## Known limits
IC-Light has no notion of light sources (a campfire layer gets dimmed under "night"); the shadow recipe assumes a ground plane. ImageDirector (UIST '26) found that full regeneration can light objects more naturally than compositing: harmonisation has to close that gap.

Results: [`../../prototype/kaggle/results_local_relight/`](../../prototype/kaggle/results_local_relight/), [`../../prototype/kaggle/results_local_shadow/`](../../prototype/kaggle/results_local_shadow/).
