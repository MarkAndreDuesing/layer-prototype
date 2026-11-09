# Move / resize with re-harmonisation (`relocate()`)

> **Role: concept (stable explanation).** How it works and why. Current numbers live in [../status.md](../status.md) and the factsheets in [../models/](../models/README.md); history in [../log.md](../log.md).

**Problem:** moving or scaling a layer as pixels leaves its old lighting and no contact with the new spot; regenerating the whole scene moves the object but changes everything else (drift). No released model re-harmonises a moved layer (gap G3). Our training-free method follows DreamLayer's harmonisation step.

Code: `LayerDiffuseChain.relocate(scene, layer, box, prompt, seed, strength=0.5, pin_frac=0.4, …, band=9, paste_back=0.0)` in [`../../prototype/layerproto/backends/layerdiffuse_chain.py`](../../prototype/layerproto/backends/layerdiffuse_chain.py).

## Steps
1. **Warp in pixel space.** Scale and shift the layer's RGBA into the target box and alpha-composite it over the background (the composite of all *other* layers). Right position, old lighting.
2. **Encode and re-noise** to strength s (s = 0.3 redoes the last 30% of the schedule).
3. **Denoise with the scene as condition:** SDXL base + LayerDiffuse `bg2ble`, **Euler** sampler, so lighting, glow and style come from the scene.
4. **Pin the interior:** in every step except the last 2, the latents inside the object (alpha > 40, eroded by a band of a few pixels) are replaced by the original object's latents re-noised to the current step. Only the rim and the surroundings change (contact, glow, edges); identity is kept.
5. **Extract the layer by mask:** the object's warped alpha plus every pixel that changed noticeably within a ring around it (new glow/contact).
6. **Optional paste-back:** blend the original object pixels back inside its alpha (0–1) to keep identity exactly.

## What the parameters do
- **strength:** 0.3 keeps identity and adds light/glow from the new spot; 0.5 also redraws details (faces); higher with a stochastic sampler broke down.
- **pin_frac / band:** how long and how tightly the interior is held.
- **paste_back:** 0.8 kept the moved skeleton's look while its surroundings were still re-harmonised.

## Why these choices (from the versions, see [../log.md](../log.md) Phase 5)
- v1 used DPM++ SDE (stochastic: fresh noise every step fights the pinned latents → artifacts) and `bgble2fg` extraction (translucent or wrong object). v2 switched to Euler and mask extraction: about 4–5 s per edit.
- v3/v4 seemed to show pinning had no effect; the cause was a bug (alpha > 128 eroded 12 px left about 42 of 4,096 latent pixels pinned). v5 fixed the mask and added paste-back.

## Results and limits
About 4 s per edit at 512 px on a T4 once SDXL is loaded. Identity largely kept; thin parts at the edge (chair legs) can still change; contact shadows weak → [harmonisation.md](harmonisation.md) adds a shadow layer and IC-Light relighting. The old position needs inpainting when the background itself still contains the object (RevealLayer's filled background avoids this).

Design sources: [../../research_notes/Layered methods deep dive/spatial_edit_mechanics.md](../../research_notes/Layered%20methods%20deep%20dive/spatial_edit_mechanics.md), survey parts 1–3 in the same folder.
