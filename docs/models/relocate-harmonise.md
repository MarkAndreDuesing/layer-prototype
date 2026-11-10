# relocate(): move / resize a layer with re-harmonisation

> **Role: current factsheet** (an approach we built, not a released model). Template: [TEMPLATE.md](TEMPLATE.md). Verdicts are the user's.

> **User assessment (first look, 2026-11-11):** Interesting and very fast, but moving and resizing is probably best done by direct user interaction with individual layers. Re-harmonising could still be useful.

**Type:** layer + new box + the scene below it → the layer re-drawn at its new place, matching the new surroundings.

## What it does

Training-free move and resize of one RGBA layer with re-harmonisation (after DreamLayer's harmonisation step): paste the object into its new box, re-noise the composite to a low strength, denoise with SDXL + LayerDiffuse `bg2ble` while the object's interior latents are pinned for most steps, then extract the object again by mask (alpha plus the ring that changed). The object keeps its identity, its rim and the light around it adapt, and contact shadows can grow.

## Licence & weights

Uses the LayerDiffuse SDXL chain ([layerdiffuse-sdxl.md](layerdiffuse-sdxl.md): SDXL base, LayerDiffuse weights; non-commercial terms of those models apply). No extra weights. Optional: [relight-ic-light.md](relight-ic-light.md) and [shadow-layers.md](shadow-layers.md).

## How we run it

1. Code: `LayerDiffuseChain.relocate()` in [`layerdiffuse_chain.py`](../../prototype/layerproto/backends/layerdiffuse_chain.py); remote route `POST /chain/relocate` and `RemoteChain.relocate`; app: `Engine.relocate_layer` and the button **Fit + re-harmonise (GPU)** (tested with a fake chain, `tools/test_relocate_route.py`).
2. Defaults: `strength` 0.3–0.45, `pin_frac` 0.4 (interior pinned for all but the last steps), `band` 4–9 px, Euler sampler, mask extraction, `paste_back` 0.8.
3. Canvas-only alternative (no model): **Fit to planned box** / the move sliders, instant.

## Fox & cottage benchmark

Input: RevealLayer's split of the scene (filled background + five layers). The fox is moved 28 % left, 4 % up and scaled 1.15×.

![original | naive paste | relocate | + shadow | + IC-Light](../img/fox_move_strip.png)

| | Compute (T4, 512 px) | Notes |
|---|---|---|
| **A. Best case** (strength 0.3 / 0.45, paste-back 0.8) | 3.4 s per edit (first call 133 s includes the SDXL download) | fox kept, edges blended into the path; 0.3 and 0.45 look nearly alike |
| **B. Live-demo budget (about 30 s)** | the same 3.4 s | already far under budget; the shadow layer adds 0.04 s |

The whole loop (move, harmonise, shadow, relight) took about 9 s on a T4 without IC-Light. Results: `prototype/kaggle/results_fox_move_v2/`.

## Further investigations

- Parameter sweep (strength, pin fraction, band, sampler, extraction): table in [concepts/editing.md §1](../concepts/editing.md).
- **ART+ objects harmonised onto another background:** it does not close a style gap (the glossy objects stay glossy and gain halos at strength 0.6): [art-plus.md](art-plus.md), strips `artplus_harmonise_s40.jpg` / `s60.jpg`.
- Identity drift over repeated moves (a long chain of relocations) is unsolved; pin length does not help.

## Special features

| Feature | Status | Evidence |
|---|---|---|
| Split an image into layers | n/a | takes an existing split |
| Move / resize a layer | **supported** | strip above, 3.4 s |
| Harmonise after a move | **supported** | strip above (naive paste vs relocate) |
| Story composite | n/a | see [qwen-image-2.1.md](qwen-image-2.1.md) |
| Spatial edit | partial | the box defines where the object goes; no text edit |
| Image to image | n/a | |
| Add / remove / modify a layer | partial | the moved layer is replaced by the harmonised one; the old place needs a filled background |
| Relight a layer | partial | combine with IC-Light: [relight-ic-light.md](relight-ic-light.md) |
| Effects as layers | not tried | |
| Amodal completion | n/a | |

## Best features / Weak spots

- Best: seconds per edit; identity largely kept; contact shadows and light adapt; works through the same server as the chain.
- Weak: thin parts at the edge can change; the old position needs a filled background (RevealLayer's, or [amodal-fill.md](amodal-fill.md)); cannot restyle objects from another model.

## Status / Next attempts / Sources

Works on the fox & cottage scene; in the app behind a button; the real server call is the 3.4 s measured above. Next: the harmonise step inside the profile server notebook, and a run through the tunnel. Sources: DreamLayer (harmonisation), [concepts/editing.md](../concepts/editing.md).
