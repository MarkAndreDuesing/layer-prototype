# Generate → SAM2 → LaMa (segment + inpaint)
> **Role: current factsheet.** Template: What it does · Licence & weights · How we run it (happy path) · Benchmarks · Best features · Weak spots · Status · Next attempts · Sources. Verdicts are the user's.

## What it does

Build-your-own decomposition: segment objects with SAM2, cut each out as a layer, inpaint the holes in the background with LaMa (optionally amodal completion).

## Licence & weights

SAM2 Apache-2.0; LaMa (simple-lama-inpainting, Apache-2.0); all public.

## How we run it (happy path)

Kaggle T4; SAM2 hiera-small automatic masks, the 4 largest object masks (< 35% of the image) as layers, LaMa for the background (`simple-lama-inpainting` installed with `--no-deps`). Generator: [`../../prototype/tools/make_segment_inpaint_trial.py`](../../prototype/tools/make_segment_inpaint_trial.py).

## Benchmarks

3–7 s per image; recomposite diff 0.2–2.4 (layers are cut-outs). Results: [`../../prototype/kaggle/results_segment_inpaint/`](../../prototype/kaggle/results_segment_inpaint/).

## Helper tools (licences)

| Tool | Role | Licence | Size |
|---|---|---|---|
| SAM 2.1 | prompted segmentation | Apache-2.0 | 39–224M |
| SAM 3 | text-prompted segmentation | Meta licence, gated | ≈ 3.6 GB |
| Florence-2-large | text grounding → boxes for SAM | MIT | 0.77B |
| BiRefNet | matting (clean cut-outs) | MIT | 0.2B |
| BEN2 base | matting | MIT | 94.6M |
| RMBG-2.0 | matting | CC BY-NC, gated | — |
| LaMa | background fill | Apache-2.0 | small |
| pix2gestalt | amodal completion of occluded objects | — | — |

More in [../research/hosting.md §7](../research/hosting.md#7-matting-segmentation-and-helper-tools).

## Fox & cottage (v2: one mask per named object)

| Variant | Time | Result |
|---|---|---|
| A: hand boxes → SAM2 | 8 s | village house, cottage, gate arch, tree, fox: clean cut-outs; diff 1.27 |
| **B: layer names → Grounding DINO → SAM2** (fully automatic) | 7 s | same as A except "village houses" (boxed the cottage instead); diff 1.30 |
| C: automatic masks, 5 largest | 9 s | fragments (cloud, bush, lawn piece, path) |

Weak spot of A/B: the LaMa background fill is blurry where large objects (cottage, tree) were removed; hidden parts of objects stay missing. Results: [`../../prototype/kaggle/results_segment_inpaint_v2/`](../../prototype/kaggle/results_segment_inpaint_v2/).

## Background fill: LaMa vs SDXL-inpaint (v3, hand boxes, 512 px → 1024 inpaint, T4)

| Fill | Time (whole route) | Look |
|---|---|---|
| LaMa | 13 s | blurry, smeared texture over the removed cottage + tree + fox area |
| **SDXL-inpaint 1.0** (fp16-fix VAE, 30 steps, scene prompt, negative "fox, animal, person") | 28 s (30 s with Grounding DINO boxes) | sharp, in the scene's style, plausible new cottage behind; but **invents content** (different cottage, no village houses) so the recomposite differs from the input only where objects are moved |

LaMa fill (left strip) vs SDXL fill (right strip): input | recomposite | background | five object layers.

![LaMa fill](../img/sam2_lamafill.png)

![SDXL fill](../img/sam2_sdxlfill.png)

Object masks are the same in both; the cottage mask is incomplete where the tree and hedge overlap (no amodal completion), which shows when the tree is moved. Results: `prototype/kaggle/results_segment_inpaint_v3/`.

## Best features

Very fast; exact pixels kept; every part public and light.

## Weak spots

Automatic "largest K" masks pick fragments (moon, stump, flame) instead of story objects; hidden parts stay missing.

## Status

Works by noun (v2): Grounding DINO + SAM2 gives meaningful object layers fully automatically in about 7 s.

## Next attempts

SDXL inpaint instead of LaMa for large holes; pix2gestalt amodal completion for occluded parts (cottage behind the tree); combine with RevealLayer's filled background.

## Sources

[../log.md](../log.md) Phase 5; [`../../research_notes/Layered methods deep dive/other_psd_methods.md`](../../research_notes/Layered%20methods%20deep%20dive/other_psd_methods.md) §1.
