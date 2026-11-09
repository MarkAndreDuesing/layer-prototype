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

## Best features

Very fast; exact pixels kept; every part public and light.

## Weak spots

Automatic "largest K" masks pick fragments (moon, stump, flame) instead of story objects; hidden parts stay missing.

## Status

Tried once (automatic masks).

## Next attempts

Segment by noun (points/boxes from the LLM breakdown, Grounded-SAM style); merge masks per object; LaMa vs SDXL inpaint; pix2gestalt amodal completion; fox & cottage.

## Sources

[../log.md](../log.md) Phase 5; [`../../research_notes/Layered methods deep dive/other_psd_methods.md`](../../research_notes/Layered%20methods%20deep%20dive/other_psd_methods.md) §1.
