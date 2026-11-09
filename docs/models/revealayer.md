# RevealLayer (ICML 2026)
> **Role: current factsheet.** Template: What it does · Licence & weights · How we run it (happy path) · Benchmarks · Best features · Weak spots · Status · Next attempts · Sources. Verdicts are the user's.

## What it does

Image + one box per object → RGBA layers, **including the hidden parts** (amodal completion), plus a background with the objects removed and filled in.

## Licence & weights

Code Apache-2.0 ([360CVGroup/RevealLayer](https://github.com/360CVGroup/RevealLayer)); LoRA, `layer_pe`, refiner and transparent decoder on HF `qihoo360/RevealLayer`; base FLUX.1-dev (non-commercial).

## How we run it (happy path)

1. Kaggle 2× T4; GGUF Q4_K_S FLUX.1-dev re-classed into RevealLayer's transformer; encoders/VAE from `ostris/Flex.1-alpha`; `transformers==4.57.1`; bundled diffusers 0.34.
2. Patches: ViT-B/32 path, attention-mask fill −1e4 (not −1e9 / finfo.min), guarded refiner residual, fp32 VAE decode, fp32 inputs to the transparent decoder; `fp16_mixed` on the transformer.
3. 512 px, 12 steps, `true_gs` 1.0, boxes in pixels.
4. Generator: [`../../prototype/tools/make_reveallayer_trial.py`](../../prototype/tools/make_reveallayer_trial.py).

## Benchmarks

| Setting | Time | Peak | Result |
|---|---|---|---|
| forest, 3 boxes, 512 px, 12 steps, true_gs 1 | **134 s** (+ about 4 min load) | 11.4 / 11.2 GB | background with hidden areas filled + 3 clean layers (pines, bare trees, campfire + chair) |

Results: [`../../prototype/kaggle/results_reveallayer_v11/`](../../prototype/kaggle/results_reveallayer_v11/).

## Code facts relevant to fp16 (from the research notes)

| Hazard in RevealLayer's code | Our fix |
|---|---|
| ROI attention mask filled with −1e9 (later `finfo.min`) → −inf in fp16 | fill with −1e4 |
| `layer_pe` kept fp32 and added to fp16 hidden states | cast to fp16 at load |
| refiner `delta` residual unguarded | fp32 add + clamp |
| VAE decode in fp16 (authors' comment: "odd purple output") | fp32 VAE, inputs cast to fp32 |
| hard-coded ViT-B/32 weight path | download + patch path |
| refiner loader ignores missing keys | check the load message |

The official pipeline runs only in bf16 at 1024 px, 30 steps. A ComfyUI port (Ltamann/ComfyUI-TBG-RevealLayer, Nov 2026) accepts fp16/bf16/fp32 FLUX but not GGUF. Full notes: [flux_layered_t4_and_weights.md](../../research_notes/Layered%20loop%20open%20paths/flux_layered_t4_and_weights.md).

## Best features

- Fills in what was behind each object: the background stays complete when an object is moved.
- Layers follow the given boxes exactly (controllable grouping, e.g. from the LLM breakdown).

## Weak spots

Slow (about 2 min per split); needs boxes; heavy setup; FLUX.1-dev licence.

## Status

Works on 2× T4 since run 11 (forest). Fox & cottage: runs 12–13 failed on fp16 inputs to the now-fp32 scene VAE; run 14 fixed that but ran out of memory in the fp32 decode (5 boxes → 7 frames); run 15 offloads the transformer before decoding.

## Next attempts

fox & cottage with LLM boxes; `true_gs` 3.5 / 16 steps; second scene; 768 px if memory allows.

## Sources

[`../../research_notes/Layered methods deep dive/flux_models_t4_adaptation.md`](../../research_notes/Layered%20methods%20deep%20dive/flux_models_t4_adaptation.md); [`../../research_notes/Layered loop open paths/flux_layered_t4_and_weights.md`](../../research_notes/Layered%20loop%20open%20paths/flux_layered_t4_and_weights.md); [../log.md](../log.md) Phases 6–7.
