# Qwen-Image-2.1 (RGBA)
> **Role: current factsheet.** Template: What it does · Licence & weights · How we run it (happy path) · Benchmarks · Best features · Weak spots · Status · Next attempts · Sources. Verdicts are the user's.

## What it does

7B text-to-image and editing model with a native 4-channel (RGBA) VAE: generates transparent images, **edits a transparent layer**, **extracts a subject** from an image, local and multi-reference edits. No multi-layer split mode.

## Licence & weights

Qwen Research licence (non-commercial). `Qwen/Qwen-Image-2.1(-Turbo)`; GGUF from `unsloth/Qwen-Image-2.1-Turbo-GGUF` (Q4_K_S 3.91 GB); fp16-safe VAE `madebyollin/texture-fix-vae-for-qwen-image-2.1` (the stock VAE gives NaNs in fp16 on transparent pixels).

## How we run it (happy path)

1. Kaggle 2× T4; diffusers `main`, transformers ≥ 5.17.
2. Turbo GGUF Q4_K_S transformer on GPU 0 + dequantise its non-Linear GGUF parameters + `fp16_mixed`; Qwen3-VL text encoder fp16 split over both GPUs; texture-fix VAE.
3. 512 px, 8 steps (Turbo schedule), CFG 1. Transparent output: wrap the prompt as "This is an RGBA image with transparency. … The image has alpha channel and the background is transparent." Edits: pass the image(s) as `image=[...]`.
4. Generator: [`../../prototype/tools/make_qwen21_rgba_trial.py`](../../prototype/tools/make_qwen21_rgba_trial.py).

## Benchmarks

| Task (512 px, Turbo) | Time | Result |
|---|---|---|
| Text → transparent object | 17–23 s | 37% transparent / 50% opaque; dark smoky halo |
| Edit transparent layer ("blue flames") | 23 s | edit applied, alpha kept |
| Extract object from a scene | 22 s | clean layer at its original position |
| Text → opaque scene | 18 s | good |
| Scene-wide edit ("rainy night") | 22 s | composition kept, effect subtle |

Load 247 s; peak 9.6–10.7 GB. Results: [`../../prototype/kaggle/results_qwen21_v2/`](../../prototype/kaggle/results_qwen21_v2/).

## Best features

- One model covers three loop steps: new layer from text, re-prompt a layer keeping transparency, cut an object out of a scene.
- Fast (about 20 s) and follows prompts well; candidate base-image generator.

## Weak spots

Halo around generated transparent objects; scene-wide edits are subtle; alpha reported as near-binary by an API provider; non-commercial licence.

## Status

Works on 2× T4 (run 2).

## Next attempts

Stronger event edits (prompt wording, more steps); halo fix (prompt, matte clean-up); use as base-image generator for fox & cottage; re-prompt Qwen-Image-Layered layers.

## Sources

[`../../research_notes/Layered loop open paths/qwen_image_21_rgba.md`](../../research_notes/Layered%20loop%20open%20paths/qwen_image_21_rgba.md); [../log.md](../log.md) Phase 7.
