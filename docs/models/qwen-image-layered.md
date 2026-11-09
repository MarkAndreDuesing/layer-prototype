# Qwen-Image-Layered
> **Role: current factsheet.** Template: What it does · Licence & weights · How we run it (happy path) · Benchmarks · Best features · Weak spots · Status · Next attempts · Sources. Verdicts are the user's.

## What it does

Splits one finished image into N full-canvas RGBA layers (N chosen per run, e.g. 3–8) that composite back to the input; hidden areas behind objects are filled. 20B MMDiT, image → layers (not text → layers).

## Licence & weights

Apache-2.0, [Qwen/Qwen-Image-Layered](https://huggingface.co/Qwen/Qwen-Image-Layered). We use a GGUF Q4_K_S transformer (12.4 GB) plus the original text encoder and VAE, cached as public Kaggle notebook outputs (`markandreduesing/qwen-layered-weights-transformer`, `-base`).

## How we run it (happy path)

1. Kaggle 2× T4, transformer split over both GPUs (blocks 0–29 / 30–59), `fp16_mixed` applied ([../concepts/fp16-on-t4.md](../concepts/fp16-on-t4.md)).
2. **384 px, 4 layers, 4 steps, `true_cfg=1.0`, `offload=False`** → 27–35 s per split.
3. Code: [`../../prototype/layerproto/qwen_layered.py`](../../prototype/layerproto/qwen_layered.py) (`decompose`, `recompose`, `cached_encode`, `HeunVelocity`); notebook generator [`../../prototype/tools/make_qwen_layered_batch.py`](../../prototype/tools/make_qwen_layered_batch.py); step-by-step guide [../guides/qwen-on-kaggle.md](../guides/qwen-on-kaggle.md).

## Benchmarks

| Setting | Time per split | Recomposite diff | Notes |
|---|---|---|---|
| 640 px, 4 layers, 50 steps, CFG 4 (defaults) | 35 min (42 s/step) | 4.1 | reference quality |
| 512 px, 3 layers, 12 steps, CFG 1 | 139 s | — | same split as CFG 4 |
| 384 px, 3 layers, 3 steps, CFG 1 | 20 s | — | softer edges |
| **384 px, 4 layers, 4 steps, CFG 1** | **27–35 s** | about 7 | recommended |
| 512 px, 4 layers, 4 steps, CFG 1 | 48 s | — | sharper |
| 384 px, 4 layers, 4 **Heun** steps | 58 s | 7.0 | 2 calls per step |
| 384 px, **2 layers**, 4 steps | 25.7 s | 5.9–6.5 | background + one foreground layer |

Peak memory 8.0 / 7.6 GB (fast settings). Measured on the forest, pixel-art and sketch scenes (sweeps v3–v7, [`../../prototype/kaggle/results_qwen_sweep_v5/`](../../prototype/kaggle/results_qwen_sweep_v5/), [`results_qwen_v7_layers/`](../../prototype/kaggle/results_qwen_v7_layers/)). Fox & cottage: pending.

## Best features

- Most meaningful semantic layers of all routes tried so far (background, tree groups, campfire group, characters).
- Layers recomposite to the input almost exactly; background filled behind objects.
- Works across painterly, pixel-art and sketch styles.

## Weak spots

- Sometimes groups unrelated objects into one layer (skeleton + tree); thin or low-contrast subjects may stay in the background at 3 layers.
- At 384 px detail is limited; old object positions may leave faint ghosts in the background layer.

## Status

Working; main decomposer. Heun and 2-layer variants measured (v9). Fox & cottage sweep queued.

## Next attempts

Heun at 4–8 steps; 2 layers for speed; fox & cottage benchmark; text-encoder cache in the batch notebook; TPU route ([../research/hosting.md](../research/hosting.md#tpu)).

## Sources

Paper arXiv 2512.15603; history in [../log.md](../log.md) Phases 3–5; notes [`../../research_notes/Coherent layered generation pipeline/qwen_layered_and_layerdiffuse_papers.md`](../../research_notes/Coherent%20layered%20generation%20pipeline/qwen_layered_and_layerdiffuse_papers.md).
