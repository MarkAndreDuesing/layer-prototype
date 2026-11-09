# Qwen-Image-Layered on free Kaggle GPUs (2× T4)

> **Role: guide.** How to run Qwen-Image-Layered on Kaggle. Why the patch is needed: [../concepts/fp16-on-t4.md](../concepts/fp16-on-t4.md); results and settings: [../models/qwen-image-layered.md](../models/qwen-image-layered.md).

Splits one image into RGBA layers with [Qwen/Qwen-Image-Layered](https://huggingface.co/Qwen/Qwen-Image-Layered) (Apache-2.0) on Kaggle's free "GPU T4 x2".

**Status (2026-11-09):** works, no NaNs.
- **Recommended fast settings: 384 px, 4 layers, 4 steps, CFG 1 (`true_cfg=1.0`), `offload=False` → 27–35 s per split** on the free 2× T4 (sweeps v3–v7, [findings log](../log.md), Phase 4).
- The defaults (640 px, 50 steps, CFG 4) take 42 s per step, 35 min per split.
- Options in `qwen_layered.py`: `sampler="heun"` (exact Heun via `HeunVelocity`, 2 model calls per step), `layers=2`, and `cached_encode()` to skip the text encoder for repeated captions.

## Public links

- Code: https://github.com/MarkAndreDuesing/layer-prototype
- Batch notebook: https://www.kaggle.com/code/markandreduesing/qwen-layered-batch
- Cached weights (attach as notebook inputs): https://www.kaggle.com/code/markandreduesing/qwen-layered-weights-transformer and https://www.kaggle.com/code/markandreduesing/qwen-layered-weights-base

## Why it needs a patch

Qwen-Image is built for **bf16**. The T4 has no bf16, and in fp16 the model's internal values exceed the fp16 limit (65504), giving all-NaN output. ComfyUI also lists Qwen-Image as bf16/fp32 only.

[`layerproto/fp16_mixed.py`](../../prototype/layerproto/fp16_mixed.py) fixes this. It keeps the fast fp16 matrix multiplications but scales their inputs into a safe range, and runs everything else in fp32. Apply it with one call after loading the transformer:

```python
from layerproto.fp16_mixed import apply_fp16_mixed
apply_fp16_mixed(transformer)
```

## Pieces

| What | Where |
|---|---|
| Pipeline code (load, encode, decompose, recompose) | [`layerproto/qwen_layered.py`](../../prototype/layerproto/qwen_layered.py) |
| fp16 → bf16 workaround | [`layerproto/fp16_mixed.py`](../../prototype/layerproto/fp16_mixed.py) |
| Notebook generator: debugging version, cell by cell in the Kaggle editor | [`tools/make_qwen_layered_test.py`](../../prototype/tools/make_qwen_layered_test.py) → `kaggle/upload_qwen_test/` |
| Notebook generator: unattended batch run | [`tools/make_qwen_layered_batch.py`](../../prototype/tools/make_qwen_layered_batch.py) → `kaggle/upload_qwen_batch/` |
| Cached weights (Kaggle notebook outputs, attached as inputs) | `markandreduesing/qwen-layered-weights-transformer` (Q4_K_S GGUF, 12.4 GB), `markandreduesing/qwen-layered-weights-base` (text encoder, VAE, configs, 16.9 GB) |
| Notebook that builds those caches | `kaggle/upload_qwen-layered-weights-*` |
| Last good results | `kaggle/results_qwen_v7_layers/` (fast settings, every RGBA layer), `kaggle/results_qwen_sweep_v3`–`v6`, Stable-Layers `kaggle/results_stable_layers_v8/` |

## Run it yourself

1. Install the Kaggle CLI in its own venv and sign in:
   ```bash
   python -m venv kagglecli && kagglecli\Scripts\pip install kaggle
   ```
   Create an API token at https://www.kaggle.com/settings (API → Create New Token) and save it as `%USERPROFILE%\.kaggle\kaggle.json`.
2. The two weight notebooks are public, so the batch notebook can attach them as they are. If you prefer your own copies, build them yourself: push `kaggle/upload_qwen-layered-weights-transformer` and `-base` under your username (CPU only, no GPU quota). Then change the `kernel_sources` usernames in both generators.
3. Generate the notebooks (any Python with only the standard library works):
   ```bash
   python prototype/tools/make_qwen_layered_test.py
   ```
   ```bash
   python prototype/tools/make_qwen_layered_batch.py
   ```
4. In `prototype/kaggle/upload_qwen_batch/kernel-metadata.json`, change `id` to `<your-username>/qwen-layered-batch`.
5. Push and run it (the GPU time depends on the sweep in the SWEEP cell; the fast settings take well under a minute per split, plus about 5 min of loading):
   ```bash
   cd prototype/kaggle/upload_qwen_batch && python -m kaggle.cli kernels push -p .
   ```
6. Check its status, then download the results:
   ```bash
   python -m kaggle.cli kernels status <your-username>/qwen-layered-batch
   ```
   ```bash
   python -m kaggle.cli kernels output <your-username>/qwen-layered-batch -p results
   ```
   You get `layer_*.png`, `recomposite.png`, `strip.png` and `stats.json`.

To debug cell by cell instead, use the Kaggle editor: **File → Import Notebook** → `kaggle/upload_qwen_test/qwen-layered-feasibility.ipynb`, set **Settings → Accelerator → GPU T4 x2**, and run the cells one at a time. Re-importing keeps the kernel alive. The toolbar's circular-arrow button restarts the whole session.

## Known gotchas

- **Text encoder load stalls at 0/729 tensors** when read straight from `/kaggle/input`. The notebook copies the base weights to `/tmp` first, which takes about 100 s.
- **Memory:** the transformer is split across both T4s (blocks 0–29 on GPU 0, blocks 30–59 on GPU 1). Peak use is 8.8 / 8.4 GB.
- **Measured speeds on the T4:** fp16 19 TFLOPS, bf16 2 (emulated), fp32 3.7.
