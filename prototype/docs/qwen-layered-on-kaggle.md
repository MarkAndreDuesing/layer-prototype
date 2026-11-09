# Qwen-Image-Layered on free Kaggle GPUs (2× T4)

Splits one image into RGBA layers with [Qwen/Qwen-Image-Layered](https://huggingface.co/Qwen/Qwen-Image-Layered) (Apache-2.0) on Kaggle's free "GPU T4 x2".

**Status (2026-11-09):** works, no NaNs, 4 clean layers. It takes **42 s per step, so 35 min for the default 50 steps**. That is far too slow for gameplay; see [../../reports/research_speed_and_apis.md](../../reports/research_speed_and_apis.md) and [../../reports/research_tpu_route.md](../../reports/research_tpu_route.md).

## Public links

- Code: https://github.com/MarkAndreDuesing/layer-prototype
- Batch notebook: https://www.kaggle.com/code/markandreduesing/qwen-layered-batch
- Cached weights (attach as notebook inputs): https://www.kaggle.com/code/markandreduesing/qwen-layered-weights-transformer and https://www.kaggle.com/code/markandreduesing/qwen-layered-weights-base

## Why it needs a patch

Qwen-Image is built for **bf16**. The T4 has no bf16, and in fp16 the model's internal values exceed the fp16 limit (65504), giving all-NaN output. ComfyUI also lists Qwen-Image as bf16/fp32 only.

[`layerproto/fp16_mixed.py`](../layerproto/fp16_mixed.py) fixes this. It keeps the fast fp16 matrix multiplications but scales their inputs into a safe range, and runs everything else in fp32. Apply it with one call after loading the transformer:

```python
from layerproto.fp16_mixed import apply_fp16_mixed
apply_fp16_mixed(transformer)
```

## Pieces

| What | Where |
|---|---|
| Pipeline code (load, encode, decompose, recompose) | [`layerproto/qwen_layered.py`](../layerproto/qwen_layered.py) |
| fp16 → bf16 workaround | [`layerproto/fp16_mixed.py`](../layerproto/fp16_mixed.py) |
| Notebook generator: debugging version, cell by cell in the Kaggle editor | [`tools/make_qwen_layered_test.py`](../tools/make_qwen_layered_test.py) → `kaggle/upload_qwen_test/` |
| Notebook generator: unattended batch run | [`tools/make_qwen_layered_batch.py`](../tools/make_qwen_layered_batch.py) → `kaggle/upload_qwen_batch/` |
| Cached weights (Kaggle notebook outputs, attached as inputs) | `markandreduesing/qwen-layered-weights-transformer` (Q4_K_S GGUF, 12.4 GB), `markandreduesing/qwen-layered-weights-base` (text encoder, VAE, configs, 16.9 GB) |
| Notebook that builds those caches | `kaggle/upload_qwen-layered-weights-*` |
| Last good result | `kaggle/results_qwen_batch_v2/` |

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
5. Push and run it (uses about 45 min of GPU quota):
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
