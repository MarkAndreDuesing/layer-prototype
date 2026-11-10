# Running models on free Kaggle GPUs

> **Role: guide (medium).** Step by step: set up the CLI once, run any trial, and the Qwen-Image-Layered specifics. Why the fp16 patch is needed: [../concepts/fp16-on-t4.md](../concepts/fp16-on-t4.md). Settings and results per model: [../models/](../models/README.md).

## Public links
| What | Link |
|---|---|
| Code | https://github.com/MarkAndreDuesing/layer-prototype |
| Qwen-Image-Layered batch notebook | https://www.kaggle.com/code/markandreduesing/qwen-layered-batch |
| Cached weights (attach as notebook inputs) | https://www.kaggle.com/code/markandreduesing/qwen-layered-weights-transformer (Q4_K_S GGUF, 12.4 GB) · https://www.kaggle.com/code/markandreduesing/qwen-layered-weights-base (text encoder, VAE, configs, 16.9 GB) |

## 1. One-time setup
1. Install the Kaggle CLI in its own venv:
   ```bash
   python -m venv kagglecli
   ```
   ```bash
   kagglecli\Scripts\pip install kaggle
   ```
2. Create an API token at https://www.kaggle.com/settings (API → Create New Token) and save it as `%USERPROFILE%\.kaggle\kaggle.json`. Phone-verify the account (needed for GPU and internet).

## 2. Run any trial
Every model has a notebook generator `prototype/tools/make_<name>.py` that writes `prototype/kaggle/upload_<name>/` (notebook + `kernel-metadata.json`).
1. Generate the notebook:
   ```bash
   fastsdcpu\env\Scripts\python.exe prototype\tools\make_qwen21_rgba_trial.py
   ```
2. If you use your own account, change `id` in `kernel-metadata.json` to `<your-username>/<notebook>`.
3. Push it from inside its folder (the CLI has a path bug otherwise):
   ```bash
   cd prototype\kaggle\upload_qwen21_rgba_trial && python -m kaggle.cli kernels push -p .
   ```
4. Check the status, then download the outputs (`--file-pattern` skips cloned repos):
   ```bash
   python -m kaggle.cli kernels status markandreduesing/qwen21-rgba-trial
   ```
   ```bash
   python -m kaggle.cli kernels output markandreduesing/qwen21-rgba-trial -p prototype/kaggle/results_qwen21_vN
   ```
5. Record it: add a row to `prototype/kaggle/benchmarks.csv`, run `python prototype/tools/build_benchmarks.py`, update the factsheet and [../status.md](../status.md), append to [../log.md](../log.md).

| Generator | Model | Notes |
|---|---|---|
| `make_qwen_layered_batch.py` | Qwen-Image-Layered, Stable-Layers, Lightning | SWEEP cell lists the configs; embeds `qwen_layered.py` via `make_qwen_layered_test.py` (re-run that first after changing the module) |
| `make_reveallayer_trial.py` | RevealLayer | reads boxes from `prototype/test_scene/fox_cottage.json` |
| `make_artplus_trial.py` | ART+ | weights from Drive, HF mirror fallback |
| `make_layerdiffuse_flux_trial.py` | LayerDiffuse-Flux | one T4 is enough |
| `make_qwen21_rgba_trial.py` | Qwen-Image-2.1 | texture-fix VAE |
| `make_flux_control_trial.py` | plain FLUX.1-dev | fp16 sanity check |
| `make_segment_inpaint_v2.py` (v1: `make_segment_inpaint_trial.py`) | SAM2 + LaMa | v2: one mask per named object (hand boxes / Grounding DINO / automatic) on fox & cottage |
| `make_layerd_v2.py` (v1: `make_layerd_trial.py`) | LayerD | CPU notebook, no GPU quota; v2 sweeps 5 variants |
| `make_test_scene.py` | shared test scene | SDXL-Turbo on the laptop CPU, about 2.5 min per image |
| `make_qwen_relocate_trial.py`, `make_relocate_trial.py` | `relocate()` | uses the `qwen-v7-layers` dataset |
| `make_tpu_probe.py` | TPU probe | CLI runs got no TPU so far |

## 3. Qwen-Image-Layered specifics
- **Recommended settings:** 384 px, 4 layers, 4 steps, `true_cfg=1.0`, `offload=False` → 27–35 s per split. Defaults (640 px, 50 steps, CFG 4) take 35 min. Options: `sampler="heun"`, `layers=2`, `cached_encode()`.
- **Code:** [`qwen_layered.py`](../../prototype/layerproto/qwen_layered.py) (load, encode, decompose, recompose) + [`fp16_mixed.py`](../../prototype/layerproto/fp16_mixed.py).
- **Weights:** the two public weight notebooks are attached via `kernel_sources`. To use your own copies, push `kaggle/upload_qwen-layered-weights-transformer` and `-base` (CPU only, no GPU quota) and change the usernames in both generators.
- **Debug cell by cell:** Kaggle editor → File → Import Notebook → `kaggle/upload_qwen_test/qwen-layered-feasibility.ipynb`, Accelerator GPU T4 x2. Re-importing keeps the kernel alive; the toolbar's circular-arrow button restarts the whole session.
- **Gotchas:** reading the text encoder straight from `/kaggle/input` stalls at 0/729 tensors (copy to `/tmp` first, about 100 s); the transformer is split over both T4s (blocks 0–29 / 30–59, peak 8.8 / 8.4 GB); T4 speeds: fp16 19 TFLOPS, bf16 2 (emulated), fp32 3.7.

## 3b. Serve the app's /decompose route from Kaggle (your steps)

The notebook is built but not pushed (pushing would start a run and use GPU quota). Its models (Grounding DINO + SAM2 + LaMa) were tested through the real route: 5–12 s per image after a 45 s load ([../models/sam2-lama.md](../models/sam2-lama.md)).

1. On the laptop run `fastsdcpu\env\Scripts\python.exe prototype	ools\make_decompose_server_notebook.py` (already done; the file is [decompose-server.ipynb](../../prototype/kaggle/decompose_server/decompose-server.ipynb)).
2. Kaggle → New Notebook → File → Import Notebook → that file. Settings: Accelerator **GPU T4 x1**, Internet **On**.
3. Run all cells (about 3 min). The 4th cell prints two lines.
4. Copy them into `prototype/config.json`: `{"remote_url": "https://….trycloudflare.com", "remote_token": "…"}`. Keep `image_backend` as it is: the decomposer uses `remote_url` on its own.
5. Start the app, open the tab **Decompose an image (GPU)**, upload an image, method `sam2-lama`, type the object names (comma separated) and press Decompose.
6. Stop the Kaggle notebook when done (the tunnel and the GPU session end with it).

## 4. Rules learned
| Rule | Why |
|---|---|
| `machine_shape: NvidiaTeslaT4` | gives 2× T4 |
| at most **2 batch GPU sessions** at once | a third push fails ("Maximum batch GPU session count of 2 reached"); CPU notebooks don't count |
| decode in fp32 with the big transformer offloaded first | fp32 VAE decodes of many layers ran out of memory next to the transformer (ART+, RevealLayer) |
| keep notebooks well under 1.5 MB | larger ones are rejected (HTTP 400): embed images as JPEG or attach data |
| 25-min SIGALRM cap in every trial | a stall still ends with a log; cancelled runs keep no output |
| check kaggle.com/work/code for an outage banner before re-pushing | the 2026-11-09 outage looked like account problems |
| smoke-test with 2 steps; verify locally with tiny models | GPU quota is 30 h/week |
| TPU via CLI (`TpuV5E8`) | stored, but batch runs got no TPU ([../research/hosting.md](../research/hosting.md#tpu)) |
