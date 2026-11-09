# Faster image generation on the CPU laptop (researched 2026-11-09)

> **Role: research (reference).** Ways to speed up the CPU laptop; measured CPU numbers are in [../log.md](../log.md).

Ways to cut the time per image on the team laptop (Intel Core 7 150U, 15.7 GB RAM, Iris Xe, no dedicated GPU). Part A covers the FastSD CPU paths from Phase 1; Part B covers the LayerDiffuse chain from Phase 2. Nothing here has been measured on the laptop yet. **[est]** = estimate; **[U]** = unverified.

## Why the Phase 1 runs were slower than the published numbers

The research-time estimates in [Image generation models hosting and costs.md](../archive/2026-11-05%20Image%20generation%20models%20hosting%20and%20costs.md) §9 scaled FastSD CPU's fastest published timings. Those used OpenVINO, the tiny decoder and 1 step. The tested setup ([../docs/fastsd-cpu/USAGE.md](../guides/fastsd-cpu/USAGE.md)) used none of the three:

| Setting | Tested setup | Published fastest |
|---|---|---|
| Engine | LCM mode, PyTorch, float32 | OpenVINO (int8 for SDXL) |
| Decoder | Full VAE ("Use Tiny AutoEncoder" unticked) | TAESD / TAESDXL |
| Steps | 4 (Image to Image at Strength 0.8 runs 3) | 1 |

FastSD CPU's own benchmark (Core i7-12700, 512 px, 1 step):

| Model | PyTorch | OpenVINO | OpenVINO + tiny decoder |
|---|---|---|---|
| SDXS-512-0.9 | 4.8 s | 3.8 s | 0.82 s |
| SD-Turbo | 7.8 s | 5 s | 1.7 s |
| SDXL-Turbo | 10 s | 5.6 s | 2.5 s |

The laptop's CPU is about 2.2× slower than the i7-12700 (Cinebench R23 multi-core 8,883 vs 19,714). Measured on the laptop: about 20 s (SD-Turbo) and 58 s (SDXL-Turbo) per image.

## A. FastSD CPU (Phase 1 paths)

Ordered from least to most effort. Each step can be tested alone with the same prompt and seed, then compared against the Phase 1 favourites in [../experiments/favourites/](../../experiments/favourites/).

| # | Change | How | Expected effect | Risks / unknowns |
|---|---|---|---|---|
| A1 | **Tiny decoder** | Tick **Use Tiny AutoEncoder** in Generation Settings | FastSD CPU states about 1.4× faster and about 2 GB less RAM | "Moderate" quality loss. Its weights (TAESD / TAESDXL) must be downloaded once for offline use [U] |
| A2 | **Fewer steps** | Text to Image: Inference Steps 1–2. Image to Image: Steps 2 at Strength 0.8 runs 1 real step (`int(2 × 0.8) = 1`) | The UNet passes drop from 3–4 to 1–2, so roughly 2–3× [est] | Turbo models are made for 1 step, but Phase 1 never tested fewer than 4 steps |
| A3 | **OpenVINO mode** | Select OpenVINO in the app and use `rupeshs/sd-turbo-openvino` or `rupeshs/sdxl-turbo-openvino-int8` (+ `rupeshs/taesdxl-openvino`) | Published: SD-Turbo 7.8 → 5 s, SDXL-Turbo 10 → 5.6 s; with the tiny decoder 1.7 s and 2.5 s. On the laptop about 4 s and 6 s at 1 step [est] | The int8 SDXL model is 4.4 GB instead of about 10 GB, which also relieves RAM. Image to Image in OpenVINO mode is not stated on the model card [U]. For offline use, download the folders as in INSTALL B1/C2 and list them in `configs/openvino-lcm-models.txt` [U]. Same Stability non-commercial research licence |
| A4 | **Iris Xe iGPU** | Start with `set DEVICE=GPU` (OpenVINO only) | Unknown. Reports on Intel laptops are mixed: sometimes slower than the CPU; the first run compiles | Test once against A3 on the CPU |
| A5 | **Other 1–2-step models** (FastSD CPU OpenVINO list) | `rupeshs/hyper-sd-sdxl-1-step-openvino-int8`, `rupeshs/SDXL-Lightning-2steps-openvino-int8`, SDXS-512-0.9 | Published at 768 px: Hyper-SD SDXL 1 step 6.3 s with TAESDXL; SDXL-Lightning 2 steps 10 s. SDXS 0.82 s at 512 px | SDXS is text-to-image only. Quality against the Phase 1 favourites untested |
| A6 | **pix2pix-turbo** (img2img-turbo) | Separate tool: one-step sketch-to-image built on SD-Turbo; OpenVINO has an official notebook for it | 0.29 s on an A6000 (authors). No CPU figure published | Not part of FastSD CPU. Code MIT; SD-Turbo's licence applies to the weights. Directly matches the sketch + prompt goal, so it is worth one test |
| A7 | **stable-diffusion.cpp** | Separate C++ engine with quantized GGUF models, CPU or Vulkan | No laptop timing published. One Iris Xe report: SD 1.5 at 20 steps took 321 s via Vulkan | Different setup and model files; only worth it if A1–A5 disappoint |

**Suggested test order:** A1 → A2 → A3 (CPU) → A4 (iGPU) → A6. A1 and A2 need no new setup.

## B. LayerDiffuse chain (Phase 2 prototype)

The chain needs SDXL base at 20 steps and CFG 7 (on SDXL-Turbo the extract step failed, see [log.md](../log.md) Phase 2). That takes about 11 min per element on the laptop and about 18 s per pass on a Kaggle T4. OpenVINO and the FastSD CPU models do not apply, because the chain uses its own patched UNet and transparent decoder.

| # | Idea | Expected effect | Risks / unknowns |
|---|---|---|---|
| B1 | **Run on the Kaggle T4** (already built) | Measured: about 18 s per pass at 1024 px | Needs a session; already the main route |
| B2 | **Few-step LoRA on SDXL base**: LCM-LoRA SDXL (4 steps) or Hyper-SD SDXL LoRA | 20 → 4–8 steps, roughly 3–5× [est] | Whether `bg2ble` / `bgble2fg` keep working is undocumented. SDXL-Turbo broke extraction, so this may too. Test on one element first |
| B3 | **DeepCache** (`pip install DeepCache`, `DeepCacheSDHelper`) | Diffusers' docs: 1.9–3.2× on SD 2.1 at 50 steps (GPU). Supports `StableDiffusionXLPipeline` | No CPU figures. The chain's patched UNet forward may not be compatible. Gains shrink at few steps |
| B4 | **Lower CFG** | Guidance above 1 runs the UNet twice per step, so CFG 1 halves the passes | The released LayerDiffuse settings use CFG 7; quality at lower CFG untested |
| B5 | **Tiny decoder for the blend image only** | Saves one full VAE decode per element [est] | The alpha layer still needs LayerDiffuse's own transparent decoder; a lossy blend decode may hurt the extract step. Untested |

## Ruled out

- **Intel Extension for PyTorch (IPEX):** end-of-life since the end of March 2026; Intel's CPU optimisations are now in plain PyTorch.
- **bf16 on this CPU:** speedups need AMX or AVX-512 BF16, which a 15 W Core 7 150U is not expected to have [est]. Not worth testing before A1–A3.

## Sources

- [FastSD CPU README](https://github.com/rupeshs/fastsdcpu) (benchmarks, OpenVINO models, tiny decoder, `DEVICE=GPU`)
- [rupeshs/sd-turbo-openvino](https://huggingface.co/rupeshs/sd-turbo-openvino) · [FastSD CPU `configs/openvino-lcm-models.txt`](https://huggingface.co/spaces/rupeshs/fastsdcpu/blob/main/configs/openvino-lcm-models.txt)
- [OpenVINO in Diffusers](https://huggingface.co/docs/diffusers/v0.39.0/optimization/open_vino.md) · [Optimum Intel with diffusers](https://huggingface.co/docs/optimum/v1.23.3/intel/openvino/tutorials/diffusers)
- [pix2pix-turbo with OpenVINO (notebook)](https://docs.openvino.ai/2025/notebooks/sketch-to-image-pix2pix-turbo-with-output.html) · [img2img-turbo paper](https://arxiv.org/abs/2403.12036) · [img2img-turbo repo listing](https://awesome.ecosyste.ms/projects/github.com%2Fgaparmar%2Fimg2img-turbo) · [Sketch demo Space](https://huggingface.co/spaces/gparmar/img2img-turbo-sketch)
- [DeepCache in Diffusers](https://huggingface.co/docs/diffusers/optimization/deepcache)
- [SDXL in 4 steps with LCM LoRAs](https://huggingface.co/blog/lcm_lora) · [Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD)
- [Intel Extension for PyTorch (EOL notice)](https://pytorch-extension.intel.com/)
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) · [issue #1449 (Iris Xe)](https://github.com/leejet/stable-diffusion.cpp/issues/1449) · [OpenVINO issue #26391 (iGPU slower than CPU)](https://github.com/openvinotoolkit/openvino/issues/26391)
- [Intel Core 7 150U](https://nanoreview.info/en/cpu/intel-core-7-150u) · [Intel Core i7-12700](https://nanoreview.info/en/cpu/intel-core-i7-12700)
