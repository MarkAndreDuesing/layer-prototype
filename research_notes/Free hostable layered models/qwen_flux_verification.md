# Qwen and FLUX Layer-Capable Models: Verification (as of 2026-11-07)

## Qwen-Image-Layered: repo, license, size, VRAM, quants, T4 feasibility, hosted price

### Takeaway
The model is at https://huggingface.co/Qwen/Qwen-Image-Layered. It is Apache-2.0 licensed, has about 20B parameters, and has a diffusers `QwenImageLayeredPipeline`. It is image-to-layers only, and the card says nothing about text-to-layers (T2L). Full BF16 does not fit a 16 GB T4. A 4-bit SDNQ or GGUF quant with CPU offload is the only realistic free route, and nobody has shown it working.

### Cited Findings
- License: Apache 2.0. Size: about 20B parameters in BF16. Load it with `QwenImageLayeredPipeline.from_pretrained("Qwen/Qwen-Image-Layered")`. It needs diffusers built from GitHub and transformers >=4.51.3. The card recommends 4 layers at 640 resolution. It gives no VRAM figure — [HF model card](https://huggingface.co/Qwen/Qwen-Image-Layered)
- The card takes an RGBA image as input and accepts an optional caption. It does not describe text-to-layers generation — [HF model card](https://huggingface.co/Qwen/Qwen-Image-Layered)
- Paper: arXiv 2512.15603, dated 17 Dec 2025 — [HF model card](https://huggingface.co/Qwen/Qwen-Image-Layered)
- The model tree lists 7 quantized repos: unsloth/Qwen-Image-Layered-GGUF, T5B/Qwen-Image-Layered-FP8, zimengxiong/Qwen-Image-Layered-6bit, Disty0/Qwen-Image-Layered-SDNQ-uint4-svd-r32, llmrobot/qwen-image-layered-int8-sdnq, OzzyGT/Qwen_Image_Layered_sdnq_dynamic_4bit and OzzyGT/Qwen_Image_Layered_sdnq_dynamic_8bit — [HF quantized listing](https://huggingface.co/models?other=base_model:quantized:Qwen/Qwen-Image-Layered)
- Comfy-Org repackages the model as bf16 and fp8mixed diffusion models plus a VAE. This came from a search snippet; I did not open the page — [Comfy-Org/Qwen-Image-Layered](https://huggingface.co/Comfy-Org/Qwen-Image-Layered)
- DataCamp tried it on a Colab T4. The session ran out of RAM before the download finished. The article cites more than 57 GB of weights — [DataCamp tutorial, via search snippet](https://www.datacamp.com/sv/tutorial/qwen-image-layered)
- A community guide claims Q4 GGUF needs about 8 GB VRAM and Q4_K_M is about 3.2 GB. That file size is implausible for a 20B model, where Q4 would be about 11-12 GB, so treat the claim as unreliable — [DEV Community guide](https://dev.to/gary_yan_86eb77d35e0070f5/how-to-use-qwen-image-layered-gguf-in-comfyui-complete-installation-and-usage-guide-2en3)
- fal.ai price: $0.05 per image. The page is marked for commercial use — [fal.ai model page](https://fal.ai/models/fal-ai/qwen-image-layered)
- Replicate has a `qwen/qwen-image-layered` page (Apache 2.0). The fetched page showed no price or hardware — [Replicate](https://replicate.com/qwen/qwen-image-layered)

### Inferences
- About 20B parameters in BF16 is roughly 40 GB of transformer weights. That matches the "40-45 GB" reports. The "13 GB" figure most likely refers to a quantized build or to FLUX.2 Klein, which is about 13 GB.
- Colab free (about 12 GB RAM) is effectively ruled out because the RAM is too small to stage the weights.
- Kaggle (about 29 GB RAM, T4 16 GB) might run a 4-bit SDNQ or Q4 GGUF transformer with CPU offload of the Qwen2.5-VL text encoder. Expect minutes per decomposition. This has not been tested.

### Gaps
- No published measured VRAM or timing on a T4/P100 for any quant.
- Text-to-layers weights: the card mentions none, and I found no release. Not confirmed.
- I did not find a Replicate price. The page did not render one.
- I did not open the unsloth Layered GGUF file sizes.

## Qwen-Image-2.1: existence, license, RGBA, size, quants, T4

### Takeaway
It exists at https://huggingface.co/Qwen/Qwen-Image-2.1. The license is the non-commercial Qwen Research License, not Apache. The official card confirms native RGBA output from text prompts. The generator is 7B, and GGUF quants exist. This is the most T4-feasible layer-capable Qwen model.

### Cited Findings
- License: "Qwen Research License Agreement" (metadata `qwen-research`). The generator is a 7B DiT with 32 single-stream layers. The card lists text-to-transparent (RGBA) generation, transparent layer editing and subject extraction. Its prompts state that the image has an alpha channel and a transparent background, and the example runs at 2048x2048 with 40 steps. It needs diffusers built from GitHub, transformers >=5.17 and torch >=2.4. It suggests `enable_model_cpu_offload()`. The model tree lists 172 quantized models. No release date is given — [HF model card](https://huggingface.co/Qwen/Qwen-Image-2.1)
- unsloth GGUF denoiser sizes: Q4_K_M 4.2 GB, Q5_K_M 5.39 GB, Q8_0 7.64 GB, F16 14.2 GB. The Qwen3-VL-8B text encoder at Q4 is about 5.0-5.15 GB, and a VAE is also needed. The page gives no VRAM figures — [unsloth/Qwen-Image-2.1-GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF)
- There is a ComfyUI GGUF repack — [vantagewithai](https://huggingface.co/vantagewithai/Qwen-Image-2.1-ComfyUI-GGUF)
- A blog claims it runs on 12 GB. This is secondary and I did not open it — [openclawdc](https://openclawdc.com/blog/can-i-run-qwen-image-2-1-locally/)

### Inferences
- Q8 denoiser (7.6 GB) plus Q4 text encoder (5 GB) should fit within a 16 GB T4, or run with the encoder offloaded. This should work on both Kaggle and Colab free. The T4 lacks bf16, so fp16 or GGUF is the safer path, and 2048 px will be slow. Use 1024 px.
- The non-commercial license matters if outputs are for commercial use.

### Gaps
- I could not find the exact release date on the card.
- No measured T4 timings.

## "FLUX 3 Image Klein": existence, license, size, transparency

### Takeaway
It exists. BFL announced it on 15 Oct 2026. It has 5.6B parameters and uses the FLUX Kommunity License, which allows free commercial use under $5M revenue. Neither the blog nor the card mentions transparency or layers.

### Cited Findings
- Blog dated 15 Oct 2026. It describes text-to-image, editing, up to 10 reference images, native 4K, bounding-box placement and pixel restoration. Base and distilled variants exist, plus FP8, INT8 and NVFP4 builds. Weights are 11.2 GB in BF16 and 5.6 GB in FP8. INT8 runs on 8 GB RTX GPUs. The blog does not mention alpha or layers — [BFL blog](https://bfl.ai/blog/flux-3-image-klein)
- HF repo (gated): the transformer has 5.6B parameters, and the text encoder is a separate Qwen3-VL-4B-Instruct (Apache-2.0). The license is FLUX Kommunity License v1.0. Peak memory at 1024 px is about 31 GB in BF16, 26 GB in FP8 and 25 GB in NVFP4. The optional 9B prompt upsampler is extra — [HF card](https://huggingface.co/black-forest-labs/flux-3-image-klein)

### Inferences
- The card's peak-memory figures exceed 16 GB, so a T4 would need INT8 plus CPU offload of the text encoder. FP8 and NVFP4 are not hardware-accelerated on a T4 (Turing).
- It is not a layer or transparency model.

### Gaps
- I did not check docs.comfy.org.
- No T4 reports.

## FLUX.2 Klein 4B

### Takeaway
Confirmed at https://huggingface.co/black-forest-labs/FLUX.2-klein-4B. It is Apache 2.0, has 4B parameters and needs about 13 GB VRAM. I found no transparency LoRA.

### Cited Findings
- Apache 2.0, a 4B rectified flow transformer, and "as little as 13GB VRAM". The card does not mention alpha or transparency — [HF card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- Runware's LayerDiffuse page says it supports FLUX Dev only. The page may be dated — [Runware blog](https://runware.ai/blog/introducing-layerdiffuse-generate-images-with-built-in-transparency-in-one-step)

### Inferences
- About 13 GB fits a T4, so it is feasible on Kaggle and Colab. Transparency would have to come from post-hoc background removal.

### Gaps
- I did not run a dedicated search for a Klein transparency LoRA. Its existence is unconfirmed.

## Z-Image Turbo transparency adapter

### Takeaway
I found no RGBA LoRA or LayerDiffuse-style adapter for Z-Image Turbo.

### Cited Findings
- Searches turned up only general Z-Image Turbo LoRA training and hosting pages, with no alpha output — [RunComfy](https://www.runcomfy.com/auth/trainer/ai-toolkit/z-image-turbo-lora-training), [Runware LayerDiffuse](https://runware.ai/models/layerdiffuse)

### Inferences
- Use background removal, or use Qwen-Image-2.1 for native RGBA.

### Gaps
- I did not search Civitai or GitHub directly.

## Existing Kaggle/Colab notebooks

### Takeaway
I found no verified Kaggle or Colab notebook for Qwen-Image-Layered. The one documented Colab T4 attempt failed with an out-of-memory (RAM) error.

### Cited Findings
- DataCamp's Colab T4 attempt ran out of RAM — [DataCamp](https://www.datacamp.com/sv/tutorial/qwen-image-layered)
- The "Colab/Kaggle" links on the HF cards are generic templates, not working notebooks — [Comfy-Org card](https://huggingface.co/Comfy-Org/Qwen-Image-Layered)

### Gaps
- I did not search for Qwen-Image-2.1, FLUX.2 Klein or FLUX 3 Klein notebooks. Kaggle's own search was not queried.
