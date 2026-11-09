# Heavy models and open paths: re-check, 2026-11-09 (evening)

Done while Kaggle was down. A banner on kaggle.com/work/code read: "issue preventing notebooks from starting or staying connected".

Purpose: replace the guesses marked [U] in `heavy_models_inference_plans.md` with facts read from the repos and from the Hugging Face API, and look into two open voice-note paths. All sizes come from the HF tree API.

## 1. Qwen-Image-2.1: now the most promising heavy model

Facts, from the [model card](https://huggingface.co/Qwen/Qwen-Image-2.1), the HF tree, and the diffusers source on `main`:

**Model and features**
- The visual generator is 7B, built from 32 single-stream DiT layers. The text encoder is **Qwen3-VL**, about 17.5 GB in bf16, stored in 4 shards.
- The model card lists these features:
  - text to RGBA
  - **editing transparent layers**
  - **extracting subjects from photographs**
  - up to 10 reference images
  - local edits marked by circles, painted annotations or masks
  - identity preservation
- Together these cover three steps of our loop in one model: make a new layer, re-prompt a layer, and split an object out of an image.

**Architecture and licence**
- The VAE (`AutoencoderKLQwenImage21`) has **4 input and output channels**, so RGBA is native and no separate transparency decoder is needed. It downsamples 16× spatially and has 64 latent channels. At 512 px that means only 32×32 = 1,024 image tokens.
- Licence: Qwen Research Licence, non-commercial. That is fine for this project.

**Inference**
- `QwenImage21Pipeline` is in diffusers `main` (`pipelines/qwenimage21`). `single_file_model.py` maps `QwenImage21Transformer2DModel`, so GGUF loading through `from_single_file` should work.
- The model card requires `transformers>=5.17`.
- The pipeline accepts RGBA input images. For the vision encoder it composites the alpha over white; the comment in the code says the model was trained that way.
- **Turbo:** `Qwen/Qwen-Image-2.1-Turbo` runs in **8 steps at CFG 1**. The schedule is saved in the checkpoint, and `num_inference_steps` does not override it; only an explicit `sigmas` does.
- **GGUF:** both `unsloth/Qwen-Image-2.1-GGUF` and `unsloth/Qwen-Image-2.1-Turbo-GGUF` exist:

| Quant | Size |
|---|---|
| Q2_K | 2.47 GB |
| Q4_K_S | **3.91 GB** |
| Q4_K_M | 4.20 GB |
| Q6_K | 6.27 GB |
| Q8_0 | 7.64 GB |
| F16 | 14.23 GB |

  Community repos also exist: Lightning, DMD2 2-step, ControlNet-Union, and assorted LoRAs.

**Kaggle plan**, ready to push as [tools/make_qwen21_rgba_trial.py](../../prototype/tools/make_qwen21_rgba_trial.py):
- Turbo Q4_K_S transformer with `fp16_mixed` on cuda:0.
- Qwen3-VL in fp16 split across both T4s with `device_map` (5 GiB on GPU 0, 14 GiB on GPU 1). The VAE also sits on cuda:0.
- 512 px, 8 steps, 25-minute cap.
- Five tests:
  1. text to RGBA campfire
  2. text to opaque scene
  3. edit the RGBA campfire ("blue flames"): does the alpha survive?
  4. extract the campfire from our forest scene
  5. scene-wide "rainy night" edit

**Risks**
- The text encoder in fp16 could overflow. Qwen2.5-VL in fp16 was fine for us.
- `output_resolution` (default 1024) may not combine with 512 px.
- diffusers `main` can change from day to day.
- The `_execution_device` override.

## 2. CLD: the code is now readable, and ART+ comes along with it

Sources: [CLD repo](https://github.com/monkek123King/CLD) (`infer/infer.py`, `infer/infer.yaml`, `README.md`) and [HF thuteam/CLD](https://huggingface.co/thuteam/CLD).

**Inputs and outputs**
- **Inputs:** a whole image (fed to the adapter), a caption, and layer boxes. Boxes are snapped to 16 px.
- **Pipeline call:** `CustomFluxPipelineCfgLayer`, with `adapter_conditioning_scale` 0.9 and CFG 4.0 by default.
- **Outputs:** an RGBA whole image, an RGBA background, and one RGBA layer per box, decoded by a transparent VAE (the same `AutoencoderKLTransformerTraining` class as RevealLayer).

**How the model is assembled**
- Base: FLUX.1-dev. Two LoRAs are **fused** into it: `pre_trained_LoRA` (the ART base) and `prism_ft_LoRA` (the PrismLayers fine-tune). A `layer_pe` and a trained "decouple" LoRA are added on top.
- The multi-layer adapter is `alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha` (4.28 GB), with its own small LoRA (75 MB) fused in.
- **File sizes:**

| File | Size |
|---|---|
| each transformer LoRA | 598 MB |
| `trans_vae/0008000.pt` | 1.02 GB |
| `layer_pe.pth` | 0.6 MB |

**What this means for ART+.** `pre_trained_LoRA` and `prism_ft_LoRA` appear to be the ART and ART+ (PrismLayersPro) LoRAs, now hosted on HF by the CLD authors. That would avoid the Google Drive quota and the questionable mirror. To confirm, compare their keys and shapes with the PrismLayers Drive files, or with the unofficial mirror `24yearsold/PrismLayers_ckpt`.

**Kaggle adaptation**, the same pattern as RevealLayer:
- Load the GGUF Q4_K_S transformer and swap its class to CLD's `CustomFluxTransformer2DModel`.
- Do **not fuse** the LoRAs; GGUF weights cannot be fused. Attach all three unmerged with `fp16_mixed.add_lora`, which costs about 1.8 GB of fp16 LoRA weights.
- Put the ControlNet adapter in fp16 on cuda:1, next to T5.
- 512 px, about 16 steps, at most 3 boxes.
- Expected memory: GPU 0 about 10–12 GB, GPU 1 about 14 GB with T5 + ControlNet. The ControlNet may need to be quantised, or T5 freed after encoding.
- **Not pushed yet:** test the RevealLayer port first, since both use the same code base.

## 3. OmniPSD, LayerDiffuse-Flux, RevealLayer

- OmniPSD: unchanged since the earlier check (code public, no weights).
- LayerDiffuse-Flux: notebook pushed (`layerdiffuse-flux-trial`), waiting for Kaggle.
- RevealLayer:
  - Run 5 works end to end: 69 s for 12 steps, peak 10.5 / 11.2 GB.
  - Its output was pure noise, because FLUX overflows fp16 in `FluxSingleTransformerBlock`.
  - Runs 6 and 7 add `fp16_mixed` and are waiting for Kaggle.
  - Later the same evening: three fp16 hazards were found in RevealLayer's own code (−1e9 attention-mask fill, unguarded refiner residual, fp16 VAE decode) and patched for run 8; a plain-FLUX control notebook was added. See `../Layered loop open paths/flux_layered_t4_and_weights.md`.

## 4. Open voice-note paths

**Harmonising placed objects (IC-Light)**
- [IC-Light](https://github.com/lllyasviel/IC-Light) v1 consists of SD1.5 checkpoints in two kinds: foreground plus text, and foreground plus background. Licence: OpenRAIL-M.
- v1 runs easily on a T4.
- The background-conditioned model fits our "relight a moved layer to match the scene" step, as a cheaper alternative or complement to the SDXL `relocate()` re-harmonisation.
- IC-Light v2 (FLUX): as far as we found, only a hosted demo; no weights released.
- It does not produce alpha. We would relight the RGB and keep the layer's own alpha.

**Scene-wide story events (rain, night, fire)**
- No layered paper we found applies one global edit consistently across all layers. The closest work:
  - [LASAGNA](https://arxiv.org/abs/2601.15507): foreground with shadows and reflections
  - [DreamLayer](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_DreamLayer_Simultaneous_Multi-Layer_Generation_via_Diffusion_Model_ICCV_2025_paper.pdf): global-image-guided harmonisation
  - [SceneCrafter](https://arxiv.org/pdf/2506.19488): weather and time of day, but for driving scenes and without layers
- This is another research gap. See G-new in `reports/Research gap and questions.md`.
- Training-free options:
  - (a) Edit the composite (Qwen-Image-2.1 or SDXL img2img), then re-split it with Qwen-Image-Layered.
  - (b) Edit each RGBA layer with the same instruction, using Qwen-Image-2.1's RGBA editing.
  - (c) Relight each layer against the edited background with IC-Light.
- Test (a) and (b) on the "rainy night" case once Qwen-Image-2.1 runs. The trial notebook already includes it.
