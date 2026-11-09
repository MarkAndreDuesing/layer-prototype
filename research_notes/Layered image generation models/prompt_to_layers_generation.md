# Prompt-to-Multi-Layer (RGBA / Transparent) Image Generation — as of Nov 2026

Note: research run on 2026-11-07 with a limited tool budget (~12 calls). Items marked "(from prior knowledge, verify)" were not re-confirmed by a source fetched in this run.

## LayerDiffuse / LayerDiffusion and its FLUX ports

### Takeaway
LayerDiffuse (Zhang & Agrawala, 2024) is still the most mature open way to get a transparent layer from a text prompt. It works on SD1.5 and SDXL, and there is an open FLUX.1-dev port with a LoRA and a transparent VAE. Its fg/bg conditional modes are the closest match to "regenerate one layer while keeping the others fixed".

### Cited Findings
- The official repo is "Transparent Image Layer Diffusion using Latent Transparency" and lists ComfyUI support — [GitHub layerdiffusion/LayerDiffuse](https://github.com/layerdiffusion/LayerDiffuse)
- A ComfyUI node pack (huchenlei, comfyui-layerdiffuse) is listed as supported on Comfy Cloud — [comfy.org](https://comfy.org/cloud/supported-nodes/comfyui-layerdiffuse.md); example workflow at [RunComfy](https://runcomfy.com/comfyui-workflows/generating-transparent-images-with-layerdiffuse-in-comfyui)
- How it works (from prior knowledge, verify): it encodes alpha as a small offset in the latent space ("latent transparency"), so the base model is barely disturbed, and a transparent VAE decoder reconstructs RGBA. Released modes: foreground only; joint fg+bg; background given foreground; foreground given background. The SDXL models are the most complete, and SD1.5 has joint 2- and 3-layer attention-sharing models. Weights are on Hugging Face at [LayerDiffusion/layerdiffusion-v1](https://huggingface.co/LayerDiffusion/layerdiffusion-v1), with a Forge extension `sd-forge-layerdiffuse` — [GitHub](https://github.com/layerdiffusion/LayerDiffuse)
- **FLUX port (FireRedTeam / RedAIGC):** it trains a new transparent VAE adapted to Flux plus a LoRA that makes FLUX.1-dev produce transparent images. Weights are `layerlora.safetensors` and `TransparentVAE.pth` on HF [RedAIGC/Flux-version-LayerDiffuse](https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse). It includes `demo_t2i.py` and `demo_i2i.py` scripts — [GitHub FireRedTeam/LayerDiffuse-Flux](https://github.com/FireRedTeam/LayerDiffuse-Flux)
- The community ComfyUI wrapper ComfyUI_FluxLayerDiffuse (leeguandong) was last updated 2025-03-17 and has few stars — [RunComfy node page](https://www.runcomfy.com/comfyui-nodes/ComfyUI_FluxLayerDiffuse)

### Inferences
- For the game: generate the background normally, then generate each object with SDXL LayerDiffuse "fg conditioned on bg" (or the FLUX LoRA in fg-only mode), and composite. Regenerating one object then only needs a rerun of that layer.
- Known limits (prior knowledge): SD1.5/SDXL quality ceiling, and weak semi-transparency and hair in some modes. The FLUX port inherits FLUX.1-dev's non-commercial license.

### Gaps
- I could not confirm whether huchenlei's ComfyUI node supports FLUX, or what license and VRAM the FLUX port needs.
- No SD3, FLUX.2 or Z-Image LayerDiffuse port was found.

## Multi-layer generators (ART, PrismLayers, DreamLayer, LayerFusion, PSDiffusion, OmniPSD, Text2Layer, LayeringDiff)

> Merged into [joint_text_to_layers.md](../Layered%20methods%20deep%20dive/joint_text_to_layers.md#paper-facts-per-method) on 2026-11-10 (moved verbatim, with its citations).

## Qwen-Image-Layered, Qwen-Image-2.1 and other modern base models

### Takeaway
Qwen-Image-2.1 (released 2026-09-20) is the only mainstream base model found with **native RGBA output** from a prompt. Qwen-Image-Layered (Dec 2025, Apache 2.0) is mainly a decomposer of an image into a variable number of RGBA layers. No native or LoRA RGBA support was found for FLUX.2 Klein or Z-Image Turbo.

### Cited Findings
- **Qwen-Image-Layered**, arXiv 2512.15603 (CVPR 2026): uses an RGBA-VAE and VLD-MMDiT for variable-length layer decomposition. Trained in stages: text-to-RGBA, then text-to-multi-RGBA, then image-to-multi-RGBA. Qwen says text-to-multi-RGBA quality is limited because the model is tuned for decomposition — [arXiv](https://arxiv.org/html/2512.15603v1), [CVPR](https://openaccess.thecvf.com/content/CVPR2026/papers/Yin_Qwen-Image-Layered_Towards_Inherent_Editability_via_Layer_Decomposition_CVPR_2026_paper.pdf). It is Apache 2.0 and has ComfyUI support — [ComfyUI Wiki](https://comfyui-wiki.com/en/news/2025-12-19-qwen-image-layered-release). HF: [Qwen/Qwen-Image-Layered](https://huggingface.co/Qwen/Qwen-Image-Layered) (link from prior knowledge, verify)
- **Qwen-Image-2.1** (2026-09-20): a 7B DiT (32 single-stream layers) with a Qwen3-VL-8B encoder and a 64-channel RGBA autoencoder. It outputs RGB or RGBA depending on the prompt (the recommended phrase is like "This is an RGBA image with transparency"). It can also extract a subject from a photo as a transparent layer and edit transparent assets. ComfyUI templates exist. License is reported as the Qwen Research License (non-commercial), and sources conflict, so verify — [eWeek](https://www.eweek.com/news/alibaba-qwen-image-2-1-native-transparency/), [Cinevva](https://app.cinevva.com/news/2026-09-20-qwen-image-2-1-transparent-sprites), [ComfyUI Wiki](https://comfyui-wiki.com/en/news/2026-09-21-qwen-image-2-1), [Qwen blog](https://qwen.ai/blog?id=qwen-image-2.1)
- **FLUX.2 Klein:** comes in 4B and 9B versions. The 4B distilled model is Apache 2.0, and there is a base-4B checkpoint for LoRA training. No transparency adapter was found — [Civitai developer docs](https://developer.civitai.com/orchestration/recipes/flux2)

### Inferences
- The best fit for the game today: generate each object layer separately with Qwen-Image-2.1 RGBA (or SDXL/FLUX LayerDiffuse if commercial use matters). Generate the background as an opaque image. Then optionally run a low-denoise img2img or relight pass on the composite to harmonize lighting. Qwen-Image-Layered can split a generated full scene into layers as an alternative path.

### Gaps
- No evidence was found on "FLUX 3 Image Klein" or Z-Image Turbo transparency or layer LoRAs. The FLUX 3 name could not be confirmed at all.
- Qwen-Image-2.1's HF model card URL and VRAM needs were not fetched.
