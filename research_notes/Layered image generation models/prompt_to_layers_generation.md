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

### Takeaway
Research methods (2023–2026) generate several RGBA layers in one pass from a global prompt, either with an anonymous layout (ART) or with per-layer prompts (PSDiffusion, DreamLayer, LayerFusion). Most give harmonized lighting because the layers are denoised jointly. Few have usable open weights: ART's were withdrawn.

### Cited Findings
- **ART (Anonymous Region Transformer)**, arXiv 2502.18364 (Feb 2025, CVPR 2025, Microsoft Research Asia): built on FLUX.1[dev], using 3D RoPE and region-wise attention. It is more than 12x faster than full attention and has fewer layer conflicts — [arXiv](https://arxiv.org/abs/2502.18364v1), [HF paper](https://huggingface.co/papers/2502.18364), [MSR](https://www.microsoft.com/en-us/research/?p=1144107)
  - Inputs: a global prompt plus an anonymous region layout (boxes with no per-layer captions), from an LLM planner or set by hand. It claims "50+ layers". The license is MIT, but **weights and code were removed on 2025-07-23** because the training data "may have come from illegal sources" — [GitHub microsoft/art-msra](https://github.com/microsoft/art-msra)
- **PrismLayers / PrismLayersPro**, arXiv 2505.22523 (May 2025): an open synthetic dataset of about 200K multi-layer images, with a 20K "Pro" subset. Images average about 7 layers (range 3–14). Built with LayerFLUX (single transparent layer) and MultiLayerFLUX (composes layers from a layout given by a reference image or an LLM). ART fine-tuned on it ("ART+") won 60% of user-study comparisons — [arXiv PDF](https://arxiv.org/pdf/2505.22523), [CatalyzeX](https://www.catalyzex.com/paper/prismlayers-open-data-for-high-quality-multi). Dataset: [HF artplus/PrismLayersPro](https://huggingface.co/datasets/artplus/PrismLayersPro) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2505.22523)
- **PSDiffusion**, arXiv 2505.11468 (WACV 2026): takes a global prompt plus layer-specific prompts (evaluated with 3 foreground + 1 background + global). Layer prompts can come from an LLM or the user. Layout is taken from cross-attention maps of a full RGB image generated together with the layers, which keeps the layers harmonized — [arXiv](https://arxiv.org/html/2505.11468), [WACV PDF](https://openaccess.thecvf.com/content/WACV2026/papers/Huang_PSDiffusion_Harmonized_Multi-Layer_Image_Generation_via_Layout_and_Appearance_Alignment_WACV_2026_paper.pdf)
- **DreamLayer**, arXiv 2503.12838 (ICCV 2025): generates transparent fg and bg layers simultaneously, using Context-Aware Cross-Attention, Layer-Shared Self-Attention and Information-Retained Harmonization. Also supports image-to-layer decomposition and latent editing — [arXiv](https://arxiv.org/html/2503.12838v1), [ICCV](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_DreamLayer_Simultaneous_Multi-Layer_Generation_via_Diffusion_Model_ICCV_2025_paper.html)
- **LayerFusion**, arXiv 2412.04460 (Dalva et al., Dec 2024): harmonized multi-layer text-to-image using generative priors on latent diffusion — [fugumt](https://fugumt.com/fugumt/paper_check/2412.04460v1_enmode)
- **LayeringDiff**: generates a composite image, then extracts layers from it, which avoids large-scale training — [fugumt summary](https://fugumt.com/fugumt/paper_check/2412.04460v1_enmode)
- **Text2Layer** (2023): an autoencoder for layered images, with diffusion producing bg, fg, mask and composite together — same source
- **OmniPSD**, arXiv 2512.09247 (Dec 2025, NUS + Lovart): Flux-based in-context text-to-PSD and image-to-PSD. Reportedly trained on about 200K real PSDs (from a review, unverified). Project page: [showlab.github.io/OmniPSD](https://showlab.github.io/OmniPSD/) — [arXiv](https://arxiv.org/pdf/2512.09247)
- **TransPixar / TransPixeler** (Adobe + HKUST, arXiv 2501.03006): RGBA text-to-video. Adds alpha tokens with a LoRA on a DiT. A CogVideoX-5B LoRA is released and Mochi training code is open — [HF wileewang/TransPixar](https://huggingface.co/wileewang/TransPixar), [project](https://wileewang.github.io/TransPixar)
- Newer 2026 work seen only as titles: **LiWi: Layering in the Wild** (arXiv 2605.14552) and "Controllable layered image generation for real-world editing" (arXiv 2601.15507) — [LiWi PDF](https://arxiv.org/pdf/2605.14552)

### Inferences
- Joint methods (PSDiffusion, DreamLayer, ART) give the best cross-layer lighting harmony. But regenerating one layer means re-running everything, or using inpainting-style tricks with a fixed seed.
- Of the joint methods, PSDiffusion's input format (global + per-layer prompts) is the closest to the game's per-object prompt design.

### Gaps
- Weights, licenses and VRAM were not verified for DreamLayer, PSDiffusion, LayerFusion, OmniPSD or LayeringDiff. Assume code-only or no release until checked.
- Not covered by search: LayerCraft, LayerTracer (vector/SVG layers), Alfie, MuLAN dataset, and "Generative Image Layer Decomposition" (LayerDecomp, Adobe).

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
