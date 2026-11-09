# Build scenes from layers, not splits

No single open model yet does exactly what the storytelling-game prototype needs: one layer per object, each with its own prompt, movable, and regenerable on its own. The most practical design is a **modular pipeline**. Generate each object on a plain background, or as a native transparent layer, then cut it out with a matting model. Composite the layers and run an optional harmonization pass that never overwrites the stored layers. Native layered models fill specific roles inside that pipeline. **Qwen-Image-2.1** (Sept 2026) is the only mainstream base model found that outputs **RGBA from a prompt**, so it is a strong per-object generator. **LayerDiffuse** (SDXL, plus a FLUX.1-dev port) is the mature fallback. **Qwen-Image-Layered** is the reference **image-to-layers decomposer**, useful for importing a finished scene. It splits *images*, not prompts: its text prompt describes the whole picture, and the model decides the groupings itself. Joint multi-layer generators (PSDiffusion, DreamLayer, ART) give better lighting harmony, but regenerating one layer means re-running the whole stack, and most have no verified public weights. Z-Image Turbo and FLUX.2 Klein 4B have no native layering. They can still serve as per-object generators or harmonizers in the modular pipeline. "FLUX 3 Image Klein" could not be confirmed to exist.

Throughout, **[unverified]** marks claims that came from the researchers' prior knowledge or from secondary sources and were not confirmed against a primary source.

## Prompt-to-layers: one native RGBA base model, several research generators

The longest-standing open option is **LayerDiffuse** (Zhang & Agrawala, arXiv 2402.17113). It encodes alpha as "latent transparency" for SD1.5 and SDXL, and has Forge and ComfyUI integrations ([arXiv](https://arxiv.org/abs/2402.17113); [GitHub](https://github.com/layerdiffusion/LayerDiffuse)). Weights are at [huggingface.co/LayerDiffusion/layerdiffusion-v1](https://huggingface.co/LayerDiffusion/layerdiffusion-v1). Its released modes are foreground only, joint foreground plus background, background given foreground, and **foreground given background** [unverified detail]. The last of these is the closest open match to "regenerate one object so it fits the existing scene." A **FLUX.1-dev port** from FireRedTeam/RedAIGC adds a LoRA and a new transparent VAE, with text-to-image and image-to-image demo scripts ([GitHub](https://github.com/FireRedTeam/LayerDiffuse-Flux); [HF RedAIGC/Flux-version-LayerDiffuse](https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse)). It inherits FLUX.1-dev's non-commercial license, and its VRAM needs were not checked. The SDXL version needs roughly 8–12 GB of GPU memory [unverified] and has no practical CPU path. No LayerDiffuse port for SD3, FLUX.2 or Z-Image was found.

**Qwen-Image-2.1** (released 2026-09-20) is the most significant new option. It is a **7B DiT** with a Qwen3-VL-8B text encoder and a **64-channel RGBA autoencoder**. It outputs RGB or RGBA depending on the prompt, for example with a phrase like "This is an RGBA image with transparency". It can also extract a subject from a photo as a transparent layer and edit transparent assets ([eWeek](https://www.eweek.com/news/alibaba-qwen-image-2-1-native-transparency/); [ComfyUI Wiki](https://comfyui-wiki.com/en/news/2026-09-21-qwen-image-2-1); [Qwen blog](https://qwen.ai/blog?id=qwen-image-2.1)). Two things are unverified: its **license** (reported as the non-commercial Qwen Research License, but sources conflict) and its **Hugging Face URL and VRAM needs**.

Several research methods generate many layers in one joint pass, and their output looks more coherent because all layers are denoised together.

- **PSDiffusion** (arXiv 2505.11468, WACV 2026) takes a global prompt plus **per-layer prompts**. Its evaluation used 3 foregrounds, 1 background and a global prompt. It gets the layout from the cross-attention maps of an RGB image it generates alongside the layers ([arXiv](https://arxiv.org/html/2505.11468)). This input format matches the game's per-object prompt design most closely.
- **DreamLayer** (arXiv 2503.12838, ICCV 2025) generates foreground and background layers together, and also supports image-to-layer decomposition ([arXiv](https://arxiv.org/html/2503.12838v1)).
- **LayerFusion** (arXiv 2412.04460) is a harmonized multi-layer text-to-image method built on latent-diffusion priors ([summary](https://fugumt.com/fugumt/paper_check/2412.04460v1_enmode)).
- **ART** (arXiv 2502.18364, CVPR 2025, Microsoft) is built on FLUX.1-dev. It takes anonymous region boxes, claims **50+ layers**, and is more than 12x faster than full attention ([arXiv](https://arxiv.org/abs/2502.18364v1)). However, its **weights and code were withdrawn on 2025-07-23** over training-data concerns ([GitHub](https://github.com/microsoft/art-msra)).
- **PrismLayersPro** is a ~20K-image multi-layer dataset (the full set is ~200K) that is useful for fine-tuning ([HF dataset](https://huggingface.co/datasets/artplus/PrismLayersPro); [arXiv](https://arxiv.org/pdf/2505.22523)).
- **OmniPSD** (arXiv 2512.09247) does Flux-based text-to-PSD and image-to-PSD, but it targets graphic design ([project](https://showlab.github.io/OmniPSD/)).

Weights, licenses and VRAM for PSDiffusion, DreamLayer, LayerFusion and OmniPSD were **not verified**. Treat them as paper-only until checked.

## Qwen-Image-Layered splits images, and the model picks the layers

**Qwen-Image-Layered** (arXiv 2512.15603, CVPR 2026) is a **20B, Apache-2.0** fine-tune of Qwen-Image. It turns one RGB image into a **variable number of RGBA layers** and fills in the content each layer hides ([arXiv](https://arxiv.org/html/2512.15603v1); [HF](https://huggingface.co/Qwen/Qwen-Image-Layered)). Two design choices set it apart:

- An **RGBA-VAE** widens the Qwen-Image VAE's input and output from 3 to 4 channels, so RGB and RGBA share one latent space.
- A **VLD-MMDiT** adds a "Layer3D RoPE" layer-index axis, which lets the number of layers vary.

It was trained in three stages: text to RGBA, then text to multiple RGBA layers ("T2L"), then image to multiple RGBA layers ("I2L"). The training data was **real Photoshop files** with up to 20 layers ([arXiv](https://arxiv.org/html/2512.15603v1)).

**It splits images, not prompts.** The optional prompt should describe the whole image, including hidden content, and it does not control what goes on each layer ([ComfyUI Wiki](https://comfyui-wiki.com/en/tutorial/advanced/image/qwen/qwen-image-layered)). You choose only the layer count. Replicate allows 2–8 and suggests 3–4 for most images ([Replicate](https://replicate.com/qwen/qwen-image-layered)). Any output layer can be decomposed again ("infinite decomposition"). Qwen itself says text-to-multi-layer quality is limited because the model is tuned for decomposition, and whether the T2L weights were released is unconfirmed.

On the Crello design benchmark it beat LayerD on both measures:

| Measure | Qwen-Image-Layered | LayerD |
|---|---|---|
| RGB L1 error, no merges (lower is better) | **0.0594** | 0.0709 |
| Alpha soft-IoU (higher is better) | **0.8705** | 0.7520 |

The model is **heavy**. One test peaked at about **45 GB VRAM**, taking ~120 s per 1024px image on a 96 GB card. A 24 GB RTX 4090 is nearly full, so FP8 is recommended ([ComfyUI Wiki](https://comfyui-wiki.com/en/tutorial/advanced/image/qwen/qwen-image-layered)). A small Q4 GGUF (~3.2 GB) that would fit 8–12 GB cards is claimed by a third-party guide and is **unverified**. Hosted inference costs about **$0.05 per image** on fal (aggregator figure, unverified) ([CostGoat](https://costgoat.com/pricing/qwen-image)), and Replicate also offers it.

If the game needs to say "give me Bob's layer", other methods take target-specific input:

- **Referring Layer Decomposition** (arXiv 2602.19358, ICLR 2026) takes an image plus a text or spatial prompt and returns that target's complete RGBA layer. It comes with the ~430K-image RefLade dataset ([arXiv](https://arxiv.org/html/2602.19358v1)).
- **CLD** (arXiv 2511.16249) is built on FLUX.1-dev and produces **one layer per user bounding box** ([arXiv](https://arxiv.org/html/2511.16249v2)).

Other decomposers cover narrower cases:

- **LayerD** (arXiv 2509.25134) peels graphic designs into layers by repeatedly removing the top layer.
- **LayerDecomp** (Adobe, CVPR 2025) splits off one foreground and keeps its **shadows and reflections** with it ([arXiv](https://arxiv.org/html/2411.17864v1)).
- **RevealLayer** (arXiv 2605.11818) targets natural images and fills in hidden content ([arXiv](https://arxiv.org/abs/2605.11818)).
- **PixelART** (arXiv 2609.22789) is a lighter pixel-space alternative ([arXiv](https://arxiv.org/abs/2609.22789)).

Code and weights for RevealLayer, PixelART, RefLayer and CLD were **not verified**.

## Modular pipelines match how HCI tools already worked

There are four modular approaches. Pipeline (a), the first in the comparison table below, is the best fit for a game.

**(a) Generate on a plain background, then matte.** Each object is prompted on a plain background and cut out. Regenerating one layer re-runs only that object, which is cheap and predictable. Matting options:

- **BEN2 Base** (MIT): about 0.13 s and 4.5 GB on an RTX 3090, according to its developer ([HF](https://huggingface.co/PramaLLC/BEN2/blob/main/README.md)).
- **BiRefNet** (MIT, with HR and matting variants) ([HF](https://huggingface.co/ZhengPeng7/BiRefNet)) [license and variants unverified].
- **RMBG-2.0**: same architecture as BiRefNet, but **CC BY-NC** ([HF](https://huggingface.co/briaai/RMBG-2.0)).
- **rembg**: wraps several of these with ONNX CPU backends, which suits the current CPU setup ([GitHub](https://github.com/danielgatis/rembg)) [unverified].

**(b) Segment, then inpaint.** This decomposes an existing scene.

- Segmentation: **SAM 3** turns text prompts into masks for every matching object ([HF](https://huggingface.co/facebook/sam3); [docs](https://huggingface.co/docs/transformers/en/model_doc/sam3)).
- Removal: **LaMa** runs on CPU [unverified]. **ObjectClear** removes objects together with their shadows ([arXiv](https://arxiv.org/abs/2505.22636)).
- Hidden parts: **pix2gestalt**-style amodal completion fills in what overlapping objects covered [unverified].

Every layer except the top one has holes, so both object removal and completion are needed.

**(c) Regional or layout generation, then harmonization.** GLIGEN, MIGC, InstanceDiffusion and regional prompting give a coherent single image, but it is flat and still needs segmenting [unverified, SD1.x era].

**(d) Edit-based decomposition.** A multi-reference edit model removes, adds or harmonizes objects. **FLUX.2 [klein] 4B** (Apache-2.0, ~13 GB VRAM) does text-to-image, editing and multi-reference composition in one checkpoint ([HF mirror](https://huggingface.co/bi1o/FLUX.2-klein-4B)). The official repo is likely [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) [path unverified]. **Qwen-Image-Edit** is another option [unverified] ([HF](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)). **IC-Light** handles relighting [unverified] ([GitHub](https://github.com/lllyasviel/IC-Light)).

The HCI precedents show that **no system used native RGBA models**. They layered at the *interaction* level, attaching prompts to regions or masks and rendering onto a flat canvas:

- **WorldSmith** (UIST 2023) introduced **hierarchical prompting**, with prompts attached to layers of a composition, plus spatial prompting and region painting, and blended the edited tiles into one image ([arXiv](https://arxiv.org/pdf/2308.13355)). The view that it used SD inpainting comes from prior knowledge [unverified].
- **SketchFlex** (CHI 2025) has an MLLM refine per-region prompts from rough sketches into canny "shape anchors" for region-conditioned generation. In an N=12 study it beat text-to-image and region-to-image baselines ([arXiv](https://arxiv.org/abs/2502.07556v1)).
- **PromptCharm** (CHI 2024) used mask brushing plus SD inpainting to regenerate only the masked pixels ([arXiv](https://arxiv.org/pdf/2403.04014)).

The data model of "prompt plus region" carries over directly. What the game adds is that objects must *move*, which requires real RGBA layers.

## Z-Image Turbo, FLUX.2 Klein, FLUX 3 and Qwen-Image-2.1 compared

| Model / method | Input → output | Per-layer prompt | Regenerate one layer | Hardware | License | HF link |
|---|---|---|---|---|---|---|
| **Qwen-Image-2.1** | prompt → 1 RGBA (native) | Yes (one call per object) | Yes | 7B DiT + 8B encoder; VRAM unverified | Possibly non-commercial [unverified] | URL unverified ([blog](https://qwen.ai/blog?id=qwen-image-2.1)) |
| **Z-Image Turbo** | prompt → RGB only | Via pipeline (a) | Yes, with matting | GPU; specifics not researched | Not researched | No layer or RGBA support found |
| **FLUX.2 Klein 4B** | prompt / edit / multi-ref → RGB | Via (a) or (d) | Yes | ~13 GB VRAM | Apache-2.0 | [mirror](https://huggingface.co/bi1o/FLUX.2-klein-4B) |
| **"FLUX 3 Image Klein"** | — | — | — | — | — | **Could not confirm this model exists** |
| LayerDiffuse SDXL | prompt → RGBA, fg given bg | Yes | Yes | ~8–12 GB [unverified] | Apache-2.0 [via index] | [layerdiffusion-v1](https://huggingface.co/LayerDiffusion/layerdiffusion-v1) |
| LayerDiffuse FLUX | prompt → RGBA | Yes | Yes | Unverified | FLUX.1-dev non-commercial | [RedAIGC](https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse) |
| PSDiffusion / DreamLayer | global + layer prompts → multiple RGBA | Yes | Whole stack re-runs | Unverified | Weights unverified | — |
| ART | prompt + boxes → 50+ layers | Global only | Whole stack re-runs | FLUX-scale | **Withdrawn** | — |
| **Qwen-Image-Layered** | image → 2–8+ RGBA layers | No (whole-image prompt) | Recurse on a layer | ~45 GB peak; FP8 on 24 GB; ~$0.05/call hosted | Apache-2.0 | [Qwen-Image-Layered](https://huggingface.co/Qwen/Qwen-Image-Layered) |
| CLD / RefLayer | image + box or prompt → that layer | Targeted | Yes | Unverified | Unverified | [CLD paper](https://huggingface.co/papers/2511.16249) |
| Pipeline (a): generation + BEN2/BiRefNet | prompt → RGB → RGBA | Yes | **Excellent** | **CPU-capable** | MIT matting | [BEN2](https://huggingface.co/PramaLLC/BEN2) |

In short, **Z-Image Turbo, FLUX.2 Klein and FLUX 3 cannot produce layers natively**. No transparency adapter or layer LoRA was found for any of them ([Civitai docs](https://developer.civitai.com/orchestration/recipes/flux2)). Z-Image Turbo and FLUX.2 Klein still work as per-object generators in pipeline (a). FLUX.2 Klein can also serve as the harmonizer or editor in pipeline (d). **Qwen-Image-2.1 is the only one with native RGBA**, but it produces one layer per call, not a joint multi-layer scene.

## Conclusion

The key design choice is to store **layers as the source of truth** and treat any coherent full-scene render as a disposable preview. That makes the trade-off manageable: joint models win on lighting harmony, modular pipelines win on control, and a harmonization pass (low-denoise img2img at about 0.2–0.35, or a FLUX.2 Klein multi-reference edit) recovers most of the harmony without locking layers together. For the prototype:

1. **Now, on CPU:** keep SDXL-Turbo and prompt each object "on a plain grey background", then cut it out with rembg running BiRefNet-lite or BEN2. Store each layer as {prompt, seed, RGBA, position}, and keep the background as its own opaque layer.
2. **On Kaggle or Colab:** swap in **Qwen-Image-2.1 RGBA** per object, after checking its license. LayerDiffuse SDXL "foreground given background" is the fallback.
3. **Harmonize** with FLUX.2 Klein 4B.
4. **Import** user-supplied scenes with Qwen-Image-Layered, hosted.

WorldSmith's scene, group and object prompt hierarchy is a good model for the UI.

What to watch: PSDiffusion-style joint generation with per-layer prompts is the closest match to the game's ideal, and it becomes worth adopting once an openly licensed checkpoint ships. Confirm the weights before building on it.
