# Qwen-Image-Layered and LayerDiffuse: how they produce coherent multi-layer images

Sources read on 2026-11-07: the arXiv HTML full text of both papers, the Qwen GitHub README and Hugging Face model card, the sd-forge-layerdiffuse README (first ~100k characters), and the local clone of rootonchair/diffuser_layerdiffuse (README and test scripts). I did not read the ComfyUI-layerdiffuse or lllyasviel/LayerDiffuse repos.

Labels used below:
- [paper]: verified from the paper text
- [repo]: verified from a README, model card or script
- [inference]: my own reasoning, not stated in a source

## Q1. Qwen-Image-Layered: inputs, outputs, modes, layering, editing

### Takeaway
The released model works on images, not text. It takes one RGB image and returns N RGBA layers (N is variable) that alpha-composite back to the input. The paper also trains a text-to-layers variant, but the official repo says text-to-multi-RGBA performance is "limited". The released workflow is decompose, edit one layer, then recombine. The paper also shows generating an image with Qwen-Image text-to-image first and then decomposing it, which gave better-looking results. For a game, this is the most direct route from one prompt to coherent PSD-like layers, because the layers are coherent by construction: they come from one image. It is a 20B BF16 model, so it is not a CPU option.

### Cited Findings
- [paper] Input is a single RGB image I. Output is N RGBA layers. Sequential alpha blending of the layers reconstructs the input (Sec. 3). — [arXiv 2512.15603](https://arxiv.org/html/2512.15603)
- [paper] The VLD-MMDiT handles "a variable number of layers". Layer3D RoPE adds a layer-index axis: target layers start at index 0 and the condition image is at -1. Training uses up to 20 layers (Sec. 3.2, 4.2). — [arXiv](https://arxiv.org/html/2512.15603)
- [paper] Layer order comes from the layer index and the compositing order. Evaluation uses order-aware DTW, which allows adjacent layers to be merged (Sec. 4.3.1). — [arXiv](https://arxiv.org/html/2512.15603)
- [paper] Occluded content: the paper gives no explicit description of how hidden regions are completed. It says the model is end-to-end "without relying on external modules". It names inpainting artifacts as a weakness of the LayerD baseline (Sec. 4.4.1). — [arXiv](https://arxiv.org/html/2512.15603)
- [repo] The README tells users to write a prompt that describes the whole image, including occluded elements. The prompt does not control individual layers. — [GitHub QwenLM/Qwen-Image-Layered](https://github.com/QwenLM/Qwen-Image-Layered)
- [paper] RGBA-VAE is a four-channel VAE extended from the Qwen-Image VAE, with alpha weights initialized to zero. It beats the LayerDiffuse VAE on AIM-500 (Sec. 3.1, Table 3). — [arXiv](https://arxiv.org/html/2512.15603)
- [paper] Training data is layers extracted from real PSD files with psd-tools. Non-overlapping layers were merged and composites were captioned with Qwen2.5-VL (Sec. 4.1). — [arXiv](https://arxiv.org/html/2512.15603)
- [paper] Text-to-Multi-RGBA (T2L) is trained in Stage 2 as "Qwen-Image-Layered-T2L" (Sec. 3.3) and shown qualitatively in Fig. 8 (Sec. 4.4.3). The paper also shows a pipeline that runs Qwen-Image text-to-image and then Qwen-Image-Layered image-to-layers, reported to "further improve visual aesthetics". — [arXiv](https://arxiv.org/html/2512.15603)
- [repo] The released checkpoint is Qwen/Qwen-Image-Layered (2025-12-19, Apache 2.0). The README describes text conditioning, but performance on text-to-multi-RGBA is "limited". It does not mention a separate T2L checkpoint. — [GitHub](https://github.com/QwenLM/Qwen-Image-Layered); [HF card](https://huggingface.co/Qwen/Qwen-Image-Layered)
- [repo] Usage: `QwenImageLayeredPipeline` in Diffusers. Parameters are `layers` (example 4; 3 and 8 are shown), `resolution` (640 recommended, or 1024), 50 steps, and `true_cfg_scale` 4.0. The caption is optional; with `use_en_prompt`, the model writes its own caption. The model is 20B BF16 on CUDA. No VRAM figure is given. — [HF card](https://huggingface.co/Qwen/Qwen-Image-Layered)
- [repo] Recursive decomposition: any output layer can be decomposed again ("infinite decomposition"). — [GitHub](https://github.com/QwenLM/Qwen-Image-Layered)
- [paper] By contrast, the paper criticizes earlier methods that need "recursive inference, leading to error propagation". Qwen produces all layers in one pass (Sec. 2.2). — [arXiv](https://arxiv.org/html/2512.15603)
- [paper] Editing: the paper names resizing, repositioning and recoloring (Fig. 1). The comparison in Fig. 7 uses "simple manual edits" on the layers and is qualitative only (Sec. 4.4.2). — [arXiv](https://arxiv.org/html/2512.15603)
- [repo] Editing workflow:
  1. `src/app.py` decomposes the image and exports PPTX, ZIP or PSD.
  2. `src/tool/edit_rgba_image.py` edits one layer with Qwen-Image-Edit (recolor, replace, text revision).
  3. `src/tool/combine_layers.py` recomposites the layers, bottom to top.

  Delete, resize and move are shown as plain layer operations. — [GitHub](https://github.com/QwenLM/Qwen-Image-Layered); [HF card](https://huggingface.co/Qwen/Qwen-Image-Layered)

### Inferences
- The pipeline best supported by the sources is any text-to-image model, then I2L decomposition, then per-layer edits, then recompositing. Image-to-layers is the released and best-evaluated mode, and the paper itself demonstrates Qwen-Image text-to-image followed by I2L.
- In principle, any text-to-image model (for example SDXL) could produce the input image, because I2L only needs an RGB image. The paper only demonstrates Qwen-Image.
- Re-prompting one layer goes through Qwen-Image-Edit on that RGBA layer. Lighting and shadow consistency with the other layers is not guaranteed after that edit.
- Hardware fit (based on the 20B size): this would need a GPU or a hosted API. It is not realistic on the user's CPU-only PC.

### Gaps
- Whether the hidden or occluded parts of each layer are actually completed, and how well: the paper does not say clearly. Check the Fig. 1/7 examples.
- No public T2L checkpoint was found. The Tongyi/ModelScope pages were not checked.
- No VRAM figures, and no information on quantized or CPU-runnable versions.

## Q2. LayerDiffuse: all modes, SDXL vs SD1.5, coherence, chaining, prompts, placement

### Takeaway
The paper describes these modes:
- foreground only
- joint foreground + background, coherent through shared attention
- foreground-conditioned background
- background-conditioned foreground
- iterative multi-layer generation: background-conditioned foreground applied repeatedly, with the current composite as the condition

For SDXL, the released weights are only:
- foreground-only (attn and conv variants)
- two-step conditional models: fg2ble, fgble2bg, bg2ble, bgble2fg

SDXL joint is "on hold" and was not released. SD1.5 has joint (3-image batch), fg2bg and bg2fg. In every mode the model decides where objects go and how big they are. No mode takes a user mask or box for placement.

### Cited Findings
- [paper] Modes: single transparent image (Sec. 3.2, 4.1); joint two-layer generation (Sec. 3.3); foreground-conditioned background and background-conditioned foreground, where the condition latent is kept clean (Sec. 3.3, 4.2); a two-stage variant using zero-initialized UNet input channels (Sec. 4.5); ControlNet compatibility (Sec. 4.4); use with community models (Sec. 4.10). — [arXiv 2402.17113](https://arxiv.org/html/2402.17113)
- [paper] Multi-layer generation is iterative: background-conditioned foreground generation is repeated, and at each step the existing layers are blended into one RGB image that becomes the condition (Sec. 4.3, Fig. 9). — [arXiv](https://arxiv.org/html/2402.17113)
- [paper] Joint generation gives each layer its own latent and its own LoRA (fg and bg). Query, key and value are concatenated across layers in every attention layer, which is the mechanism behind cross-layer coherence. The LoRAs are rank 256 (Sec. 3.3, 3.4). — [arXiv](https://arxiv.org/html/2402.17113)
- [paper] Training prompts come in three kinds: a foreground prompt, a whole-image prompt, and a background prompt (Sec. 3.4). — [arXiv](https://arxiv.org/html/2402.17113)
- [paper] Coherence: the paper claims consistent illumination and geometry and harmonious colors. Conditional examples show interaction, such as a person sitting on a bench or sofa, and reflections (Sec. 4.1, 4.2, 4.4). — [arXiv](https://arxiv.org/html/2402.17113)
- [paper] Limitation: there is a trade-off between clean, reusable elements and harmonious blending. Conditioning on the background bakes illumination into the foreground, which makes the foreground less reusable (Sec. 4.12, Fig. 17). — [arXiv](https://arxiv.org/html/2402.17113)
- [paper] Speed was measured at 30 steps on an RTX 4090. Joint generation took 5.71 s on SD1.5 and 21.5 s on SDXL; the two-stage variant took about 4.37 s × 2 on SDXL (Table 3). — [arXiv](https://arxiv.org/html/2402.17113)
- [repo] Forge model list:
  - SDXL:
    - `layer_xl_transparent_attn` (better quality)
    - `layer_xl_transparent_conv` (better prompt alignment, strong style influence)
    - `layer_xl_fg2ble`
    - `layer_xl_fgble2bg`
    - `layer_xl_bg2ble`
    - `layer_xl_bgble2fg`
  - SD1.5:
    - `layer_sd15_transparent_attn`
    - `layer_sd15_joint`
    - `layer_sd15_fg2bg`
    - `layer_sd15_bg2fg`

  SDXL joint is "on hold", not released. — [sd-forge-layerdiffuse](https://github.com/layerdiffusion/sd-forge-layerdiffuse)
- [repo] Prompts by mode:
  - SD1.5 joint: a global prompt plus optional fg, bg and blend prompts. Example: "apple" / "floor in room" / "apple on floor in room". Batch size must be divisible by 3, otherwise "you will only get noise".
  - SD1.5 fg2bg and bg2fg: batch size divisible by 2. Layer prompts are optional; bg2fg usually does not need them.

  The README warns that the global prompt can corrupt the background and suggests splitting prompts per layer. — [sd-forge-layerdiffuse](https://github.com/layerdiffusion/sd-forge-layerdiffuse)
- [repo] The blended output passes through a VAE round trip, so it can differ from the real fg and bg. The author recommends compositing the true fg and bg yourself. The SDXL bg-conditioned fg model "may be too lightweight". Examples use 20 steps and CFG 5–7. — [sd-forge-layerdiffuse](https://github.com/layerdiffusion/sd-forge-layerdiffuse)
- [repo] In the local clone, the SDXL scripts are two-step:
  - `test_diffusers_xl_bg2ble.py`: background image + prompt ("a dog sitting in room") → blended image
  - `test_diffusers_xl_bgble2fg.py`: background + that blend + prompt → transparent foreground

  The fgble2bg and fg2ble scripts are the reverse pair. Defaults: SDXL base 1.0, 896×1152, 20 steps, CFG 7. — local `prototype/vendor/diffuser_layerdiffuse/README.md` and `test_diffusers_xl_*.py` ([upstream](https://github.com/rootonchair/diffuser_layerdiffuse))
- [repo] SD1.5 scripts: `test_diffusers_joint.py` (batch of 3: fg, bg, blend), `test_diffusers_fg_bg_cond.py` (fg from a given bg), `test_diffusers_bg_fg_cond.py` (bg from a given fg). — local clone README

### Inferences
- A coherent SDXL scene recipe that fits the released weights:
  1. Generate a background as an ordinary image.
  2. For each object: run bg2ble (current composite + object prompt) to get a blended scene, then bgble2fg to extract that object as a transparent layer.
  3. Treat the new blend as the new composite and repeat.

  This is the paper's Sec. 4.3 iterative scheme, mapped onto the SDXL two-step models. The model chooses each object's placement and scale from the scene. Because each foreground carries baked-in lighting from the scene, it stays coherent but is less reusable if moved.
- To place an object in a user-chosen box, you would have to add inpainting or masking (for example, limiting img2img to the box region). The repos do not offer this; it is my suggestion.
- The current prototype generates each object independently with "a single X, solo, centered" and pastes it into a box. That bypasses every coherence mechanism LayerDiffuse provides (shared attention, or conditioning on the composite), so incoherent results are expected.

## Q3. Does SDXL-Turbo work with LayerDiffuse LoRAs?

### Takeaway
No primary source says yes or no. The repos and the paper only show SDXL base and community checkpoints at about 20–30 steps with CFG 5–7.

### Cited Findings
- [repo] Forge examples use 20 steps and CFG 5–7. The diffusers SDXL scripts default to `stabilityai/stable-diffusion-xl-base-1.0`, 20 steps, CFG 7. — [sd-forge-layerdiffuse](https://github.com/layerdiffusion/sd-forge-layerdiffuse); local clone scripts
- [paper] Applying the method to community models without retraining is claimed (Sec. 4.10). Timings use 30 steps. — [arXiv](https://arxiv.org/html/2402.17113)

### Inferences
- Turbo is a distilled model that runs at 1–4 steps with CFG around 0. That conflicts with the CFG-dependent conditional modes, and the LoRAs were trained against the base SDXL denoising trajectory. Lower quality and weaker conditioning are plausible but not verified. Running the fg-only LoRA on Turbo is likely possible because it is a LoRA on the same architecture.

### Gaps
- No official statement on Turbo or Lightning compatibility. ComfyUI-layerdiffuse issues were not checked.
- The second half of the Forge README (beyond 100k characters) and the ComfyUI repo were not read.
