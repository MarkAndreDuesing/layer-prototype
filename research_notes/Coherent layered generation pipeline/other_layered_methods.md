# Other methods for coherent layered generation and decomposition (as of 2026-11-07)

Scope note: 17 tool calls were used. Full text was read for PSDiffusion, DreamLayer, OmniPSD and LASAGNA. Only abstracts or project pages were read for CLD, RefLayer, LiWi, LayerFusion and PrismLayers. Items marked **[prior notes, not re-verified]** come from `reports/Layer based architecture models and hosting.md` and were not checked again in this pass.

## Q1. Which methods do "single prompt -> multiple coherent layers" natively, and which can run now?

### Takeaway
Several papers generate all layers jointly from a global prompt plus per-layer prompts: PSDiffusion, DreamLayer, OmniPSD, LayerFusion, LASAGNA and ART. None of them has verified, downloadable open weights today. The only runnable native text-to-RGBA options are single-layer (or fg+bg) models: LayerDiffuse, LayerDiffuse-Flux and Qwen-Image-2.1. These generate one transparent object per call, not a coordinated multi-layer stack.

### Cited Findings
- **PSDiffusion (WACV 2026, arXiv 2505.11468)**
  - **Input:** a global prompt plus per-layer prompts, which an LLM can produce or the user can write. No boxes are needed; layout comes from the global branch's cross-attention maps.
  - **Coherence:** "layer cross-attention reweighting" for placement, plus "partial joint self-attention" so each layer attends to masked regions of the global branch.
  - **Layers:** the dataset has 3–6 layers; evaluation uses 3 foreground layers + 1 background.
  - **Base:** SDXL with a rank-256 LoRA and an RGBA VAE in the LayerDiffuse style; about 60 A800 GPU-hours of training.
  - **Release:** the paper gives no link or release statement for code or weights. It says it "supports various layer-wise editing" but gives no procedure for regenerating a single layer.
  - Sources: [arXiv HTML v2](https://arxiv.org/html/2505.11468v2), [project page](https://dingbang777.github.io/PSDiffusion_Website)
- **DreamLayer (ICCV 2025, arXiv 2503.12838)**
  - **Input:** one prompt for the background and one per foreground. They are combined into a global prompt that drives a "global layer". Boxes are used only to build the dataset (GroundingDINO).
  - **Coherence:** Context-Aware Cross-Attention takes a layout from the global layer; Layer-Shared Self-Attention links all layers; Information Retained Harmonization blends them in latent space and produces shadows.
  - **Layers:** 2–4.
  - **Base:** SD1.5, fine-tuned Custom-Diffusion style (K/V only).
  - **Editing:** layers can be moved, flipped or resized at the latent level, with shadows updating to match. Single-layer regeneration is not described.
  - **Release:** only a project page is linked; no code or weights are stated.
  - Sources: [arXiv HTML](https://arxiv.org/html/2503.12838), [ICCV page](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_DreamLayer_Simultaneous_Multi-Layer_Generation_via_Diffusion_Model_ICCV_2025_paper.html), [project page](https://ll3rd.github.io/DreamLayer/)
- **OmniPSD (arXiv 2512.09247)**
  - **Text-to-PSD input:** a JSON record with a "poster" (global) description plus foreground, midground and background descriptions. Removing the per-layer prompts made results worse (FID 30.43 → 38.56).
  - **Layers:** a fixed 2×2 grid (composite, fg, mid, bg) generated jointly with FLUX.1-dev; coherence comes from in-context spatial self-attention. Image-to-PSD uses Flux-Kontext with an RGBA VAE.
  - **Editing:** poster editing in a UI; no regeneration of one layer from a prompt.
  - **Release:** the paper mentions no code or weights. Prior notes say the code is CC BY-NC-SA with no weights.
  - Sources: [arXiv HTML](https://arxiv.org/html/2512.09247v1), [project page](https://showlab.github.io/OmniPSD/)
- **LASAGNA (arXiv 2601.15507, UCSC + Adobe)**
  - **Base:** a fine-tuned ~2B DiT with an RGBA VAE and a T5 text encoder.
  - **Modes:** Text2All (composite + clean background + RGBA foreground with shadows and reflections, from text plus a location mask), FG_Gen (foreground given background) and BG_Gen (background given foreground).
  - **Release:** the dataset (LASAGNA-48K) and benchmark are promised; model release is not stated.
  - Sources: [arXiv HTML](https://arxiv.org/html/2601.15507), [project page](https://rayjryang.github.io/LASAGNA-Page/)
- **LayerFusion (arXiv 2412.04460, Virginia Tech + Adobe)**
  - **Output:** a jointly generated triplet of foreground (RGBA), background and blend, built on transparent-generation priors (LayerDiffuse).
  - **Release:** no code link was confirmed.
  - Sources: [project page](https://layerfusion.github.io), [arXiv](https://arxiv.org/html/2412.04460v1)
- **ART / ART+ and PrismLayers (arXiv 2505.22523)**
  - ART+ is ART fine-tuned on PrismLayersPro and wins 60% of head-to-head user comparisons with ART. The abstract calls it "open-source", but the HF papers page links only the dataset (creative-graphic-design/PrismLayersPro), not a model.
  - Sources: [HF papers](https://huggingface.co/papers/2505.22523), [arXiv](https://arxiv.org/abs/2505.22523)
  - **[prior notes, not re-verified]** ART takes a global prompt plus per-region boxes and prompts. Its weights were removed in July 2025, and the ART+ checkpoint is on Google Drive with no license.
- **Qwen-Image-2.1**
  - **Model:** a 7B DiT with a Qwen3-VL 8B encoder. It is the first Qwen model with native RGBA output (prompted with a fixed RGBA phrase), and it can edit transparent layers, extract subjects and take up to 10 reference images.
  - **License:** Qwen Research License (non-commercial).
  - **Not confirmed:** whether it decomposes an image into multiple layers. Secondary sources only say it "folds in" Qwen-Image-Layered's role.
  - Sources: [HF card](https://huggingface.co/Qwen/Qwen-Image-2.1), [ComfyUI wiki](https://comfyui-wiki.com/en/models/qwen-image/qwen-image-2.1)
- **LayerDiffuse-Flux (RedAIGC)** **[prior notes, not re-verified]:** single-layer RGBA on FLUX.1-dev; the repo has no license and the base model is non-commercial. Source: [HF](https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse)

### Inferences
- The "single prompt -> coherent layer stack" architecture (a global branch plus per-layer branches with shared attention) is well established in papers: PSDiffusion and DreamLayer on SD, OmniPSD on FLUX.
- PSDiffusion is architecturally closest to an SDXL+LayerDiffuse stack, so with released weights it would fit a T4. As of now it cannot be run.
- A practical stand-in is to approximate it: an LLM decomposes the scene into per-layer prompts and a layout, LayerDiffuse generates each layer, and a harmonization pass follows.

### Gaps
- No VRAM figures are published for any joint model.
- PSDiffusion, DreamLayer and LayerFusion: the GitHub orgs were not checked directly, only the paper and project-page links.
- Text2Layer, MuLAn, LayeringDiff, PixelART and RevealLayer were not researched in this pass because of the call budget.

## Q2. Which methods do the two-stage "T2I image -> decompose into layers", and which can run now?

### Takeaway
This category has runnable open options. Qwen-Image-Layered is Apache-2.0 at 20B and too large for a T4. CLD is runnable (FLUX.1-dev + LoRA, one box per layer). LayerD is small and graphics-oriented. RefLayer and LiWi are paper-only or gated. Decomposition takes a finished image, so layers are not stable between runs if the scene is regenerated.

### Cited Findings
- **CLD (arXiv 2511.16249)**
  - **Architecture:** an LD-DiT that decouples image elements into layers, plus MLCA, which injects target-image information into the multi-layer tokens.
  - **Code and weights:** MIT code on GitHub ([monkek123King/CLD](https://github.com/monkek123King/CLD)); checkpoints released December 2025 on [HF thuteam/CLD](https://huggingface.co/thuteam/CLD), whose card says Apache-2.0.
  - **Requires:** FLUX.1-dev and the alimama FLUX.1-dev-Controlnet-Inpainting-Alpha adapter. FLUX.1-dev's non-commercial license therefore applies in practice.
  - **Run:** `python -m infer.infer -c infer/infer.yaml`.
  - VRAM and layer count are not stated. Sources: [arXiv](https://arxiv.org/abs/2511.16249), [project page](https://monkek123king.github.io/CLD_page/)
  - **[prior notes]** Input is the image plus one bounding box per layer.
- **Referring Layer Decomposition / RefLayer (arXiv 2602.19358, Feb 2026)**
  - **Input:** an RGB image plus a prompt, which can be points, boxes, a mask, text, or a combination.
  - **Output:** complete RGBA layers.
  - No code or weights are linked. Source: [arXiv](https://arxiv.org/abs/2602.19358)
- **LiWi: Layering in the Wild (arXiv 2605.14552)**
  - Fine-tunes Qwen-Image-Layered on LiWi-100K (natural photos). Outputs fg, bg and an explicit shadow layer.
  - Model access is only by request form; the "Code" button links back to arXiv. Source: [project page](https://rassetmusty.github.io/LiWi)
- **OmniPSD image-to-PSD:** iterative extraction with Flux-Kontext and an RGBA VAE; not released (see Q1). Source: [arXiv HTML](https://arxiv.org/html/2512.09247v1)
- **[prior notes, not re-verified]**
  - **Qwen-Image-Layered:** 20B, Apache-2.0, variable number of layers (about 2–8); needs about 40 GB in bf16; hosted on fal for about $0.05 per image.
  - **LayerD (CyberAgent):** Apache-2.0, BiRefNet-based, repo archived, graphic design only.
  - **LayerDecomp (Adobe):** no weights.

### Inferences
- For the game, two-stage decomposition is best used as an import tool, not as the core loop.
- On a T4, CLD is the most plausible decomposer if FLUX.1-dev is run quantized (fp8/NF4). This is unmeasured.

### Gaps
- CLD's exact input schema and VRAM were not readable from the README.
- Whether Qwen-Image-2.1 can do multi-layer decomposition is unconfirmed.

## Q3. Which methods can regenerate ONE layer conditioned on the others, keeping coherence?

### Takeaway
Explicit conditional single-layer generation exists only in LASAGNA (FG_Gen and BG_Gen, unreleased) and LayerDiffuse SD1.5 (foreground given background and background given foreground, released). The joint models (PSDiffusion, DreamLayer, OmniPSD) document layer editing but not regenerating one layer while freezing the rest.

### Cited Findings
- LASAGNA's FG_Gen generates a foreground with shadows and reflections for a fixed background, and BG_Gen does the reverse. Source: [arXiv HTML](https://arxiv.org/html/2601.15507)
- DreamLayer supports moving and resizing layers in latent space with consistent shadows and completing hidden parts, but describes no per-layer re-prompting. Source: [arXiv HTML](https://arxiv.org/html/2503.12838)
- PSDiffusion claims layer-wise editing but gives no single-layer regeneration procedure. Source: [arXiv HTML](https://arxiv.org/html/2505.11468v2)
- **[prior notes]** LayerDiffuse SD1.5 has released fg-given-bg and bg-given-fg weights (OpenRAIL-M). Source: [HF](https://huggingface.co/LayerDiffusion/layerdiffusion-v1)

### Inferences
- With joint models, a regeneration that keeps the other layers fixed would need inpainting-style latent freezing. That is untested and not supported by the papers.
- The runnable path for "re-prompt one layer so it still fits" is LayerDiffuse fg-given-bg, with the background being the composite of all other layers. Qwen-Image-2.1 reference-image editing is a non-commercial alternative.

### Gaps
- No method reports quantitative results for regenerating one layer in a multi-layer stack while the rest stay frozen.
