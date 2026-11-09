# Other methods for coherent layered generation and decomposition (as of 2026-11-07)

> **Update 2026-11-09:** this is the 11-07 snapshot. Since then Qwen-Image-Layered runs on the free Kaggle 2× T4 (GGUF Q4 + `fp16_mixed`, 27–35 s per split), so "too large for a T4" no longer holds. CLD's code and weights were read in detail (`../Layered methods deep dive/heavy_models_recheck_2026-11-09.md`). Current status of every method: `reports/Layered methods - how to get them working.md`.

Scope note: 17 tool calls were used. Full text was read for PSDiffusion, DreamLayer, OmniPSD and LASAGNA. Only abstracts or project pages were read for CLD, RefLayer, LiWi, LayerFusion and PrismLayers. Items marked **[prior notes, not re-verified]** come from `reports/Layer based architecture models and hosting.md` and were not checked again in this pass.

## Q1. Native "single prompt -> coherent layers" methods

> Merged into [joint_text_to_layers.md](../Layered%20methods%20deep%20dive/joint_text_to_layers.md#paper-facts-per-method) on 2026-11-10 (moved verbatim, with its citations).

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
