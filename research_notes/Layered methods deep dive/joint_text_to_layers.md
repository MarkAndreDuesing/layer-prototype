# Joint text-to-multi-layer methods: deep dive (checked 2026-11-09)

This builds on `../Coherent layered generation pipeline/other_layered_methods.md`. That file covers paper details. This one adds checks of release status on GitHub, Hugging Face and project pages.
**[U]** = not verified this pass (taken from memory or a secondary source).

## Summary table

| Method | Base | Code | Weights | Licence | T4 (fp16, 16 GB) | Editing ops | Path for us |
|---|---|---|---|---|---|---|---|
| **PSDiffusion** (2505.11468) | SDXL + rank-256 LoRA + RGBA VAE | No. Project page says "Code (Coing Soon)" | No | - | Would fit (SDXL fp16 is fine) | "layer-wise editing" claimed, no procedure | (b) training-free approximation; (c) LoRA retrain blocked (no dataset release [U]) |
| **DreamLayer** (2503.12838) | SD1.5, K/V fine-tune | No ("Coming Soon") | No. **Dataset released** ([HF LL3RD/DreamLayer](https://huggingface.co/datasets/LL3RD/DreamLayer), ~400k samples per secondary source) | dataset licence [U] | Would fit easily | Move, flip and resize in latent space with shadows following; occlusion completion | (b) reimplement attention tricks; (c) K/V fine-tune on its dataset is feasible on Kaggle |
| **OmniPSD** (2512.09247) | FLUX.1-dev (2x2 grid, in-context) + RGBA VAE; Kontext for image-to-PSD | **Yes** ([showlab/OmniPSD](https://github.com/showlab/OmniPSD): train/inference/diffsynth) | **No** OmniPSD LoRA or RGBA-VAE link found | repo LICENSE type not readable; prior notes say CC BY-NC-SA [U] | Poor: 12B Flux in fp16 risks overflow on T4; needs NF4/fp8 + offload | Text-to-PSD, image-to-PSD; no single-layer re-prompt | (c) training code exists, but Flux LoRA training on a T4 is impractical |
| **LASAGNA** (2601.15507) | ~2B DiT (Adobe-internal), RGBA VAE, T5 | No ("Coming Soon") | No | - | Would fit by size | Text2All, FG_Gen (fg given bg, with shadows/reflections), BG_Gen | Concept only; FG_Gen ≈ LayerDiffuse SD1.5 fg-given-bg (released) |
| **LayerFusion** (2412.04460) | SDXL (RealVisXL v4) + LayerDiffuse fg weights | No ("Code (coming soon)") | **Not needed: training-free** | - | Yes. Same stack we run (paper used 1 L40) | fg/bg/blend triplet with harmonised blend | **(b) reimplement directly on our LayerDiffuse SDXL** |
| **ART** (2502.18364) | FLUX.1-dev + LoRA, anonymous-region layout [U on base] | [microsoft/art-msra](https://github.com/microsoft/art-msra): **withdrawn 2025-07-23** ("removed the model weights and inference checkpoints") | Removed | MIT (repo) | - | - | Do not use (withdrawn over data provenance) |
| **ART+ / PrismLayers** (2505.22523) | ART (Flux-based [U]) fine-tuned on PrismLayersPro | **Yes** ([redredsheep/PrismLayers](https://github.com/redredsheep/PrismLayers)) | **Yes, Google Drive**: base ART ckpt, artplus_PrismLayersPro(100k) ckpt, transparency VAE, pre-fuse LoRA (links in README); sizes not stated (expect ~24 GB Flux-class [U]) | MIT (repo); dataset [PrismLayersPro](https://huggingface.co/datasets/creative-graphic-design/PrismLayersPro) Apache-2.0 | Hard: Flux-class DiT, fp16 overflow risk, needs quantisation + CPU offload; better on LRZ GPU | Global caption (21 styles) + per-layer **boxes**, variable layer count; no single-layer regen documented | **(a) run on LRZ/A100**; T4 only with NF4 [U] |
| **CLD** (2511.16249) | FLUX.1-dev + alimama inpainting ControlNet + LoRAs + transparent VAE | Yes ([monkek123King/CLD](https://github.com/monkek123King/CLD), MIT) | Yes ([thuteam/CLD](https://huggingface.co/thuteam/CLD), Apache-2.0; Flux-dev NC licence applies) | see left | Hard (Flux + ControlNet); quantised and unmeasured | **image -> layers** (decomposition, box per layer), not joint text-to-layers | Import/decomposition tool, not the core loop |
| **LayerDiffuse-Flux** | FLUX.1-dev | weights only on HF; GitHub RedAIGC repo [U] | Yes ([HF](https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse)), AFL-3.0 on card; Spaces exist (snapplv/ginigen) | AFL-3.0 + Flux-dev NC | Same Flux problems | Single RGBA layer only | Use the HF Space for quality comparison only |
| **Qwen-Image-2.1** | 7B DiT, Qwen3-VL encoder | diffusers `QwenImage21Pipeline` | Yes ([HF](https://huggingface.co/Qwen/Qwen-Image-2.1)), **BF16** safetensors | Qwen Research (non-commercial; fine for us) | 7B ≈ 14 GB fp16, plus encoder: needs `enable_model_cpu_offload()`; fp16 overflow is a risk (same fix as our Qwen-Image-Layered workaround) | Text->RGBA (via prompt phrase), editing transparent layers, subject extraction, ≤10 reference images; **no multi-layer stack** | (a) one layer per call + reference-image consistency |

## Per method notes

### PSDiffusion
- Status: the project page shows only "Code (Coing Soon)" (sic) ([page](https://dingbang777.github.io/PSDiffusion_Website)). The WACV 2026 version is the same as [arXiv v2](https://arxiv.org/html/2505.11468v2).
- Inference: a global prompt plus N layer prompts run as parallel branches (global RGB and N RGBA). Layer cross-attention reweighting takes each layer's location from the global branch. Partial joint self-attention lets each layer attend to the global branch inside its own mask.
- Retraining: about 60 A800 GPU-hours. That is roughly 300+ T4-hours [U, rough ×5 scaling], which is not feasible on Kaggle's 30 h/week. The dataset has not been released [U].
- **(b) Training-free on our stack:** the two attention mechanisms are inference-time masks and attention sharing. Without the LoRA, the branches are untrained to cooperate, but LayerFusion (below) shows this kind of sharing works without training on LayerDiffuse SDXL.

### DreamLayer
- The code is "Coming Soon", but the **dataset is released** ([HF](https://huggingface.co/datasets/LL3RD/DreamLayer)). It is the only joint method with open multi-layer training data that fits SD-scale models.
- Its components: Context-Aware Cross-Attention (layout from the global layer), Layer-Shared Self-Attention, and Information-Retained Harmonization (latent blending that produces shadows).
- (c) It fine-tunes only K/V (Custom-Diffusion style) on SD1.5. A K/V or LoRA fine-tune on SDXL at 512–768 px with fp16 and gradient checkpointing is plausible on one T4 within tens of hours [U]. This is the most realistic "train our own joint model" option.

### OmniPSD
- Training and inference code is open ([GitHub](https://github.com/showlab/OmniPSD)). Running inference means editing `inference/infer_psd_flux.py` (set `LORA_PATH` and the RGBA-VAE path), then `python inference/infer_psd_flux.py`. The OmniPSD LoRA and the "FLUX.1-dev-alpha" VAE have **no download link**. The code cannot run without them.
- Flux is 12B. bf16 is not available on a T4, and fp16 overflows in Flux activations are known [U]. Training a Flux LoRA on a T4 is not realistic. Revisit on LRZ.

### LASAGNA
- Code is "Coming Soon" and the page links nothing else ([page](https://rayjryang.github.io/LASAGNA-Page/)). The base model is Adobe-internal, so release is unlikely.
- What it teaches us: FG_Gen and BG_Gen are exactly the "regenerate one layer conditioned on the rest" operation. Our released equivalent is LayerDiffuse SD1.5 fg-given-bg and bg-given-fg.

### LayerFusion (most directly usable idea)
- **Training-free**, on SDXL (RealVisXL v4) + LayerDiffuse foreground weights, run on one L40 ([arXiv HTML](https://arxiv.org/html/2412.04460v1)). The code is "coming soon".
- Procedure:
  1. Run the fg (LayerDiffuse), bg and blend branches in parallel.
  2. At t = 0.8T, extract a structure prior (the fg model's last self-attention map) and a content prior (cross-attention at the <EOS> token). Turn them into soft and hard masks.
  3. Blend the blend-branch and fg-branch attention outputs with these masks.
  4. Share attention in both self- and cross-attention between the blend and bg branches.
- Every component is training-free, and it is the same model family we already run on Kaggle. Extending it from 1 fg to N fg branches is our own extension: plausible, but not in the paper.

### ART / ART+ (PrismLayers)
- The original ART was withdrawn on 2025-07-23 because it "was trained using data that may have come from illegal sources" ([repo](https://github.com/microsoft/art-msra)).
- ART+ checkpoints, including a base ART checkpoint, are on Google Drive via the [PrismLayers README](https://github.com/redredsheep/PrismLayers) (MIT).
  - Run: `python multi_layer_gen/demo.py -c multi_layer_gen/demo.yaml -i inference_layout/test_1.json`.
  - The JSON gives `whole_image.caption` ("This is a [STYLE] style image. ...") plus `layout` boxes.
- Caveats:
  - The base ART checkpoint shipped here seems to be the same weights Microsoft withdrew [U]. The data-provenance concern carries over.
  - Before downloading, check the Drive folder sizes and whether FLUX.1-dev weights are needed separately.
- This is the **only joint text-to-multi-layer model with downloadable weights** found. It supports a variable number of layers and an LLM can produce the boxes, so it fits the game loop well. Hardware is the blocker: aim for LRZ, or test NF4 on a T4.

### CLD, LayerDiffuse-Flux, Qwen-Image-2.1
- These are runnable but not joint text-to-multi-layer (see table). Qwen-Image-2.1 ships BF16, so it needs the same fp16 workaround as Qwen-Image-Layered.

## Top 3 most promising for the game
1. **LayerFusion-style training-free harmonisation on LayerDiffuse SDXL.** It needs no weights or training, runs on hardware we already use, and extends to N layers by combining it with the PSDiffusion attention-mask ideas. Single-layer regeneration means rerunning one fg branch with shared attention to the frozen blend/bg.
2. **ART+ (PrismLayers) released weights.** It is the only downloadable joint model (global caption + LLM boxes → variable RGBA layers). Run it on LRZ, or try a quantised build on a T4. Check the provenance of the base ckpt and the Drive sizes before downloading.
3. **DreamLayer dataset + our own small fine-tune.** The open 400k-sample multi-layer dataset makes a K/V/LoRA fine-tune of SD-scale models on Kaggle plausible. It also adds trained latent move/resize with shadow harmonisation.

## Gaps
- No VRAM figures are published for any method.
- Sizes of the ART+ Drive checkpoints and the base model ART uses are unverified.
- The OmniPSD licence type was not read.
- The DreamLayer dataset's licence and size were not opened.
