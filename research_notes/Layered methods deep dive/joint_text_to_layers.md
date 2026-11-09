# Joint text-to-multi-layer methods: deep dive (checked 2026-11-09)

> **Update 2026-11-09 evening:** ART+ provenance is now clearer: the ART base LoRA is byte-identical (SHA-256) to `pre_trained_LoRA` in the official `thuteam/CLD` repo (Apache-2.0). A Kaggle 2× T4 notebook for ART+ (GGUF Q4 FLUX, unmerged LoRAs, `fp16_mixed`) is ready: `prototype/tools/make_artplus_trial.py`. Release status of all methods: section "Release status" below (moved in from `release_status_recheck.md`). Paper facts of all methods are collected here too, so this is the one methods note. The research text below is unchanged.

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

## Release status (verified 2026-11-09)

How I checked: the HF Hub API (models, spaces and datasets search), the GitHub repo search API (it hit the anonymous rate limit partway through), raw READMEs and repo contents, and web search. "Unverified" means I could not open it directly.

| Method | Status | URLs | Licence | Base / size | Date / notes |
|---|---|---|---|---|---|
| PSDiffusion (2505.11468) | Not released | github.com/dingbang777/PSDiffusion: README only, both TODOs open ("Release Inter-Layer Dataset", "Release inference code and pretrained model") | n/a | n/a | Last push 2025-05-15. No HF models or spaces |
| DreamLayer (2503.12838) | Dataset only | huggingface.co/datasets/LL3RD/DreamLayer | Apache-2.0 (dataset) | n/a | Dataset updated 2025-10-19. LL3RD's HF models are DreamFuse, trained-sd3 and DreamShot. None is DreamLayer. GitHub "DreamLayer" hits are unrelated (a DreamLayer-Eval UI app and others) |
| LASAGNA (2601.15507) | Not found | rayjryang.github.io/LASAGNA-Page | n/a | n/a | arXiv says LASAGNA-48K and the benchmark "will be released". Nothing is linked on HF Papers (search result, unverified on-page) |
| LayerFusion (2412.04460) | Not found | layerfusion.github.io (the layerfusion/layerfusion.github.io repo is the website only) | n/a | Training-free | No code repo found |
| OmniPSD (2512.09247) | Partial: code + dataset subset, no weights | github.com/showlab/OmniPSD (train/, inference/, diffsynth/). Dataset: huggingface.co/datasets/lc03lc/OmniPSD_Layered_Poster (mirror: creative-graphic-design-archive/OmniPSD_Layered_Poster) | Code: CC BY-NC-SA 4.0 (LICENCE file). Dataset: Apache-2.0 | FLUX.1-dev LoRA (Text-to-PSD) and 4 FLUX.1-Kontext-dev LoRAs. Needs an RGBA-VAE ("FLUX.1-dev-alpha") | Repo last pushed 2026-05-25, dataset 2026-05-25. README has placeholder `LORA_PATH` and RGBA-VAE paths and links no weights. HF user lc03lc has no OmniPSD model. ZhengNan-coder/OmniPSD-Code is empty |
| LayeringDiff (2501.01197) | Not found | none | n/a | n/a | Web search found no repo (GitHub API rate-limited for this name) |
| LayerDecomp / Generative Image Layer Decomposition with Visual Effects (Adobe) | Not released | github.com/Rayjryang/LayerDecomp: README holds only the project-page link | n/a | n/a | Last push 2025-03-03 |
| DiffDecompose | Not released | github.com/Wangzt1121/DiffDecompose: "will release the code and datasets after being accepted" | n/a | n/a | Last push 2025-05-16 |
| Text2Layer | Not found | none (GitHub "text2layers" is unrelated) | n/a | n/a | |
| Referring Layer Decomposition / RefLayer (2602.19358, ICLR26) | Not found | yaojie-shen.github.io/project/RLD (poster: "will release dataset, evaluation tools, and model") | n/a | n/a | github.com/evnlme/RefLayer is an unrelated Krita plugin |
| LayerCraft | Partial: code + OIN LoRA | github.com/PeterYYZhang/LayerCraft. README links the OIN LoRA checkpoints on Google Drive: drive.google.com/drive/folders/1OlDKIHBEs5g-OtIzX-70CZGIHTNrU_D_ | GitHub reports no licence | FLUX LoRA (Object Integration Network) | Last push 2026-05-03. Drive folder size not checked (unverified) |
| LiWi | Not found | none | | | Name too generic for HF/GH search. Nothing layer-related found |
| PixelART | Not found | none | | | Search only hits pixel-art LoRAs |
| ART (microsoft/art-msra, withdrawn) | Withdrawn. One unofficial ART+ checkpoint mirror exists | huggingface.co/ART-Release/ART_v1.0 now returns 401 (gone or private). github.com/microsoft/art-msra returned 504 (unverified). Unofficial mirror: huggingface.co/24yearsold/PrismLayers_ckpt ("Unofficial hf model repo for PrismLayers", github.com/redredsheep/PrismLayers) | Mirror has no licence tag. Original withdrawn over training-data concerns (ComfyUI Wiki, July 2025; single source) | FLUX.1-dev. Contains base/ and artplus_PrismLayersPro100k_ckpt/ LoRA + layer_pe, trans_vae, pre_fuse_lora, TIPS ckpt. About 5.56 GB total | Created 2026-04-24. Provenance and legality unverified. Treat with caution |

Not covered: Papers-with-Code (now redirects to HF Papers). ComfyUI custom-node searches turned up nothing for these names on HF or the web. The GitHub search for ComfyUI nodes was rate-limited, so absence there is not confirmed.

## Paper facts per method

From the 2026-11-07 rounds ([other_layered_methods.md](../Coherent%20layered%20generation%20pipeline/other_layered_methods.md) Q1 and [prompt_to_layers_generation.md](../Layered%20image%20generation%20models/prompt_to_layers_generation.md)); release statements there are superseded by the table above.

#### Takeaway
Several papers generate all layers jointly from a global prompt plus per-layer prompts: PSDiffusion, DreamLayer, OmniPSD, LayerFusion, LASAGNA and ART. None of them has verified, downloadable open weights today. The only runnable native text-to-RGBA options are single-layer (or fg+bg) models: LayerDiffuse, LayerDiffuse-Flux and Qwen-Image-2.1. These generate one transparent object per call, not a coordinated multi-layer stack.

#### Cited Findings
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

#### Inferences
- The "single prompt -> coherent layer stack" architecture (a global branch plus per-layer branches with shared attention) is well established in papers: PSDiffusion and DreamLayer on SD, OmniPSD on FLUX.
- PSDiffusion is architecturally closest to an SDXL+LayerDiffuse stack, so with released weights it would fit a T4. As of now it cannot be run.
- A practical stand-in is to approximate it: an LLM decomposes the scene into per-layer prompts and a layout, LayerDiffuse generates each layer, and a harmonization pass follows.

#### Gaps
- No VRAM figures are published for any joint model.
- PSDiffusion, DreamLayer and LayerFusion: the GitHub orgs were not checked directly, only the paper and project-page links.
- Text2Layer, MuLAn, LayeringDiff, PixelART and RevealLayer were not researched in this pass because of the call budget.

### Additional methods (from prompt_to_layers_generation.md)

- **ART (Anonymous Region Transformer)**, arXiv 2502.18364 (Feb 2025, CVPR 2025, Microsoft Research Asia): built on FLUX.1[dev], using 3D RoPE and region-wise attention. It is more than 12x faster than full attention and has fewer layer conflicts — [arXiv](https://arxiv.org/abs/2502.18364v1), [HF paper](https://huggingface.co/papers/2502.18364), [MSR](https://www.microsoft.com/en-us/research/?p=1144107)
  - Inputs: a global prompt plus an anonymous region layout (boxes with no per-layer captions), from an LLM planner or set by hand. It claims "50+ layers". The license is MIT, but **weights and code were removed on 2025-07-23** because the training data "may have come from illegal sources" — [GitHub microsoft/art-msra](https://github.com/microsoft/art-msra)
- **PrismLayers / PrismLayersPro**, arXiv 2505.22523 (May 2025): an open synthetic dataset of about 200K multi-layer images, with a 20K "Pro" subset. Images average about 7 layers (range 3–14). Built with LayerFLUX (single transparent layer) and MultiLayerFLUX (composes layers from a layout given by a reference image or an LLM). ART fine-tuned on it ("ART+") won 60% of user-study comparisons — [arXiv PDF](https://arxiv.org/pdf/2505.22523), [CatalyzeX](https://www.catalyzex.com/paper/prismlayers-open-data-for-high-quality-multi). Dataset: [HF artplus/PrismLayersPro](https://huggingface.co/datasets/artplus/PrismLayersPro) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2505.22523)
- **LayeringDiff**: generates a composite image, then extracts layers from it, which avoids large-scale training — [fugumt summary](https://fugumt.com/fugumt/paper_check/2412.04460v1_enmode)
- **Text2Layer** (2023): an autoencoder for layered images, with diffusion producing bg, fg, mask and composite together — same source
- **TransPixar / TransPixeler** (Adobe + HKUST, arXiv 2501.03006): RGBA text-to-video. Adds alpha tokens with a LoRA on a DiT. A CogVideoX-5B LoRA is released and Mochi training code is open — [HF wileewang/TransPixar](https://huggingface.co/wileewang/TransPixar), [project](https://wileewang.github.io/TransPixar)
- Newer 2026 work seen only as titles: **LiWi: Layering in the Wild** (arXiv 2605.14552) and "Controllable layered image generation for real-world editing" (arXiv 2601.15507) — [LiWi PDF](https://arxiv.org/pdf/2605.14552)

Short 11-07 summaries of methods also covered above (kept for their extra details):

Research methods (2023–2026) generate several RGBA layers in one pass from a global prompt, either with an anonymous layout (ART) or with per-layer prompts (PSDiffusion, DreamLayer, LayerFusion). Most give harmonized lighting because the layers are denoised jointly. Few have usable open weights: ART's were withdrawn.

- **PSDiffusion**, arXiv 2505.11468 (WACV 2026): takes a global prompt plus layer-specific prompts (evaluated with 3 foreground + 1 background + global). Layer prompts can come from an LLM or the user. Layout is taken from cross-attention maps of a full RGB image generated together with the layers, which keeps the layers harmonized — [arXiv](https://arxiv.org/html/2505.11468), [WACV PDF](https://openaccess.thecvf.com/content/WACV2026/papers/Huang_PSDiffusion_Harmonized_Multi-Layer_Image_Generation_via_Layout_and_Appearance_Alignment_WACV_2026_paper.pdf)
- **DreamLayer**, arXiv 2503.12838 (ICCV 2025): generates transparent fg and bg layers simultaneously, using Context-Aware Cross-Attention, Layer-Shared Self-Attention and Information-Retained Harmonization. Also supports image-to-layer decomposition and latent editing — [arXiv](https://arxiv.org/html/2503.12838v1), [ICCV](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_DreamLayer_Simultaneous_Multi-Layer_Generation_via_Diffusion_Model_ICCV_2025_paper.html)
- **LayerFusion**, arXiv 2412.04460 (Dalva et al., Dec 2024): harmonized multi-layer text-to-image using generative priors on latent diffusion — [fugumt](https://fugumt.com/fugumt/paper_check/2412.04460v1_enmode)
- **OmniPSD**, arXiv 2512.09247 (Dec 2025, NUS + Lovart): Flux-based in-context text-to-PSD and image-to-PSD. Reportedly trained on about 200K real PSDs (from a review, unverified). Project page: [showlab.github.io/OmniPSD](https://showlab.github.io/OmniPSD/) — [arXiv](https://arxiv.org/pdf/2512.09247)

#### Inferences
- Joint methods (PSDiffusion, DreamLayer, ART) give the best cross-layer lighting harmony. But regenerating one layer means re-running everything, or using inpainting-style tricks with a fixed seed.
- Of the joint methods, PSDiffusion's input format (global + per-layer prompts) is the closest to the game's per-object prompt design.

#### Gaps
- Weights, licenses and VRAM were not verified for DreamLayer, PSDiffusion, LayerFusion, OmniPSD or LayeringDiff. Assume code-only or no release until checked.
- Not covered by search: LayerCraft, LayerTracer (vector/SVG layers), Alfie, MuLAN dataset, and "Generative Image Layer Decomposition" (LayerDecomp, Adobe).

## Code and weights table (2026-11-07)

From [layerdiffuse_and_research_models.md](../Free%20hostable%20layered%20models/layerdiffuse_and_research_models.md) §3; older than the release table above, kept for rows it does not cover (See-through, LayerD details).

#### Takeaway
Of the research models, only CLD, ART (originally), LayerD, See-through, OmniPSD (code only), and PrismLayers (code and data) have public releases. All the generative ones sit on FLUX.1-dev, which means a non-commercial license and a hard fit on 16 GB. LayerD is the outlier: it is lightweight and runs on CPU.

#### Cited Findings
| Model | Code | Weights | License | Base | Source |
|---|---|---|---|---|---|
| CLD (2511.16249) | Yes, MIT | HF thuteam/CLD (LoRAs + trans_vae, Dec 2025) | MIT code, FLUX.1-dev NC for base | FLUX.1-dev + alimama FLUX ControlNet-Inpainting-Alpha | [monkek123King/CLD](https://github.com/monkek123King/CLD) |
| ART | microsoft/art-msra (search snippet only) | HF ART-Release/ART_v1.0. It returned HTTP 401 on 2026-11-07, so it may now be gated or removed | Not confirmed | FLUX.1-dev (search snippet only) | [HF ART_v1.0 discussion](https://huggingface.co/ART-Release/ART_v1.0/discussions/1/files) |
| PrismLayers / ART+ | github.com/redredsheep/PrismLayers | Dataset on HF. A separate ART+ weights link is not confirmed | Not stated | ART fine-tune | [prism-layers.github.io](https://prism-layers.github.io/) |
| OmniPSD (2512.09247) | Yes, showlab/OmniPSD, with train/infer scripts | No download link in README | LICENCE file present, type not shown | FLUX.1-dev (T2PSD), FLUX.1-Kontext-dev (I2PSD) | [showlab/OmniPSD](https://github.com/showlab/OmniPSD) |
| LayerD (2509.25134, ICCV25) | Yes, Apache-2.0. The repo is archived (read-only) | HF cyberagent/layerd-birefnet | Apache-2.0 code | BiRefNet matting + inpainting. Has a CPU device example | [CyberAgentAILab/LayerD](https://github.com/CyberAgentAILab/LayerD) |
| See-through (anime, SIGGRAPH 2026) | Yes, shitagaki-lab/see-through | HF layerdifforg/*, plus an NF4 Marigold variant for low VRAM | Apache-2.0 code, Open RAIL++-M on weights (search snippet only) | LayerDiff 3D + Marigold | [shitagaki-lab/see-through](https://github.com/shitagaki-lab/see-through) |
| DiffDecompose (2505.21541) | Not released. README: "We will release the code and datasets after being accepted" | None | None | Not stated | [Wangzt1121/DiffDecompose](https://github.com/Wangzt1121/DiffDecompose) |
| PSDiffusion (2505.11468, WACV26) | No repo found | None | n/a | n/a | [project page](https://dingbang777.github.io/PSDiffusion_Website) |
| RefLayer (2602.19358, ICLR26) | No code link on arXiv | None | n/a | n/a | [arXiv](https://arxiv.org/abs/2602.19358) |
| LiWi (2605.14552) | Project page only. Abstract says "will soon release" code and dataset | None | n/a | n/a | [arXiv](https://arxiv.org/abs/2605.14552) |
| 2601.15507 (LASAGNA, Adobe co-authors) | No code link. Releases the LASAGNA-48K dataset and LASAGNA-BENCH benchmark | None found | n/a | n/a | [arXiv](https://arxiv.org/abs/2601.15507) |
| LayerDecomp (Adobe, 2411.17864) | The project page lists only paper links | None found | n/a | n/a | [project page](https://rayjryang.github.io/LayerDecomp/) |
| RevealLayer (2605.11818), PixelART (2609.22789) | Listed as "paper only" | None | n/a | n/a | [awesome-image-layer-decomposition](https://github.com/discopotato3376-oss/awesome-image-layer-decomposition) (third-party list) |
| DreamLayer, LayerFusion | No repo found in search | None found | n/a | n/a | [search: OmniPSD similar papers](https://arxiv.deeppaper.ai/papers/2512.09247v1/similar) |
| Related: Qwen-Image-Layered | QwenLM/Qwen-Image-Layered | HF Qwen/Qwen-Image-Layered (Dec 19 2025) | Apache-2.0, 20B params (per list) | Qwen-Image | [awesome list](https://github.com/discopotato3376-oss/awesome-image-layer-decomposition) |

#### Inferences
- **Fits a free T4 easily:** LayerDiffuse SD1.5/SDXL, LayerD (BiRefNet, ~1 GB, CPU-capable), and See-through (NF4 variant).
- **Needs quantization or offload on 16 GB, so Kaggle is preferred:** LayerDiffuse-Flux, CLD, ART, and OmniPSD, all on FLUX.1-dev. The T4's lack of bf16 adds friction.
- **Unlikely to be practical on a free T4:** Qwen-Image-Layered (20B).
- **Commercial use:** the FLUX.1-dev Non-Commercial License restricts every FLUX-based option. Research and prototype use is fine.

#### Gaps
- No project publishes VRAM figures.
- I did not confirm official ComfyUI or diffusers support for the FLUX-based research models, or for ready Kaggle/Colab notebooks for any of them.
- ART's HF status is unclear (401).
- PrismLayers license and ART+ weights are unconfirmed.
- I did not open the OmniPSD license contents or the official DreamLayer and LayerFusion pages.

## Top 3 most promising for the game
1. **LayerFusion-style training-free harmonisation on LayerDiffuse SDXL.** It needs no weights or training, runs on hardware we already use, and extends to N layers by combining it with the PSDiffusion attention-mask ideas. Single-layer regeneration means rerunning one fg branch with shared attention to the frozen blend/bg.
2. **ART+ (PrismLayers) released weights.** It is the only downloadable joint model (global caption + LLM boxes → variable RGBA layers). Run it on LRZ, or try a quantised build on a T4. Check the provenance of the base ckpt and the Drive sizes before downloading.
3. **DreamLayer dataset + our own small fine-tune.** The open 400k-sample multi-layer dataset makes a K/V/LoRA fine-tune of SD-scale models on Kaggle plausible. It also adds trained latent move/resize with shadow harmonisation.

## Gaps
- No VRAM figures are published for any method.
- Sizes of the ART+ Drive checkpoints and the base model ART uses are unverified.
- The OmniPSD licence type was not read.
- The DreamLayer dataset's licence and size were not opened.
