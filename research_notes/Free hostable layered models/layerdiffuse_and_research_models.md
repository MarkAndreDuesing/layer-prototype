# LayerDiffuse and layered-generation / layer-decomposition research models (checked 2026-11-07)

Pages were opened 2026-11-07 unless marked "(search snippet only)". VRAM numbers were not published by any of these projects. Where this file gives VRAM figures, they are inferences from the base model.

## 1. LayerDiffuse (SD1.5 / SDXL): availability, license, modes, tooling

### Takeaway
This is the most practical free option. Code is Apache-2.0, weights are CreativeML OpenRAIL-M, and it supports all four modes. Mature diffusers and ComfyUI ports exist. SD1.5 and SDXL both fit on a T4 16 GB, and SDXL has a CPU-offload flag.

### Cited Findings
- The main repo is Apache-2.0. It links to the Forge implementation (layerdiffusion/sd-forge-layerdiffuse) and to LayerDiffuse_DiffusersCLI. The official "Gradio + Diffusers + Colab" and the HF Space are still marked "Coming soon". Training code and dataset are only planned. — [lllyasviel/LayerDiffuse](https://github.com/lllyasviel/LayerDiffuse)
- The HF weights repo LayerDiffusion/layerdiffusion-v1 is licensed creativeml-openrail-m, which allows research and prototype use. — [HF LayerDiffusion/layerdiffusion-v1](https://huggingface.co/LayerDiffusion/layerdiffusion-v1)
- The diffusers port is MIT-licensed. SD1.5 supports fg-only, joint fg+bg+blend (batch 3), fg on a given bg, and bg for a given fg (batch 2), and works with ControlNet and IP-Adapter. SDXL supports fg-only via attention injection (LoRA) or conv injection, inpainting, fg/bg-to-blend, and blend-to-fg/bg. SDXL scripts have a `--cpu-offload` flag (Accelerate). The README does not mention Colab. — [rootonchair/diffuser_layerdiffuse](https://github.com/rootonchair/diffuser_layerdiffuse)
- ComfyUI-layerdiffuse is Apache-2.0 and supports "only SDXL/SD15". The joint fg+bg+blend workflows are SD15-only. It has no Flux support, and its issue tracker had 66 open issues and 6 open PRs when checked. — [huchenlei/ComfyUI-layerdiffuse](https://github.com/huchenlei/ComfyUI-layerdiffuse)

### Inferences
- **SD1.5 (~4 GB fp16):** runs comfortably on a Kaggle or Colab T4.
- **SDXL (~7 GB fp16 UNet):** fits on a T4 at fp16. CPU offload makes it safer on Colab's ~12 GB system RAM.
- **Recommended free path:** a Kaggle or Colab notebook that pip-installs rootonchair/diffuser_layerdiffuse, or runs ComfyUI with huchenlei's nodes.

### Gaps
- I did not verify a maintained, working public Colab or Kaggle notebook. The official one is still "Coming soon".

## 2. LayerDiffuse-Flux (FireRedTeam)

### Takeaway
Code and weights are released: a LoRA plus TransparentVAE on FLUX.1-dev. The FLUX.1-dev Non-Commercial License governs any use, so it is OK for research and prototypes but not commercial use. Running it on a T4 is possible only with heavy offload or quantization.

### Cited Findings
- The repo shows no LICENSE file. Weights are on HF RedAIGC/Flux-version-LayerDiffuse (`layerlora.safetensors` and `TransparentVAE.pth`). It needs a FLUX.1-dev checkpoint and provides T2I and I2I demo scripts. Requirements are torch 2.3.0 and cu121. No VRAM figure is given. — [FireRedTeam/LayerDiffuse-Flux](https://github.com/FireRedTeam/LayerDiffuse-Flux)
- The HF weights repo is labelled license afl-3.0 and has no hardware notes. — [HF RedAIGC/Flux-version-LayerDiffuse](https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse)

### Inferences
- FLUX.1-dev is a 12B DiT, about 24 GB at bf16. A T4 has no bf16 and only 16 GB, so it would need fp8/NF4/GGUF quantization or sequential CPU offload. Colab's 12 GB system RAM is a likely bottleneck. Kaggle (~29 GB RAM) is the more realistic option.
- The same FLUX.1-dev constraints apply to ART, CLD, and OmniPSD below.

### Gaps
- I found no official ComfyUI node or notebook. T4 runtime has not been verified.

## 3. Research models: code and weights status

### Takeaway
Of the research models, only CLD, ART (originally), LayerD, See-through, OmniPSD (code only), and PrismLayers (code and data) have public releases. All the generative ones sit on FLUX.1-dev, which means a non-commercial license and a hard fit on 16 GB. LayerD is the outlier: it is lightweight and runs on CPU.

### Cited Findings
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

### Inferences
- **Fits a free T4 easily:** LayerDiffuse SD1.5/SDXL, LayerD (BiRefNet, ~1 GB, CPU-capable), and See-through (NF4 variant).
- **Needs quantization or offload on 16 GB, so Kaggle is preferred:** LayerDiffuse-Flux, CLD, ART, and OmniPSD, all on FLUX.1-dev. The T4's lack of bf16 adds friction.
- **Unlikely to be practical on a free T4:** Qwen-Image-Layered (20B).
- **Commercial use:** the FLUX.1-dev Non-Commercial License restricts every FLUX-based option. Research and prototype use is fine.

### Gaps
- No project publishes VRAM figures.
- I did not confirm official ComfyUI or diffusers support for the FLUX-based research models, or for ready Kaggle/Colab notebooks for any of them.
- ART's HF status is unclear (401).
- PrismLayers license and ART+ weights are unconfirmed.
- I did not open the OmniPSD license contents or the official DreamLayer and LayerFusion pages.
