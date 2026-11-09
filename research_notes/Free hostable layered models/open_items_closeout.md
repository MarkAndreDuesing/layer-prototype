# Open items closeout (checked 2026-11-07)

## 1. Licenses
| Item | Finding | Source |
|---|---|---|
| OmniPSD code | CC BY-NC-SA 4.0 (LICENCE file), NON-COMMERCIAL | https://github.com/showlab/OmniPSD , https://raw.githubusercontent.com/showlab/OmniPSD/main/LICENCE |
| OmniPSD weights | No own weights linked in README; only dataset subset lc03lc/OmniPSD_Layered_Poster. Depends on FLUX.1-dev + FLUX.1-Kontext-dev (non-commercial) | same |
| PrismLayers / ART+ code | MIT; ART+ ckpts hosted on Google Drive (license not stated separately) | https://github.com/redredsheep/PrismLayers |
| PrismLayersPro dataset | Apache-2.0 (gating not visible) | https://huggingface.co/datasets/artplus/PrismLayersPro |
| PrismLayersPlus datasets | Apache-2.0 | https://huggingface.co/datasets/artplus/PrismLayersPlus |
| ART original (microsoft/art-msra) | MIT code, but weights + inference code REMOVED 23 Jul 2025 (training data concern) | https://github.com/microsoft/art-msra |
| ART-Release/ART_v1.0 (HF) | HTTP 401: removed or private | https://huggingface.co/ART-Release/ART_v1.0 |
| SAM 3 | "SAM License" (Meta, updated 19 Nov 2025). Royalty-free use/modify/distribute; does not explicitly forbid commercial use; bans military/weapons/ITAR uses; derivatives must carry same license. HF repo gated (share contact info, Meta privacy policy) | https://github.com/facebookresearch/sam3/blob/main/LICENSE , https://huggingface.co/facebook/sam3 |
| OmniEraser HF weights | Card says Apache-2.0, but it is a FLUX.1-dev finetune/LoRA | https://huggingface.co/theSure/Omnieraser |
| OmniEraser GitHub | No LICENSE file seen | https://github.com/PRIS-CV/Omnieraser |
| FLUX.1-dev base | Non-commercial license; commercial use needs BFL license. Outputs may be used commercially (with conditions) | https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md |
| FLUX.2 klein 4B | Apache-2.0, ~13GB VRAM | https://huggingface.co/black-forest-labs/FLUX.2-klein-4B |

Practical: OmniEraser's Apache tag does not remove the FLUX.1-dev non-commercial restriction on running the base model.

## 2. Notebooks
- FLUX.2 klein 4B: HF auto-generated Colab/Kaggle notebook links on card (https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/colab, /kaggle). Model card says ~13GB, so a T4 (15GB) is borderline. No T4 timing found.
- ComfyUI on Kaggle: kayas881/comfyui-kaggle (T4 15GB, Pinggy tunnel; 1 star, unverified). https://github.com/kayas881/comfyui-kaggle
- LayerDiffuse: no Colab/Kaggle in rootonchair/diffuser_layerdiffuse (MIT, has --cpu-offload), huchenlei/ComfyUI-layerdiffuse (Apache-2.0), sd-forge-layerdiffuse (Apache-2.0). Official repo listed Colab as "coming soon". HF Space: radames/LayerDiffuse-gradio-unofficial.
- Qwen-Image-Layered: no dedicated notebook found; Comfy-Org fp8mixed repack exists. https://huggingface.co/Comfy-Org/Qwen-Image-Layered . T4 has no native bf16.
- Qwen-Image-2.1: no notebook found; Unsloth doc says GGUF ~11GB VRAM. https://unsloth.ai/docs/jp/moderu/qwen-image-2.1.md

## Unconfirmable
- No T4 VRAM/timing measurements were published for any of these models.
- ART+ Google Drive ckpt license: not stated; base ART weights were withdrawn, so redistribution status is unclear.
- Gating of PrismLayersPro: not visible without login.
- SAM 3 commercial use: implied by license text, not explicitly stated. Not legal advice.
- Kaggle code search is not indexed well by web search, so some notebooks may have been missed.
