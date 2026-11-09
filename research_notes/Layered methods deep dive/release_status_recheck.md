# Release status re-check: layered-image methods (2026-11-09)

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
