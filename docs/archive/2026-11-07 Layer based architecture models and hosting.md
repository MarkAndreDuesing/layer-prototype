# Layer-based architecture: models and hosting

> **Status update (2026-11-09).** Written 11-07, before any T4 runs. Measured since (details in [findings-log.md](../log.md)): Qwen-Image-Layered 27–35 s per split on 2× T4; LayerDiffuse SDXL chain about 18 s per pass at 1024 px on one T4; `relocate()` about 4 s per move; RevealLayer (FLUX GGUF Q4) runs end to end at 69 s for 12 steps (output still being fixed). The "[est]" figures below for those models are superseded.

Reference for the storytelling-game prototype: every scene object is its own prompted, individually regenerable RGBA layer. Merges "Layered image generation models" and "Free hostable layered models", updated with the 2026-11-07 open-items closeout.

**[est]** = estimate from parameter counts or file sizes, not a tested run. No T4 timings or VRAM measurements are published for any model below.

## 1. Summary

- **Decision:** generate per object. The layer boundary must exist *before* generation, so re-rolling one object never touches the others. Layers (prompt, seed, RGBA, position) are the source of truth; any full-scene render is a disposable preview.
- **Generator:** LayerDiffuse on SDXL gives native alpha, has commercially usable licenses (Apache-2.0 code, OpenRAIL-M weights) and fits a T4. Start locally with SDXL-Turbo on CPU, then move to Kaggle or Modal T4.
- **Prompting:** a local Ollama LLM refines prompts and breaks a scene into objects. Later this moves to an API or to Kaggle/Colab.
- **Why not image-to-layers:** Qwen-Image-Layered and similar decomposers split a *finished image*. Regenerating one object means regenerating and re-splitting the whole scene, and the layers are not stable between runs.
- **Known cost:** objects generated in isolation don't share lighting. Mitigations are LayerDiffuse fg-given-bg, IC-Light SD1.5, or a low-denoise harmonization pass on the preview only.
- **Safety net:** any RGB generator plus MIT matting (BiRefNet, BEN2 base) produces the same layer format.

## 2. Architectures that work

**A. Per-object transparent generation (chosen).** Each object prompt goes straight to an RGBA output. LayerDiffuse encodes alpha as "latent transparency" for SD1.5 and SDXL. SD1.5 also supports joint fg+bg and **foreground given background**, the closest open match to "regenerate this object so it fits the scene". Qwen-Image-2.1 does the same natively, through a 64-channel RGBA VAE, but under a non-commercial license. Re-rolling a layer is one cheap call, and the generator can be swapped without changing the data model.

**B. Generate on a plain background, then cut out.** Prompt the object "on a plain grey background" with any model (SDXL-Turbo, FLUX.2 Klein 4B, Z-Image Turbo), then matte it with BiRefNet or BEN2 base. It runs on CPU, works with any generator, and is the fallback if LayerDiffuse quality or alpha edges disappoint. Edges are slightly worse on hair and glass, and there is one extra step per layer.

**C. Image-to-layers decomposition (import tool only).** This splits an existing picture into layers, for example when a user uploads a finished illustration. **Qwen-Image-Layered** (20B, Apache-2.0) takes one RGB image and returns a variable number of RGBA layers (2–8 on Replicate; the card suggests 4 at 640 px), and fills in hidden content. **It splits images, not prompts.** The optional prompt describes the whole image, and the model decides the groupings itself. You choose only the layer count, though any output layer can be decomposed again. For targeted extraction ("give me the fox"), CLD takes one bounding box per layer, or use Florence-2 + SAM 2.1/SAM 3 to segment and LaMa to fill the hole. None of these belong in the core loop.

## 3. Models (usable or feasible only)

T4 = 15–16 GB VRAM, no native bf16.

| Model | Role | License (commercial?) | VRAM / T4 fit | Link |
|---|---|---|---|---|
| **LayerDiffuse SD1.5** | Native RGBA object; joint fg+bg; fg-given-bg | Code Apache-2.0, weights OpenRAIL-M, diffusers port MIT (yes) | ~4 GB fp16, fits [est] | [LayerDiffusion/layerdiffusion-v1](https://huggingface.co/LayerDiffusion/layerdiffusion-v1) |
| **LayerDiffuse SDXL** | Native RGBA object (main generator) | Same (yes) | ~7 GB fp16, fits; `--cpu-offload` available [est] | same; [diffusers port](https://github.com/rootonchair/diffuser_layerdiffuse), [ComfyUI nodes](https://github.com/huchenlei/ComfyUI-layerdiffuse) |
| LayerDiffuse-Flux | Native RGBA on FLUX.1-dev | No repo license; FLUX.1-dev base (no) | Needs heavy quantization or offload [est] | [RedAIGC/Flux-version-LayerDiffuse](https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse) |
| Qwen-Image-2.1 | Native text-to-RGBA (7B DiT), transparent editing, subject extraction | Qwen Research License (no) | Q8 GGUF 7.64 GB + Q4 encoder ~5 GB; Unsloth says ~11 GB; likely fits [est] | [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1), [GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF) |
| Qwen-Image-Layered | Image-to-layers import (20B) | Apache-2.0 (yes) | ~40–45 GB BF16. **Measured (2026-11-09): GGUF Q4_K_S split over Kaggle 2× T4 with `fp16_mixed` works, 27–35 s per split** (384 px, 4 layers, 4 steps). Hosted on fal at $0.05/image | [Qwen/Qwen-Image-Layered](https://huggingface.co/Qwen/Qwen-Image-Layered), [Comfy fp8](https://huggingface.co/Comfy-Org/Qwen-Image-Layered) |
| FLUX.2 Klein 4B | RGB object/background generator; multi-ref edit or harmonizer | Apache-2.0 (yes) | Card: "as little as 13 GB"; borderline on T4. Card links Colab/Kaggle notebooks | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) |
| FLUX 3 Image Klein (5.6B) | RGB only, no transparency | FLUX Kommunity v1.0 (free under $5M revenue); gated | Card peak 25–31 GB; poor T4 fit | [flux-3-image-klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) |
| Z-Image Turbo | RGB only (architecture B) | Not researched | Not researched | no RGBA/layer adapter found |
| CLD | One layer per bounding box (import) | MIT code; FLUX.1-dev base (no) | Needs quantization or offload [est] | [thuteam/CLD](https://huggingface.co/thuteam/CLD) |
| LayerD | Graphic-design image-to-layers (import) | Apache-2.0 code, repo archived (yes) | Small, BiRefNet-based; CPU OK [est] | [cyberagent/layerd-birefnet](https://huggingface.co/cyberagent/layerd-birefnet) |
| **BiRefNet** | Matting/cut-out | MIT (yes) | 0.2B; CPU OK | [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) |
| **BEN2 base** | Matting/cut-out | MIT (yes; Full model is API-only) | 94.6M; CPU OK | [PramaLLC/BEN2](https://huggingface.co/PramaLLC/BEN2) |
| InSPyReNet | Matting (rembg option) | MIT (yes) | Small; CPU OK | via [rembg](https://github.com/danielgatis/rembg) |
| SAM 2.1 | Prompted segmentation | Apache-2.0, not gated (yes) | 39M–224M | [GitHub](https://github.com/facebookresearch/sam2) |
| SAM 3 | Text-prompted segmentation | "SAM License": royalty-free, commercial use not explicitly banned, bans military/ITAR uses, gated | ~3.6 GB F32; likely fits [est] | [facebook/sam3](https://huggingface.co/facebook/sam3) |
| Florence-2-large | Text grounding to drive SAM; also captioning | MIT (yes) | 0.77B | [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) |
| LaMa | Fill the hole behind removed objects | Apache-2.0 (yes) | CPU OK | [smartywu/big-lama](https://huggingface.co/smartywu/big-lama) |
| IC-Light SD1.5 | Relight a foreground to match a background | Apache-2.0 (yes) | SD1.5-sized, fits [est] | [GitHub](https://github.com/lllyasviel/IC-Light) |

**rembg trap:** rembg's code is MIT, but its default model (bria-rmbg) is non-commercial. Select a BiRefNet model explicitly.

**Captioning** (alt text, or describing imported layers for the LLM): Florence-2 (MIT) covers it. The Ollama LLM can turn its captions into editable per-layer prompts.

## 4. Hosting options

| Option | Cost | Limits | Fit |
|---|---|---|---|
| Local CPU | Free | Slow; SDXL-Turbo at 512 px, 4 steps; LayerDiffuse on CPU not yet benchmarked | Draft 1: SDXL-Turbo + LayerDiffuse SDXL LoRA, Ollama (matting as fallback) |
| Kaggle | Free | ~30 GPU-h/week, ~12 h sessions (secondary sources); T4; tunnel policy unverified | Batch asset generation with LayerDiffuse SDXL. Community [kayas881/comfyui-kaggle](https://github.com/kayas881/comfyui-kaggle) runs ComfyUI on T4 via Pinggy (untested, 1 star) |
| Modal (Starter) | $30/month credit (~50 T4-h at ~$0.59/h) | Pay beyond credit | ToS-clean live HTTPS endpoint; weights cached in a Volume. Best for a live game backend |
| HF ZeroGPU | Free (account >30 days, verified email) | 2 Spaces, 5 GPU-min/day, half an RTX Pro 6000 (48 GB) | Matting and segmentation calls via `gradio_client`; not diffusion |
| Colab (free) | Free | **ToS bans** remote proxies, web services and web-UI-first use on free runtimes; sessions can be killed | Interactive notebook batch jobs into Drive only. Never as a backend |
| Lightning AI | Free ~30 credits (~50 interruptible GPU-h) | Interruptible | Fallback GPU |
| fal.ai / Replicate | ~$0.05/image (Qwen-Image-Layered on fal) | Paid | Occasional scene import |

## 5. Recommended stack

**Draft 1 (local, CPU)**
1. Ollama LLM: story text to an object list to one refined prompt per object.
2. **LayerDiffuse SDXL** (transparent LoRA + alpha decoder) on SDXL-Turbo, float32 on CPU, one prompt per object (architecture A). The background is its own opaque layer (LoRA off). If CPU speed or alpha edges disappoint, fall back to SDXL-Turbo on a plain background + BiRefNet/BEN2 cut-out (architecture B).
3. Store each layer as `{prompt, seed, RGBA, position, z}` and composite in the game.
4. Re-roll = regenerate one layer with a new seed or prompt, or img2img it (transparent encoder) from its current image or a sketch.

**Upgrade path**
1. **Kaggle or Modal T4:** run the same LayerDiffuse SDXL backend in fp16 on GPU behind the prototype's HTTP server (switch to full SDXL base for quality). Use Kaggle for tests and Modal for a stable live endpoint.
2. **Scene fit:** LayerDiffuse SD1.5 fg-given-bg, or IC-Light SD1.5 relighting. Optional preview-only harmonization (img2img at ~0.2–0.35 denoise, or a FLUX.2 Klein 4B multi-ref edit) that never overwrites stored layers.
3. **Backgrounds:** plain SDXL or FLUX.2 Klein 4B.
4. **Quality option (non-commercial only):** Qwen-Image-2.1 GGUF for native RGBA objects. Its license must be replaced before any commercial release.
5. **Import:** Qwen-Image-Layered via fal for user-supplied scenes, or Florence-2 + SAM 2.1 + LaMa for single objects.
6. **LLM:** move Ollama to an API or to a Kaggle/Colab notebook (Colab interactive only).
7. **Watch:** PSDiffusion-style joint generation with per-layer prompts, if open weights ship.

## 6. Ruled out

- **ART** (Microsoft): weights and inference code removed 23 Jul 2025; HF repo returns 401. ART+ checkpoints are on Google Drive with no license.
- **OmniPSD**: code is CC BY-NC-SA 4.0, no weights, FLUX.1-dev/Kontext base; aimed at graphic design.
- **PSDiffusion**: best match on paper (per-layer prompts), but no released weights.
- **DreamLayer**: no verified weights; joint generation re-runs the whole stack.
- **LayerFusion**: paper only; joint generation.
- **RefLayer** (Referring Layer Decomposition): no released weights.
- **DiffDecompose**: no released weights.
- **LiWi**: no released weights.
- **LASAGNA**: no released weights.
- **LayerDecomp** (Adobe): no released weights.
- **RMBG-2.0**: CC BY-NC 4.0, gated. Use BiRefNet, which has the same architecture and an MIT license.
- **ObjectClear**: NTU S-Lab License, non-commercial.
- **OmniEraser**: card says Apache-2.0, but it is a FLUX.1-dev finetune, so effectively non-commercial. GitHub has no LICENSE file.
- **pix2gestalt**: amodal completion is not needed when objects are generated whole.
- **IC-Light v2**: weights unreleased, announced as non-commercial.
- **Colab tunnelling**: banned by the Colab FAQ on free runtimes.

(PrismLayersPro/Plus datasets are Apache-2.0 and usable for any future fine-tuning, but the model built on them, ART+, has no license.)

## 7. Remaining unknowns

- Real T4 VRAM and speed for LayerDiffuse SDXL, Qwen-Image-2.1 and FLUX.2 Klein 4B. Nothing is published; measure it ourselves.
- Kaggle quota and whether its ToS allows tunnels such as Pinggy (official pages unreadable).
- No public LayerDiffuse or Qwen Kaggle/Colab notebook exists; we must write our own. The official LayerDiffuse Colab is still listed as "coming soon".
- Z-Image Turbo license and VRAM (not researched).
- SAM 3 commercial use is implied, not stated (not legal advice).
