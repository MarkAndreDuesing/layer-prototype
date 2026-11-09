# Modular Pipelines for Layered Images and How HCI Systems Implemented Layers

Note on sourcing: claims marked "[verified]" were checked in web searches this session (2026-11-07). Items marked "[prior knowledge, not re-verified]" come from the researcher's background knowledge (cutoff mid-2026). The report writer should treat those as lower confidence. Repo and Hugging Face links in that category are canonical paths I know of but did not open this session.

## Q1. Best matting and background-removal models (quality, size, CPU, license)

### Takeaway
For per-object layer cutouts, **BEN2 Base (MIT)** and **BiRefNet (MIT, several variants)** are the best license-clean choices. **RMBG-2.0** is the same BiRefNet architecture trained on extra proprietary data, but it is CC BY-NC. Use SAM 2 or SAM 3 when you need to pick *which* object to cut, and pass its mask to a matting model for clean edges. All of these run on a 16 GB GPU. BiRefNet and BEN2 also run on CPU at a few seconds per 1024px image, which suits occasional layer regeneration.

### Cited Findings
- **RMBG-2.0** (briaai) is built on the BiRefNet architecture plus proprietary training data. License: CC BY-NC 4.0, and commercial use needs a BRIA agreement. [verified] https://huggingface.co/briaai/RMBG-2.0 — [Model card](https://huggingface.co/briaai/RMBG-2.0/blob/refs%2Fpr%2F6/README.md)
- **BEN2** (PramaLLC) uses "Confidence Guided Matting", which refines the pixels the base model is least sure about. BEN2 Base is MIT-licensed. The stronger "enhanced" model is proprietary and available only through the API. [verified] — [HF README](https://huggingface.co/PramaLLC/BEN2/blob/main/README.md), [GitHub](https://github.com/PramaLLC/BEN2/)
- Vendor-reported speed on an RTX 3090: BEN2 Base takes about 0.130 s per image and about 4.5 GB VRAM. RMBG2/BiRefNet takes about 0.185 s and 5.6 GB. These are the developer's own numbers, not independent tests. The developer also says BEN2 Base is roughly on par with InSPyReNet on DIS5k. — [r/LocalLLaMA thread (developer post)](https://axebps-redlib.hf.space/r/LocalLLaMA/comments/1icwira/ben2_new_open_source_stateoftheart_background)
- **BiRefNet** (ZhengPeng7) is MIT-licensed. Variants include general, general-lite, HR (2048px), matting, HR-matting and portrait. It is about 220M params with the Swin-L backbone. The lite version uses Swin-T. [prior knowledge, not re-verified] https://huggingface.co/ZhengPeng7/BiRefNet , https://github.com/ZhengPeng7/BiRefNet
- **InSPyReNet** (transparent-background package) is MIT-licensed, is strong at high resolution, and is older (2022). [prior knowledge, not re-verified] https://github.com/plemeri/InSPyReNet
- **rembg** is a wrapper library with ONNX CPU backends, including u2net, isnet, birefnet-* and bria-rmbg. It is the easiest path to CPU inference. [prior knowledge, not re-verified] https://github.com/danielgatis/rembg
- **SAM 3** (Meta) does open-vocabulary "promptable concept segmentation". A text or exemplar prompt such as "yellow school bus" returns instance masks for all matching objects. It was added to HF Transformers on 2025-11-19, and the checkpoints are gated under Meta's SAM license. [verified] — [Transformers docs](https://huggingface.co/docs/transformers/en/model_doc/sam3), https://huggingface.co/facebook/sam3 , [PyPI sam3](https://pypi.org/project/sam3/)
- **SAM 2 / SAM 2.1** is Apache-2.0 and comes in tiny to large sizes (about 39M–224M). It takes point and box prompts and also handles video. The tiny and small sizes are feasible on CPU. [prior knowledge, not re-verified] https://huggingface.co/facebook/sam2.1-hiera-large
- **MatAnyone** (CVPR 2025) is video matting propagated from a first-frame mask. It is relevant only if layers are animated. Its license is S-Lab non-commercial. [prior knowledge, not re-verified] https://github.com/pq-yang/MatAnyone

### Inferences
- Pipeline (a), "generate object on a plain background, then cut it out", works best like this. Prompt the object "on a plain white/grey background, full body, centered". Run BEN2 Base or BiRefNet-HR-matting. Optionally erode or defringe the alpha to remove halos from the background color. Segmentation-type models such as BiRefNet-general give near-binary alpha. Use the matting variants for hair and fur.
- A license-safe commercial stack is BiRefNet or BEN2 Base (MIT) plus SAM 2.1 (Apache). Avoid RMBG-2.0 and MatAnyone unless the use is non-commercial.
- For CPU use, rembg with an ONNX birefnet-general-lite or isnet model is practical. Full BiRefNet at 1024px takes a few seconds or more per image on CPU (estimate, not measured).

### Gaps
- I found no independent 2026 benchmark comparing BEN2, BiRefNet-HR and RMBG-2.0. I also did not confirm whether newer releases (for example BEN3 or BiRefNet 2026 updates) exist.
- I did not verify BiRefNet-HR's license or measured CPU timings.

## Q2. Open-vocabulary segmentation plus inpainting and object removal (decomposition)

### Takeaway
Segment-then-inpaint decomposition is: detect or segment each object, cut out its RGBA layer, then remove it from the plate and fill in the background. Use SAM 3 text prompts, or Grounding DINO/Florence-2 plus SAM 2, to segment. For removal, LaMa is the CPU-friendly baseline. ObjectClear and OmniEraser are better when the object's shadows and reflections must also go. The occluded parts of the cut-out object need amodal completion (pix2gestalt-style), or a regeneration on a plain background, before the layer can be moved freely.

### Cited Findings
- **ObjectClear** (NTU S-Lab, arXiv 2505.22636) uses an Object-Effect Attention module to remove the object *and* its shadows and reflections. It was trained on the OBER dataset (camera-captured plus simulated data with object-effect masks). [verified] — [arXiv](https://arxiv.org/abs/2505.22636)
- **OmniEraser** (arXiv 2501.07397) is a LoRA on FLUX.1-dev. It was trained on Video4Removal (over 100k real video frames) and needs only an object mask as input. Because it builds on FLUX.1-dev, it inherits the FLUX.1-dev non-commercial license (inference). [verified for method; license is an inference] — [arXiv](https://arxiv.org/html/2501.07397v3)
- **SAM 3** text prompts return all instances of a concept, so a single model can produce per-object masks from the scene's object list. [verified] — [Transformers docs](https://huggingface.co/docs/transformers/en/model_doc/sam3)
- **Grounded-SAM 2** (IDEA-Research) combines Grounding DINO / Florence-2 with SAM 2 for text-to-box-to-mask segmentation. **Florence-2** (MIT, 0.23B/0.77B) runs on CPU. **LaMa** (Apache-2.0, about 51M params, Fourier convolutions) does fast removal on CPU, but texture filling over large holes is blurry. [prior knowledge, not re-verified] https://github.com/IDEA-Research/Grounded-SAM-2 , https://huggingface.co/microsoft/Florence-2-large , https://github.com/advimman/lama
- **pix2gestalt** (Ozguroglu et al., CVPR 2024) does amodal completion: it synthesizes the whole object behind occluders, fine-tuned from Stable Diffusion's image variations model. [prior knowledge, not re-verified] https://gestalt.cs.columbia.edu/

### Inferences
- When decomposing an existing scene into movable layers, every layer except the topmost has holes where other objects overlapped it. Moving a layer exposes the "background" left in the gap by removal, so both steps are needed: object removal for the plate, and amodal completion for the object.
- In a game prototype, it is usually cheaper to generate each object separately (pipeline a) than to decompose a full scene (pipeline b). Use decomposition only to import user-supplied or full-scene images.

### Gaps
- "MagicEraser" did not come up in my searches, and I could not confirm a specific model by that name.
- I did not verify current SOTA amodal-completion follow-ups after pix2gestalt.
- I did not confirm the ObjectClear license or HF weights.

## Q3. Layout/regional generation, multi-reference composition and harmonization

### Takeaway
There are two families. **Layout-conditioned single-pass generation** (GLIGEN, MIGC, InstanceDiffusion, regional prompting / attention couple) keeps a coherent scene, but it outputs one flat image, which you must then segment. **Compositing**, which pastes RGBA layers and then harmonizes them, keeps true layers, but it needs a harmonization pass. As of 2026, the strongest open practical route is to composite the layers, then run a multi-reference edit model (**FLUX.2 [klein] 4B**, Apache-2.0, about 13 GB VRAM; or Qwen-Image-Edit) or low-denoise img2img or IC-Light to unify lighting. Keep the original layers as the source of truth.

### Cited Findings
- **FLUX.2 [klein] 4B** was released 2026-01-15 under Apache-2.0. It is the only Apache-licensed FLUX.2 model. One 4B checkpoint does text-to-image, image editing and multi-reference generation. It needs about 13 GB VRAM (RTX 3090/4070 class). A non-distilled "base" variant exists for LoRA training. [verified] — [HF mirror card](https://huggingface.co/bi1o/FLUX.2-klein-4B), [Vercel](https://vercel.com/ai-gateway/models/flux-2-klein-4b/about), [bf16 card](https://huggingface.co/YuCollection/FLUX.2-klein-4B-bf16). Official repo is likely https://huggingface.co/black-forest-labs/FLUX.2-klein-4B [path not verified].
- I found no source for "FLUX 3 Klein bounding boxes" from the brief. It may be a misnomer for FLUX.2 [klein]. Treat it as unverified.
- **GLIGEN** (CVPR 2023) adds gated self-attention for box-plus-phrase grounding on SD1.4. **MIGC** (CVPR 2024) and **InstanceDiffusion** (CVPR 2024) handle multi-instance attribute and position control, with boxes, masks, points or scribbles for InstanceDiffusion. All are SD1.x-era, so their quality lags modern DiT models. **Regional prompting / attention couple** are training-free ways to give each region its own prompt by masking cross-attention. They are available in ComfyUI and Forge. [prior knowledge, not re-verified] https://github.com/gligen/GLIGEN , https://github.com/limuloo/MIGC , https://github.com/frank-xwang/InstanceDiffusion
- **IC-Light** (lllyasviel) relights a foreground to match a background or light condition, which suits harmonizing composited layers. It is Apache-2.0 for code. The SD1.5 FC/FBC models run on about 8 GB, and an IC-Light v2 (FLUX-based) exists mainly as a demo. [prior knowledge, not re-verified] https://github.com/lllyasviel/IC-Light
- **Qwen-Image-Edit (2509/2511)** supports multi-image input (about 1–3 references) for composition. It is Apache-2.0 and 20B parameters, so it needs quantization or offload on 16 GB. [prior knowledge, not re-verified] https://huggingface.co/Qwen/Qwen-Image-Edit-2509

### Inferences
- A recommended prototype loop:
  1. Generate each object on a plain background (FLUX.2 klein, or SDXL/SD-Turbo on CPU).
  2. Matte it with BEN2 or BiRefNet into an RGBA layer.
  3. Let the user move or scale the layers.
  4. Composite the layers into a flat preview.
  5. Optionally run a "harmonize" pass for the final render: low-denoise (0.2–0.35) img2img or a multi-reference edit, plus a shadow pass. This pass is never written back into the layers.
- When a single layer is regenerated, only that layer's step 1–2 reruns. That makes it deterministic and cheap compared with re-running regional generation for the whole scene.
- For CPU-only use (the user's FastSD CPU setup), the realistic options are SD-Turbo/LCM generation plus rembg cutouts plus low-denoise img2img harmonization. Edit models like FLUX.2 klein need a GPU.

### Gaps
- I did not check whether FLUX.2 klein officially supports bounding-box or layout conditioning.
- I found no 2026 head-to-head comparison of harmonization methods (IC-Light vs edit-model vs img2img).

## Q4. How WorldSmith, SketchFlex, PromptCharm and PortalInk implemented layers and regions

### Takeaway
None of these HCI systems uses a native layered (RGBA) model. They layer at the *interaction* level: prompts attached to regions, masks or hierarchy levels, which are realized with inpainting or region-conditioned generation onto a flat canvas. WorldSmith's "hierarchical prompting" is the closest to the prototype's layer-per-object design.

### Cited Findings
- **WorldSmith** (Dang, Brudy, Fitzmaurice, Anderson; UIST 2023, DOI 10.1145/3586183.3606772) supports novice world-building with "layered edits and hierarchical compositions". It introduces *hierarchical prompting*, which attaches prompts to different layers of the composition, and *spatial prompting*, where users specify spatial relations through direct input. Users pick one of four tiles in a Global Tile View, then edit it iteratively with text, sketching and region painting. The edited tiles are blended into one image. Studies: formative N=4, first-use N=13. [verified] — [arXiv 2308.13355](https://arxiv.org/pdf/2308.13355), [Autodesk Research](https://www.research.autodesk.com/publications/worldsmith/)
- The underlying WorldSmith model was not stated in the search results. Prior knowledge says it was built on Stable Diffusion inpainting with region masks [not re-verified].
- **SketchFlex** (Lin, Ye, Xia, Zeng; CHI 2025, arXiv 2502.07556) lets users sketch regions and label each with a prompt. An MLLM turns the sketch and prompt into refined per-region prompts. Rough sketches are refined into canny-edge "shape anchors" by decompose-and-recompose, then fed to region-conditioned generation. In a 12-participant study it beat text-to-image and region-to-image baselines on quality, cohesiveness and intent alignment, with lower cognitive load. [verified] — [arXiv](https://arxiv.org/abs/2502.07556v1)
- **PromptCharm** (Wang et al., CHI 2024, arXiv 2403.04014) does automatic prompt refinement plus attention-map visualization. Its regional editing is mask brushing, followed by a Stable Diffusion inpainting pipeline that regenerates only the masked pixels, with an optional prompt. N=12 study. [verified] — [arXiv](https://arxiv.org/pdf/2403.04014)
- **PortalInk** (Ji Won Chung et al.) is a tool for artists to craft and export 2.5D graphical stories while staying in 2D, using SVG transitions. It is vector and parallax layering, not generative. I could not confirm its venue in this session. [partially verified] — [author page](https://www.bohrium.com/scholar/2540k011/Ji_Won_Chung)

### Inferences
- Every one of these systems keeps a *prompt-per-region* data model, with masks or sketches plus text. That maps directly onto a "layer = prompt + mask/position" schema. The difference for a game prototype is that moving objects requires *real* RGBA layers, which these papers do not provide. Pipeline (a) or (b) is what supplies them.
- The interaction patterns worth borrowing:
  - hierarchical prompts (scene, then group, then object), from WorldSmith;
  - MLLM-refined per-region prompts from rough sketches, from SketchFlex;
  - mask-local regeneration, from PromptCharm.

### Gaps
- I did not get implementation details from the WorldSmith PDF body (model, how layers blend).
- I could not confirm PortalInk's publication venue or year. Other related systems (for example Inkspire, story-visualization tools) were seen only in passing.

## Q5. Modular pipelines vs native layered models (Qwen-Image-Layered, LayerDiffuse)

### Takeaway
Native layered models give coherent RGBA layers in one shot, with consistent lighting and occlusion. Their weakness is the targeted per-layer prompt regeneration and free movement that the prototype needs. Modular pipelines win on control, swappability, CPU fallback and licensing, and they lose on scene coherence (lighting, shadows, contact). The best hybrid is modular per-object generation plus a harmonization pass, with Qwen-Image-Layered used to import or decompose existing full-scene images.

### Cited Findings
- **Qwen-Image-Layered** (arXiv 2512.15603, Apache-2.0) decomposes an image into a variable number of RGBA layers (for example 3 or 8). Any layer can itself be decomposed further. It uses an RGBA-VAE and a variable-layer MMDiT, and has a diffusers pipeline on HF. [verified] — [GitHub](https://github.com/QwenLM/Qwen-Image-Layered), [Replicate](https://replicate.com/qwen/qwen-image-layered). HF path is likely https://huggingface.co/Qwen/Qwen-Image-Layered [not verified].
- **LayerDiffuse** (Zhang and Agrawala, arXiv 2402.17113) encodes alpha into the latent space ("latent transparency") of SD1.5/SDXL. It generates a transparent foreground directly, or foreground and background jointly or conditionally. The license is reported as Apache-2.0 via indexes. It has Forge and ComfyUI integrations. [verified via third-party index] — [arXiv](https://arxiv.org/abs/2402.17113), [Diffusers CLI](https://github.com/lllyasviel/LayerDiffuse_DiffusersCLI), [Forge ext](https://github.com/layerdiffusion/sd-forge-layerdiffuse)

### Inferences
- LayerDiffuse on SDXL is a strong alternative to "generate then matte" because the alpha is native (semi-transparency, glass, hair), and it needs about 8–12 GB of GPU. It is SDXL-quality, though, and has no CPU path in practice.
- Comparison of the pipelines:

| Pipeline | Per-layer regen | Move layers | Coherence | CPU | License-clean option |
|---|---|---|---|---|---|
| (a) gen-on-plain + matte | Excellent | Excellent | Weak (needs harmonize) | Yes (SD-Turbo + rembg) | BiRefNet/BEN2 MIT |
| (b) segment + inpaint | Poor (rebuild plate) | Good after amodal fill | Starts coherent | Partly (LaMa, SAM2-tiny) | SAM2 + LaMa |
| (c) regional gen + composite | Re-run region | Requires segmentation | Good | Slow | SD regional prompting |
| (d) edit-model "remove X" | Good via "add/remove" | Plates clean | Very good | No (GPU) | FLUX.2 klein Apache |
| Native (Qwen-Layered / LayerDiffuse) | Weak / limited | Good | Best | No | Apache-2.0 both |

### Gaps
- I did not verify Qwen-Image-Layered's VRAM needs, or whether it runs on 16 GB with quantization.
- I found no evaluation comparing modular vs native layered pipelines directly.

## Q6. Matting, segmentation, inpainting and harmonisation for hosting (moved from Free hostable layered models/hosting_and_matting.md §B)

### Takeaway
For commercial-safe cut-outs, use BiRefNet (MIT) or the BEN2 base model (MIT). SAM 2.1 (Apache-2.0) and Florence-2 (MIT) cover prompted segmentation. LaMa (Apache-2.0) covers background fill. RMBG-2.0, ObjectClear and IC-Light v2 are non-commercial or unreleased. SAM 3 is gated under a custom licence.

### Cited Findings
- **BiRefNet** (`ZhengPeng7/BiRefNet`): MIT, 0.2B parameters, default 1024×1024 input with fp16 CUDA in the example. A HF collection holds 18 task- and scale-specific variants. — [HF](https://huggingface.co/ZhengPeng7/BiRefNet)
  - rembg ships these BiRefNet ONNX variants: general, general-fp16, general-lite, portrait, dis, hrsod, cod and massive. — [rembg](https://github.com/danielgatis/rembg)
- **BEN2** (`PramaLLC/BEN2`): MIT, 94.6M parameters, runs on CUDA or CPU. Only the Base model is open. The Full model (proprietary 22K-image dataset, better hair and 4K handling) is available only through the API or web demo; commercial access goes through sales. — [HF](https://huggingface.co/PramaLLC/BEN2)
- **RMBG-2.0** (`briaai/RMBG-2.0`): CC BY-NC 4.0 (bria-rmbg-2.0). Gated behind a form. Built on the BiRefNet architecture, 0.2B parameters, 1024 input. Commercial use requires a paid agreement. — [HF](https://huggingface.co/briaai/RMBG-2.0)
- **InSPyReNet** (via the `transparent-background` package): MIT. Modes are `base` (1024), `fast` (384), `base-nightly` and `custom`. CPU install instructions are provided. — [GitHub](https://github.com/plemeri/transparent-background)
- **rembg:** the code is MIT, but each weight file keeps its own licence. The default is `bria-rmbg`, which is non-commercial. Supported models: u2net, u2netp, u2net_human_seg, u2net_cloth_seg, silueta, isnet-general-use, isnet-anime, sam, the BiRefNet set above, and bria-rmbg. ViTMatte refiners are also listed. — [GitHub](https://github.com/danielgatis/rembg)
- **SAM 2 / 2.1:** Apache-2.0 and not gated. Checkpoint sizes are 38.9M (tiny), 46M (small), 80.8M (base+) and 224.4M (large). — [GitHub](https://github.com/facebookresearch/sam2)
- **SAM 3** (`facebook/sam3`): gated (you must share your HF info). The licence is listed as "other" (custom Meta). It takes text-phrase and exemplar prompts for images and video. About 0.9B parameters stored as F32, so roughly 3.6 GB of weights. — [HF](https://huggingface.co/facebook/sam3)
- **Grounded-SAM-2:** Apache-2.0 plus component licences (SAM 2, GroundingDINO, cctorch). Grounding DINO and Florence-2 run locally. Grounding DINO 1.5/1.6 and DINO-X need an API token. The README does not mention SAM 3. — [GitHub](https://github.com/IDEA-Research/Grounded-SAM-2)
- **Florence-2-large:** MIT, about 0.77B parameters. Tasks include captioning, phrase grounding, object detection, dense region captioning, OCR and segmentation. — [HF](https://huggingface.co/microsoft/Florence-2-large)
- **LaMa:** Apache-2.0. Weights: `smartywu/big-lama` on HF, or Google Drive. — [LICENSE](https://raw.githubusercontent.com/advimman/lama/main/LICENSE), [GitHub](https://github.com/advimman/lama)
- **ObjectClear:** NTU S-Lab License 1.0 (non-commercial). Built on SDXL-Inpainting and has an fp16 flag. Weights download through a repo script; there is a HF demo Space `jixin0101/ObjectClear`. — [GitHub](https://github.com/zjx0101/ObjectClear)
- **OmniEraser:** weights at `theSure/Omnieraser` and `theSure/Omnieraser_Controlnet_version`. FLUX-based (FLUX.1-dev ControlNet inpainting). The repo states no licence. — [GitHub](https://github.com/PRIS-CV/OmniEraser)
- **pix2gestalt:** amodal completion fine-tuned from Lambda Labs' image-conditioned SD. Weights are at `cvlab/pix2gestalt-weights`. The Gradio app needs 22 to 28 GB VRAM. The dataset is research-only. — [GitHub](https://github.com/cvlab-columbia/pix2gestalt)
- **IC-Light:**
  - The SD1.5 repo is Apache-2.0, with models `fc`, `fcon` and `fbc` (foreground plus background conditioning). Its default remover, RMBG-1.4, is non-commercial, and the repo suggests BiRefNet instead. — [GitHub](https://github.com/lllyasviel/IC-Light)
  - IC-Light v2 (FLUX): the weights are not released, and the maintainer said the licence "will be non-commercial only". It is reachable only through the fal.ai API, and the HF Space was reported as 404 or broken in 2025. — [Discussion #98](https://github.com/lllyasviel/IC-Light/discussions/98)

### Inferences
- **CPU-feasible:** BEN2 base, BiRefNet-lite, InSPyReNet fast, u2net/isnet, SAM 2.1 tiny/small, LaMa, and probably Florence-2-base.
- **GPU-only in practice:** SAM 3, ObjectClear (SDXL), OmniEraser (FLUX, likely 24 GB+), pix2gestalt (22 to 28 GB) and IC-Light SD1.5 (about 6 to 8 GB, typical for SD1.5).
- ZeroGPU's 48 GB slice can run all of these.

### Gaps
- The exact HF repo IDs for BiRefNet_lite, BiRefNet_HR and BiRefNet-matting were not opened individually; only the collection reference was seen on the main card.
- **No VRAM figures are published** for BiRefNet, SAM 3, ObjectClear, OmniEraser or IC-Light. The CPU/GPU split above is my estimate.
- **OmniEraser** has no licence in its repo, so treat it as all rights reserved.
- **The SAM 3 licence text** was not opened.
- **The pix2gestalt code licence** was not stated in the README.
