# Qwen-Image-2.1: RGBA generation, editing and extraction, and use on Kaggle 2x T4

Research date: 2026-11-09. This is the second, deeper pass.

Primary sources read:
- HF model card and GitHub README/tree
- HF Space `app.py`
- `prompt_rewrite/pe_core.py`
- diffusers `pipeline_qwenimage21.py`
- HF LICENSE
- Comfy official docs page
- Unsloth GGUF card and docs
- HF discussions #6, #32, #35
- madebyollin texture-fix VAE card

Not reachable or not found:
- The qwen.ai blog is a JS page and returned no text.
- No arXiv tech report or paper was found.
- No official DashScope or Model Studio page was found.
- No QbitAI or Zhihu article surfaced.
- No Reddit threads surfaced in search.

## 1. Official RGBA design, prompts and code paths (transparent generation, layer editing, extraction, decomposition, masks/circles, multi-reference)

### Takeaway
RGBA is native. The VAE takes 4-channel RGBA in and out and encodes it to a 64-channel latent at 16x spatial compression, so "4-channel VAE" means 4 image channels, not 4 latent channels. Generation and editing share one pipeline.

- **Only official prompt:** the transparent-generation wrapper is the only official prompt anywhere (card, README, Space). No official code or prompt exists for editing a transparent layer or extracting a subject.
- **Extraction:** ComfyUI's official background-removal template uses "Remove the background, and output a PNG image."
- **No decomposition mode:** there is no layer-decomposition mode in the repo, Space or pipeline. One call returns one image.

### Cited Findings
- **VAE:** a "64-channel RGBA autoencoder with 16x spatial compression, supporting native transparency" — [GitHub README](https://github.com/QwenLM/Qwen-Image-2.1). Diffusers sets `latent_channels = vae.config.z_dim` with fallback 64 — [diffusers pipeline](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py). The ComfyUI VAE decodes 4 channels, so outputs carry alpha — [Comfy docs](https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1)
- **Condition images in diffusers:**
  - All condition images are converted to RGBA, and the VAE reads all four channels.
  - Only the Qwen3-VL vision-encoder copy is alpha-composited over white ("The checkpoint was trained with the alpha composited over white for the vision encoder").
  - There is no mask argument. Circles and masks are just condition images.
  - A flat image list is shared across the batch.
  - The output aspect ratio follows the **last** image.
  - Source: [diffusers pipeline](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py)
- **Transparent-generation template (official Space UI):** "This is an RGBA image with transparency. xxxxx The image has alpha channel and the background is transparent." The Chinese version is "这是一张带有透明度的RGBA图像。.... 该图像具有alpha通道，背景是透明的。" — [HF Space app.py](https://huggingface.co/spaces/Qwen/Qwen-Image-2.1/raw/main/app.py); the same wrapper is in the [HF card](https://huggingface.co/Qwen/Qwen-Image-2.1)
- **Space app.py behaviour:**
  - Inputs are converted to RGBA and kept as PNG, and the output uses `image_mode="RGBA"`.
  - Defaults: 40 steps, 1024 px "speed" or 2048 px "quality", negative prompt " ", true-CFG 1.0.
  - KV cache is on only if it fits a 10 GB budget.
  - Edit mode is triggered simply by the presence of images.
  - There is no extract, mask or layer mode.
  - Source: [app.py](https://huggingface.co/spaces/Qwen/Qwen-Image-2.1/raw/main/app.py)
- **Repo contents:**
  - The repo has only `assets/`, `prompt_rewrite/`, LICENSE and README.
  - There are no example scripts for editing, extraction or layers, and the word "layered" does not appear.
  - The README says it can "edit transparent layers" and "extract subjects from photographs".
  - Source: [GitHub tree](https://github.com/QwenLM/Qwen-Image-2.1/tree/main)
- **Prompt rewriter (`pe_core.py`):**
  - It has two profiles, `t2i` and `edit`.
  - Edit output is `{"rewritten_prompt", "wh_ratio", "ratio_follow": "<image1>"}`.
  - Reference images are addressed as `<image1>`, `<image2>` in input order.
  - The system prompts live in the checkpoints, not in the repo.
  - Source: [pe_core.py](https://raw.githubusercontent.com/QwenLM/Qwen-Image-2.1/main/prompt_rewrite/pe_core.py)
- **ComfyUI official nodes:**
  - "Text Encode Qwen Image 2.1" has 16 reference slots, but the templates wire 10. `image_1` is the image being edited.
  - The background-removal template prompt is "Remove the background, and output a PNG image."
  - SplitImageWithAlpha / JoinImageWithAlpha are used for alpha, with mask 1 = transparent.
  - 3-channel nodes downstream may error on RGBA.
  - Source: [Comfy docs](https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1)
- **Local edits:** circles, painted annotations or a separate mask image. Showcase edits are circle-guided (remove a watch, recolour hair, replace clothing) — [GitHub](https://github.com/QwenLM/Qwen-Image-2.1). Third-party prompt pattern: "replace the text inside the red circle..." — [eachlabs](https://www.eachlabs.ai/blog/qwen-image-2-1-transparent-layers-ten-references)
- **Multi-reference:**
  - Up to 10 references. The official example is "These three characters are sitting around a campfire in a forest.", and showcases include a 6-portrait group photo and a 5-item outfit — [GitHub](https://github.com/QwenLM/Qwen-Image-2.1)
  - Third-party pattern: "the bottle from image 1 on the marble counter from image 2", plus a list of what must stay unchanged — [eachlabs](https://www.eachlabs.ai/blog/qwen-image-2-1-transparent-layers-ten-references)
- **Training data:** not disclosed in the card or README. Chinese coverage also gives no data details — [DataLearner](https://www.datalearner.com/ai-models/pretrained-models/qwen-image-2-1); [ComfyUI wiki zh](https://comfyui-wiki.com/zh/news/2026-09-21-qwen-image-2-1)
- **Hosted APIs:** Krea and RunComfy expose a `background: transparent|opaque` parameter. No official DashScope parameter was found [unverified] — [Krea API](https://www.krea.ai/docs/api-reference/image/qwen-image-21.md); [RunComfy](https://www.runcomfy.com/models/qwen/qwen-image-2.1/text-to-image/api)

### Inferences
- To edit an existing layer, pass the RGBA layer as `image_1` and use the RGBA wrapper plus the instruction. Alpha survives through the VAE path.
- To extract a subject, use the photo plus "Remove the background, and output a PNG image." (official Comfy template). Optionally add the RGBA wrapper and explicitly list shadow and floor removal.
- To decompose a scene, run one extraction per subject, then do a background-fill edit ("remove X"). This is N+1 calls, versus a single call with Qwen-Image-Layered.

### Gaps
- There is no paper or blog text, so the RGBA training recipe, alpha loss and official limitations are unknown.
- It is undocumented whether the RGBA wrapper is needed when editing an already-transparent input.

## 2. Quality and limitations (alpha edges, semi-transparency, shadows, hair, text) and benchmarks

### Takeaway
No measured RGBA benchmark exists, and no user reports specifically about alpha surfaced (HF discussion titles have none). One provider guide reports near-binary alpha (glass about 99% opaque). Vendor samples look clean.

### Cited Findings
- **Runware guide:**
  - Alpha is "subject or nothing". A glass bottle averages 99% opacity inside its outline.
  - Pale fringes show up on dark backgrounds.
  - Shadows and floors must be excluded by name ("no floor and no shadow"). Leave an empty margin.
  - JPG drops alpha.
  - Source: [Runware](https://runware.ai/docs/models/alibaba-qwen-image-2-1-pro/guides/transparent-backgrounds). This is the "Pro" endpoint and may differ from the open weights.
- **Release samples:** clean edges with no white halo, judged by viewing vendor samples — [Cinevva](https://app.cinevva.com/news/2026-09-20-qwen-image-2-1-transparent-sprites)
- **Reviewer caution:** hair, glass, fur and motion blur are still hard or untested [unverified, search snippet only] — [MindStudio](https://www.mindstudio.ai/blog/qwen-image-2-1-hands-on-review/); [eesel](https://www.eesel.ai/blog/qwen-image-2-1-review)
- **fp16 VAE NaNs:** the original VAE decoder NaNs in fp16 mostly in **fully transparent regions**, then spreads (75/185 stress inputs; max activation about 3.5M versus the fp16 maximum of 65504) — [texture-fix VAE](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1)
- **Official benchmark:** only VDEBench text editing (0.81 local, 0.83 global) — [HF card](https://huggingface.co/Qwen/Qwen-Image-2.1). Vendor overall ranking: 60.28, 7th of 30 — [Cinevva](https://app.cinevva.com/news/2026-09-20-qwen-image-2-1-transparent-sprites)
- **Comparisons:** no comparison against LayerDiffuse, Qwen-Image-Layered or ART was found — [Mixed News](https://mixed-news.com/en/qwen-image-2-1-transparent-rgba-7b-open-weights-research-licence). For reference, Qwen-Image-Layered beats LayerD on alpha soft-IoU — [alphaxiv 2512.15603](https://alphaxiv.org/resources/2512.15603v1)
- **Rewriter side effects:** the rewriter can add details you didn't ask for, so turn it off for precise prompts. State what must stay unchanged — [eachlabs](https://www.eachlabs.ai/blog/qwen-image-2-1-transparent-layers-ten-references)

### Inferences
- Use it for opaque props and characters. Keep glass, smoke and soft shadows in the background, or add them during compositing. Check every layer against both black and white backgrounds.

### Gaps
- No independent edge or hair test exists for the open weights.

## 3. Composition, spatial placement and harmonisation

### Takeaway
Multi-reference composition is official (10 references in the templates; 16 Comfy encoder slots). Placement is controlled by prompt and circle/mask images. There are no coordinate controls. Harmonising a pasted object is plausible: pass the rough composite as `image_1` and ask for a blend. This is untested [unverified].

### Cited Findings
- 10 references, circle/mask local edits — [GitHub](https://github.com/QwenLM/Qwen-Image-2.1)
- 16 encoder slots, 10 wired, `<imageN>` indexing — [Comfy docs](https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1)
- No bbox or position arguments in diffusers — [diffusers pipeline](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py)

### Inferences
- Do move and resize deterministically with PIL. Call 2.1 only to re-prompt a layer, optionally for a harmonise pass, and for re-extraction.

### Gaps
- There is no published harmonisation example.

## 4. Community runs: ComfyUI, GGUF, low VRAM, fp16 on T4/20-series, speeds, distillation LoRAs, text-encoder quantisation

### Takeaway
There are now concrete VRAM and speed numbers and three 4-step accelerators. The fp16 risks are real and localised:
- **VAE:** NaNs in transparent regions. The texture-fix VAE solves this.
- **Earlier Qwen-Image DiTs:** overflowed in fp16.
- **Text encoder:** one report of an all-NaN encoder, even in bf16/fp32, suspected to be corrupt files.

No T4 or 20-series report was found. City96 GGUF support is disputed.

### Cited Findings
- **Speeds and VRAM:**
  - RTX 4060 Ti 16 GB, ComfyUI, INT8 DiT + INT8 Qwen3-VL + bf16 VAE, 1024², 25 steps: about 20 s per generation and about 60 s per edit. Peak VRAM about 15 GB with no offload — [HF disc #35](https://huggingface.co/Qwen/Qwen-Image-2.1/discussions/35)
  - RTX 5090, NF4 DiT and text encoder: about 10.6 GB resident, 15.2 GB peak at 1024², 19.1 GB with two references. 40 steps took 19.2 s at 1024² and 118.8 s at 2048². fp8: 16.6 GB resident. Time grows about 6x for 4x pixels — [HF disc #32](https://huggingface.co/Qwen/Qwen-Image-2.1/discussions/32)
  - Edits are about 0.3 s/it at about 1 MP versus about 6 s/it at about 12 MP on a 5090 — [Comfy docs](https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1)
  - Other community ports (titles only, not read): under 4 GB, browser at 4 GB, ncnn/Vulkan "2 GB", Android MNN int4 + Turbo 6-step + Tiny VAE — [HF discussions](https://huggingface.co/Qwen/Qwen-Image-2.1/discussions)
- **ComfyUI KV cache:** the KV-cache node offers `int8` (half the size, about bf16 accuracy) or `int4` (a quarter, about 2x per-step error), on device auto/gpu/cpu/off. Steps: base 25; Turbo 8; Lightning 4; local edits are fine at 4–8; full-frame edits need about 25 — [Comfy docs](https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1)
- **Accelerators:**
  - Turbo checkpoint / Turbo LoRA (rank about 178), 8 steps CFG 1 — [GitHub](https://github.com/QwenLM/Qwen-Image-2.1)
  - LightX2V Lightning 4-step LoRA (about 67 MB, T2I + edit, 2026-10-13) — [ComfyUI wiki](https://comfyui-wiki.com/en/news/2026-10-13-qwen-image-2-1-lightning)
  - Lightning v2 RL (a full distilled transformer, 2026-11-02) — [ComfyUI wiki](https://comfyui-wiki.com/en/news/2026-11-02-qwen-image-2-1-lightning-v2-rl)
  - Alibaba PAI Fun-Acc 4-step LoRA (346 MB) — [ComfyUI wiki](https://comfyui-wiki.com/en/news/2026-09-24-qwen-image-2-1-fun-acc-lora)
  - Not verified on HF directly.
- **GGUF:**
  - Unsloth quants run from Q2_K 2.47 GB, Q4_K_S 3.91, Q4_K_M 4.2, Q5_K_M 5.39 and Q8_0 7.64 up to F16 14.2 GB.
  - Text encoder `Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf`: versus Q4_K_M, LPIPS 0.029 / SSIM 0.959.
  - The sd.cpp `sd-cli` example uses 20 steps, CFG 6, 1024².
  - Source: [unsloth GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF)
  - Unsloth estimates about 11 GB for GGUF, and says its figures "are estimates, not tested minimums" — [unsloth docs](https://unsloth.ai/docs/models/qwen-image-2.1)
- **City96 ComfyUI-GGUF:** reports conflict. Some users get "Unknown model architecture" and are told to use the leejet fork; others say recent builds work [unverified, aggregator snippets] — [codersera](https://codersera.com/blog/how-to-run-qwen-image-2-1-locally-2026/); [laozhang](https://blog.laozhang.ai/zh/posts/qwen-image-2-1-local). The official Comfy docs don't mention GGUF; they ship int8_convrot and bf16 files — [Comfy docs](https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1). Diffusers GGUF loading for `QwenImage21Transformer` was not confirmed.
- **fp16 VAE fix:**
  - The madebyollin texture-fix VAE was finetuned to cut activation magnitudes, reducing the max fp16 activation from about 3.5M to about 1.3k. It produced 0 NaN in the stress tests and works with Comfy `--fp16-vae`.
  - Quality: rFID improves (3.38 to 2.06) while PSNR drops slightly (33.29 to 32.79).
  - Diffusers usage: load `AutoencoderKLQwenImage21` from that repo and pass it as `vae=`.
  - Source: [texture-fix VAE](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1)
- **fp16 DiT history:** earlier Qwen-Image models produce NaN latents in fp16 in ComfyUI — [ComfyUI #10751](https://github.com/Comfy-Org/ComfyUI/issues/10751). A partial fp16 fix PR was tested on V100 — [ComfyUI PR #12522](https://github.com/Comfy-Org/ComfyUI/pull/12522). These were not checked for 2.1.
- **Text-encoder NaN report:** an all-NaN text encoder gave black images on an RTX 3070 even in bf16/fp32. The reporter suspects the served safetensors don't match their hashes. There has been no staff reply — [HF Turbo disc #6](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/discussions/6)
- **Reference-image VRAM:** two references add about 4 GB of peak VRAM (NF4) — [HF disc #32](https://huggingface.co/Qwen/Qwen-Image-2.1/discussions/32)

### Inferences (T4 plan)
- **GPU0:** Qwen3-VL-8B Q4_K_XL GGUF or NF4 (about 5–6 GB), plus the VAE.
- **GPU1:** Q4_K_S/Q5_K_M DiT, Lightning or Turbo LoRA (4–8 steps), working at 1024².
- **Always use the texture-fix VAE in fp16.** The original VAE is most likely to NaN exactly on transparent pixels, which is our core use case.
- Reuse our Qwen-Image-Layered fp16 patch on the DiT (fp32 norms, softmax and residual).
- Check `prompt_embeds` for NaN as a sanity test, and verify text-encoder file hashes.
- Use the KV cache int8, or turn it off.
- T4 time will likely be several times slower than the 4060 Ti's ~20 s for 25 steps [unverified]. At 4–8 steps, roughly tens of seconds per 1024² layer is plausible [unverified].

### Gaps
- No T4 or 20-series run, no fp16 DiT test on 2.1, and diffusers GGUF and city96 support are unconfirmed.

## 5. Layered decomposition: is a multi-layer mode planned, and are there community LoRAs?

### Takeaway
No. There is no official multi-layer mode for 2.1 and no announced "Qwen-Image-Layered 2.x". No community decomposition LoRA on 2.1 was found. Qwen-Image-Layered (Dec 2025, Apache-2.0, 3–8 layers, recursive) remains the only Qwen decomposer.

### Cited Findings
- Qwen-Image-Layered: released 2025-12-19, decomposes into 3 or 8 RGBA layers with recursive decomposition, Apache 2.0 — [GitHub Qwen-Image-Layered](https://github.com/QwenLM/Qwen-Image-Layered); [ComfyUI wiki](https://comfyui-wiki.com/en/news/2025-12-19-qwen-image-layered-release)
- The only LoRA-based decomposition fine-tunes found are research papers, and they target Qwen-Image-Layered, not 2.1 — [arXiv 2603.14925](https://arxiv.org/pdf/2603.14925)
- MRT claims to beat Qwen-Image-Layered on many-layer flat designs — [arXiv 2605.27235](https://arxiv.org/pdf/2605.27235)
- The 2.1 repo and Space have no layered or decomposition mode — [GitHub tree](https://github.com/QwenLM/Qwen-Image-2.1/tree/main); [app.py](https://huggingface.co/spaces/Qwen/Qwen-Image-2.1/raw/main/app.py)

### Inferences
- A hybrid is best. Use Qwen-Image-Layered (already working on our T4s) for the initial scene decomposition. Use 2.1 for per-layer re-prompting, new RGBA props and subject extraction. 2.1's licence is stricter than Layered's Apache 2.0.

### Gaps
- No roadmap statement from the Qwen team was found. Twitter/X was not searchable.

## 6. Licence (qwen-research) for a personal non-commercial project

### Takeaway
Non-commercial use is allowed, defined as "for research or evaluation purposes only". A free personal prototype fits a permissive reading.

### Cited Findings
- Non-commercial grant (§1i, §2a). Commercial licences come from model-business@notice.qwencloud.com (§2b).
- If you distribute: include the licence, mark changes, and add a NOTICE (§3). Derived models must show "Built with Qwen" (§4b). "Qwen" cannot be the primary product name (§4c).
- Output ownership is not addressed explicitly. Chinese law applies, with courts in Hangzhou (§8). Breach means you must delete the Materials (§7b).
- Source: [LICENSE](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)
- Contrast: Qwen-Image-Layered is Apache 2.0 — [GitHub](https://github.com/QwenLM/Qwen-Image-Layered)

### Inferences
- A free, unmonetised story prototype is fine. Keep 2.1 out of any paid path.

### Gaps
- There is no explicit clause on rights to generated outputs.
