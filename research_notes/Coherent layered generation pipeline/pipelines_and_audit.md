# Coherent layered generation: candidate pipelines and prototype audit (Nov 2026)

## Q1. Does the current prototype meet the original goal, and do its layers share context?

### Takeaway
No real shared context. Layers share only a text style suffix and the same size, steps and guidance. Each layer is a separate text-to-image call with its own random seed and no image conditioning. "Placement" is a post-hoc paste into LLM-guessed boxes. Steps 1, 3, 4 and 5 of the goal exist in a basic form (text input, variants, picking a variant, per-layer regen). Coherence is missing: lighting, perspective, scale, shadows and contact. Users can move or resize a layer only by typing box numbers into a table. Rotation is absent.

### Cited Findings (code citations; file paths relative to `prototype/`)
- **Per-layer prompts with no other objects.** The LLM is told to write "a standalone text-to-image prompt for ONLY that layer... Objects must be described alone (no other objects, no scenery)". The background must have "no characters or main objects". See `layerproto/llm.py` lines 81-83.
- **Shared context, item 1 (style text only).** `_full_prompt` appends `scene.style` to every layer prompt when `step_style` is on. See `layerproto/engine.py` lines 167-169 and `config.py` line 68. The style is empty unless the user named one (`llm.py` line 76).
- **Shared context, item 2 (identical generation parameters).** Every layer uses the same width, height, steps and guidance (`engine.py` lines 176-177). Defaults are 512px and 4 steps (`config.py` lines 26-28).
- **Object wrapper.** Object prompts become `"a single " + prompt + ", solo, centered"` (`engine.py` lines 173-175; `config.py` lines 51-54). This pushes objects to be generated in isolation, centred in their own frame, at an arbitrary scale.
- **Independent seed per variant.** `seed = random.randint(...)` is drawn per variant (`engine.py` line 186). There is no shared or scene seed.
- **No image conditioning.**
  - `generate_object` calls the plain `StableDiffusionXLPipeline` with only `prompt`, `negative_prompt`, steps, guidance, size and generator (`backends/layerdiffuse_sdxl.py` lines 104-111).
  - No background latent, composite or mask is passed.
  - The background is generated with the LoRA disabled, as an independent opaque image (lines 113-123).
- **Refinement ignores the scene.**
  - `refine` / `sketch` modes img2img the layer's own current image or the sketch (`engine.py` lines 88-98; backend lines 125-147).
  - The composite and neighbouring layers are never inputs.
- **Placement is post-hoc.** `compose` works in four steps (`engine.py` lines 127-158):
  1. Crop each object to its alpha bbox.
  2. Scale it uniformly to fit the box.
  3. Centre it horizontally.
  4. Stand it on the box bottom edge and `alpha_composite` it.
- **Boxes come from the LLM** (`llm.py` line 84; normalisation lines 257-261). In the no-LLM fallback the default is `[0.25, 0.25, 0.5, 0.5]` (`llm.py` line 240).
- **Scale is only box-relative.** It is `min(bw/obj.w, bh/obj.h)` (`engine.py` line 149). Nothing relates an object's size to the background's perspective or depth.
- **No lighting or shadow sharing.**
  - No relighting, harmonization, shadow or contact pass exists anywhere in engine.py or the backend.
  - The `relations` field (subject/relation/object) is only stored on the Scene (`engine.py` line 49). It is never used in generation or composition.
- **User layer transforms.**
  - Boxes can be edited as text "x,y,w,h" in a table, then recomposed (`app.py` lines 48, 60, 86-98, 349-356).
  - Z-order can be moved up or down (`engine.py` lines 114-125; `app.py` lines 359-363, 530-531).
  - No drag, rotate or flip exists (grep for "rotat"/"drag" in app.py found nothing).
- **Step 5 (regen one layer) works mechanically** (`update_layer`, `engine.py` lines 76-101). Only that layer changes. But the new content is still blind to the scene, so it can clash.

### Inferences
- The architecture is "sticker collage": coherence relies only on a shared style word. Mismatched lighting, scale and perspective, and floating objects without shadows, are the expected failures, not bugs.
- The reference workflow (generate a whole image, then decompose) gets coherence for free, because all layers come from one coherent image. The prototype inverts that order.

### Gaps
- I did not run the prototype or measure visual coherence. These conclusions come from reading the code.

## Q2. Pipeline A: generate the whole image, then decompose with Qwen-Image-Layered

### Takeaway
This is the best match to the user's reference and gives the highest coherence. It is GPU-only in practice. It does not fit the laptop. It is marginal on a Kaggle T4 (an FP8 or GGUF quant with offload is needed, and it would be slow; this is an estimate). It fits a hosted API, Comfy Cloud, or a rented 24-48 GB GPU.

### Cited Findings
- **What the model does.** It decomposes an image into multiple RGBA layers, and each layer can be edited independently. The layer count is variable: 4 by default, 3 and 8 are shown, and recursive decomposition is supported. Resolution buckets are 640 or 1024, with 640 recommended. It is 20B parameters in BF16, under Apache-2.0. Shown edits include recolour, swapping a subject, moving, resizing and deleting, some using Qwen-Image-Edit. Defaults are 50 steps and true CFG 4.0. Source: [HF model card](https://huggingface.co/Qwen/Qwen-Image-Layered).
- **ComfyUI support and cost.** ComfyUI supports it natively, with these files:
  - qwen_2.5_vl_7b_fp8 text encoder
  - qwen_image_layered_bf16 or fp8mixed
  - the layered VAE

  Measured cost: about 45 GB peak VRAM, and about 120 s at 1024px on an RTX Pro 6000. A 4090 nearly fills its VRAM. Source: [ComfyUI Wiki](https://comfyui-wiki.com/en/tutorial/advanced/image/qwen/qwen-image-layered).
- **Community GGUF quants exist (QuantStack).** One guide claims Q4 runs in 8 GB VRAM. Reported file sizes conflict (3.2 GB vs 13 GB), so treat them as unverified. Sources: [dev.to guide](https://dev.to/gary_yan_86eb77d35e0070f5/how-to-use-qwen-image-layered-gguf-in-comfyui-complete-installation-and-usage-guide-2en3); [aistudynow](https://aistudynow.com/how-to-use-qwen-image-layered-model-the-best-alternative-to-photoshop/).

### Concrete steps
1. Turn the prompt (optionally LLM-refined) into one image with a T2I model: SDXL-Turbo locally, or Krea/Flux/Qwen-Image on a GPU or API.
2. Show N interpretations. This covers goal steps 3 and 4: generate several whole images and let the group pick one.
3. Run Qwen-Image-Layered with N layers to get RGBA layers.
4. Users move, resize or rotate layers on a canvas. This is pure 2D and cheap.
5. To re-prompt one layer, either:
   - (a) edit that layer image with Qwen-Image-Edit, or
   - (b) inpaint the composite in the layer's mask region and re-decompose. Pick (b) for coherence.
6. Optional: a low-strength img2img or harmonize pass over the final composite.

### Inferences
- **Coherence.** Lighting, style and scale stay consistent because they come from one image. After a user moves layers, holes are revealed and shadows no longer match. The decomposition fills occluded background, according to the model card's showcase claims, but shadows do not follow moved objects, so a final low-denoise pass helps.
- **Laptop (a).** Not feasible: the model is 20B, 16 GB RAM is not enough, and runtime would be hours if it runs at all (estimate).
- **Kaggle T4 (b).** Possible only with fp8 or GGUF plus CPU offload. The T4 lacks native FP8 compute, so expect several minutes per decomposition at 640px. This estimate is unverified.
- **Hosted (c).** Comfy Cloud (the user's reference) or another Qwen-Image-Layered endpoint is the most practical route.

### Gaps
- I found no measured timings on a T4. I did not verify GGUF sizes or quality, or whether a Lightning step-distilled LoRA exists for the Layered variant.

## Q3. Pipeline B: chained LayerDiffuse conditional generation (background first, foreground conditioned on it)

### Takeaway
SDXL LayerDiffuse supports exactly the needed chain: bg2ble ("blend given background") and bgble2fg ("foreground from background plus blend"). The vendored diffusers clone already includes them. This is the cheapest route to real context sharing and a direct upgrade of the current backend. The joint 3-way and batched fg+blend modes are SD1.5-only.

### Cited Findings
- **The local clone provides SDXL scripts** for `fg2ble`, `bg2ble`, `fgble2bg` and `bgble2fg` using converted delta weights hosted at `rootonchair/diffuser_layerdiffuse`. See `vendor/diffuser_layerdiffuse/README.md` lines 16-18 and 132-135.
- **How bgble2fg conditions the UNet.** It concatenates background and blend latents into the UNet via `enable_unet_extra_concat_condition(pipeline.unet, torch.cat([background_latents, blend_latents], dim=1))`. Its default base model is `stabilityai/stable-diffusion-xl-base-1.0` (`test_diffusers_xl_bgble2fg.py` lines 51, 113-119).
- **SD1.5-only modes.** `test_diffusers_joint.py` (fg+bg+blend, batch 3) and the `fg_bg_cond` / `bg_fg_cond` scripts (batch 2) are SD1.5 (README lines 124-128). An SDXL inpaint/img2img transparent script also exists (README lines 131, 355-368).
- **ComfyUI-layerdiffuse mode restrictions.** It lists FG only, Blend given FG/BG, Extract FG/BG from blended, and FG from BG. Only the joint batch modes are marked "Currently only for SD15". It notes colour variation in extracted FG, and RGBA decode needs sizes that are multiples of 64. The last dated update is 2024-3-9. Source: [ComfyUI-layerdiffuse](https://github.com/huchenlei/ComfyUI-layerdiffuse).

### Concrete steps
1. Generate the background from the full refined prompt.
2. For each object, in back-to-front order, at the user's box (mask or position):
   1. Run **bg2ble** with the current composite as the background and the object prompt plus its location. This produces the blended image, where the object sits in the scene with lighting and shadows.
   2. Run **bgble2fg** to extract the transparent foreground.
   3. The blended image becomes the new composite for the next object.
3. To regenerate one layer, re-run steps 2.1 and 2.2 for that layer only, with a background equal to the composite of all other layers.
4. Placement variant: run the blend step as an SDXL inpaint restricted to the user's box mask. This is not in the vendored scripts, so it would need wiring. Alternatively, paste a rough object or sketch and run low-strength img2img.

### Inferences
- **Coherence.** Lighting, style and scale are conditioned on the actual scene. Shadows can appear in the blend (though bgble2fg separates them unpredictably).
- **Caveats.**
  - Conditional weights were trained for SDXL base. Compatibility with SDXL-Turbo's few-step schedule is untested. Expect to need base SDXL at about 20+ steps, which is about 5x slower than Turbo on CPU (estimate).
  - The blend model does not take an explicit position, so steer it with an inpaint mask or prompt.
- **Laptop (a).** Feasible but slow. Each object needs about 2 SDXL passes. With Turbo that is about 160 s per object; with base at 20 steps it is roughly 10+ min per object (estimate).
- **Kaggle T4 (b).** Good fit: about 10-20 s per pass at 1024 fp16 (estimate).
- **Hosted (c).** Needs a custom endpoint. I know of no hosted API for these modes.

### Gaps
- I did not test whether converted SDXL conditional deltas work with sdxl-turbo. I did not run the quality check myself.

## Q4. Pipeline C: generate the whole image, then segment, cut layers and inpaint holes

### Takeaway
This is the modular, CPU-capable cousin of A. Use a grounded detector (Florence-2 or GroundingDINO) plus SAM 2.1 masks to cut per-object layers, and LaMa to fill the background holes. Coherence is as good as the source image. Edges and partially occluded objects are weaker than in Qwen-Image-Layered. It is feasible on the laptop.

### Cited Findings
- I did not fetch the primary docs this session. Repos to cite and check:
  - [SAM 2](https://github.com/facebookresearch/sam2)
  - [Florence-2 (HF)](https://huggingface.co/microsoft/Florence-2-large)
  - [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
  - [LaMa](https://github.com/advimman/lama)
- I have not verified their sizes or CPU timings this session.

### Concrete steps
1. Run T2I to get the whole image.
2. For each layer noun from the LLM breakdown, ground it with Florence-2 or GroundingDINO to get a box.
3. Run SAM 2.1 to get a mask, then build an RGBA layer, optionally with matting for soft edges.
4. Inpaint the background under the masks with LaMa (fast) or SDXL inpaint (better).
5. Users move layers.
6. To re-prompt a layer, inpaint the composite in that mask (SDXL inpaint, or LayerDiffuse bg2ble) and re-segment.
7. Run a harmonize pass.

### Inferences
- **Laptop (a).** Feasible. Florence-2-base, SAM 2.1 tiny or small, and LaMa are small models that run in seconds to tens of seconds on CPU (estimate, unverified). The T2I step is still about 80 s.
- **Kaggle T4 (b) and Hosted (c).** Easy on both.
- **Main failure.** Occluded parts of objects are missing after cutting, so a moved object shows a bite. Fix it by outpainting the layer with LayerDiffuse img2img (the vendored inpaint script).

### Gaps
- No measured CPU timings for SAM 2.1 or Florence-2 on this laptop.

## Q5. Pipeline D: native joint multi-layer models

### Takeaway
Research-grade. ART has official Microsoft code. I found no confirmed released code for PSDiffusion. These are not practical for this prototype in Nov 2026 compared with A and B.

### Cited Findings
- **ART (Anonymous Region Transformer)** generates variable multi-layer transparent images from one global prompt plus an anonymous region layout. It claims over 50 layers and has an official repo, microsoft/art-msra. Sources: [sourcepulse listing](https://www.sourcepulse.org/projects/1834720); [paper summary](https://arxiv.deeppaper.ai/papers/2502.18364v1).
- **PSDiffusion** (arXiv 2505.11468) generates one RGB background plus several RGBA foregrounds in a single pass. I found no code link. Source: [paper page](https://fugumt.com/fugumt/paper_check/2505.11468v2_enmode).
- **OmniPSD** (arXiv 2512.09247) generates layered PSDs with a DiT. Source: [paper page](https://fugumt.com/fugumt/paper_check/2512.09247v1_enmode).

### Inferences
- ART's layout input maps naturally onto user boxes. It is built on a large DiT (FLUX-class, per the paper; unverified here), so it is GPU-only and not feasible on CPU.

### Gaps
- Licences, VRAM needs and per-layer re-edit support for ART are unverified. DreamLayer was not researched.

## Q6. Pipeline E: regional/layout-conditioned single pass, then decomposition

### Takeaway
Use the LLM or user boxes as a layout to generate one coherent image with objects where the user wants them. Options are regional prompting, GLIGEN-style box conditioning, or simply drawing boxes and sketches and running img2img/ControlNet. Then decompose with A or C. This is the best way to honour user placement before generation.

### Inferences (no sources fetched this session)
- **Cheapest CPU version.** Paint the LLM boxes as coloured blobs or user sketches on the background, then run SDXL-Turbo img2img at about 0.6 strength on the whole canvas. That gives one coherent image with an approximate layout. Then run C to cut layers. This reuses the existing `refine_background` path.
- **Regional prompting and GLIGEN-style nodes** exist in the ComfyUI and diffusers ecosystems. I did not verify current SDXL support or names.

### Gaps
- No verified sources on current regional/layout tools for SDXL-Turbo.

## Q7. Keeping per-layer edits coherent after editing

### Takeaway
Regenerate in context, never in isolation. Edit through the composite, then re-extract the layer.

### Inferences (synthesised from Q2-Q4 sources)
1. **Contextual regen.** Background = composite minus the edited layer. Use mask = the user's new box or sketch. Then run SDXL inpaint, or LayerDiffuse bg2ble followed by bgble2fg. The result is a new RGBA layer that matches the lighting.
2. **Harmonization pass.** Run a low-denoise img2img (0.2-0.35, estimate) on the flattened composite for display, keeping the editable layers separate. Alternatively, re-decompose (A) when the user wants the harmonized result back as layers.
3. **Move or resize.** These are pure 2D, but afterwards fill the revealed holes (LaMa or the inpaint layer) and optionally re-run harmonization for shadows.
4. **Coherence that needs no new models** comes from three changes:
   - a shared scene seed and style per scene, not per variant;
   - dropping "solo, centered" in contextual modes;
   - passing the background prompt into object prompts.

### Gaps
- No measured quality comparison between the approaches. All timings above are estimates except the measured laptop figure of about 80 s per SDXL-Turbo 512px image.
