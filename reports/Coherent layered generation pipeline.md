# Generate one image first, then split it

> **Status update (2026-11-09).** This report is the 11-07 analysis and stays as written. Since then:
> - Qwen-Image-Layered **runs on the free Kaggle 2× T4** (GGUF Q4 + `fp16_mixed`): 27–35 s per split at 384 px, 4 layers, 4 steps. Item 6 of "What to test first" is answered.
> - The LayerDiffuse scene-to-layer chain works on a T4 (about 18 s per pass at 1024 px), and `relocate()` moves/resizes a layer with re-harmonisation in about 4 s.
> - Segment + inpaint (SAM2 + LaMa) and LayerD were tried once; next attempts are listed in [findings-log.md](../docs/findings-log.md).
> - Current status of every method: [Layered methods - how to get them working.md](Layered%20methods%20-%20how%20to%20get%20them%20working.md).

The current storytelling prototype **does not achieve coherent layered generation**. It makes a sticker collage. Each layer is a separate SDXL-Turbo text-to-image call with its own random seed and no image conditioning. Each object is pushed into isolation by the "a single X, solo, centered" wrapper and then pasted into a box the LLM guessed. The layers share only a style suffix and the same generation settings. Qwen-Image-Layered works the other way round. Its released model takes **one finished RGB image** and splits it into N RGBA layers that composite back to that image, so the layers are coherent because they all come from the same picture. Your Comfy Cloud reference (Krea2-Turbo T2I → Qwen-Image-Layered with 6 layers → recomposite) is therefore the right design, and it should be the primary pipeline. Qwen-Image-Layered is a 20B model, so it cannot run on the laptop. There are two local fallbacks:

- chained SDXL LayerDiffuse (bg2ble → bgble2fg), which builds the scene one object at a time, each conditioned on the scene so far;
- whole-image T2I followed by segmentation and inpainting.

Whichever pipeline you use, a layer edit stays coherent only if it is **regenerated through the composite**, never generated in isolation. The LLM split stays as an optional add-on. The "solo, centered" wrapper should be deleted.

## The prototype collages independent stickers (code evidence)

File paths are relative to `prototype/`. The findings come from reading the code. I did not run the prototype or measure output quality.

**What the layers share is only text and settings.** `_full_prompt` adds `scene.style` to every layer prompt (`layerproto/engine.py` 167-169, `config.py` 68), and that style is empty unless the user names one (`llm.py` 76). Every layer uses the same width, height, steps and guidance (`engine.py` 176-177; defaults are 512px and 4 steps, `config.py` 26-28). Nothing else is shared:

- **Seeds:** each variant draws `seed = random.randint(...)` (`engine.py` 186). There is no scene seed.
- **Prompts:** the LLM is told to describe objects "alone (no other objects, no scenery)", and the background must contain "no characters or main objects" (`llm.py` 81-83).
- **The wrapper:** object prompts become `"a single " + prompt + ", solo, centered"` (`engine.py` 173-175, `config.py` 51-54). Each object is rendered centred in its own frame at an arbitrary scale.
- **Conditioning:** `generate_object` calls a plain `StableDiffusionXLPipeline` with prompt, negative prompt, steps, guidance, size and generator only (`backends/layerdiffuse_sdxl.py` 104-111). No background latent, composite or mask is passed in. The background is a separate opaque generation with the LoRA turned off (113-123).

**Placement and scale happen after generation, in 2D.** `compose` (`engine.py` 127-158) crops each object to its alpha bounding box, scales it by `min(bw/obj.w, bh/obj.h)` (line 149), centres it horizontally and stands it on the bottom edge of an LLM box (`llm.py` 84, 257-261; when there is no LLM the fallback box is `[0.25,0.25,0.5,0.5]`, line 240). Size is never related to the background's perspective. No code pass handles relighting, shadows or contact, and the `relations` field is stored (`engine.py` 49) but never used. Users can move or resize a layer only by typing "x,y,w,h" into a table (`app.py` 48, 60, 86-98, 349-356) or by changing z-order (`engine.py` 114-125). Rotate and drag do not exist.

**Per-layer regeneration ignores the scene.** `update_layer` (`engine.py` 76-101) changes only the target layer, as intended. But refine and sketch modes run img2img on that layer's own image (`engine.py` 88-98; backend 125-147), so the composite and the neighbouring layers are never inputs. From this code, mismatched lighting, scale and perspective, and objects floating without shadows, are the **expected behaviour, not bugs**. The prototype never uses either coherence mechanism LayerDiffuse offers: shared attention, or conditioning on the composite.

## What Qwen-Image-Layered and LayerDiffuse actually do

### Qwen-Image-Layered: image → layers, not text → layers

The paper defines the task as one RGB image in and N RGBA layers out, where blending the layers in order reproduces the input ([arXiv 2512.15603](https://arxiv.org/html/2512.15603)). Its VLD-MMDiT handles **a variable number of layers** (trained with up to 20) and produces all of them in a single pass. The authors present this as an advantage over recursive methods that "lead to error propagation". The model uses its own RGBA-VAE, which beats the LayerDiffuse VAE on AIM-500 ([arXiv](https://arxiv.org/html/2512.15603)). The paper did train a **text-to-layers (T2L) variant**, but the official repo describes text-to-multi-RGBA performance as "limited" and releases no separate T2L checkpoint. The paper's own best-looking text route is **Qwen-Image T2I followed by Qwen-Image-Layered I2L**, reported to "further improve visual aesthetics" ([GitHub](https://github.com/QwenLM/Qwen-Image-Layered)). That is the shape of your reference workflow.

The released model is 20B parameters in BF16 under **Apache-2.0** and runs through `QwenImageLayeredPipeline`. The `layers` parameter is set per run: 4 by default, and examples show 3 and 8. Resolution is 640 (recommended) or 1024, with 50 steps and true CFG 4.0 ([HF card](https://huggingface.co/Qwen/Qwen-Image-Layered)). The prompt describes the whole image, including hidden parts. It does not control individual layers. Any output layer can be decomposed again. ComfyUI supports the model natively, at **about 45 GB peak VRAM and about 120 s at 1024px on an RTX Pro 6000** ([ComfyUI Wiki](https://comfyui-wiki.com/en/tutorial/advanced/image/qwen/qwen-image-layered)). Community GGUF quants exist; one guide claims Q4 runs in 8 GB, but reported file sizes conflict, so treat this as unverified ([dev.to](https://dev.to/gary_yan_86eb77d35e0070f5/how-to-use-qwen-image-layered-gguf-in-comfyui-complete-installation-and-usage-guide-2en3)).

The official editing loop has three steps: decompose, edit one RGBA layer with **Qwen-Image-Edit**, then recombine bottom-to-top. Move, resize and delete are plain layer operations ([GitHub](https://github.com/QwenLM/Qwen-Image-Layered)). There are two gaps:

- The paper does not say clearly how occluded regions are completed.
- An edit made with Qwen-Image-Edit on an isolated layer has no guarantee of matching the other layers' lighting. This is an inference: the edit never sees the composite.

### LayerDiffuse: the paper's coherent modes mostly shipped for SD1.5 only

The paper describes these modes ([arXiv 2402.17113](https://arxiv.org/html/2402.17113)):

- single transparent foreground;
- **joint fg+bg generation**, kept coherent by concatenating Q/K/V across the layers' latents in every attention layer;
- foreground-conditioned background, and background-conditioned foreground;
- **iterative multi-layer generation**: repeat background-conditioned foreground generation, each time using the blended composite so far as the condition.

The paper names a trade-off: conditioning on the background bakes the scene's lighting into the foreground, so the foreground becomes **less reusable once moved**.

The released weights split by base model ([sd-forge-layerdiffuse](https://github.com/layerdiffusion/sd-forge-layerdiffuse)):

| Base | Released modes | Not released |
|---|---|---|
| **SDXL** | fg-only (`attn`, `conv`); two-step conditional pairs `fg2ble`/`fgble2bg` and **`bg2ble`/`bgble2fg`** | joint ("on hold") |
| **SD1.5** | fg-only; **joint** (batch of 3: fg/bg/blend); `fg2bg`; `bg2fg` | none |

In the prototype's vendored diffusers clone, `test_diffusers_xl_bg2ble.py` turns a background plus a prompt into a blended scene. `test_diffusers_xl_bgble2fg.py` then extracts the transparent foreground by adding the background and blend latents as extra UNet input channels. Both default to SDXL base 1.0 at 20 steps and CFG 7 ([upstream](https://github.com/rootonchair/diffuser_layerdiffuse)). **No mode accepts a user box or mask**, so the model decides where objects go and how big they are. Whether these LoRAs work with SDXL-Turbo (1-4 steps, CFG ≈ 0) is **undocumented**. I infer that the conditional modes will degrade on Turbo and will need base SDXL at about 20 steps.

## Most joint "prompt → layer stack" models cannot be run today

The idea of one global prompt producing a coherent layer stack through shared attention is well established in papers, but none of these models can be downloaded and run:

- **PSDiffusion** ([arXiv](https://arxiv.org/html/2505.11468v2)): SDXL + LayerDiffuse-style VAE, global prompt plus per-layer prompts, 3-6 layers. No code or weights.
- **DreamLayer** ([arXiv](https://arxiv.org/html/2503.12838)): SD1.5, 2-4 layers. Moves and resizes layers with matching shadows. Project page only.
- **OmniPSD** ([arXiv](https://arxiv.org/html/2512.09247v1)): FLUX in a 2×2 grid (composite, fg, mid, bg). No weights.
- **LASAGNA** ([arXiv](https://arxiv.org/html/2601.15507)): roughly 2B DiT with Text2All, FG_Gen and BG_Gen modes. Only the dataset is promised.
- **LayerFusion** ([project](https://layerfusion.github.io)): no code confirmed.
- **ART+** ([HF papers](https://huggingface.co/papers/2505.22523)): only the dataset is linked.

None of these papers describes how to regenerate one layer while freezing the rest.

The runnable options are all two-stage decomposers or single-layer generators:

| Method | What it does | Runnable? | Licence | Fit |
|---|---|---|---|---|
| **Qwen-Image-Layered** | image → N RGBA layers | Yes | Apache-2.0 | Hosted or large GPU |
| **CLD** ([GitHub](https://github.com/monkek123King/CLD), [HF](https://huggingface.co/thuteam/CLD)) | image + one box per layer → layers | Yes | MIT/Apache, but needs **FLUX.1-dev (non-commercial)** | Quantized T4 (unmeasured) |
| **LiWi** ([page](https://rassetmusty.github.io/LiWi)) | Qwen-Image-Layered fine-tune, adds a shadow layer | By request form only | n/a | n/a |
| **RefLayer** ([arXiv](https://arxiv.org/abs/2602.19358)) | layers selected by point, box, mask or text | No | n/a | n/a |
| **Qwen-Image-2.1** ([HF](https://huggingface.co/Qwen/Qwen-Image-2.1)) | native RGBA single-layer generation and editing, up to 10 reference images | Yes | Research licence (non-commercial) | GPU; multi-layer decomposition **unconfirmed** |
| **LayerDiffuse SDXL/SD1.5** | fg-only and conditional chains | Yes | OpenRAIL-M ([HF](https://huggingface.co/LayerDiffusion/layerdiffusion-v1)) | Laptop (slow), T4 |
| **Segmentation stack**: Florence-2 / GroundingDINO + SAM 2 + LaMa ([SAM 2](https://github.com/facebookresearch/sam2), [Florence-2](https://huggingface.co/microsoft/Florence-2-large), [LaMa](https://github.com/advimman/lama)) | cut objects out of an image, fill the holes | Yes | Permissive (not re-verified) | Laptop CPU (estimate) |

## Five candidate pipelines compared

All timings are **estimates** unless marked measured. The only measured laptop figure is about 80 s for one SDXL-Turbo 512px image.

| | Pipeline | Coherence | Per-layer re-prompt | Move/resize | Laptop CPU | Kaggle T4 | Hosted | Licence |
|---|---|---|---|---|---|---|---|---|
| **A** | T2I (Krea/Flux/Qwen-Image/SDXL) → **Qwen-Image-Layered** | Highest: all layers come from one image | Inpaint the composite in the layer mask, then re-decompose; or Qwen-Image-Edit on the layer (less coherent) | 2D, cheap. Shadows don't follow and holes need filling | No (20B) | Marginal: fp8/GGUF + offload, minutes per run (unverified) | **Yes**: Comfy Cloud (your reference), fal | Apache-2.0 (T2I model's licence applies separately) |
| **B** | Background → per object **bg2ble → bgble2fg**, each conditioned on the composite so far | High: lighting and shadows are conditioned on the real scene | Re-run both steps for that layer with background = composite of the other layers | Model chooses placement. A user box needs an added inpaint mask (not in the repos). Moving breaks baked lighting | Feasible but slow: about 2 SDXL passes per object, around 10+ min per object at 20 base steps | Good: about 10-20 s per pass | Needs a custom endpoint | OpenRAIL-M |
| **C** | T2I → ground (Florence-2/GroundingDINO) → SAM 2 masks → LaMa/SDXL background fill | As good as the source image. Weaker edges, and occluded parts are missing | Inpaint the composite in the mask (SDXL inpaint or bg2ble), then re-segment | 2D. A moved object shows a "bite" where it was hidden unless outpainted | **Yes**: segmenters take seconds to tens of seconds plus about 80 s T2I (unverified) | Easy | Easy | Mostly permissive |
| **D** | Native joint models (PSDiffusion, DreamLayer, ART+, LASAGNA) | High in the papers | Not described | DreamLayer: in latent space | No | No weights | No | Unreleased or unclear |
| **E** | Layout first (user or LLM boxes painted as blobs or sketch) → whole-canvas img2img about 0.6 → decompose with A or C | High, and respects the user's placement | Same as A or C | Placement fixed before generation | Yes, reuses `refine_background` | Yes | Yes | Same as A or C |

## Recommendation: decompose a coherent image and edit through the composite

**Primary core pipeline (A, mirroring your reference):**

1. Prompt → several whole-image T2I candidates. The group picks one.
2. Qwen-Image-Layered (hosted on Comfy Cloud or fal, layers set to about 4-6 at 640px) → RGBA layers, saved as PNGs, or as a PSD/ZIP via the official app.
3. Canvas editing: move, resize, rotate and delete as pure 2D operations, then fill any revealed holes.
4. Final pass: a light img2img harmonization over the flattened image, about 0.2-0.35 denoise (estimate), for display only. The layers stay editable.

**Re-prompting a layer while keeping coherence.** Never regenerate the layer on its own:

1. Background = composite of all other layers.
2. Mask = the layer's current or new region.
3. Inpaint with the new prompt: SDXL inpaint, or LayerDiffuse bg2ble.
4. Extract the new layer: bgble2fg, a segmenter, or re-decompose with Qwen-Image-Layered.

Qwen-Image-Edit applied to the bare RGBA layer is quicker, but it never sees the scene.

**Local/free fallback.** On a laptop or Kaggle T4, use **C (or E → C)**: SDXL-Turbo makes the whole image, a segmenter cuts the layers, and LaMa fills the holes, all on CPU. Use **B** when you need newly added objects to pick up the scene's lighting. On a T4 that is about 10-20 s per pass (estimate). On the laptop it is only practical for one or two objects.

**Remove from the current code:**

- the `"a single … , solo, centered"` wrapper (`engine.py` 173-175, `config.py` 51-54);
- the "describe alone, no scenery" instruction to the LLM (`llm.py` 81-83);
- per-layer independent T2I as the main path (`layerdiffuse_sdxl.py` 104-111);
- per-variant random seeds (`engine.py` 186);
- the post-hoc fit-to-box scaling in `compose`, as a generation strategy (`engine.py` 127-158). Keep it only as a 2D canvas transform.

Keep the LLM split only as an optional helper. It can provide layer names for grounding in C, region boxes for layout in E, or a whole-image caption for Qwen-Image-Layered.

## What to test first

1. **Baseline.** Run your Krea2-Turbo → Qwen-Image-Layered workflow on 5-10 story prompts. Check whether occluded background is actually filled and whether layers survive a move. The paper is unclear on hole completion.
2. **Edit loop.** Re-prompt one layer through the composite and re-decompose. Check that the other layers stay stable and that the new layer matches the scene's lighting.
3. **Cost.** Measure Comfy Cloud or fal cost and latency per decomposition. The earlier estimate of about $0.05 per image on fal comes from prior notes and has not been re-verified.
4. **Turbo compatibility.** Test whether the SDXL bg2ble/bgble2fg deltas work with SDXL-Turbo at 4-8 steps, or need base SDXL at 20 steps.
5. **CPU timings.** Time Florence-2-base + SAM 2.1-tiny + LaMa on the laptop.
6. **Quantized decomposition.** Test whether a Qwen-Image-Layered GGUF Q4 runs on a T4 at 640px in usable time.
7. **Qwen-Image-2.1.** Check whether it can decompose an image into multiple layers (unconfirmed).

## Conclusion

The core insight is about **order**: coherence comes from generating a scene together, not from assembling pieces generated apart. Every method that holds up either produces one image first and splits it (Qwen-Image-Layered, CLD, segmentation), or conditions each new layer on the composite so far (LayerDiffuse chaining, LASAGNA). The prototype did neither. Fixing it is a rewrite of the generation order, not a matter of tuning prompts.

A second point follows. Once layers come from one image, coherence becomes a property you have to **maintain**, not just create. Every move or re-prompt slightly breaks lighting, shadows and occlusion. The durable design treats the composite as the source of truth: edits go through it, and layers are re-extracted from it. That keeps a PSD-like editing experience without pretending the layers are independent assets.
