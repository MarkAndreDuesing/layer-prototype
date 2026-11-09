# How layered and object-relocation papers do spatial editing

How to read this note:
- **[V]** means I read the method section of the arXiv HTML myself during this pass (2026-11-09).
- **[U]** means the claim comes from prior knowledge, an abstract or the project README and was not re-checked this pass. Check it before you rely on it.
- The arXiv HTML link for any paper is `https://arxiv.org/html/<id>`.

## 1. Layered papers

| Paper | Spatial edit? | Mechanism | Training | Code/weights | What we can borrow |
|---|---|---|---|---|---|
| **Qwen-Image-Layered** 2512.15603 [V] | Yes, but trivially. It decomposes the image into RGBA layers, then a user moves, resizes or recolours a layer by hand and the layers are alpha-composited again. | No re-generation step after an edit. The paper does not describe any harmonisation, shadow update or amodal completion. Its claim is that the layers "inherently" support basic operations. The occluded parts of lower layers are already complete because the model generates full layers. | Heavy: three stages of roughly 1.3M steps, on top of the Qwen-Image 20B MMDiT with an RGBA-VAE | Released on QwenLM GitHub/HF. Too large for a T4 without heavy quantisation [U]. | The pixels of a moved layer are **never redrawn**. Identity holds because no model touches the moved layer. That is the baseline our relocate() should match. |
| **Collage Diffusion** 2303.00262 [V] | Yes. You edit the collage (layer positions), then it regenerates. | SDEdit on the composite, **with a separate noise level t_i per layer**. At each step a mask blends the solver output with the re-noised composite. t_i=0 freezes a layer. Masked cross-attention (positive or negative bias on QKᵀ per layer token). Per-layer textual inversion. Per-layer ControlNet weight map. | Training-free, apart from per-image textual inversion, which is optional | No official code found [U] | **Per-layer strength map**: moved object at ~0.05–0.1, its boundary ring at ~0.3, background at 0.3–0.5. This is the core fix. |
| **DesignEdit** 2403.14487 [V] | Yes: move, resize, flip, remove, pan, zoom, occlusion-aware editing. | Proximal-guidance inversion of the source. **Key-masking self-attention** for the first K=40 of 50 steps fills the hole. After every step the latent is reset outside the mask to the inversion trajectory. Resizing is done at **image level and then re-encoded**, because latent resizing blurs faces. At step T−K the shifted object latents are pasted onto the canvas latent. The last ~10 steps harmonise the boundaries. | Training-free, SDXL-1.0 | Project page design-edit.github.io; GitHub repo [U] | Paste the **inverted latent** of the object, not a VAE re-encode plus noise, and only for the last ~20% of steps. Key-masking for the revealed hole. |
| **LayerDiffuse** (base we use) [U] | No spatial editing of its own | Latent transparency offset. Its fg/bg/blend conditional models (bg2ble, fg2ble) | LoRA plus conditioning, released | Yes (SDXL) | We already use bg2ble. fg-conditioned blending (fg2ble) conditions on the **object** and so keeps identity better than bg2ble. |
| **Stable-Layers** 2605.30257 [U] | Claims layer-consistent editing | Not verified. I could not read the method in this pass. | ? | ? | Read before citing. |
| **DreamLayer** 2503.12838 [U] | Yes: layer-wise generation and editing | Context-sharing attention between layers, plus a layer-shared self-attention and information retained across layers. Editing works by regenerating chosen layers while keeping the shared attention to the others. | Trained (fine-tuned SD) | Code unclear | Shared-attention injection of the K/V of the unchanged layers into the changed one. |
| **PSDiffusion** 2505.11468 [U] | Partly (harmonised multi-layer generation) | A single model generates layers jointly, with cross-layer attention and position/shadow priors. It produces plausible shadows and occlusion between layers. | Trained | Code announced [U] | Ideas only. Shadows are a separate layer, so a move should also move or regenerate the shadow layer. |
| **LayerFusion** 2412.04460 [U] | Not focused on editing | Training-free fg/bg generation on LayerDiffuse. Attention-level blending with masks derived from the transparency of the fg layer. Harmonised blend. | Training-free on SDXL+LayerDiffuse | [U] | **Closest to our stack.** Fuse the fg branch's attention into the blended image so the object stays the fg version. |
| **ART / ART+ PrismLayers** 2502.18364, 2505.22523 [U] | Generation via a layout. Editing means changing the layout and regenerating. | Anonymous region layout, one transformer over all layers with a layer-wise mask attention. A multi-layer transparent VAE. | Trained (FLUX-based) | Some weights [U] | Too heavy for a T4. Borrow only the region-masked attention idea. |
| **OmniPSD** 2512.09247 [U] | Decompose and edit PSD | DiT with in-context layers [U] | Trained | [U] | Nothing practical. |
| **LASAGNA** 2601.15507 [U] | Layered generation and editing with lighting/shadow effects [U] | Not verified | Trained [U] | [U] | Check whether it outputs a separate effects layer. |
| **RevealLayer** (ICML26, 360CVGroup) [U] | Decomposition with **amodal completion** of occluded regions | Not verified | Trained | GitHub 360CVGroup/RevealLayer | Use it to complete hidden object parts *before* a move, so the revealed side is not hallucinated per edit. |
| **CLD** [U] | Not verified | Not verified | Not verified | Not verified | Not verified |
| **LayerD** 2509.25134 [U] | Decomposition (graphic design). Edits by recompositing. | Iteratively extracts the top layer (matting) and inpaints the background | Trained matting plus an off-the-shelf inpainter | Code from CyberAgent [U] | Peel-and-inpaint loop for the background hole. Qwen-Image-Layered notes LayerD's inpainting artifacts [V]. |
| **LayerCraft** [U] | Agentic (LLM) layout, with layer-wise generation and editing | Not verified | Mostly training-free plus LoRA [U] | Not verified | LLM planning, which we already have via Ollama. |
| **LayerDecomp** [U] | Yes. Decomposes into bg plus a fg that includes shadows and reflections, then moves it. | DiT trained on fg/bg pairs including visual effects | Trained | No [U] | Treat shadows as part of the fg layer, or as a child layer, when moving. |
| **Generative Image Layer Decomposition with Visual Effects** [U] | This is the LayerDecomp paper (same work) [U] | Not verified | Not verified | Not verified | Not verified |
| **Text2Layer** [U] | No (generates a 2-layer fg/bg) | Joint autoencoder for fg, bg and mask | Trained | Partial | Nothing new. |

## 2. Non-layered object relocation

| Method | Mechanism | Training | Code | Borrow |
|---|---|---|---|---|
| **DragonDiffusion / DiffEditor** [U] | Inversion. Energy guidance on UNet features: the source object's features at the old position are pulled to match the generated features at the new position. **Memory bank** of the source self-attention K/V, injected into the edit branch. DiffEditor adds regional SDE noise and image-prompt embeddings. | Training-free (SD1.5) | Yes (MC-E/DragonDiffusion) | **Inject the reference K/V** of the object (mutual self-attention) for identity |
| **Diffusion Self-Guidance** [U] | Gradient guidance on attention centroid, size and appearance (feature mean inside the attention map) | Training-free, Imagen | No | Appearance-preservation loss as a cheap guidance term |
| **ObjectDrop / ObjectMove** (Google) [U] | Counterfactual dataset (photos with and without the object) trains removal that also removes shadows and reflections. Bootstrapped synthetic data trains insertion with effects. Move = remove + insert. | Trained, closed | No | Concept: move = remove (with effects) + insert (with re-shadowing) |
| **Magic Fixup** [U] | User coarse-edits (warps) the image. A model conditioned on the coarse edit and on **reference features of the original** (two-branch, with cross-frame attention from detail) synthesises a realistic result. Trained on video frames. | Trained | Code from Adobe [U] | Exactly our setup (warp → refine). The detail branch is mutual self-attention to the original image. |
| **AnyDoor** [U] | DINOv2 ID tokens plus a high-frequency detail map into a ControlNet-like branch. Insertion at a box. | Trained, SD2.1 | Yes | Using the high-frequency map as control (e.g. a Canny/tile ControlNet on the object) |
| **PrimeComposer** [U] | Training-free. Object-prior attention (self-attn K/V from the object's inversion) plus region-constrained cross-attention, progressively | Training-free | Yes [U] | Region-constrained attention plus prior attention for identity |
| **TF-ICON** [U] | **Exceptional prompt** inversion of the composite (accurate inversion). Composite self-attention maps injected from the fg and bg inversions for the first steps. | Training-free SD | Yes | Accurate inversion of the composite, then attention injection, instead of plain img2img noise |
| **Move Anything / ObjMove** [U] | Not verified | Not verified | Not verified | Not verified |
| **GeoDiffuser** [U] | Geometric transforms applied as a **warp of the attention (shared K/V) between the source and the edit branch**, plus losses for the revealed regions | Training-free | Yes | Warp the reference attention maps with the same affine transform we apply to the RGBA layer |
| **RegionDrag** [U] | Region-to-region copy of inverted latents and self-attention features in one step, fast | Training-free | Yes | Copy the inverted latent of the region (fast, no gradients) |

## 3. Why our relocate() redraws the object

Our current pipeline: warp the RGBA layer → SDXL img2img on the whole frame at strength 0.3 via bg2ble → early-step interior pinning → mask extraction.

At strength 0.3 the object interior is still re-noised and denoised with **no reference to its original appearance** except the pinned early steps. bg2ble also conditions on the background, not the foreground. The model is therefore free to "re-interpret" the object. Every method above that keeps identity does at least one of three things:
- (a) never re-denoises the object interior (Qwen-Image-Layered, Collage t_i=0);
- (b) re-injects the object's own K/V or latents throughout (DragonDiffusion, PrimeComposer, TF-ICON, GeoDiffuser, Magic Fixup);
- (c) pastes the object's latents late and only harmonises the boundary (DesignEdit).

## 4. Ranked recommendation: three mechanisms to implement next

1. **Per-layer noise map / mask blend at every step (Collage Diffusion t_i, plus DesignEdit's late fusion).**
   - **Zone strengths:**
     - Object interior: effective strength ≈0, replaced with the warped object's latent re-noised to the current t at **every** step, not just the early ones.
     - Eroded-edge band (~8–16 px): strength 0.3.
     - Background: 0.3.
     - Hole: higher, or inpaint it.
   - **Final steps:** free only the boundary band for the last 2–3 steps.
   - **Effort:** ~30 lines in a diffusers callback (`callback_on_step_end`). No training, and it runs on a T4.
   - **Expected effect:** fixes most of the identity drift on its own.

2. **Mutual self-attention / reference K/V injection of the object (DragonDiffusion memory bank, PrimeComposer, TF-ICON; GeoDiffuser if transformed).**
   - **Reference branch:** run the warped RGBA object (on neutral bg, or the fg2ble path) through a parallel UNet pass at the same t.
   - **Edit branch:** in the decoder self-attention layers, concatenate the reference K/V, restricted to the object mask (or warped with the same affine).
   - **Effect:** keeps texture identity even in the boundary band, where 1 lets the model draw.
   - **Effort:** training-free, but needs about 2× UNet compute. On a T4 in fp16 at 1024 that is fine.

3. **Accurate inversion instead of noise-and-denoise img2img (DesignEdit proximal guidance / TF-ICON exceptional-prompt / DDIM inversion of the composite).**
   - **Change:** invert the warped composite, then denoise with the inverted trajectory as the reset target outside the editable band (DesignEdit Eq. 2).
   - **Effect:** stochastic noise no longer gives the model licence to redraw.
   - **Revealed hole:** pair it with key-masked self-attention to fill the hole left at the old position.

**Optional later step:** move shadows as a child layer (LayerDecomp/PSDiffusion idea). For amodal completion of the revealed side, pre-complete the layer once (RevealLayer or Qwen-Image-Layered decomposition) instead of letting each relocate() hallucinate it.

## Sources
- https://arxiv.org/html/2512.15603 (Qwen-Image-Layered, Sec. 3, 4.4.2) [V]
- https://arxiv.org/html/2403.14487 (DesignEdit, key-masking, Eq. 2–4, K=40/50) [V]
- https://arxiv.org/html/2303.00262 (Collage Diffusion, per-layer noise, masked cross-attention) [V]
- Everything else: https://arxiv.org/abs/<id> for the IDs given in the brief, and the GitHub repos named. Not re-read in this pass [U].
