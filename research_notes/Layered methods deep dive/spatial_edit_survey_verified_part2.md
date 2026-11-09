# Spatial editing in layered-image papers: verified survey, part 2

Sources: arXiv HTML full text, fetched 2026-11-09 through WebFetch, which returns a summary of the page. Quotes are the excerpts WebFetch returned, so they are close to the paper but check them against the paper before citing. Nothing below comes from memory.
Caveats: for LASAGNA and LayerCraft, only the first 100k characters were read, so the appendix tail is [NOT READ]. No GitHub repos were opened, so the code status is what the paper or the search result says.

## Headline finding
None of the 8 papers gives an algorithm for move, resize or reorder. Each one decomposes or generates layers, then edits them **only by alpha-compositing the RGBA layers again** (translate or scale the layer and blend). No paper re-runs the model after an edit. None describes shadow re-casting, occlusion reasoning or amodal completion at edit time. The only "fill revealed region" mechanism is a **clean background layer** produced during decomposition, either by inpainting or by learned background prediction.

---

## 1. OmniPSD (2512.09247)
- **Spatial editing in the paper:** No. Closest text: "associated with editable metadata (e.g., bounding box, visibility, stacking order)" (dataset, §3.5).
- **Algorithm:** Image-to-PSD runs a loop on Flux-Kontext with two LoRAs. The *extraction* LoRA takes (I0, mask or bbox) and returns the foreground RGBA. The *erasure* LoRA "removes the current foreground, restores the occluded background". The loop repeats until text, foreground and background are separate. Text-to-PSD trains Flux-dev on a 2×2 grid (composite, fg, mid, bg) and generates the layers jointly "in a single pass" through native self-attention. An RGBA-VAE (from AlphaVAE) encodes the layers. Text layers are re-rendered as vector text (PaddleOCR plus a font classifier).
- **Training:** Flow matching. LoRA rank 128, 30k steps at 1024².
- **Code:** project page showlab.github.io/OmniPSD. The paper does not state a code release.
- **Limitations:** no limitations section. Timesteps, inversion and inpainting details are not specified.
- **What we can borrow:** (a) **extract → erase → repeat** with a background-restoration step yields a hole-free background plate behind every layer. With Qwen-Image-Layered on a T4, we can run one inpaint pass on the background layer once, then move layers freely by compositing. (b) For relocate(): do not warp and then img2img the whole scene. Keep a restored background plate and paste the warped RGBA on top. Identity is untouched because no diffusion runs over the object.

## 2. LASAGNA (2601.15507)
- **Spatial editing in the paper:** Yes, as compositing only. "via alpha compositing alone, without invoking any model post-edit". "such as translation and resizing—without requiring model re-inference" (§5). Spatial editing "manipulates object position or scale" (§4.4).
- **Algorithm:** A 2B DiT with a T5 text encoder and an RGBA-VAE jointly denoises the composite, the FG layer and the BG layer. Each frame is tagged with type, IO, 3D-RoPE and timestep embeddings, under "full bidirectional self-attention". Inference is DDIM with 50 steps. **The model puts shadows and reflections inside the FG RGBA layer**, so they move with the object. No inversion and no edit-time inpainting.
- **Training:** flow matching, lr 1.2e-5, 20k iterations. The dataset is built with LayerDecomp and dilated masks.
- **Code:** project page rayjryang.github.io/LASAGNA-Page. The paper contradicts itself on whether the data is released.
- **Limitations:** single FG over BG only. Multi-object scenes and complex geometry are future work.
- **What we can borrow:** **bake shadow and contact effects into the object's alpha layer** so a move needs no re-harmonisation. (b) For LayerDiffuse: generate the object with its shadow in the transparent layer (prompt for the shadow, or keep the low-alpha shadow pixels) and skip the 0.3 img2img. (a) Check whether Qwen-Image-Layered layers already carry shadows. If they do, translate-and-composite is enough.

## 3. RevealLayer (ICML 2026, arXiv 2605.11818)
- **Spatial editing in the paper:** Not move, resize or reorder. It shows "region-specific manipulation, where targeted edits maintain consistency with the surrounding context" (Fig. 17). Multi-round interactive editing is future work.
- **Algorithm:** FLUX.1-dev MM-DiT, LoRA rank 64. Bounding boxes crop each layer's latent into variable-length token sequences. **Region-Aware Attention** is a mask in which layer-i tokens see their own tokens plus the global-image tokens inside box i, to stop leakage between layers. **Occlusion-Guided Adapter** runs self-attention over the layer latent plus masked conditional-image features. Its masks: BG = outside all boxes, FG_i = box_i minus the other boxes. Losses: flow matching, a focal alpha loss (τ 0.95, γ 1.5) and a cosine orthogonality loss between layers. **No explicit amodal-completion procedure.** Hidden content is recovered implicitly, through the BG layer and the attention design.
- **Training:** Prodigy optimizer, 50k iterations, batch size 8. Data: RevealLayer-100K. Inference for 3 layers at 1024² takes 122 s and 60 GB, against 418 s and 76 GB for Qwen-Image-Layered.
- **Code:** the paper's text does not state a release. The GitHub repo 360CVGroup/RevealLayer exists [NOT READ].
- **Limitations:** inaccurate boxes, heavy occlusion, transparency and dense repeated textures.
- **What we can borrow:** **region-masked attention**, so each layer attends only to its own box of the source image, keeps identity from leaking. (b) The SDXL equivalent is to restrict the img2img to the object's dilated mask (masked latent blend at each step) rather than the whole image. (a) Needs 60 GB, so it is not runnable on a T4.

## 4. CLD — Controllable Layer Decomposition (arXiv 2511.16249; HF thuteam/CLD)
- **Spatial editing in the paper:** Downstream only. Editing in PowerPoint, "including removing layers, copying and adding layers, and adjusting layouts" (§4.6). The paper gives no algorithm for it.
- **Algorithm:** LD-DiT on FLUX.1-dev crops input tokens to the user bboxes and prepends the composite token. LA-RoPE encodes (layer, h, w). The MLCA adapter takes VAE features of the conditioning image, crops them by bbox and adds them to the hidden states. Dual-condition CFG keeps the image condition in the unconditional branch and drops only the text. Decoding uses ART's multi-layer transparent autoencoder.
- **Training:** LoRA rank 64, Prodigy, batch size 4, 25k iterations at 1024², on PrismLayersPro. Flow-matching loss only.
- **Code:** LoRA checkpoints on HF thuteam/CLD (Apache-2.0). Code at GitHub monkek123King/CLD (from the search result, [NOT READ]). Inference takes about 97 s and 62 GB (RevealLayer Table 3).
- **Limitations:** heavy occlusion: "the image provides insufficient information". Small or fine content.
- **What we can borrow:** **dual-condition CFG**: keep the scene or reference image in both the conditional and unconditional branches so guidance doesn't erode fidelity to it. (b) For SDXL img2img, use the IP-Adapter or reference image in both CFG branches, or lower the CFG. (a) Not runnable on a T4.

## 5. LayerD (2509.25134)
- **Spatial editing in the paper:** Yes, minimal: "simple operations such as color conversion, translation, and resizing at the layer level". Layers can be split "into connected components for more granular editing". Reorder: the order is kept as a z-index. No reorder operation is described.
- **Algorithm:** loop over three steps: BiRefNet matting of the top layer, **LaMa** completion of the background behind it, then recovery of the FG RGB by inverting the alpha blend. Stops when the matte is empty (at most 3 iterations). The background completion "should not insert new objects". FLUX.1 Fill "often introduces unwanted objects". A palette refinement assumes flat graphic-design regions.
- **Training:** BiRefNet Swin-L, 60 epochs. LaMa is used off the shelf.
- **Code:** "Code and models are publicly available" (cyberagentailab.github.io/LayerD).
- **Limitations:** tiny text, ambiguous granularity, transparent layers. Built for graphic design, not photos.
- **What we can borrow:** **LaMa, not a generative model, to fill revealed regions**. It is cheap (runs on CPU or a T4), deterministic and never invents objects. Use it on (a) the Qwen background layer, and (b) the hole left at the old position in relocate(). **Recovering FG colour by inverting the alpha blend** gives a cleaner layer extraction.

## 6. LayerDecomp (2411.17864)
- **Spatial editing in the paper:** Yes, high-level. Effects "move naturally with the editing targets". Recomposition is by alpha blend. The ablation shifts the object by ΔX, ΔY ∈ [−0.3, 0.3] and scales by ΔS ∈ [0.5, 1.5].
- **Algorithm:** a 5B DiT conditioned on the composite latent and the object-mask latent jointly denoises a clean BG and an RGBA FG with shadows and reflections. The RGBA-VAE follows LayerDiffusion. **Consistency loss:** predict x0, decode, alpha-blend FG over BG, take L1 against the input composite. This trains the effects without FG ground truth. DDIM, 50 steps. Occlusion and revealed regions are not described beyond the BG branch predicting an object-free background.
- **Training:** 80k iterations, batch size 128, 16 A100s. Data: simulated triplets (entity segmentation plus synthetic shadows) and about 6k real with/without photo pairs.
- **Code:** the paper does not state a release (page rayjryang.github.io/LayerDecomp).
- **Limitations:** synthetic data lacks realism. Smoke and mist are not handled.
- **What we can borrow:** the **composite-consistency check**: after relocate(), decode, recomposite and measure L1 against the expected composite outside the edit region to catch drift. Also the same principle as LASAGNA: the shadow belongs in the FG alpha.

## 7. Text2Layer (2307.09781)
- **Spatial editing in the paper:** Only as motivation: "a user is able to scale or move the foreground and replace foreground or background" (intro). Reorder is not mentioned.
- **Algorithm:** CaT2I-AE encodes (F, B, m) jointly with separate decoder heads for F, B and m plus a composite branch. An SD-style UNet runs in that latent space. DDIM with 50 steps, CFG 7.5. Training backgrounds are filled by diffusion inpainting with an empty prompt (dataset only).
- **Training:** autoencoder 250k steps on 4×8 A100s. LAION-L2I is 57M images.
- **Code:** not stated.
- **Limitations:** imperfect masks, 256² evaluation, two layers only.
- **What we can borrow:** little beyond the 2-layer F/B/m formulation that LayerDiffuse already covers. **Inpainting with an empty prompt** fills backgrounds without inventing objects (relevant to the old-position hole).

## 8. LayerCraft (arXiv 2504.00010, NeurIPS 2025)
- **Spatial editing in the paper:** Placement by bbox only. No explicit move or resize. Applications are batch collage editing and narrative scenes.
- **Algorithm (Object Integration Network):** FLUX with two LoRAs, one for background inpainting and one for subject-driven object generation, then a dual-branch attention with separate QKV per branch. Tokens are merged per step: "replacement of the masked region's latent sequence X1 with X2 according to the bounding box mask". Text tokens are averaged. Reference tokens get OminiControl-style biased positional encodings. Bboxes are expanded (+15% bottom, +10% sides) in the data, to leave room for shadows and contact. Timesteps, harmonisation and losses are not specified.
- **Training:** LoRA rank 4, 20k iterations, 4×A6000, IPA300K.
- **Code:** "Code will be released" (GitHub PeterYYZhang) [NOT READ].
- **Limitations:** agent overhead. Depends on bbox accuracy. Weighted-sum attention gave boundary artifacts and background changes.
- **What we can borrow:** **latent replacement inside the box at every step**: object-branch latents fill the box and background latents fill the rest. That is the most direct cure for both kinds of identity drift. (b) In SDXL relocate(): denoise with the scene latent re-injected outside the dilated new mask at every step (Blended-Latent style), and use a reference adapter (IP-Adapter) for the object inside the mask. **Expand the mask downward by about 15%** so contact shadows can form. Full OIN needs FLUX, so it won't fit a T4 without heavy offload.

---

## Synthesis for our two pipelines
**(a) Qwen-Image-Layered on a T4**
1. Decompose once.
2. Fill the background layer's holes with LaMa (LayerD), not a generative inpaint.
3. Move or resize layers by alpha compositing only (LASAGNA, LayerD, LayerDecomp, CLD all do this). No model re-run, so zero identity drift.
4. If shadows sit in the FG layer they travel with it. If not, accept that or add a separate shadow layer.

**(b) SDXL + LayerDiffuse relocate(), fixing identity drift**
1. Keep the object as its LayerDiffuse RGBA, generated with its shadow inside alpha (LASAGNA, LayerDecomp).
2. Fill the vacated area with LaMa, or an SD inpaint with an empty prompt (LayerD, Text2Layer).
3. Composite the warped RGBA at the new position.
4. Optional harmonise pass:
   - Restrict it to a dilated mask, expanded about 15% downward (RevealLayer, LayerCraft).
   - Re-inject the original latents outside the mask at every step (LayerCraft).
   - Keep the object pixels locked inside the alpha core. Diffuse only the boundary ring and the shadow area.
   - Keep the reference conditioning in both CFG branches (CLD).
5. Verify with a composite-consistency L1 outside the edit region (LayerDecomp).
