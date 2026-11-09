# Spatial editing survey (verified, part 1)

Source: arXiv HTML full text for each paper, read through WebFetch, which returns model-summarised extracts. Quotes are as the fetch returned them. Not read: ART HTML past character 100k (including supplementary Algorithms 1 and 2), Stable-Layers past character 100k (including Appendix F), and the PDFs. Code repositories were not checked.

## 1. DreamLayer (2503.12838): describes spatial editing
- What the paper says: "we can edit the layers within the latent space, ensuring a smoother, more harmonious fusion of the layers"; Sec 4.4: "we perform these edits within IRH at the latent level". In Eq. 13, op(.) covers "resizing, flipping, and moving".
- How it works: op(.) is applied to each foreground latent and its alpha. The layers are composited in latent space as bg*prod(1-a_j) + sum a_i*z_i*prod_{j above i}(1-a_j); occlusion is handled through layer order. The fused latent is re-noised and denoised in Information Retained Harmonization (IRH). Fusion happens between T_H=600 and T_H'=400 out of 1000 steps; Appendix B states the reverse order, so the paper contradicts itself. Re-denoising runs for T-T_H steps with 50 DDIM steps. The model uses Context-Aware Cross-Attention (CACA) and Layer-Shared Self-Attention (LSSA).
- Shadows: shadows are not modelled explicitly. During re-denoising, IRH "generates corresponding shadow details", for example a chair's floor shadow follows the chair to its new position. Regions revealed by a move get no inpainting step. The paper claims amodal completion for objects moved past the image border.
- Training: fine-tuned from SD1.5 using the Custom Diffusion approach (K and V layers only), 4 days on 2 A100s. Edits need no extra training.
- Code: only a project page is given (ll3rd.github.io/DreamLayer); the paper makes no statement about releasing code.
- Limitations: alpha accuracy gets less attention than layout and harmony, at most 4 layers, and too few LSSA steps cause layer generation to fail.

## 2. PSDiffusion (2505.11468): METHOD section does not describe spatial editing
- The only claim is in the introduction: "our model supports various layer-wise editing for users to implement precise layer control". Layout is decided while generating, from a global cross-attention map: "we can precisely derive and control the location of foreground instance".
- How it works: SDXL with a rank-256 LoRA and a LayerDiffuse-style RGBA VAE. Eqs. 2-4 reweight each layer's attention map with the normalised global map (M~ = Norm(M^g) * M^l, then rescale). Extra K_GL/V_GL projections in every self-attention block give partial joint self-attention between layers. Masks come from filtering, blurring and binarising the global attention map at resolution 16; thresholds are not given. Layers are composited in depth order (Eq. 1).
- Shadows: built by hand during dataset curation; the method does not generate them.
- Training: about 60 A800 GPU-hours, 10k iterations. The fetched text says nothing about code.
- Limitations: the METHOD section lists none. The dataset depends on manual curation.

## 3. LayerFusion (2412.04460): spatial editing only mentioned outside METHOD
- Mentions: Fig. 1 says it "enables performing spatial editing with the generated image layers in a straight-forward manner". Moving, resizing, reordering and filling are not described anywhere in the text read.
- How it works (training-free, built on SDXL RealVisXL_V4.0 with LayerDiffuse foreground weights):
  - Structure prior from the last self-attention layer (up.1.attns.2.block.1): sparsity s_i = 1/sum_j m_ij^2, then density s' = 1 - norm(s).
  - Content-confidence prior c from the EOS token's cross-attention probability, averaged over heads.
  - mask_soft = norm(s*c); mask_hard = sigmoid(10*(mask_soft - 0.5)).
  - Blended latent: a'_B = a_FG*mask_soft + a_B*(1-mask_soft).
  - Foreground latent: a'_FG = a'_B*mask_hard + a_FG*(1-mask_hard).
  - The background and blended branches share self- and cross-attention.
  - Masks are shown at t=0.8T; the full timestep range is not stated.
- Results and limitations: better shadows and lighting than baselines. Only two layers, and output quality depends on how good the masks are.
- Code: only a project page is given.

## 4. ART (2502.18364): no move, resize or reorder in METHOD
- The paper says it "avoids the need for inpainting algorithms to complete missing parts of the occluded layers". In the supplementary section on editing, layers are edited by "regenerating the layers that need to be edited, and freezing the remaining layers".
- How it works: FLUX.1-dev MMDiT with an LLM (LLaMA-3.1-8B) that plans an anonymous region layout. Each layer is a tight crop with sides divisible by 16. Layout-conditioned multi-layer 3D RoPE (Eq. 6). A multi-layer transparency VAE with a ViT-B decoder. LoRA rank 64.
- Limitations: the model sometimes generates repeated layers or merges layers together. Code: not found in the text read.

## 5. PrismLayers / ART+ (2505.22523): placement only
- Placement: layers are "resized to fit the corresponding bounding boxes" and "composited according to the layer-order annotations".
- Not covered: harmonisation, shadows and occlusion. The conclusion lists "lighting and occlusion effects" as an open problem.
- Pipeline: per-layer FLUX generation "isolated on a gray background", with RMBG-2.0 producing the alpha. ART+ is trained on FLUX.1-dev for 20k iterations at 512 and 10k at 1024.
- Limitations: "cannot fully guarantee inter-layer consistency". Dataset: artplus/PrismLayersPro on Hugging Face. No code repository link was found.

## 6. Stable-Layers (2605.30257): no spatial editing
- What it is: Qwen-Image-Layered fine-tuned with Flow-GRPO RL, using a VLM as the reward model. Training uses LoRA r16 on all attention and FFN layers, G=16, KL 1e-3, LR 1e-5, 600 steps on 8 H200s.
- Rewards and results: one reward checks "background inpainting (layer 0 is a plausible scene completion)". The results say "RL fine-tuning can reorder layers without changing decomposition quality".
- Code: "Code release is under consideration"; the authors plan to release the LoRA adapters. Limitation: the reward model is a proprietary VLM accessed through a paid API.

## What we can borrow
(a) Qwen-Image-Layered on a T4:
- Do the spatial edit directly on the RGBA layers: affine warp, then alpha-over composite in depth order (DreamLayer Eq. 13 and PrismLayers both do this).
- Layer 0 is already an inpainted background (Stable-Layers' reward checks for this), so moving an object reveals a filled background without extra inpainting.
- For harmony and shadows, run one short, low-strength img2img or SDEdit pass on the composite. Paste the original layer pixels back inside its alpha, so only the surroundings and shadows change. ART's "freeze other layers, regenerate edited one" points the same way.
- Re-running Qwen-Image-Layered itself on a T4 to re-harmonise was not examined; no paper here does that.

(b) SDXL+LayerDiffuse relocate() and its identity drift:
- DreamLayer IRH (closest match): warp the layer's latent and alpha, composite in latent space, add noise only to about t=600/1000 (strength ~0.4), then denoise. Its attention modules are trained (CACA and LSSA), so on our side this stage can only be approximated.
- The main fix for drift is the LayerFusion blend applied at each step: a'_B = a_FG*mask_soft + a_B*(1-mask_soft). Here a_FG is the noised latent of the warped original layer (the forward-noise of x0 at timestep t) and mask_soft is its warped alpha, softened. Inside the object the latent is re-imposed from the original, which stops identity drift; outside it, shadows and contact can still be generated. This is the same idea as Blended Latent Diffusion; it is training-free and fits SDXL.
- Optionally, do the hard re-imposition only for t > ~0.5T and let the soft mask handle later steps, so edges can blend.
- Last step: alpha-composite the original RGBA layer pixels back over the decoded image, keeping only the regenerated shadow halo.
- Mask extraction: LayerFusion's self-attention sparsity x EOS cross-attention gives a soft mask with no extra model. We already get alpha from LayerDiffuse, so this mask is only needed for shadow regions.
