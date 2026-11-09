# Spatial edit + re-harmonise: mechanics from LayerFusion, DreamLayer, PSDiffusion

Date: 2026-11-09. Sources were read through the arXiv HTML pages using a summarising fetch tool. Exact layer names and numbers are as reported by that tool. Check them against the PDF before relying on them.

## 1. What each paper actually does

### LayerFusion (arXiv 2412.04460, https://arxiv.org/html/2412.04460, project https://layerfusion.github.io)
- **Training-free:** yes. Sec 3.1 calls it "a training-free solution".
- **Weights:** LayerDiffuse is *not* enough on its own.
  - The FG (RGBA) branch uses LayerDiffuse SDXL (latent transparency offset plus transparent VAE).
  - The BG and blend branches use a plain SDXL. The paper reports RealVisXL_V4.0 "unless otherwise stated".
  - All three branches denoise together in one batch.
- **Priors from the FG branch:**
  - Structure prior. It comes from the self-attention map of the last self-attn layer, `up.1.attns.2.block.1` (App. A.3).
    1. Sparsity per token: s_i = 1 / Σ_j m_ij².
    2. Invert to favour dense tokens: s' = 1 − normalize(s).
  - Content-confidence prior, c. It is the cross-attention column for the `<EOS>` token of the FG prompt, averaged over heads (Alg. 2).
- **Masks:**
  - mask_soft = normalize(s' · c)
  - mask_hard = sigmoid(d · (mask_soft − 0.5)), with d = 10.
  - Masks are taken at t = 0.8T (Fig. 3 caption).
- **Attention blending (Eq. 1, 2):**
  - `a'_Blended = a_FG * mask_soft + a_Blended * (1 - mask_soft)`
  - `a'_FG = a'_Blended * mask_hard + a_FG * (1 - mask_hard)`
  - On top of this, attention is shared between the blend and BG hidden states in both self- and cross-attention (Alg. 3).
  - **Uncertain:** the paper does not clearly say which layers or timestep range the blending covers. Read Alg. 3 in the PDF before implementing.
- **Spatial editing:** the paper only says layered outputs allow editing "in a straight-forward manner". It gives **no move/resize algorithm**.
- **Code:** no GitHub repository found (web search on 2026-11-09, and the project page/paper link none).

### DreamLayer (arXiv 2503.12838, https://arxiv.org/html/2503.12838, project https://ll3rd.github.io/DreamLayer/)
- **Training:** *trained*.
  - Base model is SD1.5, fine-tuned with Custom Diffusion on the attention K/V layers.
  - Lr 2e-6, 60k steps, on 2×A100 for about 4 days.
  - Its attention modules are learned: CACA at resolution 16, and LSSA in the up-blocks, which concatenates all layer latents into a joint K/V.
  - Image-to-layer decomposition is training-free (DDIM inversion plus mask attention).
- **Information Retained Harmonization (IRH), the part we can borrow:**
  - 50-step DDIM, T = 1000.
  - Layers are generated separately down to T_H = 600 (latents kept between 600 and 400).
  - The latents are then alpha-composited at the latent level (Eq. 12):
    `z_m = z_bg · Π(1−α_i) + Σ_i z_i · α_i · Π_{f in front}(1−α_f)`
  - Denoising continues from the composite. This is where shadows and contact appear. Without IRH, objects "appear to float".
- **Move/flip/resize (Eq. 13):** apply op(·) to the FG latents *and* alphas before the Eq. 12 fusion, then re-denoise. The paper's chair example shows the floor shadow following the new position.
- **Code:** none linked.

### PSDiffusion (arXiv 2505.11468, https://arxiv.org/html/2505.11468)
- **Partial joint self-attention:**
  - Q = W_Q z_i
  - K = W_K [K_GL, K_L]
  - V = W_V [V_GL, V_L]
  - with K_GL = K_G·Mask_i^g and V_GL = V_G·Mask_i^g.
  - So each layer attends to its own tokens plus the global image's tokens, restricted to its own region.
  - The mask is the global cross-attention map, filtered, blurred and binarised. Thresholds are not given.
  - It runs in **all** self-attention layers.
- **Training:** *trained*. SDXL LoRA rank 256, about 60 A800-hours, Inter-Layer dataset of 30K samples. No code is linked.
- **What we can borrow:** only the idea of masked joint K/V. Without the trained K_GL/V_GL projections, the mechanism is a heuristic.

## 2. Takeaway
- None of the three gives a ready-made training-free *spatial edit* algorithm for SDXL LayerDiffuse.
- The closest recipe is DreamLayer's IRH: transform the FG, composite at latent level at an intermediate t, then re-denoise.
- That recipe maps directly onto our existing `bgble2fg` / `bg2ble` modes plus SDEdit-style partial re-noising.
- Optionally, add LayerFusion-style attention masking to keep FG identity stable.

## 3. Design: training-free "move/resize + re-harmonise"

**Inputs**
- Background image B (RGB)
- Layer L (RGBA)
- Old box b0 and new box b1
- Prompts: FG, BG and blend
- Strength s, default 0.4–0.55, matching DreamLayer's T_H = 600/1000 → 0.6. Tune it.

**Steps**
1. **Pixel-space transform.** Warp L's RGB and alpha from b0 to b1 (affine: scale + translate, optional flip). Paste over B → naive composite C0, alpha A1.
   - Pixel space is preferred to shifting latents, because SDXL latents are /8. Resizing a latent blurs it. A pure shift by a multiple of 8 px could be done in latent space instead.
2. **Shadow cleanup at the old location.** If the background B was itself a blend containing the old object, inpaint region b0 \ b1 first (SDXL inpaint at strength ~0.9). In the normal case B is a clean `bg` layer, so skip this.
3. **Encode** C0 with the SDXL VAE → z0.
4. **Partial re-noise.**
   - Choose t_s = s·T. Set z_ts = scheduler.add_noise(z0, ε, t_s) (SDEdit).
   - Equivalent DreamLayer variant: re-noise the B and L latents separately to t_s, apply Eq. 12 with A1 downsampled to 128×128, and share one noise ε.
5. **Denoise with the `bg2ble` blend pipeline** from t_s to 0, so the blend is conditioned on B. Keep the FG identity with:
   - (a) Latent preservation: in the first k% of the remaining steps, at each step set `z = M_core·noised(z0_fg) + (1−M_core)·z`.
     - M_core = eroded A1, so object interiors are pinned.
     - The ring and the area outside it stay free, so the model can paint shadows and contact.
   - (b) Optional LayerFusion-style attention hook: in the up-block self-attention, give the blend branch's attention probabilities a soft mask (A1 blurred), mixing toward a reference pass on the naive composite. This is Eq. 1 with "a_FG" taken from a parallel reference run on z0 noised. It doubles the cost.
6. **Re-extract the FG.** Run `bgble2fg` with (B, new blend) → RGBA. This gives a harmonised FG layer with the relit object and its contact shadow in alpha.
   - Optionally preserve the old alpha shape by intersecting with dilated A1.
7. **Output:** the new blend plus the new RGBA layer. B is unchanged.

**Pseudocode (diffusers)**
```python
class MaskedMixProc(AttnProcessor2_0):  # registered on up_blocks self-attn only
    def __init__(self, store, mask_tokens, mode): ...
    def __call__(self, attn, hs, enc=None, attention_mask=None, **kw):
        q,k,v = attn.to_q(hs), attn.to_k(hs if enc is None else enc), attn.to_v(...)
        # reshape to heads; compute probs p = softmax(qk^T/sqrt d)
        if self.mode=="record": self.store[attn.name]=p.detach()          # reference pass
        elif self.mode=="mix" and attn.name in self.store:
            m = self.mask_tokens[p.shape[-2]]                              # (N,1) A1 at this res
            p = m*self.store[attn.name] + (1-m)*p                          # LayerFusion Eq.1 analogue
        return attn.to_out[0](rearrange(p@v)) 

def move_and_reharmonise(bg, layer_rgba, b0, b1, s=0.5, steps=20, k=0.4):
    comp, A1 = paste_affine(bg, layer_rgba, b0, b1)
    z0 = vae.encode(comp).latent_dist.mean * vae.config.scaling_factor
    sched.set_timesteps(steps); ts = sched.timesteps[int((1-s)*steps):]
    eps = randn_like(z0); z = sched.add_noise(z0, eps, ts[:1])
    Mcore = downsample(erode(A1, 8px), 1/8)
    for i,t in enumerate(ts):
        z = bg2ble_step(z, t, cond_bg=bg)           # existing LayerDiffuse bg->blend UNet step
        if i < k*len(ts): z = Mcore*sched.add_noise(z0, eps, t_next(t)) + (1-Mcore)*z
    blend = vae.decode(z)
    fg_rgba = bgble2fg(bg, blend, prompt_fg)         # existing mode, full 20 steps
    return blend, fg_rgba
```

**Cost on a T4 (16 GB, fp16).** These are estimates, not measured.
- SDXL at 1024² runs at about 1.6–2.2 s/it with CFG. The bg-conditioned LayerDiffuse UNet adds a little.
- Re-harmonise at s = 0.5 is 10 steps ≈ 20 s. bgble2fg at 20 steps ≈ 40 s. VAE and transparent-VAE decode add ≈ 5 s. Total ≈ **65–75 s**.
- With the attention reference pass, the re-harmonise part costs about 2× (+20 s).
- Needs `enable_model_cpu_offload` / VAE tiling to fit. A custom processor that materialises probabilities rules out SDPA in those layers: more memory, slower. Limit it to up_blocks.1 (32² tokens at 1024).

## 4. Uncertainties
- LayerFusion's exact layer/timestep schedule (Alg. 3) and its sampler/steps were not extracted. The summary may be lossy, so verify in the PDF.
- A pure-SDXL BG/blend branch (as in LayerFusion) versus our bg-conditioned `bg2ble` UNet: we use the latter. That is a deviation, but it is the more natural fit for our pipeline.
- Strength s has a trade-off. Low s keeps the object but gives weak shadows and stale shadows at the old position. High s gives correct lighting but drifts the identity. DreamLayer's 0.6 equivalent is a starting point only, and it is for SD1.5 with trained modules.
- Occlusion with *other* objects needs a z-order of all layers composited per Eq. 12. Single-layer moves are well-posed. Multi-object scenes will still "stack" without a trained joint attention (PSDiffusion and DreamLayer both had to train for that).
- Resizing up will soften the object. Low s will not fix that, so consider re-generating the FG at the new size.
- None of the three papers has public code (as far as the 2026-11-09 search shows), so all of the above is reimplementation.

## Sources
- LayerFusion: https://arxiv.org/html/2412.04460 , https://layerfusion.github.io
- DreamLayer: https://arxiv.org/html/2503.12838 , https://ll3rd.github.io/DreamLayer/
- PSDiffusion: https://arxiv.org/html/2505.11468
