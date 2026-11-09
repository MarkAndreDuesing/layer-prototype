# Spatial-edit survey, part 3: object relocation and identity preservation (verified)

Source: arXiv HTML full text, Method sections and appendices, read 2026-11-09 with a summarising fetcher. Where a field says "not stated", the paper text I read does not give it.
GitHub check: MC-E/DragonDiffusion returned HTTP 200. The other repos returned 504 (GitHub throttling), so they are **[CODE NOT VERIFIED]**. The repo URL listed for each is the one printed in the paper.

## Per-method facts

| Method | Training-free | Inversion | Attention injection | Latent paste/blend | Revealed background | Shadows | Cost | Base | Code |
|---|---|---|---|---|---|---|---|---|---|
| DragonDiffusion 2307.02421 | Yes | DDIM | Decoder self-attention. K/V come from a memory bank recorded during inversion, at every step. For paste/appearance tasks the reference K/V are concatenated. No attention mask. | None. Guidance is an energy gradient on decoder blocks 2+3 (cosine similarity plus a global mean term), applied in the first n=30 of 50 steps. Object moving adds an inpaint term w_i=2.5. | Energy term pushes the vacated region away from the object. For shrinking, the vacant area is padded with 0. | Not stated | A100: 3.6 s prep + 15.9 s infer | SD1.5, 50 DDIM steps, CFG 5 | github.com/MC-E/DragonDiffusion (verified) |
| DiffEditor 2402.02583 | Sampler yes. The image-prompt encoder (QFormer + linear) is trained. | DDIM | Same memory bank. Image-prompt K/V are added with weight γ. Layers not stated. | Regional SDE: η=0.4 inside the edit mask and 0.2 outside, during τ_SDE (interval not given). Regional gradient guidance in the first n steps, every 2nd step. Time travel rolls back with DDIM inversion U=3 times. | Not stated | Not stated | A100: 3.5 s + 13.9 s | SD1.5 | Same repo as DragonDiffusion |
| Self-Guidance 2306.00986 | Yes | Not central | Guidance, not injection. Centroid, size and shape come from all 36 cross-attention maps (8/16/32 px). Appearance comes from the penultimate decoder activation × the final cross-attention map, with stop-grad on the attention. | None. Guidance applied on the first 3N/16 steps, alternate steps after that, none on the last N/32. Scale v=7500. Move weights: w0 0.5–2, w1 0.03–0.3, w2 0.5–5. | Regenerated | Not stated | Not quantified (needs backprop) | Imagen 1024 | None stated |
| TF-ICON 2307.12493 | Yes | DPM-Solver++ ODE inversion with an "exceptional prompt" (token ids = 7788, no positional embedding), used for inversion only | Injects self-attention **maps** only (V injection made results rigid). Composite map: reference self-attention on object patches, main-Q × reference-K cross map on transition patches. Applied t∈[0.4T, T], i.e. the early 60%. Layers not stated. | x_T = ref⊙M_seg + main⊙(1−M_user) + noise⊙transition. Background replaced at each step from the main reconstruction (τ_B=0). | Background is kept | Not stated | About 8 s on A100, 3 parallel ODEs, 20 steps, CFG 2.5 | SD | github.com/Shilin-LU/TF-ICON [NOT VERIFIED] |
| PrimeComposer 2403.05053 | Yes | DPM-Solver++. Object pixel-pasted with M_obj before inversion. | Correlation Diffuser: Q from the pasted composite latent, K from the object latent. Object-appearance map in the decoder only, plus a bg–object cross map. α=0.2, but the text says "initial" steps while the algorithm says t≤αT (inconsistent). RCA: object tokens get −∞ outside M_obj in cross-attention. | z = fg⊙M_fg + noised_bg⊙(1−M_fg) at every step. Transition (M_obj XOR M_fg) filled with noise. | Kept by blending | Not addressed | 16.2 s on A100 (vs 30.6 s for TF-ICON in the same table) | SD, DPM++ 20 steps | github.com/CodeGoat24/PrimeComposer [NOT VERIFIED] |
| GeoDiffuser 2404.14403 | Yes (zero-shot optimisation of latents and text embedding) | Direct inversion, null prompt | Shared attention in **all** UNet blocks up to step 45. Transform F is applied to the reference **queries**, then attended against the reference K/V. Output blended with F(M_obj). | Loss optimisation every other step for the first 32 steps (background, placement, distance-weighted removal, smoothness, amodal). Correction in the last 15 steps. | Removal loss pulls from nearby background (94% preferred over LaMa) | Implicit: shadows and reflections move with the attention | 30–45 s on RTX 3090 at 512 | SD | Project page only [NOT VERIFIED] |
| ObjectDrop 2403.18818 | **No.** Fine-tuned on 2.5k counterfactual pairs plus a 350k bootstrap set. | No | Not applicable | Object pasted as-is (identity kept by design). The model generates only shadows and reflections. | Learned | **Learned (main contribution)** | Heavy training | Internal SDXL-like / SD-Inpaint v1 | Not released |
| Magic Fixup 2403.13044 | **No.** Two SD1.4 UNets fine-tuned 120k steps on 5M video pairs. | Starts from a heavily noised coarse edit | Detail-extractor features before each of 11 self-attention blocks feed the synthesiser through extra cross-attention, every step | Coarse edit + mask concatenated as input | Learned | Learned implicitly | 5 s on A100, 50 DDIM steps | SD1.4 | Promised [NOT VERIFIED] |
| AnyDoor 2307.09481 | **No** (trained once) | No | DINOv2 257 tokens → linear → 1024-dim cross-attention context. Sobel high-frequency collage → ControlNet-style encoder → concatenated to decoder features. | Not applicable | Generated | Not addressed | Not given | SD2.1 | github.com/ali-vilab/AnyDoor [NOT VERIFIED] |
| RegionDrag 2407.18247 | Yes | DDPM inversion to t'=500/1000 (10 steps) | Mutual self-attention: K/V replaced with cached inversion K/V in the self-attention blocks, all timesteps. Layers not stated. | **Copy-paste of handle latents onto target regions at each step from t'=500 to t''=200.** Handle region mixed with noise (α=1) first. | Not stated | Not stated | **1.5 s on V100** | SD1.5, 20-step DDPM, 512 | Project page only [NOT VERIFIED] |
| SceneDiffusion "Move Anything" 2404.07178 (no "ObjMove" paper found) | Yes | None. The reference is forward-noised and used as an anchor with weight 1e4. | None | Layered feature maps. N=8 random layouts denoised and least-squares averaged, for τ=25 of 50 steps. | Background layer optimised jointly. Fill not described. | Not stated | 17.3 s optimisation + 0.8 s per layout | 512 SD-type, DDIM 50 | Not stated |
| DesignEdit 2403.14487 | Yes | Proximal-guidance inversion | **Key-masking self-attention** (mask on K only; masking Q blurs, masking V distorts) in all 70 SDXL self-attention blocks, first K=40 of 50 steps. Context reset from the inversion latent each step. | Layers pasted onto a canvas latent with a Move op at step T−K, then harmonised for the last T−K steps | Layer-0 background inpainted with key-masking | Not stated | Not reported | **SDXL 1.0, 1024** | Project page only [NOT VERIFIED] |
| Collage Diffusion 2303.00262 | Partly (per-layer Textual Inversion token) | None (SDEdit from the noised collage) | Alpha-mask positive/negative bias added to QKᵀ before softmax for the layer tokens | **Per-layer noise level t_i** (0.5–0.8), blurred noise map, Blended-Diffusion mask | Regenerated | Not stated | Not given | SD2.1, Euler-a, 50 steps | None found |

## Why the object gets redrawn in our relocate()
Our current pipeline is img2img at strength 0.3, which is uniform SDEdit. Every pixel, including the object, is re-denoised, and nothing ties the object back to its source. Every fidelity-preserving training-free method adds one or both of two things:
- **Latent pinning:** RegionDrag, PrimeComposer, TF-ICON, Collage
- **Self-attention K/V sharing from a source trajectory:** DragonDiffusion, RegionDrag, DesignEdit, GeoDiffuser, TF-ICON

Gradient-guidance methods are too slow for a T4. Trained methods (ObjectDrop, Magic Fixup, AnyDoor) are out of scope.

## Ranked implementation plan (SDXL + LayerDiffuse bg2ble, 512 px, Euler, 20 steps, T4)

1. **Region latent pinning with a release step.** Combines RegionDrag copy-paste, Collage per-layer noise and the PrimeComposer blend. Near-zero cost.
   1. Re-noise the warped-object latent to the current sigma and paste it under an eroded object mask for the first ~60% of the img2img steps.
   2. Leave the edge ring, shadow ring and revealed hole free.
2. **Mutual self-attention K/V injection (decoder/up_blocks only).** Combines the DragonDiffusion memory bank, RegionDrag mutual attention and DesignEdit for SDXL. Costs one extra UNet batch half.
   1. Build the reference branch by forward-noising the warped composite (as RegionDrag does; at strength 0.3 no full inversion is needed).
   2. Object-box queries attend to [own K/V ; reference K/V]. Background queries see only their own K/V.
   3. Enable it only while pinning is active.
3. If step 2 is too rigid, inject self-attention maps only, not V (TF-ICON).
4. **Revealed hole:** DesignEdit key-masking. Set keys at the old object location to −inf in all self-attention layers for the first 80% of steps, or run a separate inpaint pass first.
5. **Shadows:** no training-free paper solves them explicitly (GeoDiffuser: they move with attention). Keep a dilated, unpinned ring below the object.

### Pseudocode 1: latent pinning (diffusers step-end callback)
```python
z_obj0 = vae.encode(warped_rgb).latent_dist.mean * vae.config.scaling_factor
m_pin  = F.interpolate(erode(alpha_warped, 3), size=z_obj0.shape[-2:], mode="bilinear")
noise  = torch.randn_like(z_obj0)          # fixed noise, reused every step
release_frac = 0.4
def cb(pipe, i, t, kw):
    ts = pipe.scheduler.timesteps; n = len(ts)
    if i < int(n * (1 - release_frac)) and i + 1 < n:
        ref = pipe.scheduler.add_noise(z_obj0, noise, ts[i+1:i+2])  # Euler: x0 + sigma*noise
        kw["latents"] = m_pin * ref + (1 - m_pin) * kw["latents"]
    return kw
out = pipe(..., strength=0.3, num_inference_steps=20,
           callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
```

### Pseudocode 2: mutual self-attention processor (SDXL up_blocks attn1)
```python
class MutualSelfAttn:
    def __init__(self, store): self.s = store
    def __call__(self, attn, hs, encoder_hidden_states=None, attention_mask=None, **kw):
        if encoder_hidden_states is not None or not self.s.active:
            return self.s.base(attn, hs, encoder_hidden_states, attention_mask, **kw)
        # batch layout per CFG half: [ref, edit]
        q, k, v = attn.to_q(hs), attn.to_k(hs), attn.to_v(hs)
        r, e = slice(0, None, 2), slice(1, None, 2)
        N = hs.shape[1]
        obj = self.s.obj_mask(N)                     # (1,N) new box, 1 = object query
        hole = self.s.hole_mask(N)                   # (1,N) old location
        k_e = torch.cat([k[e], k[r]], 1); v_e = torch.cat([v[e], v[r]], 1)
        bias = torch.zeros(1, N, 2 * N, device=hs.device, dtype=hs.dtype)
        bias[:, :, N:] = torch.where(obj[..., None] > .5, 0., -1e4)   # only obj queries see ref
        bias[:, :, :N] += torch.where(hole[:, None, :] > .5, -1e4, 0.) * self.s.keymask_on
        H = attn.heads
        sp = lambda x: x.view(x.shape[0], -1, H, x.shape[-1] // H).transpose(1, 2)
        o_r = F.scaled_dot_product_attention(sp(q[r]), sp(k[r]), sp(v[r]))
        o_e = F.scaled_dot_product_attention(sp(q[e]), sp(k_e), sp(v_e), attn_mask=bias[:, None])
        o = torch.stack([o_r, o_e], 1).flatten(0, 1)            # re-interleave
        o = o.transpose(1, 2).reshape(hs.shape[0], N, -1)
        return attn.to_out[1](attn.to_out[0](o))
procs = {n: MutualSelfAttn(store) if n.startswith("up_blocks") and "attn1" in n else p
         for n, p in unet.attn_processors.items()}
unet.set_attn_processor(procs)
# Each step: z_ref_t = scheduler.add_noise(z_warped_composite, same_noise, t); run the UNet on interleaved [z_ref_t, z_edit].
# store.active = i < n*(1-release_frac); store.base = the existing processor (wrap the LayerDiffuse one if present).
```
Caveats:
- The LayerDiffuse bg2ble conditioning must be identical for both halves.
- Cross-attention is untouched.
- The reference branch output is discarded.
- Untested; written from the paper descriptions only.
