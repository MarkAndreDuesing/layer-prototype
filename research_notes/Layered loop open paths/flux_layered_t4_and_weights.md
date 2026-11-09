# FLUX-based layered models on fp16-only GPUs (T4) and weight provenance (Nov 2026)

Second pass, Nov 9 2026. Items marked [unverified] have no fetched source. Items marked [coordinator] were supplied by the coordinator from an HF LFS hash comparison and were not re-checked by me.

## FLUX fp16 overflow: where it occurs, guards in diffusers/ComfyUI, and whether our fp16_mixed is enough

### Takeaway
Both diffusers and ComfyUI handle FLUX in fp16 with one thing only: a clamp to ±65504 after the residual add. Diffusers clips image hidden states in the single block and the encoder (text) stream in the double block; ComfyUI does the same with `nan_to_num` and adds one more on the final img. Neither upcasts modulation, norm_out or the embedders. The only fp32 or fp64 work in either is RoPE frequency construction and the sinusoidal timestep embedding. So plain FLUX output is usable with nothing more than "clamp inf back to 65504", and our fp16_mixed design (fp32 residual, norms and gates, plus absmax-scaled linears) is strictly stronger.

That makes our pure-noise result unlikely to be a base-FLUX overflow. RevealLayer's own code (fp32 `layer_pe`, -1e9 mask in the hidden dtype, an unguarded refiner `delta`, a bf16-only pipeline) or the GGUF/LoRA path is the more likely cause.

### Cited Findings
**diffusers (`transformer_flux.py`, main):**
- FluxSingleTransformerBlock: `if hidden_states.dtype == torch.float16: hidden_states = hidden_states.clip(-65504, 65504)`, after the residual add. — [diffusers transformer_flux.py](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/models/transformers/transformer_flux.py)
- FluxTransformerBlock (double): the same clip, but only on `encoder_hidden_states` (the text stream). — [same](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/models/transformers/transformer_flux.py)
- norm_out (AdaLayerNormContinuous) has no fp16 check or upcast at the call site. Timestep and guidance are cast to the hidden dtype and multiplied by 1000. FluxPosEmbed computes RoPE with `ids.float()` and float64 freqs where the device supports them. — [same](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/models/transformers/transformer_flux.py)
- The clamp came from PR #9097, which fixed "Flux inference with torch.half outputs NaN values". — [diffusers PR 9097](https://github.com/huggingface/diffusers/commit/f771be1d7b403ef2de2d6250976aa9185b6be060); [issue/PR 9097](https://redirect.github.com/huggingface/diffusers/issues/9097)
- A Sana analogue (not FLUX) shows the semantics: overflow in the attention output and hidden_states reaches about 5e5 to 1.3e6. The maintainer says the model was trained with the clamp and "never saw value out of the scope of (-65504, 65504)", so clamping is part of the model's expected behaviour, not just damage control. — [diffusers issue 10336](https://github.com/huggingface/diffusers/issues/10336)

**ComfyUI (`comfy/ldm/flux/`):**
- layers.py DoubleStreamBlock: `txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)` when fp16; the img stream is not guarded. SingleStreamBlock: the same on `x`. — [ComfyUI layers.py](https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/ldm/flux/layers.py)
- layers.py Modulation (`self.lin(silu(vec))`), RMSNorm/QKNorm (outputs cast `.to(v)`) and LastLayer (`modulated_norm`) have no fp32 casts. `timestep_embedding` builds freqs in float32 and casts back to `t`'s dtype. — [same](https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/ldm/flux/layers.py)
- model.py: time_in and guidance_in are fed `timestep_embedding(...).to(img.dtype)`. img_ids and txt_ids are float32. A final `img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)` runs when fp16, and there is no cast around final_layer. — [ComfyUI model.py](https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/ldm/flux/model.py)
- math.py `rope()` builds the scale in float64 (falling back to CPU if fp64 is unsupported) and returns float32. `attention()` does no casts and dispatches to `optimized_attention`. — [ComfyUI math.py](https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/ldm/flux/math.py)

**Community fp16 FLUX builds:**
- Disty0 FLUX.1-dev qint4 "with FP16 calculations". — [Disty0](https://huggingface.co/Disty0/FLUX.1-dev-qint4-fp16_compute)
- An NF4 package made for T4 with fp16 compute. — [sayakpaul/flux.1-dev-nf4-pkg](https://huggingface.co/sayakpaul/flux.1-dev-nf4-pkg)

### Inferences
- Our fp16_mixed already keeps the residual, norms and gates in fp32, which is beyond either reference implementation. Optional extras for safety [unverified as necessary]: run norm_out/AdaLayerNormContinuous and proj_out in fp32, keep the time, guidance and text embedders' linear inputs in fp32 (cheap), and keep RoPE in fp32 (both references already do).
- Cheapest diagnostic: run our loader with plain FLUX (RevealLayer LoRA, layer_pe and refiner off) using only the diffusers clip. If that gives a clean image, base FLUX in GGUF+fp16 is fine and the fault is in RevealLayer-specific code (see the RevealLayer section).

### Gaps
- I did not read diffusers' AdaLayerNormContinuous or the attention processor source, and I did not check ComfyUI's `--force-fp32` / `--force-upcast-attention` effect on FLUX.
- No GitHub issue found that ties FLUX black images specifically to T4/P100 fp16. The closest, with no fix, is [city96 schnell GGUF disc. 4](https://huggingface.co/city96/FLUX.1-schnell-gguf/discussions/4).

## Real reports of FLUX dev GGUF on T4 / P100 / free Colab

### Takeaway
FLUX dev runs on free-Colab T4 in ComfyUI with fp8 or GGUF Q4/Q5 (stated to need about 15 GB VRAM), but I found no published T4 s/it figure and no reports of fp16 NaNs on T4 in ComfyUI. That absence suggests ComfyUI's clamp is enough there [inference].

### Cited Findings
- The sayantan-2/comfyui_colab_Flux notebook targets the free-tier Tesla T4 (about 15 GB VRAM). FP8 is the recommended default (works without LoRA). GGUF Q5 and Q4 (ComfyUI-GGUF) were tested and work with LoRAs. No speeds given. — [comfyui_colab_Flux](https://github.com/sayantan-2/comfyui_colab_Flux)
- Free Colab: about 12 GiB system RAM with a T4. RAM crashes are a common failure. — [HighCWu FLUX 4bit disc. 9](https://huggingface.co/HighCWu/FLUX.1-dev-4bit/discussions/9)
- Closest speed analogue (not T4): RTX 3060 Ti 8 GB, Q4_K_S, `--lowvram`, about 90–150 s per 1024², 20 steps. RTX 3060 12 GB: about 50–90 s. (Search-summary figures from a secondary guide.) — [localaimaster low-VRAM guide](https://localaimaster.com/blog/run-flux-on-low-vram-gpu)
- stable-diffusion.cpp FLUX GGUF runs on 4–6 GB GPUs. — [leejet/FLUX.1-dev-gguf](https://huggingface.co/leejet/FLUX.1-dev-gguf)
- Forge supports GGUF FLUX. — [Forge disc. 1050](https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/1050)
- Our own data point [coordinator]: RevealLayer at 512 px, 12 steps took 69 s on a Kaggle T4.

### Gaps
- No Kaggle or P100 FLUX benchmarks found, and no T4 s/it numbers.

## CLD weights vs ART / ART+ (PrismLayers) and the 24yearsold mirror

### Takeaway
[coordinator, by SHA-256] CLD's `pre_trained_LoRA` is byte-identical to the mirror's `base/pytorch_lora_weights.safetensors`, which is the ART base LoRA. That means the ART base LoRA is officially on HF in thuteam/CLD under Apache-2.0. CLD's `prism_ft_LoRA` differs from the mirror's ART+ 100k LoRA.

The CLD paper says CLD fine-tunes FLUX.1-dev on the PrismLayersPro training split (LoRA rank 64, 25K iters, 1024²), and the config loads it as `artplus_lora_dir`. So `prism_ft_LoRA` is either the official ART+ PrismLayersPro (20K) checkpoint or CLD's own fine-tune; the paper does not settle which. CLD needs an input image, so it is image-to-layers, not text-to-layers.

### Cited Findings
- CLD HF card: Apache-2.0. Files: `decouple_LoRA/{adapter/pytorch_lora_weights.safetensors, layer_pe.pth, transformer/pytorch_lora_weights.safetensors}`, `pre_trained_LoRA/…`, `prism_ft_LoRA/…`, `trans_vae/0008000.pt`. Config: `pretrained_lora_dir`, `artplus_lora_dir`, `transp_vae_path`. — [thuteam/CLD README](https://huggingface.co/thuteam/CLD/blob/main/README.md)
- CLD paper: the backbone is FLUX.1[dev], LoRA rank 64, Prodigy lr 1, batch 4, 25K iters at 1024×1024 on the PrismLayersPro training split. It adapts ART's multi-layer transparent autoencoder as the RGBA decoder. ART+ is not mentioned. Conditioning is bounding boxes plus the original image plus text, and the image is central. No VRAM or time reported. — [arXiv 2511.16249v2](https://arxiv.org/html/2511.16249v2)
- The code is at GitHub monkek123King/CLD; github.com/thuteam/CLD returns 404. — [HF CLD card](https://huggingface.co/thuteam/CLD)
- PrismLayers README Drive folders: `base_ckpt` (ART), `artplus_PrismLayersPro_ckpt`, `artplus_PrismLayersPro100k_ckpt` (both ART+ at 1024 px), `transp_vae_ckpt`, `pre_fuse_lora_dir`, `classifier_ckpt`, `TIPS_ckpt`. MIT licence. — [redredsheep/PrismLayers](https://github.com/redredsheep/PrismLayers)
- 24yearsold/PrismLayers_ckpt: 5.56 GB, one uploader, 86-byte README, no licence. Contents: `artplus_PrismLayersPro100k_ckpt/`, `base/` (with layer_pe.safetensors), `pre_fuse_lora/`, `trans_vae/`, `TIPS_ckpt.safetensors` (3.94 GB), `classifier_head_epoch_9.safetensors`. It lacks the non-100k ART+ variant. — [mirror tree](https://huggingface.co/24yearsold/PrismLayers_ckpt/tree/main)

### Inferences
- The CLD paper's training recipe (PrismLayersPro split, 1024², LoRA) matches how the PrismLayers README describes the ART+ PrismLayersPro checkpoint, while the 100k variant is excluded by hash. The likeliest reading is that `prism_ft_LoRA` is the ART+ `artplus_PrismLayersPro_ckpt` reused by CLD as its starting point, with the CLD-specific weights in `decouple_LoRA`. This is [unverified]. To confirm, hash the file in Drive folder 1d-lIRYVyhKqpdZO50h4p4qZz9mXU3w4H.
- Licences: CLD weights are Apache-2.0 (HF metadata). PrismLayers code is MIT; the Drive weights have no separate licence I could find. The mirror is unlicensed. All are fine for personal non-commercial use. FLUX.1-dev's own non-commercial licence still applies to the base.
- Note: CLD's paper has rank 64 and 25K iters, while ART+ in the PrismLayers paper has 20K iters at 512² then 10K at 1024². These differing recipes argue mildly that CLD trained its own LoRA [inference].

### Gaps
- I could not download from Drive to hash the non-100k ART+ checkpoint.

## ART / ART+ (PrismLayers): input, text-only mode, base, low-VRAM

### Takeaway
ART+ is FLUX.1-dev based and does text+layout to layers with no input image. Layouts come from a reference image or an LLM. No GGUF or low-VRAM runs were found.

### Cited Findings
- ART was trained from FLUX.1-[dev] on the 200K PrismLayers set. ART+ was fine-tuned from it on PrismLayersPro: 20K iters at 512² then 10K at 1024², batch 4, Prodigy lr 1.0. It denoises a concatenation of global and regional visual tokens. Layouts come from a reference image or an LLM (unnamed), and LLM layouts lag human ones. No time or VRAM given. Pre-fuse LoRA and the 100k checkpoint are not described in the paper. — [PrismLayers paper, arXiv 2505.22523](https://ar5iv.labs.arxiv.org/html/2505.22523)
- ART+ beats ART in about 57–60% of user-study comparisons. — [arXiv 2505.22523 abs](https://arxiv.org/abs/2505.22523v1)
- Input JSON: `whole_image` {caption "This is a [STYLE] style image. …", height, width} plus a `layout` list of boxes, with the first two boxes full canvas. Run via `multi_layer_gen/demo.py`. There is an undocumented `layout_planner/` folder. — [PrismLayers README](https://github.com/redredsheep/PrismLayers)

### Inferences
- ART+ with Ollama-generated boxes fits a text-only pipeline. On T4 it would need the same fp16 treatment, and 1024² with many regional tokens is heavy. Starting at 512² is reasonable because ART's first training stage was 512² [inference].

### Gaps
- ART's custom attention (anonymous region masking) code was not read for fp16 risks.

## LayerDiffuse-Flux (RedAIGC)

### Takeaway
It is a LoRA plus TransparentVAE on FLUX.1-dev, licensed AFL-3.0 per the HF card, with a ComfyUI port (leeguandong). There are no formal quality reports; it has 14 likes and 1 HF discussion.

### Cited Findings
- HF card: licence `afl-3.0`, card text only "weights of Flux-version-LayerDiffuse", 1 discussion, 14 likes, used by 2 Spaces. — [RedAIGC/Flux-version-LayerDiffuse HF](https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse)
- GitHub: `layerlora.safetensors` plus `TransparentVAE.pth`, `demo_t2i.py` and `demo_i2i.py`, PyTorch 2.3 / CUDA 12.1, 11 open issues. — [GitHub](https://github.com/RedAIGC/Flux-version-LayerDiffuse)
- ComfyUI_FluxLayerDiffuse nodes: T2I, I2I, model loader. — [RunComfy](https://www.runcomfy.com/comfyui-nodes/ComfyUI_FluxLayerDiffuse)
- A vendor calls LayerDiffuse "outdated" and pushes AlphaVAE (vendor claim). — [comfyai.run](https://comfyai.run/custom_node/ComfyUI-AlphaVAE)

### Gaps
- I did not read the GitHub issue contents, so there are no independent quality reports.

## RevealLayer (360CVGroup): issues, code fp16 risks, ComfyUI port

### Takeaway
The official pipeline is bf16-only (no fp16 path, no offload, 1024 px, 30 steps). Its custom code adds no fp16 guards and contains several fp16 hazards. The new ComfyUI-TBG-RevealLayer port officially allows FP16 FLUX.1-dev but says quantized (GGUF) transformers are unsupported.

### Cited Findings
- infer.py: transformer and VAE are `torch.bfloat16`; the transparent decoder dtype is unset; no offload; 30 steps; resolution hardcoded to 1024; `--max_layer` 12. `_load_refiner` loads with `strict=False`, prints and skips on errors, and does not check missing keys. — [infer.py](https://raw.githubusercontent.com/360CVGroup/RevealLayer/main/infer.py)
- custom_model_mmdit.py hazards:
  - (1) `build_additive_spatial_roi_mask_v2` fills `torch.full(..., -1e9, dtype=hidden_states.dtype)`, which becomes -inf in fp16. Rows are not fully masked, so it is NaN-safe only while that holds.
  - (2) `layer_pe` is an fp32 Parameter that is only moved to the device (`.to(device)`), so `hidden_states + layer_pe` silently promotes to fp32 or mismatches.
  - (3) Timestep and guidance are cast to the hidden dtype ×1000.
  - (4) `hidden_states_part2 = hidden_states[:, l0_len:] + delta` (refiner output) is unguarded.
  - (5) A `Clamp` module is defined but never used.
  - (6) It reuses diffusers' FluxSingleTransformerBlock (so it inherits the diffusers clip only if the installed bundled diffusers has it) and adds no clamp itself.
  - — [custom_model_mmdit.py](https://raw.githubusercontent.com/360CVGroup/RevealLayer/main/models/custom_model_mmdit.py)
- The repo bundles its own copy of `diffusers/` (vendored source tree). — [repo tree](https://api.github.com/repos/360CVGroup/RevealLayer/git/trees/main?recursive=1)
- adapter.py HybridRefiner: 4 FluxRefinerBlocks (dim 384, 8 heads, head_dim 48). LayerNorm without affine, eps 1e-6, no QK-norm. Plain SDPA without mask. adaLN modulation `x*(1+scale)+shift` and gated residuals all in the input dtype (fp16 if fed fp16). Only RoPE is computed in float32. compute_dtype falls back to bf16. — [adapter.py](https://raw.githubusercontent.com/360CVGroup/RevealLayer/main/models/adapter.py)
- custom_pipeline.py: no VAE upcast on encode/decode (a code comment mentions "odd purple output" and chunked decode of 16 latents). CFG is combined in the latent dtype. Position ids are in the model dtype (fp16 integers are exact only to 2048). `overlap_counter` is fp32. — [custom_pipeline.py](https://raw.githubusercontent.com/360CVGroup/RevealLayer/main/models/custom_pipeline.py)
- Issues: #3 (layer_pe 12 vs Refiner 32). The maintainer says this is expected because the Refiner has its own `max_layers`=32, so both load at the default 12. — [issue #3](https://github.com/360CVGroup/RevealLayer/issues/3). #5 Table 3 metrics not reproducible; #7 links the TBG port. None is about VRAM or fp16. — [issues](https://github.com/360CVGroup/RevealLayer/issues)
- Licence is Apache-2.0 (code). Weights are at HF qihoo360/RevealLayer; the datasets are research-use. — [RevealLayer README](https://github.com/360CVGroup/RevealLayer)
- ComfyUI-TBG-RevealLayer (Ltamann): "Use standard guidance-embedded FLUX.1-dev in FP16, BF16, or FP32". FLUX.2 and quantized transformers are not supported. It auto-downloads 4 files (about 697 MB) from qihoo360/RevealLayer and applies the LoRA internally at 1.0. Optional SAM3 grounding. No VRAM figures or fp16 overflow notes. — [ComfyUI-TBG-RevealLayer](https://github.com/Ltamann/ComfyUI-TBG-RevealLayer)

### Inferences
- Likely causes of our pure noise, in order to check:
  - (a) Our fp16_mixed patch may not cover the bundled `diffusers/` copy or the HybridRefiner's own SDPA and LayerNorm path. Make sure the patch targets the vendored module classes.
  - (b) `layer_pe` dtype promotion interacting with GGUF dequant.
  - (c) The refiner `delta` overflowing.
  - (d) The -inf mask combined with our V-scaled custom attention: if the custom attention computes softmax manually in fp16 with -inf plus large logits, NaNs follow. Clamp the mask to -1e4 or build it in fp32.
- The TBG port runs inside ComfyUI, so FLUX blocks get ComfyUI's nan_to_num. It "officially" accepts FP16 FLUX, which makes it a cheap test path on T4. Its no-quantized-transformer rule means using fp16 full weights (about 24 GB, which needs CPU offload across 2×T4) or trying GGUF anyway [unverified].

### Gaps
- I did not read the TBG port's source for dtype casts.

## Newer 2026 FLUX.2 / Kontext / Qwen ports

### Takeaway
I found no sourced FLUX.2, Kontext or Qwen port of RevealLayer, CLD, ART or LayerDiffuse-Flux. The TBG port explicitly excludes FLUX.2.

### Cited Findings
- The TBG RevealLayer port does not support FLUX.2. — [ComfyUI-TBG-RevealLayer](https://github.com/Ltamann/ComfyUI-TBG-RevealLayer)
- A May 2026 paper studies synthetic layered data for decomposition; its base model was not checked. — [arXiv 2605.15167](https://arxiv.org/pdf/2605.15167)

### Gaps
- Qwen-Image-Layered (from memory, Dec 2025) [unverified]. It is the most T4-friendly candidate, given that our fp16_mixed already works on Qwen-Image, and it should be researched next.
