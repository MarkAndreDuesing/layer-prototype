# Heavy layered models: reduced-spec Kaggle 2x T4 plans

Target: Kaggle free, 2x T4 (16 GB each, fp16 only, no bf16), ~30 GB RAM, ~70 GB disk.
Researched 2026-11-09. [U] = unverified (not read in code/page, inferred or from memory).
GitHub HTML pages timed out (504); READMEs read via raw.githubusercontent.com.

---

## 0. Common: FLUX.1-dev on T4

Facts:
- city96/FLUX.1-dev-gguf: no gating notice on page; licence `flux-1-dev-non-commercial-license` ("all the same restrictions... still apply"). Sizes: Q4_K_S 6.81 GB, Q5_K_S 8.29 GB, Q6_K 9.86 GB, Q8_0 12.7 GB, F16 23.8 GB. Page only documents ComfyUI-GGUF.
  https://huggingface.co/city96/FLUX.1-dev-gguf
- black-forest-labs/FLUX.1-dev is gated (accept terms). We cannot accept terms for the user; the user must click "Agree" with their own HF account and put an HF token in Kaggle Secrets.
- Ungated route [U]: text encoders (CLIP-L, T5-XXL) and VAE are identical in black-forest-labs/FLUX.1-schnell (Apache-2.0, ungated). Load `FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", transformer=gguf_dev_transformer)`; the scheduler config differs slightly (dev uses shift/`use_dynamic_shifting`) - copy dev scheduler config or set it by hand. Third-party full-weight mirrors of dev exist but are a licence grey zone; prefer GGUF + schnell components.
- Diffusers GGUF loading [U, from diffusers docs, not re-read today]:
  ```python
  from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
  tr = FluxTransformer2DModel.from_single_file(
      "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q4_K_S.gguf",
      quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
      torch_dtype=torch.float16)
  ```
  LoRA on top of GGUF transformer: supported in recent diffusers (`load_lora_weights` on GGUF-quantised Flux) [U - test]; PEFT LoRA layers run in compute dtype.
- fp16: ComfyUI's Flux model class lists supported inference dtypes bf16/fp16/fp32 [U]. Flux in fp16 is known to run but some activations can overflow (black/NaN images) - same mixed-precision fix as for Qwen-Image-Layered: keep norms/residual adds/final layers in fp32, or clamp hidden states. Plan for it.
- T5-XXL fp16 ~9.5 GB -> put on GPU1 or use a GGUF T5 (city96/t5-v1_1-xxl-encoder-gguf [U]) / encode on CPU then free.
- Memory layout (all FLUX methods): GPU0 = transformer Q4 (~7 GB) + LoRAs + activations at 512 px; GPU1 = T5 + CLIP + VAE (+ extra decoders). RAM 30 GB is enough if loading via `low_cpu_mem_usage`.

---

## 1. RevealLayer (360CVGroup/RevealLayer)
Source: https://raw.githubusercontent.com/360CVGroup/RevealLayer/main/README.md
- Base: FLUX.1-dev (clones `black-forest-labs/FLUX.1-dev` to models/FLUX.1-dev) -> gated.
- Weights (HF `qihoo360/RevealLayer`): `pytorch_lora_weights.safetensors`, `layer_pe.pt`, `Refiner.pt`, `xvae/transparent_decoder_ckpt.pth`. Sizes not listed [U].
- Entry: `bash infer.sh 0` (arg = GPU id [U]) -> `infer.py`; paths edited inside. `--guidance_scale 1.0` recommended.
- Inputs: JSON list: `imgid`, `full_image` (RGB path), optional `background`, `LayerInfoRaw`, and `detections: [{bbox:[x1,y1,x2,y2]}]` -> image + boxes required.
- Resolution/steps: not in README [U]; read infer.sh.
- diffusers: vendored fork in repo (`pip install .` of its `diffusers` folder); version unknown [U]. Not stock FluxPipeline -> GGUF swap needs: load fork's custom pipeline, replace `.transformer` with a GGUF FluxTransformer2DModel. Works only if the fork's transformer class is stock-compatible (layer_pe added outside?) [U]. Also requires flash-attn (T4 = sm75: flash-attn 2 unsupported) -> patch to SDPA.
- Kaggle plan: download Q4_K_S dev GGUF (6.8 GB) + schnell T5/CLIP/VAE (~10 GB) + RevealLayer ckpt; 512 px, 12-20 steps, guidance 1.0, few boxes (<=3). Est. GPU0 ~10-12 GB.
- Blockers: gated base (user accept or schnell components), custom diffusers fork, flash-attn, fp16 overflow.

## 2. ART+ / PrismLayers (redredsheep/PrismLayers)
Source: https://raw.githubusercontent.com/redredsheep/PrismLayers/main/README.md
- Checkpoints (Google Drive, sizes not listed [U]):
  - base_ckpt https://drive.google.com/drive/folders/1NltNyAaUI_9gQfD9IN4WEj5bAJvxqPC4
  - artplus_PrismLayersPro_ckpt https://drive.google.com/drive/folders/1d-lIRYVyhKqpdZO50h4p4qZz9mXU3w4H
  - artplus_PrismLayersPro100k_ckpt https://drive.google.com/drive/folders/1ymrHD3c1cfToyt_ftu6D-sL1wcq65k5H
  - transp_vae_ckpt https://drive.google.com/drive/folders/1K-d1RurP6W8LaVNc3y70Y60K3RqF2eo-
  - pre_fuse_lora_dir https://drive.google.com/drive/folders/1dHqBLJJtdCr0PMuZ5S3iq--7kSHbOISU
  - classifier / TIPS (optional quality filters).
- Base: README only says "ART pretrained"; ART (Anonymous Region Transformer) is built on FLUX.1-dev and README asks `huggingface-cli login` -> FLUX.1-dev is pulled from HF separately (gated) [U - confirm in demo.yaml].
- Entry: `python multi_layer_gen/demo.py -c multi_layer_gen/demo.yaml -i inference_layout/test_1.json`
- Inputs: `whole_image: {caption ("This is a [STYLE] style image. ..."), height, width}` + `layout: [[x1,y1,x2,y2], ...]` (first two boxes = full image). No per-layer prompts in input (anonymous regions).
- Requirements: torch 2.4.0, diffusers 0.31.0, transformers 4.44.0, accelerate 0.34.2, peft 0.12.0. diffusers 0.31 has no GGUF support (added 0.32) -> upgrade and test.
- Kaggle plan: gdown folders (use `gdown --folder`; Drive quota may block), Q4 dev GGUF, 512 px (fine-tuned at 1024; expect degradation), 20 steps, <=4 boxes. Seq length grows with region tokens; est. 11-14 GB GPU0.
- Blockers: Drive quota/size unknown, likely custom transformer (anonymous region attention) so GGUF swap requires porting; gated base.

## 3. CLD (thuteam/CLD)
Source: https://huggingface.co/thuteam/CLD ; code https://github.com/monkek123King/CLD
- Base: FLUX.1-dev + ControlNet `alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha` (~4 GB fp16 [U]).
- Files: decouple_LoRA/{adapter/pytorch_lora_weights.safetensors, layer_pe.pth, transformer/pytorch_lora_weights.safetensors}, pre_trained_LoRA/, prism_ft_LoRA/, trans_vae/0008000.pt. Sizes not listed.
- Entry: `python -m infer.infer -c infer/infer.yaml` (yaml holds all paths). Image/prompt input format not documented on HF page [U] (likely image + boxes, same lineage as PrismLayers/ART).
- Kaggle plan: transformer Q4 GGUF on GPU0 (~7 GB) + ControlNet fp16 on GPU1 with T5/VAE; or ControlNet also quantised. 512 px, 20 steps. Est. both GPUs ~12-14 GB.
- Blockers: three stacked LoRAs + ControlNet; gated base; input format to confirm.

## 4. Stable-Layers (Stability-AI/Stable-Layers)
Source: https://raw.githubusercontent.com/Stability-AI/Stable-Layers/main/README.md and .../decompose.py
- Base: Qwen/Qwen-Image-Layered (~40 GB bf16), LoRA in `./model` (adapter_config.json + adapter_model.safetensors), also HF `StabilityLabs/Stable-Layers`. Applied with `PeftModel.from_pretrained(transformer, args.lora)` (transformer only).
- Rank / target_modules: NOT obtained (raw adapter_config.json URL 404'd) [U]. Read it from the repo's `model/adapter_config.json`.
- Defaults: `--size 640`, `--num-layers 4`, `--steps 50`, `--guidance-scale 1.0`, seed = 42 + image index. README: fewer steps / other sampler / higher res garble results.
- Prompt encoding (own, not pipeline): Qwen-Image chat template around prompt, text encoder last hidden state, drop first 34 tokens (`PROMPT_TEMPLATE_DROP_IDX = 34`), pad + mask. Default prompt "a clean, well composed image". Negative = encode("") but unused at cfg 1.0.
- Conditioning: resize longest side to size, multiple of 16, [-1,1], VAE posterior mode, normalise with latents_mean/std, 2x2 pack. Latents: randn(1, (num_layers+1)*h*w, C). img_shapes = generated frames + 1 condition entry.
- Schedule: `mu = (cond_seq_len / (256*256/16/16))**0.5`; `scheduler.set_timesteps(sigmas=np.linspace(1,0,steps+1)[:-1], mu=mu)`.
- Heun loop (verbatim gist):
  ```python
  dt = sigma_next - sigma
  v1 = velocity(latents, t)          # model gets timestep/1000
  latents_mid = latents + dt * v1
  if sigma_next > 0:
      v2 = velocity(latents_mid, sigma_next * 1000)
      latents = latents + dt * 0.5 * (v1 + v2)
  else:
      latents = latents_mid
  ```
  50 steps = 99 transformer calls.
- Output: drop frame 0, decode frames 1..N; compositing back-to-front "over", alpha [-1,1]->[0,1].
- Reimplementation on our diffusers QwenImageLayeredPipeline + GGUF: use `pipe.encode_prompt` equivalents (check drop_idx 34 matches pipeline's template), `pipe.prepare_latents`, call `pipe.transformer(...)` directly in the loop above; load LoRA via `pipe.load_lora_weights` or PEFT onto GGUF transformer (PEFT keys `base_model.model.` prefix -> may need key rename) [U]. fp16 patch from our Qwen work applies.
- Kaggle estimate: our Q4 run = 384 px, 4 steps -> 30 s. Stable-Layers needs ~99 calls at 640 px with 5 frames: roughly 99/4 x (640/384)^2 ~ 70x -> 30-40 min/image [U]. Reduced: 384-512 px, 20-25 Heun steps (~45 calls) -> ~5-10 min; quality risk per README.
- Blockers: adapter_config unknown, LoRA on GGUF, runtime.

## 5. Qwen-Image-2.1 (Qwen/Qwen-Image-2.1)
Source: https://huggingface.co/Qwen/Qwen-Image-2.1
- 7B visual generator (32 single-stream DiT layers), BF16 safetensors; plus text encoder (size [U]).
- Pipeline: `QwenImage21Pipeline` from diffusers (needs very recent/main diffusers [U]).
- RGBA: prompt-driven: "This is an RGBA image with transparency. ... The image has alpha channel and the background is transparent." width/height 2048, 40 steps. (Single RGBA image, not multi-layer.)
- Licence: Qwen Research License (`qwen-research`) - non-commercial research; fine for personal use.
- GGUF: not checked [U] (search city96 / unsloth / QuantStack). 7B fits on T4 in 8-bit (~7-8 GB) even without GGUF (bitsandbytes 8-bit or fp16 with CPU offload ~14 GB).
- Kaggle plan: transformer fp16 on GPU0 (~14 GB, tight) or 8-bit; text encoder on GPU1. 768-1024 px, 20 steps.
- Blockers: diffusers version, fp16 overflow, how alpha is decoded (VAE with 4 channels?) [U].

## 6. LayerDiffuse-Flux (RedAIGC/Flux-version-LayerDiffuse)
Source: https://raw.githubusercontent.com/RedAIGC/Flux-version-LayerDiffuse/main/README.md
- Files (HF `RedAIGC/Flux-version-LayerDiffuse`): `layerlora.safetensors`, `TransparentVAE.pth`. Base FLUX.1-dev via `--ckpt_path` (local diffusers folder).
- torch 2.3.0 cu121 + requirements.txt.
- `python demo_t2i.py --ckpt_path /path/FLUX.1_dev --prompt "..."`; `demo_i2i.py --image ./imgs/causal_cut.png` (strength 0.9).
- Single transparent image (fg only), not multi-layer.
- Kaggle plan: easiest FLUX one: build a local folder = schnell components + GGUF dev transformer (or patch demo to accept transformer object), load LoRA, 512-768 px, 20 steps. GPU0 ~9 GB.
- Blockers: gated base unless schnell-component trick; LoRA-on-GGUF.

## 7. OmniPSD
Sources: https://arxiv.org/abs/2512.09247 , https://showlab.github.io/OmniPSD
- Flux-dev (text->PSD) + Flux-Kontext (image->PSD). No public GitHub code found; online demo only via Lovart [U]. Not runnable now.
- If released: Kontext-dev GGUF exists in QuantStack [U]; same plan as FLUX.

---

## Priority for Kaggle
1. Stable-Layers on our existing Qwen GGUF stack (no gating, reuses fp16 patch) - main cost is runtime.
2. LayerDiffuse-Flux (simple, single LoRA) - tests FLUX GGUF + fp16 path.
3. RevealLayer / CLD (image+boxes decomposition) - custom code ports.
4. ART+ (Drive downloads, custom attention).
5. Qwen-Image-2.1 RGBA (single image only).

Gating action for user: accept FLUX.1-dev terms on HF themselves if full-weight dev is wanted; otherwise use city96 GGUF + FLUX.1-schnell components (ungated).
