# RevealLayer and LayerDiffuse-Flux on Kaggle 2x T4, without the gated FLUX.1-dev repo

Status: **the scripts are untested.** They were written on 2026-11-09 from the source code at the pinned commits below. Nothing in this note has been run yet.

## 0. Facts checked against the source and Hugging Face

| Item | Finding |
|---|---|
| RevealLayer commit | `da68ff58889a1799a1e75efd23e72bccebbf7589` (main) |
| LayerDiffuse-Flux commit | `d806a7fee93f54684fbcb17ae6356380057096a0` (FireRedTeam/LayerDiffuse-Flux, main) |
| `city96/FLUX.1-dev-gguf` | Ungated. The file is `flux1-dev-Q4_K_S.gguf`. |
| `qihoo360/RevealLayer` | Ungated. Files: `pytorch_lora_weights.safetensors` (608 keys, BF16, prefix `transformer.module.`, attention only), `layer_pe.pt`, `Refiner.pt`, `xvae/transparent_decoder_ckpt.pth` |
| `RedAIGC/Flux-version-LayerDiffuse` | Ungated. Files: `layerlora.safetensors` (912 keys, BF16, kohya/BFL naming `lora_unet_double_blocks_0_img_attn_proj...`), `TransparentVAE.pth` |
| **`black-forest-labs/FLUX.1-schnell`** | **Gated.** The API reports `"gated":"auto"`, and an anonymous download of `vae/config.json` returns 401. It is a click-through gate under Apache-2.0: you accept it once, then download with an `HF_TOKEN`. It is not usable "without a token". |
| Ungated substitute: **`ostris/Flex.1-alpha`** | `gated:false`, Apache-2.0, diffusers layout with `text_encoder` (CLIP-L), `text_encoder_2` (T5-XXL, 2 shards), `tokenizer`, `tokenizer_2`, and `vae` (AutoencoderKL, 16 ch, scaling 0.3611, shift 0.1159, the Flux AE). Flex.1 is a schnell-derived transformer only, so its encoders and VAE are the FLUX ones. LFS sha256: vae `f5b59a26…d40a3`; T5 shards `ec87bffd…893e` and `a5640855…a00`. I could not compare these with schnell or dev directly because their tree API needs auth. With a token, confirm with `huggingface_hub.HfApi().list_repo_tree(..., expand=True)`. The schnell and dev folder layouts are identical: the same file list for `text_encoder*`, `tokenizer*`, `vae`. |
| Scheduler | Schnell and Flex ship `shift=1.0, use_dynamic_shifting=false`. **Dev needs `shift=3.0, use_dynamic_shifting=true`** (base_shift 0.5, max_shift 1.15, 256/4096). Build it explicitly. Do not take it from schnell or Flex. |
| Transformer config | Schnell has `guidance_embeds=false`. Dev (and the GGUF, which contains `guidance_in.*`) needs `guidance_embeds=true`. Write a local `transformer/config.json`, because `from_single_file` otherwise fetches the config from the gated dev repo. |
| Licence | The GGUF is a quantised FLUX.1-dev, so the **FLUX.1 [dev] Non-Commercial License** applies to it and to all outputs/derivatives under that licence's terms. RevealLayer code is Apache-2.0. Personal, non-commercial use is fine. |

### RevealLayer code facts
- `infer.sh` launches 8 processes, one per GPU, on `infer.py --cid --tols`, with `PYTHONPATH=root:root/diffusers/src`. **It relies on the bundled diffusers, version `0.34.0`, which includes `quantizers/gguf`.**
- `infer.py` does the following:
  - `_prepare_custom_transformer` loads `FluxTransformer2DModel.from_pretrained(dev/transformer, bf16)` and then calls `CustomFluxTransformer2DModel.from_config(cfg + max_layer_num=12)`.
  - It then runs `load_state_dict(orig.state_dict(), strict=False)`, followed by `_load_layer_pe` (`transformer.layer_pe.data.copy_(torch.load(layer_pe.pt))`) and `_load_refiner` (keys prefixed with `refiner.`, `strict=False`).
  - It builds `CustomFluxPipelineCfg.from_pretrained(dev, transformer=, vae=, bf16).to(cuda)`.
  - `_load_loras` renames `transformer.module.` to `transformer.` and calls `pipeline.load_lora_weights(sd, adapter_name="layer")`.
  - The transparent decoder is loaded with `CustomVAE()` and `load_state_dict(pth, strict=False)`.
- Defaults: `--steps 30`, `--cfg 1.0` (the distilled guidance embedding), `--seed 43`, `--max_layer 12`. Resolution is hard-coded `resize_with_bbox_edge(..., resolution=1024)` in `test_one_sample`. Sizes snap to multiples of 16. Boxes become `[full, full] + detections`, capped at 12.
- The prompt is fixed: `"Decompose the image into foreground and background."`.
- **`CustomFluxPipelineCfg.__call__` always runs true CFG.** Each step makes two transformer calls (positive, then the negative prompt `""`), with `true_gs` defaulting to 3.5 because `infer.py` never passes it. That doubles the cost.
- Custom classes:
  - `models/custom_model_mmdit.py::CustomFluxTransformer2DModel(FluxTransformer2DModel)` is the same block structure as Flux. It adds:
    - `layer_pe` (an nn.Parameter of shape [1,12,1,1,3072])
    - `refiner = HybridRefiner(attn_hidden=1024, num_layers=1)`
    - a new `forward(hidden_states[b,L,c,h,w], adapter_data, split_sizes, list_layer_box, ...)` that crops per-layer tokens and builds an additive ROI attention mask `build_additive_spatial_roi_mask_v2`
  - The mask is filled with **`-1e9`**, which becomes `-inf` in fp16.
  - Pipelines: `models/custom_pipeline.py` defines `CustomFluxPipeline` and `CustomFluxPipelineCfg` (the one used). It also contains `models/custom_model_xvae.py` (transparent decoder) and `models/adapter.py` (refiner).
  - Bundled diffusers changes: `transformer_flux.py` blocks take `attention_mask`, and `FluxAttnProcessor2_0` passes it to `F.scaled_dot_product_attention`. The single block already does `clip(-65504, 65504)` when the dtype is fp16.
- **Flash-attn is never imported.** The README asks for `pip install flash-attn` and requirements list xformers, but every attention path is `F.scaled_dot_product_attention` with a float mask (diffusers processor plus `adapter.py:83`). The only `flash_attention` string is the torch_xla branch. **No SDPA swap is needed.** On T4, SDPA with a float mask uses the mem-efficient kernel.
- The input JSON is a list of `{imgid, full_image, background?, LayerInfoRaw?, detections:[{bbox:[x1,y1,x2,y2]}]}` with boxes in original-image pixels.
- `pipeline.encode_prompt` is the stock `FluxPipeline.encode_prompt`, using `self.text_encoder_2(ids.to(device))`. For that reason, putting T5 on another GPU needs a wrapper.

### LayerDiffuse-Flux code facts
- `demo_t2i.py` runs:
  1. `FluxPipeline.from_pretrained(ckpt, bf16).to('cuda')`, then `load_lora_weights(layerlora.safetensors)`, then `TransparentVAE(pipe.vae, pipe.vae.dtype)` with `load_state_dict(TransparentVAE.pth, strict=False)`.
  2. `pipe(..., output_type="latent")`, then `_unpack_latents`, then `/scaling + shift`, then `trans_vae.decode(latents)`, which returns `(original_x, x)` where `x` is RGBA in [0,1].
- Defaults: 1024x1024, 50 steps, guidance 3.5, `--dtype bfloat16`. The dtype argument is parsed but bf16 is hard-coded.
- There is no flash-attn and no custom transformer. It uses stock diffusers (requirements pin 0.32.2). `lib_layerdiffuse/vae.py` uses only UNet blocks.

## 1. Code changes per model

### 1a. RevealLayer

| # | File / function | Change |
|---|---|---|
| R1 | `infer.py::_prepare_custom_transformer` | Replace it entirely. Load `FluxTransformer2DModel.from_single_file(gguf, config=<local dir>, subfolder="transformer", quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16), torch_dtype=torch.float16)`. Then **re-class it in place**: set `t.__class__ = CustomFluxTransformer2DModel`, `t.register_to_config(max_layer_num=12)`, `t.max_layer_num = 12`, `t.layer_pe = nn.Parameter(torch.load(layer_pe.pt).half())`, `t.refiner = HybridRefiner(feature_dim=3072, pixel_dim=16, time_emb_dim=3072, attn_hidden=1024, num_layers=1, patch_size=2)`, and load `Refiner.pt` into `t.refiner` in fp16. This works because the custom class adds only `layer_pe`, `refiner`, `max_layer_num` and `forward`; every Flux submodule has the same name. It avoids `load_state_dict` of GGUF-packed uint8 tensors into plain Linears, which would fail on shape. |
| R1-alt | (only if LoRA on GGUFLinear fails) | Dequantise the GGUF to fp16 with `diffusers.quantizers.gguf.utils.dequantize_gguf_tensor`, one tensor at a time, and load into `CustomFluxTransformer2DModel` built on `meta`, placing tensors straight on the GPU. Put double blocks, embedders and the refiner on cuda:0 (about 7.5 GB), and the 38 single blocks, `norm_out` and `proj_out` on cuda:1 (about 16 GB). **That does not fit**, because single blocks are about 8.7B params. A true fp16 split needs about 23.8 GB of weights in total, which is roughly 12 GB per GPU with an even split (blocks 0-18 + singles 0-9 on cuda:0, singles 10-37 on cuda:1). T5 then has to go to the CPU (fp32 on CPU needs about 19 GB RAM, which is too much on Kaggle's ~29 GB together with dequant buffers) or be run first and freed. Prompt embeds are constant, so cache them to disk and drop T5. Hidden states hop GPUs once through a forward pre-hook on `single_transformer_blocks[10]`. This is a fallback only. |
| R2 | `infer.py::_load_vae_encoder`, `initialize_pipeline` | Replace `torch.bfloat16` with `torch.float16` everywhere. Do **not** call `.to(cuda)` on the whole pipeline. Place components individually: transformer, VAE and CLIP on cuda:0; T5 (fp16, about 9.5 GB) on cuda:1. Pass `scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0, use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15, base_image_seq_len=256, max_image_seq_len=4096)`. Load from the `Flex.1-alpha` folder with `transformer=` passed in. |
| R3 | pipeline `encode_prompt` / `_execution_device` | Monkeypatch `pipe.encode_prompt` to call the original with `device=cuda:1` and `.to(cuda:0)` every output. Set `CustomFluxPipelineCfg._execution_device = property(lambda s: torch.device("cuda:0"))`. |
| R4 | `infer.py::test_one_sample` | Change `resolution=1024` to `resolution=512`. Change `dtype=torch.bfloat16` to `float16`. Add `true_gs=getattr(args,"true_gs",3.5)` to the pipeline call. |
| R5 | `models/custom_pipeline.py::CustomFluxPipelineCfg.__call__` | Optional speed patch (halves time). Change `neg_noise_pred = self.transformer(` to `neg_noise_pred = noise_pred if true_gs == 1.0 else self.transformer(`. Default leaves `true_gs=3.5`, as in the original. |
| R6 | `models/custom_model_mmdit.py::build_additive_spatial_roi_mask_v2` | Replace `-1e9` with `torch.finfo(dtype).min` (-65504 in fp16), so there is no `-inf` in the mask. |
| R7 | fp16 overflow guard | Add forward hooks on every `transformer_blocks[i]`, `single_transformer_blocks[i]` and `refiner` that apply `nan_to_num(nan=0, posinf=6e4, neginf=-6e4)` to fp16 outputs, and log the first hit. Bundled `FluxSingleTransformerBlock` already clips fp16. |
| R8 | flash-attn | Do nothing, because it is never imported. Do not install `flash-attn` or `xformers`. |
| R9 | steps | `steps=16` (range 12-20). The scheduler shift adapts to the token count via `calculate_shift`. |

### 1b. LayerDiffuse-Flux

| # | File / function | Change |
|---|---|---|
| L1 | `demo_t2i.py` `__main__` | Replace `FluxPipeline.from_pretrained(ckpt, bf16).to('cuda')` with a GGUF transformer from `from_single_file(..., config=local dev cfg, quantization_config=GGUF fp16)` plus `FluxPipeline.from_pretrained(flex_dir, transformer=t, text_encoder_2=None, tokenizer_2=None, scheduler=dev_sched, torch_dtype=fp16)` on cuda:0. CLIP stays on cuda:0 because it is small, and its dtype keeps `text_ids` fp16. |
| L2 | T5 | Use a separate text-only `FluxPipeline.from_pretrained(flex_dir, transformer=None, vae=None, fp16).to("cuda:1")` and `encode_prompt(prompt, prompt_2=None, device=cuda:1)`. Pass `prompt_embeds` and `pooled_prompt_embeds` (moved to cuda:0) into the main pipe. |
| L3 | LoRA | `pipe.load_lora_weights(layerlora.safetensors)` is unchanged. diffusers converts kohya/BFL keys, including the fused-qkv split. Then cast any `lora_` params to fp16 on cuda:0. |
| L4 | `TransparentVAE(pipe.vae, pipe.vae.dtype)` | Unchanged. It is fp16 because the VAE is fp16. If decoded RGB is black or NaN, use an fp32 VAE plus TransparentVAE and pass `latents.float()`. |
| L5 | args | `height=width=512`, `steps=20`, `guidance=3.5`. |
| L6 | flash-attn | Nothing to do. |

## 2. Kaggle notebook cells (UNTESTED)

Set the Accelerator to GPU T4 x2 and turn Internet on. No HF token is needed with the Flex.1-alpha substitute. To use schnell instead, accept its gate and add `HF_TOKEN` as a Kaggle secret.

### 2a. RevealLayer (one cell)

```python
# ===== UNTESTED: RevealLayer on Kaggle 2xT4, GGUF Q4_K_S FLUX.1-dev, fp16, 512px =====
import subprocess, sys, os, json
def sh(c): print(c); subprocess.run(c, shell=True, check=True)
sh("pip -q install gguf>=0.10 peft==0.15.2 mmengine==0.10.7 einops sentencepiece protobuf")
RL="/kaggle/working/RevealLayer"; COMMIT="da68ff58889a1799a1e75efd23e72bccebbf7589"
if not os.path.exists(RL):
    sh(f"git clone https://github.com/360CVGroup/RevealLayer {RL} && cd {RL} && git checkout {COMMIT}")

# ---- source patches (python string replace) ----
def patch(path, old, new, count=-1):
    p=f"{RL}/{path}"; s=open(p,encoding="utf-8").read()
    assert old in s, f"patch anchor not found in {path}: {old[:60]}"
    open(p,"w",encoding="utf-8").write(s.replace(old,new,count)); print("patched",path)
patch("infer.py", "torch.bfloat16", "torch.float16")
patch("infer.py", "resolution=1024,", "resolution=512,")
patch("infer.py", "guidance_scale=args.cfg,", "guidance_scale=args.cfg, true_gs=getattr(args,'true_gs',3.5),")
patch("models/custom_model_mmdit.py", "-1e9, dtype=dtype", "torch.finfo(dtype).min, dtype=dtype")
patch("models/custom_pipeline.py", "neg_noise_pred = self.transformer(",
      "neg_noise_pred = noise_pred if true_gs == 1.0 else self.transformer(")

# ---- use bundled diffusers 0.34 (has GGUF + attention_mask-aware Flux blocks) ----
sys.path[:0] = [RL, f"{RL}/diffusers/src"]
for m in [k for k in sys.modules if k.startswith("diffusers")]: del sys.modules[m]
import torch, torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download
import diffusers; print("diffusers", diffusers.__version__, diffusers.__file__)
from diffusers import FluxTransformer2DModel, AutoencoderKL, GGUFQuantizationConfig, FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file

# ---- downloads ----
gguf = hf_hub_download("city96/FLUX.1-dev-gguf", "flux1-dev-Q4_K_S.gguf")
CK = snapshot_download("qihoo360/RevealLayer", allow_patterns=["*.pt","*.safetensors","xvae/*"])
BASE = snapshot_download("ostris/Flex.1-alpha",  # ungated; FLUX VAE + CLIP-L + T5-XXL
    allow_patterns=["model_index.json","text_encoder/*","text_encoder_2/*","tokenizer/*","tokenizer_2/*","vae/*","scheduler/*"])
CFG = "/kaggle/working/flux_dev_cfg"; os.makedirs(f"{CFG}/transformer", exist_ok=True)
json.dump({"_class_name":"FluxTransformer2DModel","attention_head_dim":128,"axes_dims_rope":[16,56,56],
           "guidance_embeds":True,"in_channels":64,"joint_attention_dim":4096,"num_attention_heads":24,
           "num_layers":19,"num_single_layers":38,"patch_size":1,"pooled_projection_dim":768},
          open(f"{CFG}/transformer/config.json","w"))
dev_sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True,
            base_shift=0.5, max_shift=1.15, base_image_seq_len=256, max_image_seq_len=4096)

D0, D1, DT = torch.device("cuda:0"), torch.device("cuda:1"), torch.float16
from models.custom_model_mmdit import CustomFluxTransformer2DModel
from models.adapter import HybridRefiner
from models.custom_pipeline import CustomFluxPipelineCfg
from models.custom_model_xvae import AutoencoderKLTransformerTraining as CustomVAE
import infer  # helpers + (patched) test_one_sample

# ---- R1: GGUF transformer, re-classed into the custom model ----
t = FluxTransformer2DModel.from_single_file(gguf, config=CFG, subfolder="transformer",
        quantization_config=GGUFQuantizationConfig(compute_dtype=DT), torch_dtype=DT)
t.__class__ = CustomFluxTransformer2DModel
t.register_to_config(max_layer_num=12); t.max_layer_num = 12
t.layer_pe = nn.Parameter(torch.load(f"{CK}/layer_pe.pt", map_location="cpu").to(DT).view(1,12,1,1,3072))
t.refiner = HybridRefiner(feature_dim=3072, pixel_dim=16, time_emb_dim=3072, attn_hidden=1024, num_layers=1, patch_size=2)
print("refiner:", t.refiner.load_state_dict(torch.load(f"{CK}/Refiner.pt", map_location="cpu"), strict=False))
t.refiner.to(DT); t.to(D0).eval()

# ---- R7: fp16 overflow guard ----
_hit = {"n":0}
def guard(mod, inp, out):
    def fix(o):
        if torch.is_tensor(o) and o.dtype == torch.float16 and not torch.isfinite(o).all():
            _hit["n"] += 1
            if _hit["n"] == 1: print("[guard] non-finite fp16 activations in", type(mod).__name__)
            return torch.nan_to_num(o, nan=0.0, posinf=6e4, neginf=-6e4)
        return o
    return tuple(fix(o) for o in out) if isinstance(out, tuple) else fix(out)
for b in list(t.transformer_blocks) + list(t.single_transformer_blocks) + [t.refiner]:
    b.register_forward_hook(guard)

# ---- R2/R3: pipeline, components placed by hand ----
vae = AutoencoderKL.from_pretrained(BASE, subfolder="vae", torch_dtype=DT)
pipe = CustomFluxPipelineCfg.from_pretrained(BASE, transformer=t, vae=vae, scheduler=dev_sched, torch_dtype=DT)
pipe.vae.to(D0); pipe.text_encoder.to(D0); pipe.text_encoder_2.to(D1)
CustomFluxPipelineCfg._execution_device = property(lambda self: D0)
_orig_enc = pipe.encode_prompt
def _enc(*a, device=None, **kw):
    # run CLIP on cuda:0 inputs and T5 on cuda:1 inputs, then move outputs to cuda:0
    pipe.text_encoder.to(D1); out = _orig_enc(*a, device=D1, **kw); pipe.text_encoder.to(D0)
    return tuple(o.to(D0) if torch.is_tensor(o) else o for o in out)
pipe.encode_prompt = _enc

lora = {k.replace("transformer.module.", "transformer."): v for k, v in load_file(f"{CK}/pytorch_lora_weights.safetensors").items()}
pipe.load_lora_weights(lora, adapter_name="layer")
for n, p in pipe.transformer.named_parameters():
    if "lora_" in n: p.data = p.data.to(D0, DT)

transp_vae = CustomVAE()
print("xvae:", transp_vae.load_state_dict(torch.load(f"{CK}/xvae/transparent_decoder_ckpt.pth", map_location="cpu"), strict=False))
transp_vae.to(D0).eval()

# ---- one example (replace with your own image + boxes in ORIGINAL pixel coords) ----
from PIL import Image; import urllib.request
img_path = "/kaggle/working/test.png"
urllib.request.urlretrieve("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png", img_path)
W, H = Image.open(img_path).size
sample = {"index": "demo", "layout": [[int(W*.2), int(H*.15), int(W*.8), int(H*.95)]], "full_image_path": img_path}
from types import SimpleNamespace
args = SimpleNamespace(seed=43, cfg=1.0, steps=16, true_gs=3.5,   # true_gs=1.0 -> half time (skips neg pass)
                       output_dir="/kaggle/working/reveal_out")
import time; t0 = time.time()
with torch.inference_mode():
    infer.test_one_sample(args, sample, pipe, transp_vae, D0, shuffle=False)
print(f"done in {time.time()-t0:.0f}s; guard hits={_hit['n']}")
for i in range(2): print(f"cuda:{i} peak {torch.cuda.max_memory_allocated(i)/2**30:.1f} GiB")
sh("ls -la /kaggle/working/reveal_out/demo")
```

Note on `_enc`: stock `FluxPipeline._get_clip_prompt_embeds` and `_get_t5_prompt_embeds` both send token ids to the single `device` argument. The wrapper therefore moves CLIP (250 MB) to cuda:1 just for encoding and back again afterwards. That is cheap.

### 2b. LayerDiffuse-Flux (one cell, fresh kernel)

```python
# ===== UNTESTED: LayerDiffuse-Flux T2I on Kaggle 2xT4, GGUF Q4_K_S FLUX.1-dev, fp16, 512px =====
import subprocess, os, sys, json
def sh(c): print(c); subprocess.run(c, shell=True, check=True)
sh("pip -q install diffusers==0.34.0 gguf>=0.10 peft sentencepiece protobuf opencv-python-headless")
LD="/kaggle/working/LayerDiffuse-Flux"; COMMIT="d806a7fee93f54684fbcb17ae6356380057096a0"
if not os.path.exists(LD):
    sh(f"git clone https://github.com/FireRedTeam/LayerDiffuse-Flux {LD} && cd {LD} && git checkout {COMMIT}")
sys.path.insert(0, LD)

import torch, numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig, FlowMatchEulerDiscreteScheduler
from lib_layerdiffuse.vae import TransparentVAE

gguf  = hf_hub_download("city96/FLUX.1-dev-gguf", "flux1-dev-Q4_K_S.gguf")
lora  = hf_hub_download("RedAIGC/Flux-version-LayerDiffuse", "layerlora.safetensors")
tvae  = hf_hub_download("RedAIGC/Flux-version-LayerDiffuse", "TransparentVAE.pth")
BASE  = snapshot_download("ostris/Flex.1-alpha", allow_patterns=["model_index.json","text_encoder/*",
        "text_encoder_2/*","tokenizer/*","tokenizer_2/*","vae/*","scheduler/*"])
CFG = "/kaggle/working/flux_dev_cfg"; os.makedirs(f"{CFG}/transformer", exist_ok=True)
json.dump({"_class_name":"FluxTransformer2DModel","attention_head_dim":128,"axes_dims_rope":[16,56,56],
           "guidance_embeds":True,"in_channels":64,"joint_attention_dim":4096,"num_attention_heads":24,
           "num_layers":19,"num_single_layers":38,"patch_size":1,"pooled_projection_dim":768},
          open(f"{CFG}/transformer/config.json","w"))
dev_sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True,
            base_shift=0.5, max_shift=1.15, base_image_seq_len=256, max_image_seq_len=4096)
D0, D1, DT = "cuda:0", "cuda:1", torch.float16
PROMPT, H, W, STEPS, GUID, SEED = "glass bottle, high quality", 512, 512, 20, 3.5, 11111

# ---- L2: text encoders on cuda:1, encode once ----
tp = FluxPipeline.from_pretrained(BASE, transformer=None, vae=None, torch_dtype=DT).to(D1)
with torch.inference_mode():
    pe, ppe, _ = tp.encode_prompt(PROMPT, prompt_2=None, device=D1, max_sequence_length=512)
pe, ppe = pe.to(D0), ppe.to(D0)
tp.text_encoder_2.to("cpu"); del tp; torch.cuda.empty_cache()

# ---- L1: GGUF transformer + main pipe on cuda:0 ----
t = FluxTransformer2DModel.from_single_file(gguf, config=CFG, subfolder="transformer",
        quantization_config=GGUFQuantizationConfig(compute_dtype=DT), torch_dtype=DT)
pipe = FluxPipeline.from_pretrained(BASE, transformer=t, text_encoder_2=None, tokenizer_2=None,
        scheduler=dev_sched, torch_dtype=DT).to(D0)
pipe.load_lora_weights(lora)                                      # L3
for n, p in pipe.transformer.named_parameters():
    if "lora_" in n: p.data = p.data.to(D0, DT)

# fp16 overflow guard
hits = {"n": 0}
def guard(m, i, o):
    fx = lambda x: (hits.__setitem__("n", hits["n"]+1) or torch.nan_to_num(x, nan=0., posinf=6e4, neginf=-6e4)) \
         if torch.is_tensor(x) and x.dtype == torch.float16 and not torch.isfinite(x).all() else x
    return tuple(fx(x) for x in o) if isinstance(o, tuple) else fx(o)
for b in list(t.transformer_blocks) + list(t.single_transformer_blocks): b.register_forward_hook(guard)

trans_vae = TransparentVAE(pipe.vae, pipe.vae.dtype)              # L4
print(trans_vae.load_state_dict(torch.load(tvae, map_location="cpu"), strict=False))
trans_vae.to(D0)

import time; t0 = time.time()
with torch.inference_mode():
    lat = pipe(prompt_embeds=pe, pooled_prompt_embeds=ppe, height=H, width=W, num_inference_steps=STEPS,
               guidance_scale=GUID, output_type="latent", generator=torch.Generator(D0).manual_seed(SEED)).images
    lat = pipe._unpack_latents(lat, H, W, pipe.vae_scale_factor)
    lat = lat / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    _, x = trans_vae.decode(lat)
x = x.clamp(0, 1).permute(0, 2, 3, 1)
os.makedirs("/kaggle/working/ld_out", exist_ok=True)
Image.fromarray((x*255).float().cpu().numpy().astype(np.uint8)[0]).save("/kaggle/working/ld_out/out.png")
print(f"done {time.time()-t0:.0f}s, guard hits {hits['n']}, peak cuda:0 {torch.cuda.max_memory_allocated(0)/2**30:.1f} GiB")
```

## 3. Memory and time estimates (unmeasured), and risks

### Memory
| | cuda:0 | cuda:1 | CPU RAM peak |
|---|---|---|---|
| RevealLayer | GGUF 6.8 GB + LoRA r64 ~0.15 + refiner ~0.05 + VAE 0.17 + CLIP 0.25 + xvae ~0.1 + per-layer dequant temp ~0.1 + activations (512px, ~3-4k tokens, dense fp16 mask ~30 MB) ≈ **8-10 GB** | T5-XXL fp16 ≈ **9.5 GB** | about 12-16 GB during load (T5 fp16 + GGUF mmap) |
| LayerDiffuse-Flux | GGUF 6.8 + LoRA + VAE + CLIP + TransparentVAE UNet (~0.2 GB; aug decode runs 8 passes) ≈ **8-9 GB** | T5 9.5 GB, freed after encoding | about 12 GB |

### Time on T4, fp16 tensor cores
- Flux-12B at about 1.5k tokens: roughly 2-3 s per forward. Add 20-40% for GGUF dequant-on-the-fly with no CUDA kernels.
- **LayerDiffuse 512px, 20 steps: about 1-1.5 min**, plus about 10 s for the augmented transparent decode.
- **RevealLayer 512px, 1 box:** about 1024 + 1024 + ~400 image tokens + 512 text, so about 3k tokens and about 4-6 s per forward. Each step makes 2 forwards (true CFG), so 16 steps is **about 2.5-3.5 min**. With `true_gs=1.0` it takes about half that. Each extra box adds roughly its area in tokens.
- Building the ROI mask is a per-token Python loop on the CPU, which adds a few seconds per call at 512px.
- **Model load: about 5-8 min**, for downloads (6.8 GB GGUF + about 10 GB T5 + LoRAs) and placement.

### Risks
1. **LoRA on GGUFLinear (high).** peft must wrap `GGUFLinear`, which subclasses nn.Linear, while leaving the base layer quantised. diffusers 0.33+ supports this for Flux, but the bundled 0.34 fork is modified. If `load_lora_weights` errors or LoRA has no visible effect, fall back to R1-alt (fp16 dequant split across 2 GPUs), which is much heavier. Do not call `fuse_lora` on GGUF.
2. **`__class__` swap (medium).** It relies on the custom `__init__` adding only `layer_pe`, `refiner` and `max_layer_num`, which I verified in the source. PEFT and `load_lora_weights` key off module names, which are unchanged. `CustomFluxTransformer2DModel.from_pretrained` is bypassed, which is harmless.
3. **fp16 overflow (medium).** Flux in fp16 is known to overflow occasionally in the late single blocks, and the bundled code already clips. The guard turns NaN into 0, which hides corruption, so watch the hit counter. T5-XXL in fp16 can produce inf. Transformers' T5 clamps in fp16, but if embeddings contain NaN, run T5 on cuda:1 in fp32 (about 19 GB, does not fit) or use bf16-emulated fp32 on the CPU for the two fixed prompts and cache them.
4. **Quality drift.** Q4_K_S and fp16 differ from the bf16 originals, and LoRAs were trained on bf16 dev. 512px is half RevealLayer's training resolution (1024), and its layer_pe and ROI masks assume 16-px box snapping. Expect worse occlusion fill and alpha. Try 768 if memory allows: tokens grow about 2.25x, which is still fine on 15 GB.
5. **Using Flex.1-alpha encoders and VAE instead of dev's.** These are believed byte-identical (Flex is a transformer-only fine-tune of schnell), but I could not hash-compare them without a token. If you have one, compare `list_repo_tree(..., expand=True)` LFS sha256 values against `FLUX.1-schnell`.
6. **Bundled diffusers vs Kaggle's transformers and torch versions.** The bundled code is 0.34.0 and should work with transformers 4.4x-4.5x. The `sys.modules` purge in the RevealLayer cell only works in a **fresh kernel** before any `import diffusers`. Run the two cells in separate notebooks or sessions.
7. **`hf_hub_download` RAM and disk.** Kaggle `/kaggle/working` has about 20 GB, and the HF cache defaults to `/root/.cache`, which is fine. Total download is about 18 GB.
8. **Licence.** FLUX.1-dev Non-Commercial applies to the GGUF weights and their use. RevealLayer is Apache-2.0. The LayerDiffuse-Flux repo states no licence file in its tree.
