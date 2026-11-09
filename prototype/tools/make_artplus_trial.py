r"""Kaggle 2x T4 notebook: ART+ (PrismLayers, MIT code) text + boxes -> RGBA layers, at reduced settings.

    fastsdcpu\env\Scripts\python.exe prototype\tools\make_artplus_trial.py
    cd prototype\kaggle\upload_artplus_trial && python -m kaggle.cli kernels push -p .

Why ART+: of the FLUX layered models it is the only one that goes from TEXT + boxes to layers (no input image),
i.e. our "one prompt -> coherent layers" step; boxes can come from the LLM breakdown.

Weights (PrismLayers demo.yaml needs pre_fuse_lora + base_ckpt (both fused) + artplus_ckpt (layer_pe + "layer" LoRA)
+ transparent decoder):
  * base_ckpt LoRA  -> thuteam/CLD pre_trained_LoRA (official, Apache-2.0; SHA-256 identical to the ART base LoRA)
  * transparent VAE -> thuteam/CLD trans_vae/0008000.pt (CLD paper: reuses ART's transparent decoder)
  * pre_fuse_lora, artplus layer_pe + LoRA -> official Google Drive folders (README) via gdown; if Drive refuses
    (quota), the unofficial HF mirror 24yearsold/PrismLayers_ckpt (its base LoRA hash matches the official one,
    which supports its authenticity; its ART+ files are the PrismLayersPro100k variant)
Adaptation (as for RevealLayer): GGUF Q4_K_S FLUX.1-dev re-classed into CustomFluxTransformer2DModel; the three LoRAs
are attached UNMERGED with fp16_mixed.add_lora (GGUF weights cannot be fused); fp16_mixed against fp16 overflow;
encoders/VAE from ungated ostris/Flex.1-alpha; 512 px, 16 steps, 25-min cap. FLUX.1-dev Non-Commercial licence.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
OUT = os.path.join(PROTO, "kaggle", "upload_artplus_trial")
MIXED = open(os.path.join(PROTO, "layerproto", "fp16_mixed.py"), encoding="utf-8").read()

# our forest scene as an ART+ input: caption with style prefix + boxes (first two = full image, as in PrismLayers)
LAYOUTS = [
    {"whole_image": {"caption": "This is a painting style image. An impressionist oil painting of a forest clearing at night "
                                "with a full moon, tall pine trees, a crackling campfire of crossed logs and an empty wooden chair.",
                     "height": 512, "width": 512},
     "layout": [[0, 0, 512, 512], [0, 0, 512, 512], [40, 0, 230, 430], [300, 0, 470, 420], [90, 300, 290, 460], [300, 350, 380, 430]]},
    {"whole_image": {"caption": "This is a sketch style image. A pale pencil sketch of a skeleton sitting next to a small campfire "
                                "in a snowy forest, with a log on the ground.",
                     "height": 512, "width": 512},
     "layout": [[0, 0, 512, 512], [0, 0, 512, 512], [250, 120, 400, 400], [120, 330, 300, 450], [300, 380, 460, 450]]},
]

src = r'''import signal, time as _t
_T0 = _t.time()
def _cap(*a): raise TimeoutError("25 min cap hit after %.0fs" % (_t.time() - _T0))
signal.signal(signal.SIGALRM, _cap); signal.alarm(25 * 60)
import subprocess, os, sys, json, time, glob, argparse
def sh(c, check=True): print(c, flush=True); return subprocess.run(c, shell=True, check=check).returncode
sh("pip -q install 'transformers==4.57.1' diffusers==0.32.2 peft==0.14.0 gguf>=0.10 sentencepiece protobuf mmengine gdown einops")
PL = "/kaggle/working/PrismLayers"
if not os.path.exists(PL): sh(f"git clone --depth 1 https://github.com/redredsheep/PrismLayers {PL}")
sys.path.insert(0, f"{PL}/multi_layer_gen")
import torch, torch.nn as nn, numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig, FlowMatchEulerDiscreteScheduler
import diffusers; print("diffusers", diffusers.__version__, flush=True)

# ---- weights ----
gguf = hf_hub_download("city96/FLUX.1-dev-gguf", "flux1-dev-Q4_K_S.gguf")
BASE = snapshot_download("ostris/Flex.1-alpha", allow_patterns=["model_index.json","text_encoder/*","text_encoder_2/*","tokenizer/*","tokenizer_2/*","vae/*","scheduler/*"])
CLD = snapshot_download("thuteam/CLD", allow_patterns=["pre_trained_LoRA/*", "trans_vae/*"])
W = "/kaggle/working/art_ckpt"; os.makedirs(W, exist_ok=True)
SRC = "drive"
ok = sh(f"gdown --folder -q https://drive.google.com/drive/folders/1dHqBLJJtdCr0PMuZ5S3iq--7kSHbOISU -O {W}/pre_fuse_lora", check=False) == 0 and \
     sh(f"gdown --folder -q https://drive.google.com/drive/folders/1d-lIRYVyhKqpdZO50h4p4qZz9mXU3w4H -O {W}/artplus", check=False) == 0
def find(pat):
    f = sorted(glob.glob(f"{W}/**/{pat}", recursive=True)); return f[0] if f else None
if not (ok and find("pre_fuse_lora/*.safetensors") and find("artplus/*layer_pe*")):
    SRC = "hf-mirror 24yearsold/PrismLayers_ckpt"
    M = snapshot_download("24yearsold/PrismLayers_ckpt", allow_patterns=["pre_fuse_lora/*", "artplus_PrismLayersPro100k_ckpt/*", "base/*"])
    W = M
print("ART+ weights from:", SRC, flush=True)
pre_fuse = find("pre_fuse_lora/*.safetensors")
art_lora = find("artplus*/pytorch_lora_weights.safetensors")
art_pe = find("artplus*/layer_pe*")
print(pre_fuse, art_lora, art_pe, flush=True)
def load_any(p):
    return load_file(p) if p.endswith(".safetensors") else torch.load(p, map_location="cpu")

CFG = "/kaggle/working/flux_dev_cfg"; os.makedirs(f"{CFG}/transformer", exist_ok=True)
json.dump({"_class_name":"FluxTransformer2DModel","attention_head_dim":128,"axes_dims_rope":[16,56,56],"guidance_embeds":True,
           "in_channels":64,"joint_attention_dim":4096,"num_attention_heads":24,"num_layers":19,"num_single_layers":38,
           "patch_size":1,"pooled_projection_dim":768}, open(f"{CFG}/transformer/config.json","w"))
dev_sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True, base_shift=0.5,
                                            max_shift=1.15, base_image_seq_len=256, max_image_seq_len=4096)
D0, D1, DT = torch.device("cuda:0"), torch.device("cuda:1"), torch.float16
from custom_model_mmdit import CustomFluxTransformer2DModel
from custom_pipeline import CustomFluxPipelineCfgLayer
from custom_model_transp_vae import AutoencoderKLTransformerTraining as CustomVAE

# ---- transformer: GGUF re-classed, layer_pe, three unmerged LoRAs, fp16_mixed ----
t = FluxTransformer2DModel.from_single_file(gguf, config=CFG, subfolder="transformer",
        quantization_config=GGUFQuantizationConfig(compute_dtype=DT), torch_dtype=DT)
t.__class__ = CustomFluxTransformer2DModel
t.register_to_config(max_layer_num=52); t.max_layer_num = 52
pe = load_any(art_pe); pe = pe.get("layer_pe", next(iter(pe.values()))) if isinstance(pe, dict) else pe
t.layer_pe = nn.Parameter(pe.to(torch.float32).view(1, 52, 1, 1, -1), requires_grad=False)
exec(compile(__MIXED__, "fp16_mixed", "exec"))
apply_fp16_mixed(t)
for name, path in (("pre_fuse", pre_fuse), ("art_base(CLD)", f"{CLD}/pre_trained_LoRA/pytorch_lora_weights.safetensors"), ("artplus", art_lora)):
    hit, miss = add_lora(t, path, 1.0); print(f"LoRA {name}: matched {hit}, missing {len(miss)}", miss[:3], flush=True)
t.to(D0).eval()

pipe = CustomFluxPipelineCfgLayer.from_pretrained(BASE, transformer=t, scheduler=dev_sched, torch_dtype=DT)
pipe.vae.to(D0, torch.float32); pipe.text_encoder.to(D1); pipe.text_encoder_2.to(D1)
CustomFluxPipelineCfgLayer._execution_device = property(lambda self: D0)
_orig_enc = pipe.encode_prompt
def _enc(*a, device=None, **kw):
    out = _orig_enc(*a, device=D1, **kw)
    return tuple(o.to(D0) if torch.is_tensor(o) else o for o in out)
pipe.encode_prompt = _enc

vae_args = argparse.Namespace(max_layers=48, decoder_arch="vit", pos_embedding="rope", layer_embedding="rope", single_layer_decoder=None)
transp_vae = CustomVAE(vae_args)
tv = torch.load(glob.glob(f"{CLD}/trans_vae/*.pt")[0], map_location="cpu", weights_only=False)  # official thuteam/CLD checkpoint
print("transp vae:", transp_vae.load_state_dict(tv.get("model", tv), strict=False), flush=True)
transp_vae.to(D0, torch.float32).eval()
# VAEs run in fp32 but the pipeline hands them fp16 tensors -> cast tensor inputs (RevealLayer needed the same fix)
def _to32(m, args, kwargs):
    c = lambda x: x.float() if torch.is_tensor(x) and x.is_floating_point() else x
    return tuple(c(a) for a in args), {k: c(v) for k, v in kwargs.items()}
for _v in (transp_vae, pipe.vae):
    _v.register_forward_pre_hook(_to32, with_kwargs=True)
    for _name in ("encode", "decode"):
        if hasattr(_v, _name):
            _f = getattr(_v, _name)
            setattr(_v, _name, (lambda f: lambda *a, **k: f(*[x.float() if torch.is_tensor(x) and x.is_floating_point() else x for x in a], **k))(_f))
# run 4 OOM'd in the fp32 VAE decode with the transformer still on GPU 0: offload it before decoding, slice/tile the VAE
pipe.vae.enable_slicing(); pipe.vae.enable_tiling()
for _v in (transp_vae, pipe.vae):
    _dec = _v.decode
    def _offload_then(*a, _d=_dec, **k):
        if next(t.parameters()).device.type == "cuda":
            t.to("cpu"); torch.cuda.empty_cache()
        return _d(*a, **k)
    _v.decode = _offload_then
print("loaded", round(time.time() - _T0), "s; GiB", [round(torch.cuda.memory_allocated(i)/2**30, 1) for i in range(2)], flush=True)

def boxes16(bs):
    return [((x0//16)*16, (y0//16)*16, ((x1//16)+1)*16 if x1 < 512 else 512, ((y1//16)+1)*16 if y1 < 512 else 512) for x0, y0, x1, y1 in bs]
LAYOUTS = __LAYOUTS__
os.makedirs("/kaggle/working/art_out", exist_ok=True)
for k, inp in enumerate(LAYOUTS):
    for steps in (16,):
        try:
            t.to(D0); torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0); t0 = time.time()
            with torch.inference_mode():
                x_hat, image, _ = pipe(prompt=inp["whole_image"]["caption"], validation_box=boxes16(inp["layout"]),
                                       generator=torch.Generator(D0).manual_seed(42), height=512, width=512,
                                       num_inference_steps=steps, guidance_scale=4.0, num_layers=len(inp["layout"]), sdxl_vae=transp_vae)
            x_hat = ((x_hat + 1) / 2).squeeze(0).permute(1, 0, 2, 3).float()
            d = f"/kaggle/working/art_out/case{k}_{steps}st"; os.makedirs(d, exist_ok=True)
            image[0].save(f"{d}/whole.png"); image[1].save(f"{d}/background.png")
            merged = image[1].convert("RGBA")
            for i in range(2, x_hat.shape[0]):
                lay = Image.fromarray((x_hat[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), "RGBA")
                lay.save(f"{d}/layer_{i-2}_rgba.png"); merged = Image.alpha_composite(merged, lay)
            merged.convert("RGB").save(f"{d}/merged.png")
            print(f"case{k} {steps} steps: {time.time()-t0:.0f}s, peak GiB {torch.cuda.max_memory_allocated(0)/2**30:.1f}", flush=True)
        except BaseException as e:
            import traceback; traceback.print_exc(); print("FAILED", k, steps, flush=True)
            if isinstance(e, TimeoutError): raise
print("total", round(time.time() - _T0), "s", flush=True)
'''
src = src.replace("__MIXED__", repr(MIXED)).replace("__LAYOUTS__", json.dumps(LAYOUTS))
compile(src, "cell", "exec")
os.makedirs(OUT, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": src.strip("\n").splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(OUT, "artplus-trial.ipynb"), "w", encoding="utf-8"), indent=1)
json.dump({"id": "markandreduesing/artplus-trial", "title": "artplus-trial", "code_file": "artplus-trial.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true",
           "enable_tpu": "false", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
           "competition_sources": [], "kernel_sources": [], "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("wrote", OUT)
