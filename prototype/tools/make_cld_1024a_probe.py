r"""Kaggle 2x T4 notebook: CLD (Controllable Layer Decomposition, MIT code, Apache-2.0 weights): image + caption + boxes -> RGBA layers.

    fastsdcpu/env/Scripts/python.exe prototype/tools/make_cld_trial.py
    cd prototype/kaggle/upload_cld_1024a_probe && python -m kaggle.cli kernels push -p .

Same adaptation as ART+ (GGUF Q4_K_S FLUX.1-dev re-classed into the custom transformer, LoRAs unmerged via fp16_mixed.add_lora,
fp32 VAEs, offload before decode) plus CLD's MultiLayerAdapter (inpainting ControlNet alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha
and its LoRA) fed with the input image. Input: the fox & cottage test scene + its 5 hand boxes. 1024 px, 2 steps (scaling probe), CFG 4, scale 0.9.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
OUT = os.path.join(PROTO, "kaggle", "upload_cld_1024a_probe")
MIXED = open(os.path.join(PROTO, "layerproto", "fp16_mixed.py"), encoding="utf-8").read()

def _case(img_rel, extra):
    j = json.load(open(os.path.join(PROTO, 'test_scene', 'fox_cottage.json'), encoding='utf-8'))
    import base64, io
    from PIL import Image as _I
    buf = io.BytesIO(); _I.open(os.path.join(PROTO, *img_rel)).convert('RGB').resize((1024, 1024)).save(buf, 'JPEG', quality=90)
    return {'whole_image': {'caption': j['caption'] + extra, 'height': 1024, 'width': 1024}, 'img': base64.b64encode(buf.getvalue()).decode(),
            'layout': [[0, 0, 512, 512], [0, 0, 512, 512]] + [L['box'] for L in j['layers_back_to_front']]}


# v5: more boxes on the original fox scene (5 hand boxes + cobblestone path + hedge/garden wall): does CLD scale to 7 layers on 2x T4?
def _case7():
    c = _case(('test_scene', 'fox_cottage.png'), '')
    lay = c['layout']
    # back to front: houses, path, cottage, hedge, gate arch, tree, fox  (full-image boxes stay first)
    houses, cottage, gate, tree, fox = lay[2:]
    c['layout'] = lay[:2] + [houses, [0, 330, 300, 512], cottage, [0, 255, 512, 330], gate, tree, fox]
    return c
LAYOUTS = [_case7()]
IMG_B64 = ''

src = r'''import os as _o; _o.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import signal, time as _t
_T0 = _t.time()
def _cap(*a): raise TimeoutError("25 min cap hit after %.0fs" % (_t.time() - _T0))
signal.signal(signal.SIGALRM, _cap); signal.alarm(25 * 60)
import subprocess, os, sys, json, time, glob, argparse, base64, io
def sh(c, check=True): print(c, flush=True); return subprocess.run(c, shell=True, check=check).returncode
sh("pip -q install 'transformers==4.57.1' diffusers==0.32.2 peft==0.14.0 gguf>=0.10 sentencepiece protobuf mmengine gdown einops")
PL = "/kaggle/working/CLD"
if not os.path.exists(PL): sh(f"git clone --depth 1 https://github.com/monkek123King/CLD {PL}")
sys.path.insert(0, PL)
_f = f"{PL}/models/multiLayer_adapter.py"   # diffusers 0.32 no longer re-exports BaseOutput from models.controlnet
_t = open(_f).read().replace("from diffusers.models.controlnet import BaseOutput, zero_module", "from diffusers.utils import BaseOutput; from diffusers.models.controlnets.controlnet import zero_module")
open(_f, "w").write(_t)
import torch, torch.nn as nn, numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig, FlowMatchEulerDiscreteScheduler
import diffusers; print("diffusers", diffusers.__version__, flush=True)

# ---- weights ----
gguf = hf_hub_download("city96/FLUX.1-dev-gguf", "flux1-dev-Q4_K_S.gguf")
BASE = snapshot_download("ostris/Flex.1-alpha", allow_patterns=["model_index.json","text_encoder/*","text_encoder_2/*","tokenizer/*","tokenizer_2/*","vae/*","scheduler/*"])
CLD = snapshot_download("thuteam/CLD")
print("CLD files:", sorted(os.path.relpath(f, CLD) for f in glob.glob(f"{CLD}/**/*", recursive=True) if os.path.isfile(f)), flush=True)
ADP = snapshot_download("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha")
pre_fuse = f"{CLD}/pre_trained_LoRA/pytorch_lora_weights.safetensors"
art_lora = f"{CLD}/prism_ft_LoRA/pytorch_lora_weights.safetensors"
cld_lora = f"{CLD}/decouple_LoRA/transformer/pytorch_lora_weights.safetensors"
adp_lora = f"{CLD}/decouple_LoRA/adapter/pytorch_lora_weights.safetensors"
art_pe = f"{CLD}/decouple_LoRA/layer_pe.pth"
def load_any(p):
    return load_file(p) if p.endswith(".safetensors") else torch.load(p, map_location="cpu")

CFG = "/kaggle/working/flux_dev_cfg"; os.makedirs(f"{CFG}/transformer", exist_ok=True)
json.dump({"_class_name":"FluxTransformer2DModel","attention_head_dim":128,"axes_dims_rope":[16,56,56],"guidance_embeds":True,
           "in_channels":64,"joint_attention_dim":4096,"num_attention_heads":24,"num_layers":19,"num_single_layers":38,
           "patch_size":1,"pooled_projection_dim":768}, open(f"{CFG}/transformer/config.json","w"))
dev_sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True, base_shift=0.5,
                                            max_shift=1.15, base_image_seq_len=256, max_image_seq_len=4096)
D0, D1, DT = torch.device("cuda:0"), torch.device("cuda:1"), torch.float16
from models.mmdit import CustomFluxTransformer2DModel
from models.pipeline import CustomFluxPipelineCfgLayer
from models.multiLayer_adapter import MultiLayerAdapter
from models.transp_vae import AutoencoderKLTransformerTraining as CustomVAE

# ---- transformer: GGUF re-classed, layer_pe, three unmerged LoRAs, fp16_mixed ----
t = FluxTransformer2DModel.from_single_file(gguf, config=CFG, subfolder="transformer",
        quantization_config=GGUFQuantizationConfig(compute_dtype=DT), torch_dtype=DT)
t.__class__ = CustomFluxTransformer2DModel
t.register_to_config(max_layer_num=52); t.max_layer_num = 52
pe = load_any(art_pe); pe = pe.get("layer_pe", next(iter(pe.values()))) if isinstance(pe, dict) else pe
t.layer_pe = nn.Parameter(pe.to(torch.float32).view(1, 52, 1, 1, -1), requires_grad=False)
_raw = load_any(art_pe)
print("layer_pe file:", art_pe, "type", type(_raw).__name__, "keys", list(_raw.keys())[:5] if isinstance(_raw, dict) else None,
      "shape", tuple(pe.shape), "norm", round(float(pe.float().norm()), 3), "frame-to-frame diff", round(float((pe.float().view(52, -1)[1] - pe.float().view(52, -1)[0]).abs().mean()), 5), flush=True)
exec(compile(__MIXED__, "fp16_mixed", "exec"))
apply_fp16_mixed(t)
for name, path in (("pre_trained", pre_fuse), ("prism_ft", art_lora), ("decouple", cld_lora)):
    hit, miss = add_lora(t, path, 1.0); print(f"LoRA {name}: matched {hit}, missing {len(miss)}", miss[:3], flush=True)
t.to(D0).eval()
adapter = MultiLayerAdapter.from_pretrained(ADP, torch_dtype=DT)
apply_fp16_mixed(adapter)
hit, miss = add_lora(adapter, adp_lora, 1.0); print("LoRA adapter: matched", hit, "missing", len(miss), miss[:3], flush=True)
adapter.set_layerPE(t.layer_pe, t.max_layer_num)
adapter.to(D1).eval()                     # adapter lives on GPU 1 (text encoders are idle during denoising); inputs hop over, outputs hop back
def _mv(x, d):
    if torch.is_tensor(x): return x.to(d)
    if isinstance(x, (list, tuple)): return type(x)(_mv(i, d) for i in x)
    if isinstance(x, dict):
        y = {k: _mv(v, d) for k, v in x.items()}
        try: return type(x)(**y)
        except TypeError: return y
    return x
adapter.register_forward_pre_hook(lambda m, a, k: (_mv(a, D1), _mv(k, D1)), with_kwargs=True)
adapter.register_forward_hook(lambda m, a, k, out: _mv(out, D0), with_kwargs=True)

pipe = CustomFluxPipelineCfgLayer.from_pretrained(BASE, transformer=t, scheduler=dev_sched, torch_dtype=DT)
pipe.set_multiLayerAdapter(adapter); pipe.vae.to(D0, torch.float32); pipe.text_encoder.to(D1); pipe.text_encoder_2.to(D1)
CustomFluxPipelineCfgLayer._execution_device = property(lambda self: D0)
# PrismLayers was written for diffusers 0.31 (FLUX vae_scale_factor 16); 0.32+ uses 8 -> latents for 2x the size, boxes in one corner
pipe.vae_scale_factor = 16; print('vae_scale_factor ->', pipe.vae_scale_factor, flush=True)
_orig_enc = pipe.encode_prompt
def _enc(*a, device=None, **kw):
    out = _orig_enc(*a, device=D1, **kw)
    return tuple(o.to(D0) if torch.is_tensor(o) else o for o in out)
pipe.encode_prompt = _enc

vae_args = argparse.Namespace(max_layers=48, decoder_arch="vit", pos_embedding="rope", layer_embedding="rope", single_layer_decoder=None)
transp_vae = CustomVAE(vae_args)
_tv = glob.glob(f"{CLD}/trans_vae/*.pt")[0]
print("transparent decoder file:", _tv, flush=True)
tv = load_file(_tv) if _tv.endswith(".safetensors") else torch.load(_tv, map_location="cpu", weights_only=False)
print("transp vae:", transp_vae.load_state_dict(tv.get("model", tv) if isinstance(tv, dict) and "model" in tv else tv, strict=False), flush=True)
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
            t.to("cpu"); adapter.to("cpu"); torch.cuda.empty_cache()
        return _d(*a, **k)
    _v.decode = _offload_then
print("loaded", round(time.time() - _T0), "s; GiB", [round(torch.cuda.memory_allocated(i)/2**30, 1) for i in range(2)], flush=True)

def boxes16(bs):
    return [(((2*x0)//16)*16, ((2*y0)//16)*16, min(1024, (((2*x1)//16)+1)*16), min(1024, (((2*y1)//16)+1)*16)) for x0, y0, x1, y1 in bs]
LAYOUTS = __LAYOUTS__
os.makedirs("/kaggle/working/cld_out", exist_ok=True)
for k, inp in enumerate(LAYOUTS):
    for steps in (2,):
        try:
            t.to(D0); adapter.to(D1); torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0); t0 = time.time()
            with torch.inference_mode():
                x_hat, image, _lat = pipe(prompt=inp["whole_image"]["caption"], adapter_image=Image.open(io.BytesIO(base64.b64decode(inp["img"]))).convert("RGB"), adapter_conditioning_scale=0.9, validation_box=boxes16(inp["layout"]),
                                       generator=torch.Generator(D0).manual_seed(42), height=1024, width=1024,
                                       num_inference_steps=steps, guidance_scale=4.0, num_layers=len(inp["layout"]), sdxl_vae=transp_vae)
            _l = _lat.float().reshape(_lat.shape[0] * _lat.shape[1] if _lat.dim() == 5 else _lat.shape[0], -1) if torch.is_tensor(_lat) else None
            if _l is not None and _l.shape[0] > 1:
                print("final latents per frame: frame0-vs-frame1 diff", round(float((_l[0] - _l[1]).abs().mean()), 4), "frame std", [round(float(x), 3) for x in _l.std(dim=1)[:6]], flush=True)
            x_hat = ((x_hat + 1) / 2).squeeze(0).permute(1, 0, 2, 3).float()
            d = f"/kaggle/working/cld_out/case{k}_{steps}st"; os.makedirs(d, exist_ok=True)
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
src = src.replace("__MIXED__", repr(MIXED)).replace("__LAYOUTS__", json.dumps(LAYOUTS)).replace("__IMG__", IMG_B64)
compile(src, "cell", "exec")
os.makedirs(OUT, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": src.strip("\n").splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(OUT, "cld-1024a-probe.ipynb"), "w", encoding="utf-8"), indent=1)
json.dump({"id": "markandreduesing/cld-1024a-probe", "title": "cld-1024a-probe", "code_file": "cld-1024a-probe.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true",
           "enable_tpu": "false", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
           "competition_sources": [], "kernel_sources": [], "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("wrote", OUT)
