"""Builds the Kaggle notebook that tests Qwen-Image-Layered (4-bit GGUF) on a free T4.

    fastsdcpu\\env\\Scripts\\python.exe prototype\\tools\\make_qwen_layered_test.py [image.png]

The notebook writes prototype/layerproto/qwen_layered.py (embedded verbatim) and calls it, so it runs exactly
the code that tools/dryrun_qwen_layered.py verifies locally with tiny models. Only the real-weight loads differ:
    stage 1: Qwen2.5-VL-7B text encoder in 8-bit (bitsandbytes) on cuda:0
    stage 2: 20B transformer as unsloth GGUF Q4_K_S (fallback Q3_K_M) on cuda:0, fp16 compute
"""
import base64
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
ROOT = os.path.dirname(PROTO)
img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "experiments", "blkE__P2__textonly__sdxlturbo-steps4-g1.0__seed11__impressionist.png")
CAPTION = ("An impressionist oil painting of a forest clearing at night: tall pine trees and bare trees, a full moon and "
           "stars in a blue sky, distant mountains, a campfire with logs in the middle, a wooden chair next to it, "
           "rocks and fallen branches on the ground.")
b64 = base64.b64encode(open(img_path, "rb").read()).decode()
# extra test images (different styles / content) for generalisation checks
EXTRA = {
    "pixelart_camp": (os.path.join(ROOT, "experiments", "P3__ph4__canny-raw__neutral-scale70-steps6-g1.0__seed7__pixelart.png"),
                      "A colourful pixel-art night scene: a skeleton sitting next to a small brick campfire, stylised white, yellow and "
                      "green trees, a big yellow moon and a small orange moon in a dark blue starry sky, brown ground."),
    "sketch_skeleton": (os.path.join(ROOT, "experiments", "blkB__P2__raw__str80-steps4-g1.0__seed11__hyperreal.png"),
                        "A pale illustration of a skeleton sitting at a small wooden table next to a campfire with logs and stones, "
                        "two cut logs, misty pine trees and a few stars, all inside a thin circle on a white background."),
}
def _jpeg_b64(f):   # JPEG keeps the notebook small (the Kaggle API rejected a 1.5 MB notebook with PNGs)
    import io
    from PIL import Image
    buf = io.BytesIO(); Image.open(f).convert("RGB").save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()
EXTRA_B64 = {k: (_jpeg_b64(f), c) for k, (f, c) in EXTRA.items()}
module_src = open(os.path.join(PROTO, "layerproto", "qwen_layered.py"), encoding="utf-8").read()
mixed_src = open(os.path.join(PROTO, "layerproto", "fp16_mixed.py"), encoding="utf-8").read()
WORK = "/kaggle/working"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}


MIXED_CELL = """
# fp32 residual stream + fp16 tensor-core matmuls: see layerproto/fp16_mixed.py
import gc, sys
sys.last_traceback = sys.last_value = sys.last_type = None
from layerproto.fp16_mixed import apply_fp16_mixed
print('linears patched, params moved to fp32:', apply_fp16_mixed(tr), flush=True)
gc.collect(); torch.cuda.empty_cache(); split_across_gpus(tr)
for _i in range(torch.cuda.device_count()):
    torch.cuda.reset_peak_memory_stats(_i)
Q.decode_latents, pipe.vae.decode = _dl, _vd
try:
    _s7, _st7 = Q.decompose(pipe, tr, img, CAPTION, emb, steps=2, layers=4, resolution=640)
finally:
    Q.decode_latents, pipe.vae.decode = _orig_dl, _orig_vd
print(_st7, '| peak GB per GPU:', [round(torch.cuda.max_memory_allocated(i) / 1e9, 2) for i in range(torch.cuda.device_count())], flush=True)
Q.strip([img] + _s7).save(f'{OUT}/mixed_smoke_strip.png'); display(Q.strip([img] + _s7))
"""
cells = [
    md("""
# Feasibility test: Qwen-Image-Layered (20B) on a free Kaggle T4

Same code as the local dry run (`layerproto/qwen_layered.py`), with the real weights:
8-bit text encoder (stage 1, then unloaded) and the 4-bit GGUF transformer (stage 2), one T4 at a time.
"""),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install "diffusers==0.37.0" "transformers>=4.51" accelerate gguf
import importlib.metadata as m
print({p: m.version(p) for p in ['torch', 'diffusers', 'transformers', 'gguf', 'accelerate']})
"""),
    code(f"import os, sys\nWORK = {json.dumps(WORK)}\nos.makedirs(WORK + '/layerproto', exist_ok=True)\n"
         "open(WORK + '/layerproto/__init__.py', 'w').close()\n"
         f"open(WORK + '/layerproto/qwen_layered.py', 'w', encoding='utf-8').write({json.dumps(module_src)})\n"
         f"open(WORK + '/layerproto/fp16_mixed.py', 'w', encoding='utf-8').write({json.dumps(mixed_src)})\n"
         "sys.path.insert(0, WORK)\nfrom layerproto import qwen_layered as Q\nprint('module ok')"),
    code(f"""
import base64, io, json, time, gc
import torch
from PIL import Image
os.environ['HF_HUB_DISABLE_XET'] = '1'
OUT = WORK + '/qwen_layered_test'; os.makedirs(OUT, exist_ok=True)
CAPTION = {json.dumps(CAPTION)}
img = Image.open(io.BytesIO(base64.b64decode('{b64}'))).convert('RGB'); img.save(f'{{OUT}}/input.png')
EXTRA = {{k: (Image.open(io.BytesIO(base64.b64decode(b))).convert('RGB'), c) for k, (b, c) in {json.dumps(EXTRA_B64)}.items()}}
DT = torch.float16
import glob
_g = glob.glob('/kaggle/input/**/qwen-image-layered-Q4_K_S.gguf', recursive=True)
_b = glob.glob('/kaggle/input/**/Qwen-Image-Layered/model_index.json', recursive=True)
GGUF = _g[0] if _g else None                         # cached transformer (kernel output), else Hugging Face
SRC = os.path.dirname(_b[0]) if _b else Q.REPO       # cached text encoder / VAE / configs, else Hugging Face
print('transformer from:', GGUF or 'Hugging Face', '| base from:', SRC, flush=True)
stats = {{'weights_cached': bool(GGUF and _b)}}
def peak(tag):
    stats[tag + '_peak_GB'] = [round(torch.cuda.max_memory_allocated(i) / 1e9, 2) for i in range(torch.cuda.device_count())]
    print(tag, 'peak GB per GPU:', stats[tag + '_peak_GB'], flush=True)
def save():
    json.dump(stats, open(f'{{OUT}}/stats.json', 'w'), indent=1)
"""),
    md("## Input read-speed check (seconds of GPU time; stops the run if the cached weights are still slow)"),
    code("""
_sh = sorted(glob.glob('/kaggle/input/**/Qwen-Image-Layered/text_encoder/*.safetensors', recursive=True))
if _sh:
    _f = max(_sh, key=os.path.getsize); _t = time.time(); _n = 0
    with open(_f, 'rb', buffering=0) as _fh:
        while time.time() - _t < 10:
            _b = _fh.read(64 * 1024 * 1024)
            if not _b:
                break
            _n += len(_b)
    stats['input_read_MBps'] = round(_n / 1e6 / max(time.time() - _t, 1e-3), 1)
    print('cached text encoder read speed:', stats['input_read_MBps'], 'MB/s', flush=True); save()
    assert stats['input_read_MBps'] > 100, 'Cached inputs read too slowly (still being copied?) - stop the session and retry later'
else:
    print('no cached text encoder attached - will download from Hugging Face')
"""),
    md("## Stage 1: fp16 text encoder split over both T4s -> prompt embeddings, then unload"),
    code("""
from transformers import Qwen2_5_VLForConditionalGeneration
from diffusers import AutoencoderKLQwenImage
import shutil
# transformers 5 loads tensors with many parallel threads; on the /kaggle/input mount those reads stalled at
# 0/729 for minutes (single sequential reads are fast). Copy the base weights to local disk once per session.
t = time.time()
LOCAL = '/tmp/Qwen-Image-Layered'
os.system('df -h /tmp | tail -n 1')
if SRC.startswith('/kaggle/input'):
    shutil.copytree(SRC, LOCAL, dirs_exist_ok=True)
    SRC = LOCAL
stats['copy_to_local_s'] = round(time.time() - t, 1)
print('base weights on local disk after', stats['copy_to_local_s'], 's ->', SRC, flush=True)
t = time.time()
# no bitsandbytes: its 8-bit load sat at 0/729 tensors for minutes; fp16 (~16.6 GB) fits across two T4s
te = Qwen2_5_VLForConditionalGeneration.from_pretrained(SRC, subfolder='text_encoder', torch_dtype=DT,
        device_map='auto', max_memory={0: '13GiB', 1: '14GiB'})
print('text encoder loaded in', round(time.time() - t, 1), 's', flush=True)
vae = AutoencoderKLQwenImage.from_pretrained(SRC, subfolder='vae', torch_dtype=DT).to('cuda:0')
pipe = Q.make_pipeline(te, vae, SRC)
stats['stage1_load_s'] = round(time.time() - t, 1); t = time.time()
emb = tuple(x.to('cuda:0') for x in Q.encode(pipe, CAPTION, device='cuda:0'))
emb_x = {k: tuple(x.to('cuda:0') for x in Q.encode(pipe, c, device='cuda:0')) for k, (_, c) in EXTRA.items()}
stats['stage1_encode_s'] = round(time.time() - t, 1); peak('stage1')
del te; Q.unload_text_encoder(pipe, DT); torch.cuda.reset_peak_memory_stats()
print('after unload, GB allocated:', round(torch.cuda.memory_allocated(0) / 1e9, 2)); save(); print(stats)
"""),
    md("## Stage 2: 4-bit GGUF transformer. Smoke test (2 steps incl. decode) first, then 50 steps"),
    code("""
from huggingface_hub import hf_hub_download
from diffusers import QwenImageTransformer2DModel, GGUFQuantizationConfig
t = time.time()
tr = QwenImageTransformer2DModel.from_single_file(
    GGUF or hf_hub_download('unsloth/Qwen-Image-Layered-GGUF', 'qwen-image-layered-Q4_K_S.gguf'),
    quantization_config=GGUFQuantizationConfig(compute_dtype=DT), torch_dtype=DT,
    config=SRC, subfolder='transformer').to('cuda:0')
stats['gguf_params_dequantized'] = Q.fix_gguf_non_linear(tr, DT)  # BF16 embedding etc. (see module)
stats['transformer_load_s'] = round(time.time() - t, 1)
print('loaded in', stats['transformer_load_s'], 's; GB allocated', round(torch.cuda.memory_allocated(0) / 1e9, 2), flush=True)
# smoke test: whole path incl. decode with 2 steps (~1 min). Stops here if memory does not fit.
smoke, st = Q.decompose(pipe, tr, img, CAPTION, emb, steps=2, layers=4, resolution=640)
stats['smoke'] = st; peak('smoke'); print('SMOKE OK', st, flush=True); save()
Q.strip([img] + smoke).save(f'{OUT}/smoke_strip.png')
"""),
    md("### NaN check: fp16 overflow in the transformer (latents) or in the VAE (decoded pixels)?"),
    code("""
def _report(tag, x):
    x = x.detach().float()
    print(tag, 'nan:', torch.isnan(x).any().item(), 'inf:', torch.isinf(x).any().item(),
          'absmax:', round(x.nan_to_num(0, 0, 0).abs().max().item(), 2), flush=True)
_orig_dl, _orig_vd = Q.decode_latents, pipe.vae.decode
def _dl(p, lat, *a, **k):
    _report('latents', lat); return _orig_dl(p, lat, *a, **k)
def _vd(z, *a, **k):
    _report('vae input', z); o = _orig_vd(z, *a, **k); _report('vae output', o[0]); return o
Q.decode_latents, pipe.vae.decode = _dl, _vd
try:
    tr.to('cuda:0'); _s2, _ = Q.decompose(pipe, tr, img, CAPTION, emb, steps=2, layers=4, resolution=640)
finally:
    Q.decode_latents, pipe.vae.decode = _orig_dl, _orig_vd
"""),
    code("""
# where do NaNs start, and what would the alternatives to fp16 cost on a T4?
for i, e in enumerate(emb):
    if e is not None and e.is_floating_point():
        _report(f'emb[{i}]', e)
print('torch.cuda.is_bf16_supported():', torch.cuda.is_bf16_supported(), flush=True)
for _dt in (torch.float16, torch.bfloat16, torch.float32):
    a = torch.randn(3072, 3072, device='cuda:0', dtype=_dt); a @ a; torch.cuda.synchronize(); _t = time.time()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize(); print(_dt, 'TFLOPS', round(20 * 2 * 3072 ** 3 / (time.time() - _t) / 1e12, 2), flush=True)
del a
"""),
    code("""
# fp16 overflow probe: the first submodule (in execution order) whose output contains inf/nan, then stop
class _Found(Exception):
    pass
def _absmax(ts):
    return [round(t.detach().float().nan_to_num(0, 0, 0).abs().max().item(), 1) for t in ts if torch.is_tensor(t) and t.is_floating_point()]
def _mk(name):
    def _hook(m, i, o):
        outs = o if isinstance(o, tuple) else (o,)
        for t in outs:
            if torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all():
                kind = 'nan' if torch.isnan(t).any() else 'inf'
                raise _Found(f'{name} ({type(m).__name__}) -> {kind}; input absmax {_absmax(i)}; output absmax {_absmax(outs)}')
    return _hook
_hooks = [m.register_forward_hook(_mk(n)) for n, m in tr.named_modules() if n]
try:
    (split_across_gpus(tr) if 'split_across_gpus' in globals() else tr.to('cuda:0'))
    Q.decompose(pipe, tr, img, CAPTION, emb, steps=2, layers=4, resolution=640)
    print('no inf/nan found', flush=True)
except _Found as e:
    print('FIRST NON-FINITE:', e, flush=True)
finally:
    for h in _hooks:
        h.remove()
    tr.to('cpu'); torch.cuda.empty_cache()
"""),
    code("""
# fp16 fix: the MLP output projections overflow (block 0 img_mlp.net.2: input ~8.9e3 -> output > 65504).
# Run them on a down-scaled input (still fp16 tensor cores), rescale in fp32, saturate at the fp16 limit.
import types, gc
gc.collect(); torch.cuda.empty_cache()
def _safe_forward(self, x, _S=64.0):
    # all fp16 and in place (fp32 temporaries OOM'd next to the 12.7 GB weights): W(x/S)+b never overflows;
    # rescaling may hit inf, which clamp turns into +-65504 (only inf-inf would make NaN, and that cannot happen here)
    y = self._orig_forward(x / _S)                              # = W x / S + b
    b = self.bias if getattr(self, 'bias', None) is not None else None
    if b is not None:
        y -= b
    y.mul_(_S).clamp_(-65504, 65504)
    if b is not None:
        y.add_(b).clamp_(-65504, 65504)
    return y
_patched = 0
for _n, _m in tr.named_modules():
    # every linear layer inside the blocks (block 1 attn.to_out.0 overflowed next: input ~1.8e4)
    if _n.startswith('transformer_blocks.') and 'Linear' in type(_m).__name__:
        if not hasattr(_m, '_orig_forward'):
            _m._orig_forward = _m.forward
        _m.forward = types.MethodType(_safe_forward, _m); _patched += 1     # (re)bind the current version
print('patched linear layers in blocks:', _patched, flush=True)
Q.decode_latents, pipe.vae.decode = _dl, _vd
try:
    tr.to('cuda:0'); _s4, _st4 = Q.decompose(pipe, tr, img, CAPTION, emb, steps=2, layers=4, resolution=640)
finally:
    Q.decode_latents, pipe.vae.decode = _orig_dl, _orig_vd
print(_st4, flush=True)
"""),
    md("### Use both T4s: blocks 0-29 on GPU 0, blocks 30-59 on GPU 1 (one GPU has no headroom left for activations)"),
    code("""
import sys, gc
sys.last_traceback = sys.last_value = sys.last_type = None   # a stored OOM traceback keeps GPU tensors alive
def _move(x, d):
    if torch.is_tensor(x):
        return x.to(d, non_blocking=True)
    if isinstance(x, dict):
        return {k: _move(v, d) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_move(v, d) for v in x)
    return x
def _to_own_device(m, args, kwargs):
    d = next(m.parameters()).device
    return _move(args, d), _move(kwargs, d)
def split_across_gpus(model, devices=('cuda:0', 'cuda:1')):
    blocks = model.transformer_blocks
    half = len(blocks) // 2
    for name, child in model.named_children():
        if name != 'transformer_blocks':
            child.to(devices[0])
    for i, b in enumerate(blocks):
        b.to(devices[0] if i < half else devices[1])
    if not getattr(model, '_split_hooks', False):
        for m in list(blocks) + [model.norm_out, model.proj_out]:
            m.register_forward_pre_hook(_to_own_device, with_kwargs=True)
        model._split_hooks = True
    return model
tr.to('cpu'); gc.collect(); torch.cuda.empty_cache()
for _i in range(torch.cuda.device_count()):
    torch.cuda.reset_peak_memory_stats(_i)
split_across_gpus(tr)
print('GB allocated per GPU after split:', [round(torch.cuda.memory_allocated(i) / 1e9, 2) for i in range(torch.cuda.device_count())], flush=True)
Q.decode_latents, pipe.vae.decode = _dl, _vd
try:
    _s5, _st5 = Q.decompose(pipe, tr, img, CAPTION, emb, steps=2, layers=4, resolution=640)
finally:
    Q.decode_latents, pipe.vae.decode = _orig_dl, _orig_vd
print(_st5, '| peak GB per GPU:', [round(torch.cuda.max_memory_allocated(i) / 1e9, 2) for i in range(torch.cuda.device_count())], flush=True)
"""),
    md("### LayerNorms in fp32 (block 1 img_norm2 got a saturated fp16 input and its variance overflowed to NaN)"),
    code("""
import torch.nn.functional as F
def _ln_fp32(self, x):
    w = self.weight.float() if self.weight is not None else None
    b = self.bias.float() if self.bias is not None else None
    return F.layer_norm(x.float(), self.normalized_shape, w, b, self.eps).to(x.dtype)
_n_ln = 0
for _n, _m in tr.named_modules():
    if isinstance(_m, torch.nn.LayerNorm):
        _m.forward = types.MethodType(_ln_fp32, _m); _n_ln += 1
print('LayerNorms now computed in fp32:', _n_ln, flush=True)
gc.collect(); torch.cuda.empty_cache(); split_across_gpus(tr)
Q.decode_latents, pipe.vae.decode = _dl, _vd
try:
    _s6, _st6 = Q.decompose(pipe, tr, img, CAPTION, emb, steps=2, layers=4, resolution=640)
finally:
    Q.decode_latents, pipe.vae.decode = _orig_dl, _orig_vd
print(_st6, flush=True)
Q.strip([img] + _s6).save(f'{OUT}/fp16fix_smoke_strip.png')
"""),
    md("### Mixed precision: fp32 residual stream, fp16 tensor-core matmuls (ComfyUI also lists Qwen-Image as bf16/fp32 only)"),
    code(MIXED_CELL),
    code("""
torch.cuda.reset_peak_memory_stats(); split_across_gpus(tr)   # decompose() moved it to CPU RAM for decoding
result, st = Q.decompose(pipe, tr, img, CAPTION, emb, steps=50, layers=4, resolution=640,
                         on_step=lambda i: print('step', i, time.strftime('%H:%M:%S'), flush=True) if i % 5 == 0 else None)
stats.update(st); peak('stage2'); save(); print(json.dumps(stats, indent=1))
"""),
    md("## Results"),
    code("""
import numpy as np
canvas, info = Q.recompose(result, img.size)
for i, L in enumerate(result):
    L.save(f'{OUT}/layer_{i}.png')
canvas.save(f'{OUT}/recomposite.png')
stats['layers'] = info
stats['recomposite_mean_abs_diff'] = round(float(np.abs(np.array(canvas.convert('RGB'), dtype=int) - np.array(img, dtype=int)).mean()), 2)
s = Q.strip([img, canvas] + result); s.save(f'{OUT}/strip.png'); display(s)
save(); print(json.dumps(stats, indent=1))
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out_dir = os.path.join(PROTO, "kaggle", "upload_qwen_test")
os.makedirs(out_dir, exist_ok=True)
json.dump(nb, open(os.path.join(out_dir, "qwen-layered-feasibility.ipynb"), "w", encoding="utf-8"), indent=1)
json.dump({"id": "markandreduesing/qwen-layered-feasibility", "title": "qwen-layered-feasibility",
           "code_file": "qwen-layered-feasibility.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
           "machine_shape": "", "dataset_sources": [], "competition_sources": [], "kernel_sources": ["markandreduesing/qwen-layered-weights-transformer",
                              "markandreduesing/qwen-layered-weights-base"],
           "model_sources": []}, open(os.path.join(out_dir, "kernel-metadata.json"), "w"), indent=1)
print("wrote", out_dir)
