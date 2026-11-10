r"""Kaggle 2x T4 notebook: Qwen-Image-2.1-Turbo (7B, native RGBA VAE) at reduced settings.

    fastsdcpu\env\Scripts\python.exe prototype\tools\make_qwen21_rgba_trial.py
    cd prototype\kaggle\upload_qwen21_drift_trial && python -m kaggle.cli kernels push -p .

Why: Qwen-Image-2.1 generates RGBA from text, edits transparent layers and extracts subjects (model card,
2026). That covers three steps of our loop (new layer, re-prompt a layer, split an object out) in one model.

Setup (facts checked 2026-11-09):
  * transformer: unsloth/Qwen-Image-2.1-Turbo-GGUF Q4_K_S (3.91 GB); diffusers main maps
    QwenImage21Transformer2DModel for from_single_file. Turbo = 8 steps, CFG 1, schedule saved in the checkpoint.
  * rest (Qwen3-VL text encoder ~17.5 GB bf16, 4-channel VAE 1.35 GB, processor, scheduler) from
    Qwen/Qwen-Image-2.1-Turbo; text encoder in fp16 split over both T4s via device_map.
  * our fp16_mixed on the transformer (Qwen-family models overflow plain fp16).
  * 512 px instead of 2048 (VAE is 16x, so 32x32 = 1024 image tokens).
Qwen Research Licence (non-commercial) applies.
"""
import base64
import io
import json
import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
ROOT = os.path.dirname(PROTO)
OUT = os.path.join(PROTO, "kaggle", "upload_qwen21_drift_trial")
SCENE = "prototype/test_scene/fox_cottage.png"   # shared test scene

buf = io.BytesIO()
Image.open(os.path.join(ROOT, SCENE)).convert("RGB").resize((512, 512)).save(buf, "JPEG", quality=90)
SCENE_B64 = base64.b64encode(buf.getvalue()).decode()
MIXED = open(os.path.join(PROTO, "layerproto", "fp16_mixed.py"), encoding="utf-8").read()

src = r'''import signal, time as _t
_T0 = _t.time()
def _cap(*a): raise TimeoutError("25 min cap hit after %.0fs" % (_t.time() - _T0))
signal.signal(signal.SIGALRM, _cap); signal.alarm(25 * 60)

import subprocess, os, sys, json, time, base64, io
def sh(c): print(c, flush=True); subprocess.run(c, shell=True, check=True)
sh("pip -q install 'transformers>=5.17' gguf>=0.10 accelerate git+https://github.com/huggingface/diffusers")
import torch, numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download
import diffusers
from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel, GGUFQuantizationConfig
print("diffusers", diffusers.__version__, "torch", torch.__version__, flush=True)

REPO = "Qwen/Qwen-Image-2.1-Turbo"
BASE = snapshot_download(REPO, ignore_patterns=["transformer/*.safetensors", "vae/*.safetensors", "assets/*"])
gguf = hf_hub_download("unsloth/Qwen-Image-2.1-Turbo-GGUF", "qwen-image-2.1-turbo-Q4_K_S.gguf")
print("downloads done", round(time.time() - _T0), "s", flush=True)
D0, DT = torch.device("cuda:0"), torch.float16

exec(compile(__MIXED__, "fp16_mixed", "exec"))
try:
    t = QwenImage21Transformer2DModel.from_single_file(gguf, config=BASE, subfolder="transformer",
            quantization_config=GGUFQuantizationConfig(compute_dtype=DT), torch_dtype=DT)
    print("transformer: GGUF Q4_K_S", flush=True)
except Exception as e:  # diffusers GGUF loading for 2.1 is unconfirmed; NF4 is reported working (15 GB peak on a 5090 at 1024)
    import traceback; traceback.print_exc()
    sh("pip -q install bitsandbytes")
    from diffusers import BitsAndBytesConfig
    t = QwenImage21Transformer2DModel.from_pretrained(snapshot_download(REPO, allow_patterns=["transformer/*"]), subfolder="transformer",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DT), torch_dtype=DT)
    print("transformer: NF4 fallback", flush=True)
# diffusers' GGUF loader only dequantizes nn.Linear; other GGUF params keep packed bytes (run 1: 4096 vs 8192)
from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter, dequantize_gguf_tensor
_fixed = 0
for _m in t.modules():
    for _n, _p in list(_m.named_parameters(recurse=False)):
        if isinstance(_m, GGUFLinear) and _n == "weight":
            continue
        if isinstance(_p, GGUFParameter):
            setattr(_m, _n, torch.nn.Parameter(dequantize_gguf_tensor(_p).to(DT), requires_grad=False)); _fixed += 1
print("dequantized non-linear GGUF params:", _fixed, flush=True)
apply_fp16_mixed(t); t.to(D0).eval()

from transformers import AutoModelForImageTextToText
te = AutoModelForImageTextToText.from_pretrained(f"{BASE}/text_encoder", torch_dtype=DT, device_map="auto",
        max_memory={0: "5GiB", 1: "14GiB", "cpu": "24GiB"})
# stock 2.1 VAE gives NaNs in fp16, mostly in fully transparent regions; texture-fix VAE is fp16-safe drop-in
from diffusers import AutoencoderKLQwenImage21
vae = AutoencoderKLQwenImage21.from_pretrained("madebyollin/texture-fix-vae-for-qwen-image-2.1", torch_dtype=DT)
pipe = QwenImage21Pipeline.from_pretrained(BASE, transformer=t, text_encoder=te, vae=vae, torch_dtype=DT)
pipe.vae.to(D0)
_ep = pipe.encode_prompt
def _checked_encode(*a, **k):
    out = _ep(*a, **k)
    for o in (out if isinstance(out, tuple) else (out,)):
        if torch.is_tensor(o) and o.is_floating_point() and not torch.isfinite(o).all():
            print("[warn] non-finite prompt embeddings from the fp16 text encoder", flush=True)
    return out
pipe.encode_prompt = _checked_encode
type(pipe)._execution_device = property(lambda self: D0)
print("loaded", round(time.time() - _T0), "s; GiB", [round(torch.cuda.memory_allocated(i)/2**30, 1) for i in range(2)], flush=True)

os.makedirs("/kaggle/working/q21_out", exist_ok=True)
def alpha_stats(im):
    if im.mode != "RGBA": return f"mode {im.mode}"
    a = np.asarray(im.getchannel("A")); return f"RGBA, transparent {(a < 16).mean()*100:.0f}%, opaque {(a > 240).mean()*100:.0f}%"
def run(name, prompt, image=None, size=512, kv=True):
    try:
        torch.cuda.reset_peak_memory_stats(0); t0 = time.time()
        with torch.inference_mode():
            im = pipe(prompt=prompt, image=image, width=size, height=size, output_resolution=size, use_kv_cache=kv,
                      generator=torch.Generator(D0).manual_seed(42)).images[0]
        im.save(f"/kaggle/working/q21_out/{name}.png")
        print(f"{name}: {time.time()-t0:.0f}s, {alpha_stats(im)}, peak GiB {torch.cuda.max_memory_allocated(0)/2**30:.1f}", flush=True)
        return im
    except BaseException as e:
        import traceback; traceback.print_exc(); print("FAILED", name, flush=True)
        if isinstance(e, TimeoutError): raise

scene = Image.open(io.BytesIO(base64.b64decode("__SCENE__"))).convert("RGB")
os.makedirs("/kaggle/working/drift_B", exist_ok=True)
scene.save("/kaggle/working/drift_B/t0_composite.png")
TURNS = __TURNS__
cur = scene
for t, text in enumerate(TURNS, 1):
    t0 = time.time()
    try:
        with torch.inference_mode():
            nxt = pipe(prompt=text + " Keep every other object, the composition and the painting style exactly the same.", image=[cur],
                       width=512, height=512, output_resolution=512, use_kv_cache=True, generator=torch.Generator(D0).manual_seed(42)).images[0]
        cur = nxt.convert("RGB")
        cur.save(f"/kaggle/working/drift_B/t{t}_composite.png")
        print(f"turn {t}: {time.time()-t0:.0f}s", flush=True)
    except BaseException as e:
        import traceback; traceback.print_exc(); print("FAILED turn", t, flush=True)
        if isinstance(e, TimeoutError): raise
        break
print("total", round(time.time() - _T0), "s", flush=True)
'''
import json as _j
_cap = _j.load(open(os.path.join(PROTO, "test_scene", "fox_cottage.json"), encoding="utf-8"))["caption"]
_turns = [t["text"] for t in json.load(open(os.path.join(HERE, "drift_edits.json")))["turns"]]
src = src.replace("__TURNS__", json.dumps(_turns)).replace("__MIXED__", repr(MIXED)).replace("__SCENE__", SCENE_B64).replace("__CAPTION__", _cap.replace('"', "'"))
compile(src, "cell", "exec")

os.makedirs(OUT, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": src.strip("\n").splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(OUT, "qwen21-drift-trial.ipynb"), "w", encoding="utf-8"), indent=1)
json.dump({"id": "markandreduesing/qwen21-drift-trial", "title": "qwen21-drift-trial",
           "code_file": "qwen21-drift-trial.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
           "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [],
           "kernel_sources": [], "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("wrote", OUT, round(os.path.getsize(os.path.join(OUT, "qwen21-drift-trial.ipynb")) / 1e6, 2), "MB")
