r"""Kaggle 2x T4 notebook: can Qwen-Image-2.1 replace relocate() (LayerDiffuse SDXL) as the 'harmonise after a move' step?

    fastsdcpu/env/Scripts/python.exe prototype/tools/make_qwen21_harmonise_trial.py
    cd prototype/kaggle/upload_qwen21_harmonise_trial && python -m kaggle.cli kernels push -p .

Input: RevealLayer's fox & cottage split (filled background + 5 layers, results_reveallayer_fox_v15 gs1.0_st12), the same one the relocate() move loop
used (docs/concepts/editing.md section 4). The fox is moved 28 % left, 4 % up, 1.15x (case 1) and, as a harder case, 1.6x further down (case 2).
Jobs, all Qwen-Image-2.1 at 512 px, 8 steps:
  control        "keep the picture exactly the same" on the naive paste: how much does the editor change when told to change nothing
  harmonise      whole-scene edit of the naive paste ("blend the pasted fox in: light, colours, soft shadow, keep everything else")  (2 seeds)
  shadow_only    whole-scene edit that adds only a soft cast shadow
  refs           the fox layer and the scene without the fox as two reference images ("place the fox of the first image into the second")
  layer_relight  RGBA edit of the moved fox layer alone ("relight for a sunlit path"), composited over the scene on the CPU afterwards
  harder         the whole-scene harmonise edit for the 1.6x case
Evaluation (CPU, after the run): drift outside the moved fox (PSNR / mean abs diff against the naive paste), fox identity inside its mask, shadow darkening.
"""
import base64
import io
import json
import os
import zipfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
OUT = os.path.join(PROTO, "kaggle", "upload_qwen21_harmonise_trial")
RL = os.path.join(PROTO, "kaggle", "results_reveallayer_fox_v15", "reveal_out", "gs1.0_st12", "fox_cottage")
S = 512
bg = Image.open(f"{RL}/bg_rgba.png").convert("RGBA").resize((S, S))
objs = [Image.open(f"{RL}/layer_{i}_rgba.png").convert("RGBA").resize((S, S)) for i in range(5)]   # houses, cottage, arch, tree, fox
fox = objs[4]
bbox = fox.getchannel("A").point(lambda a: 255 if a > 40 else 0).getbbox()


def move(scale, dx, dy):
    crop = fox.crop(bbox)
    crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS)
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    layer.alpha_composite(crop, (max(0, int(bbox[0] + dx * S)), max(0, int(bbox[1] + dy * S))))
    return layer


def stack(layers):
    c = bg.copy()
    for L in layers:
        c.alpha_composite(L)
    return c


fox1 = move(1.15, -0.28, -0.04)
fox2 = move(1.6, -0.10, 0.02)
nofox = stack(objs[:4])
files = {
    "original": stack(objs).convert("RGB"),
    "naive": stack(objs[:4] + [fox1]).convert("RGB"),
    "naive2": stack(objs[:4] + [fox2]).convert("RGB"),
    "scene_nofox": nofox.convert("RGB"),
}
rgba = {"fox1": fox1, "fox2": fox2}


def enc(im, fmt, **kw):
    b = io.BytesIO()
    im.save(b, fmt, **kw)
    return base64.b64encode(b.getvalue()).decode()


DATA = {k: enc(v, "JPEG", quality=92) for k, v in files.items()}
DATA.update({k: enc(v, "WEBP", quality=92, alpha_quality=100) for k, v in rgba.items()})
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, fs in os.walk(os.path.join(PROTO, "layerproto")):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in fs:
            if fn.endswith(".py"):
                full = os.path.join(root, fn)
                z.write(full, os.path.join("prototype", os.path.relpath(full, PROTO)))
code_b64 = base64.b64encode(buf.getvalue()).decode()

HARM = ("The red fox in this picture was pasted in from another picture. Make it look naturally part of the scene: match the lighting and colours of "
        "the surroundings, add a soft cast shadow on the path under the fox, and keep every other part of the picture exactly the same, including the fox's "
        "shape, pose and markings.")
JOBS = [
    ("control", ["naive"], "Keep the picture exactly the same. Do not change anything.", 42, False),
    ("harmonise_s42", ["naive"], HARM, 42, False),
    ("harmonise_s7", ["naive"], HARM, 7, False),
    ("shadow_only", ["naive"], "Add only a soft cast shadow of the red fox on the cobblestone path below it. Keep everything else exactly the same.", 42, False),
    ("refs", ["fox1", "scene_nofox"], "Place the red fox from the first image into the second image exactly where it is in the first image: same position, same size, same pose. "
     "Light it like the second image and add a soft shadow on the path. Keep the rest of the second image exactly the same.", 42, False),
    ("layer_relight", ["fox1"], "Relight this fox as if it stands on a sunlit cobblestone path in a sunny meadow, soft light from the upper left, keep its exact shape, pose and markings", 42, True),
    ("harder_control", ["naive2"], "Keep the picture exactly the same. Do not change anything.", 42, False),
    ("harder_harmonise", ["naive2"], HARM, 42, False),
]
RGBA = "This is an RGBA image with transparency. {}. The image has alpha channel and the background is transparent."

SRC = f'''
import signal, time as _t
_T0 = _t.time()
def _cap(*a): raise TimeoutError("40 min cap")
signal.signal(signal.SIGALRM, _cap); signal.alarm(40 * 60)
import base64, io, zipfile, subprocess, sys, os, time, json
os.environ["HF_HUB_DISABLE_XET"] = "1"
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "transformers>=5.17", "gguf>=0.10", "accelerate", "git+https://github.com/huggingface/diffusers"], check=False)
os.makedirs("/kaggle/working/code", exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode("{code_b64}"))).extractall("/kaggle/working/code")
sys.path.insert(0, "/kaggle/working/code/prototype")
from PIL import Image
from layerproto.serve import qwen21
import torch
pipe = qwen21.load()
DATA = {json.dumps(DATA)}
img = lambda k: Image.open(io.BytesIO(base64.b64decode(DATA[k])))
RGBA = {json.dumps(RGBA)}
os.makedirs("/kaggle/working/harm", exist_ok=True)
for k in DATA:
    img(k).convert("RGBA" if k.startswith("fox") else "RGB").save(f"/kaggle/working/harm/input_{{k}}.png")
results = []
for name, ins, prompt, seed, rgba in {JOBS!r}:
    row = dict(job=name, inputs=ins, seed=seed)
    try:
        images = [img(k).convert("RGBA" if k.startswith("fox") else "RGB") for k in ins]
        t0 = time.time()
        with torch.inference_mode():
            out = pipe(prompt=RGBA.format(prompt) if rgba else prompt, image=images, width=512, height=512, output_resolution=512,
                       use_kv_cache=True, generator=torch.Generator(pipe._execution_device).manual_seed(seed)).images[0]
        row["compute_s"] = round(time.time() - t0, 1)
        out.save(f"/kaggle/working/harm/{{name}}.png")
        row["mode"] = out.mode
    except Exception as e:
        import traceback; traceback.print_exc(); row["error"] = repr(e)[:300]
    print(json.dumps(row), flush=True); results.append(row)
    json.dump(results, open("/kaggle/working/harm/results.json", "w"), indent=1)
print("TRIAL DONE", round(time.time() - _T0), "s", flush=True)
'''
compile(SRC, "cell", "exec")
os.makedirs(OUT, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": SRC.strip("\n").splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(OUT, "qwen21-harmonise-trial.ipynb"), "w", encoding="utf-8"), indent=1)
json.dump({"id": "markandreduesing/qwen21-harmonise-trial", "title": "qwen21-harmonise-trial", "code_file": "qwen21-harmonise-trial.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
           "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("wrote", OUT, round(os.path.getsize(os.path.join(OUT, "qwen21-harmonise-trial.ipynb")) / 1e6, 2), "MB")
