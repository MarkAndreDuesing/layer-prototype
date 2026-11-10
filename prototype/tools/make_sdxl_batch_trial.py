r"""Kaggle T4 notebook: amodal completion of the cottage (SDXL-inpaint) + LayerDiffuse effect layers, one session (tools/sdxl_batch_demo.py).

    fastsdcpu/env/Scripts/python.exe prototype/tools/make_sdxl_batch_trial.py
    cd prototype/kaggle/upload_sdxl_batch_trial && python -m kaggle.cli kernels push -p .

GPU-sparing: all masks and inputs are computed here on the CPU from the saved SAM2 layers (results_segment_inpaint_v3);
the notebook only loads SDXL-inpaint, then the LayerDiffuse chain, and runs 2 + 3 short generations.
"""
import base64
import glob
import io
import json
import os
import zipfile

import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
OUT = os.path.join(PROTO, "kaggle", "upload_sdxl_batch_trial")
SAM = os.path.join(PROTO, "kaggle", "results_segment_inpaint_v3", "seginp_v3", "layers_A_hand_boxes")
J = json.load(open(os.path.join(PROTO, "test_scene", "fox_cottage.json"), encoding="utf-8"))
BOX = [l for l in J["layers_back_to_front"] if l["name"] == "cottage"][0]["box"]

L = {os.path.basename(f).split("_", 1)[1].rsplit(".", 1)[0]: Image.open(f).convert("RGBA") for f in sorted(glob.glob(os.path.join(SAM, "*.png")))}
alpha = lambda im: np.asarray(im.getchannel("A")) > 40
cottage_a = alpha(L["cottage"])
occ = alpha(L["stone_gate_arch"]) | alpha(L["big_tree"]) | alpha(L["red_fox"])
box = np.zeros_like(cottage_a)
box[BOX[1]:BOX[3], BOX[0]:BOX[2]] = True
hidden = box & occ & ~cottage_a                           # part of the cottage's box that the front objects cover
hidden_img = Image.fromarray((hidden * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(1.5))
print("hidden share of the canvas:", round(float(hidden.mean()), 3))


def png(im):
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def webp(im):
    b = io.BytesIO()
    im.save(b, "WEBP", quality=92, alpha_quality=100)
    return b.getvalue()


def jpg(im):
    b = io.BytesIO()
    im.convert("RGB").save(b, "JPEG", quality=92)
    return b.getvalue()


buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for top in ("layerproto", os.path.join("vendor", "diffuser_layerdiffuse", "layer_diffuse")):
        for root, dirs, files in os.walk(os.path.join(PROTO, top)):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn in files:
                if fn.endswith(".py"):
                    full = os.path.join(root, fn)
                    z.write(full, os.path.join("prototype", os.path.relpath(full, PROTO)))
    z.write(os.path.join(HERE, "sdxl_batch_demo.py"), "prototype/tools/sdxl_batch_demo.py")
    z.writestr("data/bg.png", jpg(L["background"]))
    z.writestr("data/cottage.png", webp(L["cottage"]))
    z.writestr("data/tree.png", webp(L["big_tree"]))
    z.writestr("data/hidden_mask.png", webp(hidden_img.convert("RGBA")))
    z.writestr("data/scene.png", jpg(Image.open(os.path.join(PROTO, "test_scene", "fox_cottage.png"))))
code_b64 = base64.b64encode(buf.getvalue()).decode()

SRC = f'''
import signal, time as _t
_T0 = _t.time()
def _cap(*a): raise TimeoutError("25 min cap")
signal.signal(signal.SIGALRM, _cap); signal.alarm(25 * 60)
import base64, io, zipfile, subprocess, sys, os
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'diffusers==0.37.0'], check=False)
os.makedirs('/kaggle/working/code', exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode("{code_b64}"))).extractall('/kaggle/working/code')
os.chdir('/kaggle/working/code')
r = subprocess.run([sys.executable, "prototype/tools/sdxl_batch_demo.py", "--data", "data", "--out", "/kaggle/working/sdxl_batch"],
                   capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
'''
compile(SRC, "cell", "exec")
os.makedirs(OUT, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": SRC.strip("\n").splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(OUT, "sdxl-batch-trial.ipynb"), "w", encoding="utf-8"), indent=1)
json.dump({"id": "markandreduesing/sdxl-batch-trial", "title": "sdxl-batch-trial", "code_file": "sdxl-batch-trial.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true",
           "enable_tpu": "false", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
           "competition_sources": [], "kernel_sources": [], "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("wrote", OUT, round(os.path.getsize(os.path.join(OUT, "sdxl-batch-trial.ipynb")) / 1e6, 2), "MB")
