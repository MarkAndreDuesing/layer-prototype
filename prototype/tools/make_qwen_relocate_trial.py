r"""Kaggle T4 notebook: the full loop on real Qwen-Image-Layered output.
Qwen layers (from run v7, prototype/kaggle/results_qwen_v7_layers) -> move/resize one layer -> re-harmonise with
LayerDiffuseChain.relocate (SDXL base, mask extraction). Runs tools/qwen_relocate_demo.py per scene.

    fastsdcpu\env\Scripts\python.exe prototype\tools\make_qwen_relocate_trial.py
    cd prototype\kaggle\upload_qwen_relocate_trial && python -m kaggle.cli kernels push -p .
"""
import base64
import io
import json
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
OUT = os.path.join(PROTO, "kaggle", "upload_qwen_relocate_trial")
LAYERS = os.path.join(PROTO, "kaggle", "results_qwen_v7_layers")
DEMOS = [  # (layers dir, layer index, prompt, dx, dy, scale)
    ("layers_forest_384px_4L_4st", 3, "a campfire with burning logs and a small wooden chair, impressionist oil painting", 0.3, 0.0, 1.2),
    ("layers_pixelart_camp_384px_4L_4st", 3, "a colourful pixel-art skeleton sitting, and a green tree", -0.45, 0.0, 1.0),
    ("layers_sketch_skeleton_384px_4L_4st", 3, "a pale illustration of a sitting skeleton", -0.3, 0.05, 0.9),
]
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for top in ("layerproto", os.path.join("vendor", "diffuser_layerdiffuse", "layer_diffuse")):
        for root, dirs, files in os.walk(os.path.join(PROTO, top)):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn in files:
                if fn.endswith(".py"):
                    full = os.path.join(root, fn)
                    z.write(full, os.path.join("prototype", os.path.relpath(full, PROTO)))
    z.write(os.path.join(HERE, "qwen_relocate_demo.py"), "prototype/tools/qwen_relocate_demo.py")
code_b64 = base64.b64encode(buf.getvalue()).decode()
cmds = [["prototype/tools/qwen_relocate_demo.py", "--layers-dir", f"layers/{d}", "--index", str(i), "--prompt", p,
         "--dx", str(dx), "--dy", str(dy), "--scale", str(sc), "--out", "/kaggle/working/qwen_relocate", "--pin", "0.7", "--strengths", "0.3,0.45"]
        for d, i, p, dx, dy, sc in DEMOS]
SRC = f'''
import base64, io, os, subprocess, sys, zipfile, time
os.environ['HF_HUB_DISABLE_XET'] = '1'
print(subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'diffusers==0.37.0'], capture_output=True, text=True).stderr[-800:])
zipfile.ZipFile(io.BytesIO(base64.b64decode("{code_b64}"))).extractall('/kaggle/working')
os.chdir('/kaggle/working')
import glob, shutil
src = glob.glob('/kaggle/input/**/qwen_layered_test', recursive=True)   # outputs of the qwen-layered-batch run v7
print('layers from', src, flush=True)
for d in glob.glob(src[0] + '/layers_*'):
    shutil.copytree(d, 'layers/' + os.path.basename(d), dirs_exist_ok=True)
for c in {json.dumps(cmds)}:
    t = time.time()
    r = subprocess.run([sys.executable] + c, capture_output=True, text=True)
    print(r.stdout[-2000:]); print(r.stderr[-2500:]); print('step total', round(time.time() - t, 1), 's', flush=True)
'''
compile(SRC, "cell", "exec")
os.makedirs(OUT, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": SRC.strip("\n").splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(OUT, "qwen-relocate-trial.ipynb"), "w", encoding="utf-8"), indent=1)
json.dump({"id": "markandreduesing/qwen-relocate-trial", "title": "qwen-relocate-trial", "code_file": "qwen-relocate-trial.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true",
           "enable_tpu": "false", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
           "competition_sources": [], "kernel_sources": ["markandreduesing/qwen-layered-batch"], "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("wrote", OUT, "| payload", round(len(code_b64) / 1e6, 2), "MB b64")
