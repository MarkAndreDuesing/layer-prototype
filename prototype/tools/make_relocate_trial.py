"""Kaggle T4 notebook: training-free move/resize re-harmonisation (LayerDiffuseChain.relocate) with SDXL base.

    fastsdcpu\\env\\Scripts\\python.exe prototype\\tools\\make_relocate_trial.py
    cd prototype\\kaggle\\upload_relocate_trial && python -m kaggle.cli kernels push -p .

Embeds the prototype's layerproto package, the (patched) LayerDiffuse diffusers port and tools/try_relocate.py,
plus the saved chain scene (background + extracted element), then runs try_relocate.py on the GPU.
"""
import base64
import io
import json
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
OUT = os.path.join(PROTO, "kaggle", "upload_relocate_trial")

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for top in ("layerproto", os.path.join("vendor", "diffuser_layerdiffuse", "layer_diffuse")):
        for root, dirs, files in os.walk(os.path.join(PROTO, top)):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn in files:
                if fn.endswith(".py"):
                    full = os.path.join(root, fn)
                    z.write(full, os.path.join("prototype", os.path.relpath(full, PROTO)))
    z.write(os.path.join(HERE, "try_relocate.py"), "prototype/tools/try_relocate.py")
    for fn in ("base20_0_bg.png", "base20_2_layer.png"):
        z.write(os.path.join(PROTO, "scenes", "_chain", fn), f"prototype/scenes/_chain/{fn}")
code_b64 = base64.b64encode(buf.getvalue()).decode()

SRC = f'''
import base64, io, os, subprocess, sys, zipfile, time
os.environ['HF_HUB_DISABLE_XET'] = '1'
print(subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'diffusers==0.37.0'], capture_output=True, text=True).stderr[-800:])
zipfile.ZipFile(io.BytesIO(base64.b64decode("{code_b64}"))).extractall('/kaggle/working')
os.chdir('/kaggle/working')
t = time.time()
r = subprocess.run([sys.executable, 'prototype/tools/try_relocate.py', '--size', '512', '--steps', '20', '--guidance', '7',
                    '--out', '/kaggle/working/relocate'], capture_output=True, text=True)
print(r.stdout[-4000:]); print(r.stderr[-4000:]); print('total', round(time.time() - t, 1), 's')
'''
compile(SRC, "cell", "exec")
os.makedirs(OUT, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": SRC.strip("\n").splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(OUT, "relocate-trial.ipynb"), "w", encoding="utf-8"), indent=1)
json.dump({"id": "markandreduesing/relocate-trial", "title": "relocate-trial", "code_file": "relocate-trial.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true",
           "enable_tpu": "false", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
           "competition_sources": [], "kernel_sources": [], "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("wrote", OUT, "| code zip", round(len(code_b64) / 1e6, 2), "MB b64")
