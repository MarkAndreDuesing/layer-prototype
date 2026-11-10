r"""Kaggle T4 notebook: real /decompose route with Sam2Decomposer (tools/amodal_v2_demo.py).

    fastsdcpu/env/Scripts/python.exe prototype/tools/make_sam2_server_trial.py
    cd prototype/kaggle/upload_amodal_v2_trial && python -m kaggle.cli kernels push -p .
"""
import base64
import io
import json
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
OUT = os.path.join(PROTO, "kaggle", "upload_amodal_v2_trial")

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for top in ("layerproto",):
        for root, dirs, files in os.walk(os.path.join(PROTO, top)):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn in files:
                if fn.endswith(".py"):
                    full = os.path.join(root, fn)
                    z.write(full, os.path.join("prototype", os.path.relpath(full, PROTO)))
    z.write(os.path.join(HERE, "amodal_v2_demo.py"), "prototype/tools/amodal_v2_demo.py")
    z.write(os.path.join(PROTO, "test_scene", "fox_cottage.png"), "prototype/test_scene/fox_cottage.png")
    z.write(os.path.join(PROTO, "test_scene", "fox_cottage.json"), "prototype/test_scene/fox_cottage.json")
code_b64 = base64.b64encode(buf.getvalue()).decode()

SRC = f'''
import signal, time as _t
_T0 = _t.time()
def _cap(*a): raise TimeoutError("25 min cap")
signal.signal(signal.SIGALRM, _cap); signal.alarm(25 * 60)
import base64, io, zipfile, subprocess, sys, os
os.environ['SAM2_BUILD_CUDA'] = '0'
for _pkg in (['git+https://github.com/facebookresearch/sam2.git'], ['--no-deps', 'simple-lama-inpainting'], ['fire'], ['fastapi', 'httpx'], ['diffusers==0.37.0']):
    _r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + _pkg, capture_output=True, text=True)
    print(_pkg, _r.returncode, _r.stderr[-200:], flush=True)
os.makedirs('/kaggle/working/code', exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode("{code_b64}"))).extractall('/kaggle/working/code')
os.chdir('/kaggle/working/code')
r = subprocess.run([sys.executable, "prototype/tools/amodal_v2_demo.py", "--image", "prototype/test_scene/fox_cottage.png",
                    "--out", "/kaggle/working/amodal_v2"], capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
'''
compile(SRC, "cell", "exec")
os.makedirs(OUT, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": SRC.strip("\n").splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(OUT, "amodal-v2-trial.ipynb"), "w", encoding="utf-8"), indent=1)
json.dump({"id": "markandreduesing/amodal-v2-trial", "title": "amodal-v2-trial", "code_file": "amodal-v2-trial.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true",
           "enable_tpu": "false", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
           "competition_sources": [], "kernel_sources": [], "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("wrote", OUT, round(os.path.getsize(os.path.join(OUT, "amodal-v2-trial.ipynb")) / 1e6, 2), "MB")
