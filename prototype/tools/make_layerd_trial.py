"""CPU-only Kaggle notebook (no GPU quota): run LayerD (CyberAgentAILab, Apache-2.0, ICCV 2025) on our test scenes.

    fastsdcpu\\env\\Scripts\\python.exe prototype\\tools\\make_layerd_trial.py
    cd prototype\\kaggle\\upload_layerd_trial && python -m kaggle.cli kernels push -p .

LayerD peels off the top layer (BiRefNet matting), inpaints underneath and repeats. It was trained on graphic
designs, so this checks how far it carries to painted / generated game scenes, compared with Qwen-Image-Layered.
"""
import base64
import io
import json
import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
ROOT = os.path.dirname(PROTO)
OUT = os.path.join(PROTO, "kaggle", "upload_layerd_trial")
IMAGES = {
    "forest": "experiments/blkE__P2__textonly__sdxlturbo-steps4-g1.0__seed11__impressionist.png",
    "pixelart_camp": "experiments/P3__ph4__canny-raw__neutral-scale70-steps6-g1.0__seed7__pixelart.png",
    "sketch_skeleton": "experiments/blkB__P2__raw__str80-steps4-g1.0__seed11__hyperreal.png",
}


def b64(path):
    buf = io.BytesIO()
    Image.open(os.path.join(ROOT, path)).convert("RGB").save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


SRC = f'''
import subprocess, sys, time, io, base64, json, os
t0 = time.time()
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'git+https://github.com/CyberAgentAILab/LayerD.git', 'transformers<5'],  # BiRefNet remote code breaks on transformers 5
                   capture_output=True, text=True)
print('pip rc', r.returncode, r.stderr[-1500:], flush=True)
from PIL import Image
import numpy as np
from layerd import LayerD
OUT = '/kaggle/working/layerd'; os.makedirs(OUT, exist_ok=True)
IMAGES = {{k: Image.open(io.BytesIO(base64.b64decode(v))).convert('RGB') for k, v in {json.dumps({k: b64(v) for k, v in IMAGES.items()})}.items()}}
t = time.time(); model = LayerD(matting_hf_card='cyberagent/layerd-birefnet').to('cpu'); print('load', round(time.time() - t, 1), 's', flush=True)
stats = {{}}
for name, im in IMAGES.items():
    t = time.time()
    layers = model.decompose(im)
    dt = round(time.time() - t, 1)
    info = []
    for i, L in enumerate(layers):
        L = L if isinstance(L, Image.Image) else Image.fromarray(np.asarray(L))
        L.save(f'{{OUT}}/{{name}}_layer{{i}}.png')
        a = np.asarray(L.convert('RGBA'))[..., 3]
        info.append({{'size': L.size, 'mode': L.mode, 'transparent_pct': round(float((a < 10).mean() * 100), 1)}})
    stats[name] = {{'seconds_cpu': dt, 'n_layers': len(layers), 'layers': info}}
    print(name, stats[name], flush=True)
    # strip: input + layers on green
    tiles = [im.resize((256, 256))]
    for i in range(len(layers)):
        L = Image.open(f'{{OUT}}/{{name}}_layer{{i}}.png').convert('RGBA').resize((256, 256))
        g = Image.new('RGBA', L.size, (40, 160, 60, 255)); g.alpha_composite(L); tiles.append(g.convert('RGB'))
    s = Image.new('RGB', (256 * len(tiles), 256))
    for i, tile in enumerate(tiles):
        s.paste(tile, (256 * i, 0))
    s.save(f'{{OUT}}/strip_{{name}}.png')
json.dump(stats, open(f'{{OUT}}/stats.json', 'w'), indent=1, default=str)
print('total', round(time.time() - t0, 1), 's')
'''
compile(SRC, "cell", "exec")
os.makedirs(OUT, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": SRC.strip("\n").splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(OUT, "layerd-trial.ipynb"), "w", encoding="utf-8"), indent=1)
json.dump({"id": "markandreduesing/layerd-trial", "title": "layerd-trial", "code_file": "layerd-trial.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "false",
           "enable_tpu": "false", "enable_internet": "true", "machine_shape": "", "dataset_sources": [],
           "competition_sources": [], "kernel_sources": [], "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("wrote", OUT)
