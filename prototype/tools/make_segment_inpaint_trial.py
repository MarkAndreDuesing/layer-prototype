"""Kaggle T4 notebook: the "generate -> segment -> inpaint" decomposition route, as a baseline next to Qwen-Image-Layered.

    fastsdcpu\\env\\Scripts\\python.exe prototype\\tools\\make_segment_inpaint_trial.py
    cd prototype\\kaggle\\upload_segment_inpaint_trial && python -m kaggle.cli kernels push -p .

Per image: SAM2 (hiera-small, Apache-2.0) automatic masks -> keep the K largest object masks (each < 35% of the image,
low overlap) as RGBA layers -> LaMa (simple-lama-inpainting, Apache-2.0) fills their union in the background.
No amodal completion: hidden parts of objects stay missing (that's what RevealLayer / pix2gestalt would add).
"""
import base64
import io
import json
import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
ROOT = os.path.dirname(PROTO)
OUT = os.path.join(PROTO, "kaggle", "upload_segment_inpaint_trial")
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
import subprocess, sys, os, io, base64, json, time
os.environ['SAM2_BUILD_CUDA'] = '0'
for pkg in (['git+https://github.com/facebookresearch/sam2.git'], ['--no-deps', 'simple-lama-inpainting'], ['fire']):
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + pkg, capture_output=True, text=True)
    print(pkg, 'rc', r.returncode, r.stderr[-600:], flush=True)
import numpy as np, torch
from PIL import Image, ImageFilter
from sam2.build_sam import build_sam2_hf
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from simple_lama_inpainting import SimpleLama
OUT = '/kaggle/working/seginp'; os.makedirs(OUT, exist_ok=True)
IMAGES = {{k: Image.open(io.BytesIO(base64.b64decode(v))).convert('RGB') for k, v in {json.dumps({k: b64(v) for k, v in IMAGES.items()})}.items()}}
t = time.time()
sam = build_sam2_hf('facebook/sam2-hiera-small', device='cuda')
gen = SAM2AutomaticMaskGenerator(sam, points_per_side=24, pred_iou_thresh=0.8, stability_score_thresh=0.9, min_mask_region_area=200)
lama = SimpleLama()
print('load', round(time.time() - t, 1), 's', flush=True)
K = 4
stats = {{}}
for name, im in IMAGES.items():
    t = time.time()
    arr = np.asarray(im)
    masks = sorted(gen.generate(arr), key=lambda m: -m['area'])
    H, W = arr.shape[:2]
    taken = np.zeros((H, W), bool); layers = []
    for m in masks:
        seg = m['segmentation']
        if seg.mean() > 0.35 or seg.mean() < 0.005:
            continue
        if (seg & taken).sum() > 0.2 * seg.sum():
            continue
        seg = seg & ~taken; taken |= seg; layers.append(seg)
        if len(layers) == K:
            break
    t_seg = time.time() - t
    hole = Image.fromarray((taken * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9))
    bg = lama(im, hole).resize(im.size)
    t_all = time.time() - t
    out = [bg.convert('RGBA')]
    for seg in reversed(layers):        # larger objects further back
        a = Image.fromarray((seg * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1))
        L = im.convert('RGBA'); L.putalpha(a); out.append(L)
    canvas = out[0].copy()
    for L in out[1:]:
        canvas.alpha_composite(L)
    diff = float(np.abs(np.asarray(canvas.convert('RGB'), dtype=int) - arr.astype(int)).mean())
    stats[name] = dict(seconds_gpu=round(t_all, 1), seconds_segment=round(t_seg, 1), n_masks=len(masks), n_layers=len(out),
                       recomposite_diff=round(diff, 2))
    print(name, stats[name], flush=True)
    tiles = [im.resize((256, 256)), canvas.convert('RGB').resize((256, 256))]
    for L in out:
        g = Image.new('RGBA', L.size, (40, 160, 60, 255)); g.alpha_composite(L); tiles.append(g.convert('RGB').resize((256, 256)))
    s = Image.new('RGB', (256 * len(tiles), 256))
    for i, tile in enumerate(tiles):
        s.paste(tile, (256 * i, 0))
    s.save(f'{{OUT}}/strip_{{name}}.png')
json.dump(stats, open(f'{{OUT}}/stats.json', 'w'), indent=1)
'''
compile(SRC, "cell", "exec")
os.makedirs(OUT, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": SRC.strip("\n").splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(OUT, "segment-inpaint-trial.ipynb"), "w", encoding="utf-8"), indent=1)
json.dump({"id": "markandreduesing/segment-inpaint-trial", "title": "segment-inpaint-trial",
           "code_file": "segment-inpaint-trial.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
           "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
           "model_sources": []}, open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("wrote", OUT)
