r"""Kaggle T4 notebook, segment + inpaint route v2: segment BY NOUN instead of "largest K masks".

    fastsdcpu\env\Scripts\python.exe prototype\tools\make_segment_inpaint_v2.py
    cd prototype\kaggle\upload_segment_inpaint_trial && python -m kaggle.cli kernels push -p .

On the shared fox & cottage scene (prototype/test_scene/fox_cottage.json), three ways to get one mask per object:
  A  hand boxes from the scene file -> SAM2 box prompt (upper bound: what a perfect planner would give)
  B  layer NAMES -> Grounding DINO (tiny, Apache-2.0) text boxes -> SAM2 (fully automatic, uses the LLM's nouns)
  C  SAM2 automatic masks, largest K (the v1 baseline)
Each: masks become RGBA layers back to front, LaMa fills the union in the background. Strips + stats saved.
"""
import base64
import io
import json
import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
OUT = os.path.join(PROTO, "kaggle", "upload_segment_inpaint_trial")
SC = json.load(open(os.path.join(PROTO, "test_scene", "fox_cottage.json"), encoding="utf-8"))
buf = io.BytesIO()
Image.open(os.path.join(PROTO, "test_scene", SC["image"])).convert("RGB").save(buf, "JPEG", quality=93)
IMG = base64.b64encode(buf.getvalue()).decode()
LAYERS = SC["layers_back_to_front"]

SRC = f'''
import subprocess, sys, os, io, base64, json, time
os.environ['SAM2_BUILD_CUDA'] = '0'
for pkg in (['git+https://github.com/facebookresearch/sam2.git'], ['--no-deps', 'simple-lama-inpainting'], ['fire']):
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + pkg, capture_output=True, text=True)
    print(pkg, 'rc', r.returncode, r.stderr[-400:], flush=True)
import numpy as np, torch
from PIL import Image, ImageFilter
from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from simple_lama_inpainting import SimpleLama
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
OUT = '/kaggle/working/seginp_v2'; os.makedirs(OUT, exist_ok=True)
im = Image.open(io.BytesIO(base64.b64decode("{IMG}"))).convert('RGB'); arr = np.asarray(im); H, W = arr.shape[:2]
LAYERS = {json.dumps(LAYERS)}
t = time.time()
sam = build_sam2_hf('facebook/sam2-hiera-small', device='cuda'); pred = SAM2ImagePredictor(sam)
gen = SAM2AutomaticMaskGenerator(sam, points_per_side=24, pred_iou_thresh=0.8, stability_score_thresh=0.9, min_mask_region_area=200)
gd_proc = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny')
gd = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny').to('cuda')
lama = SimpleLama()
print('load', round(time.time() - t, 1), 's', flush=True)

def sam_box(box):
    pred.set_image(arr)
    m, s, _ = pred.predict(box=np.array(box, dtype=np.float32), multimask_output=False)
    return m[0].astype(bool)

def gd_box(name):
    inp = gd_proc(images=im, text=name.lower() + '.', return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = gd(**inp)
    r = gd_proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.3, text_threshold=0.25, target_sizes=[(H, W)])[0]
    if len(r['boxes']) == 0:
        return None
    i = int(r['scores'].argmax()); return [float(x) for x in r['boxes'][i].tolist()]

def build(masks, tag):
    # masks: list of (name, bool mask) back to front; later (front) layers own overlapping pixels
    taken = np.zeros((H, W), bool); owned = []
    for name, m in reversed(masks):
        mm = m & ~taken; taken |= mm; owned.append((name, mm))
    owned.reverse()
    hole = Image.fromarray((taken * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9))
    bg = lama(im, hole).resize(im.size)
    layers = [('background', bg.convert('RGBA'))]
    for name, mm in owned:
        a = Image.fromarray((mm * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
        L = im.convert('RGBA'); L.putalpha(a); layers.append((name, L))
    canvas = layers[0][1].copy()
    for _, L in layers[1:]: canvas.alpha_composite(L)
    diff = float(np.abs(np.asarray(canvas.convert('RGB'), dtype=int) - arr.astype(int)).mean())
    tiles = [im.resize((256, 256)), canvas.convert('RGB').resize((256, 256))]
    for _, L in layers:
        g = Image.new('RGBA', L.size, (40, 160, 60, 255)); g.alpha_composite(L); tiles.append(g.convert('RGB').resize((256, 256)))
    s = Image.new('RGB', (256 * len(tiles), 256))
    for i, tile in enumerate(tiles): s.paste(tile, (256 * i, 0))
    s.save(f'{{OUT}}/strip_fox_cottage_{{tag}}.png')
    os.makedirs(f'{{OUT}}/layers_{{tag}}', exist_ok=True)
    for k, (name, L) in enumerate(layers): L.save(f'{{OUT}}/layers_{{tag}}/{{k}}_{{name.replace(" ", "_")}}.png')
    return dict(layers=len(layers), recomposite_diff=round(diff, 2), coverage=[round(float(mm.mean()), 3) for _, mm in owned])

stats = {{}}
t = time.time(); A = [(l['name'], sam_box(l['box'])) for l in LAYERS]; stats['A_hand_boxes'] = dict(build(A, 'A_hand_boxes'), seconds=round(time.time() - t, 1))
t = time.time(); boxes = {{l['name']: gd_box(l['name']) for l in LAYERS}}
B = [(n, sam_box(b)) for n, b in boxes.items() if b is not None]
stats['B_grounding_dino'] = dict(build(B, 'B_grounding_dino'), seconds=round(time.time() - t, 1), found={{n: b is not None for n, b in boxes.items()}}, boxes={{n: [round(x) for x in b] if b else None for n, b in boxes.items()}})
t = time.time(); auto = sorted(gen.generate(arr), key=lambda m: -m['area'])
taken = np.zeros((H, W), bool); C = []
for m in auto:
    seg = m['segmentation']
    if seg.mean() > 0.35 or seg.mean() < 0.005 or (seg & taken).sum() > 0.2 * seg.sum(): continue
    seg = seg & ~taken; taken |= seg; C.append((f'auto{{len(C)}}', seg))
    if len(C) == 5: break
C = C[::-1]
stats['C_auto_largest'] = dict(build(C, 'C_auto_largest'), seconds=round(time.time() - t, 1))
for k, v in stats.items(): print(k, v, flush=True)
json.dump(stats, open(f'{{OUT}}/stats.json', 'w'), indent=1)
'''
compile(SRC, "cell", "exec")
os.makedirs(OUT, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": SRC.strip("\n").splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(OUT, "segment-inpaint-trial.ipynb"), "w", encoding="utf-8"), indent=1)
print("wrote", OUT, round(os.path.getsize(os.path.join(OUT, "segment-inpaint-trial.ipynb")) / 1e6, 2), "MB")
