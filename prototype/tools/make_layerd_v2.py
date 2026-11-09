r"""CPU-only Kaggle notebook: LayerD parameter sweep on the shared fox & cottage scene (no GPU quota).

    fastsdcpu\env\Scripts\python.exe prototype\tools\make_layerd_v2.py
    cd prototype\kaggle\upload_layerd_trial && python -m kaggle.cli kernels push -p .

LayerD stops when the matting alpha is empty or covers > 99% of the image (`_decompose_step`). The first attempt
returned 1 layer on painted scenes. Variants: defaults; lower hard-alpha threshold; matting at 1024 px; no fg/bg
palette refinement (meant for flat graphic designs); more iterations; plus the alpha coverage per step for diagnosis.
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
IMAGES = {"fox_cottage": os.path.join(PROTO, "test_scene", "fox_cottage.png"),
          "forest": os.path.join(ROOT, "experiments", "blkE__P2__textonly__sdxlturbo-steps4-g1.0__seed11__impressionist.png")}


def b64(p):
    buf = io.BytesIO(); Image.open(p).convert("RGB").save(buf, "JPEG", quality=93)
    return base64.b64encode(buf.getvalue()).decode()


SRC = f'''
import subprocess, sys, os, io, base64, json, time
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'git+https://github.com/CyberAgentAILab/LayerD.git', 'transformers<5'], capture_output=True, text=True)
print('pip rc', r.returncode, r.stderr[-400:], flush=True)
import numpy as np
from PIL import Image
from layerd import LayerD
OUT = '/kaggle/working/layerd_v2'; os.makedirs(OUT, exist_ok=True)
IMAGES = {{k: Image.open(io.BytesIO(base64.b64decode(v))).convert('RGB') for k, v in {json.dumps({k: b64(v) for k, v in IMAGES.items()})}.items()}}
VARIANTS = {{
    'default': dict(kw={{}}, th=None, iters=3),
    'th0.001_iter5': dict(kw={{}}, th=0.001, iters=5),
    'norefine_iter5': dict(kw=dict(fg_refine=False, bg_refine=False), th=None, iters=5),
    'norefine_nounblend_iter5': dict(kw=dict(fg_refine=False, bg_refine=False, use_unblend=False), th=None, iters=5),
    'matt1024_norefine_iter5': dict(kw=dict(fg_refine=False, bg_refine=False, matting_process_size=(1024, 1024)), th=None, iters=5),
}}
stats = {{}}
for vname, v in VARIANTS.items():
    t = time.time(); m = LayerD(matting_hf_card='cyberagent/layerd-birefnet', **v['kw']).to('cpu')
    if v['th'] is not None: m._th_alpha = v['th']
    cov = []
    orig = m.matting_model.__call__
    def spy(img, _o=orig):
        a = _o(img); cov.append(round(float((np.asarray(a) > m._th_alpha).mean()), 3)); return a
    m.matting_model.__call__ = spy
    for name, im in IMAGES.items():
        cov.clear(); t0 = time.time()
        try:
            layers = m.decompose(im, max_iterations=v['iters'])
        except Exception as e:
            print('FAILED', vname, name, repr(e)[:200], flush=True); continue
        d = f'{{OUT}}/{{vname}}/{{name}}'; os.makedirs(d, exist_ok=True)
        for i, L in enumerate(layers): L.save(f'{{d}}/layer_{{i}}.png')
        canvas = layers[0].convert('RGBA').copy()
        for L in layers[1:]: canvas.alpha_composite(L.convert('RGBA'))
        diff = float(np.abs(np.asarray(canvas.convert('RGB'), dtype=int) - np.asarray(im, dtype=int)).mean())
        stats[f'{{vname}}/{{name}}'] = dict(layers=len(layers), seconds=round(time.time() - t0, 1), recomposite_diff=round(diff, 2), alpha_coverage_per_step=list(cov))
        print(vname, name, stats[f'{{vname}}/{{name}}'], flush=True)
json.dump(stats, open(f'{{OUT}}/stats.json', 'w'), indent=1)
'''
compile(SRC, "cell", "exec")
os.makedirs(OUT, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": SRC.strip("\n").splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(OUT, "layerd-trial.ipynb"), "w", encoding="utf-8"), indent=1)
print("wrote", OUT)
