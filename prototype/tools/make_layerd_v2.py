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
# v4 showed the matting output is exactly 0 everywhere (max 0.0): BiRefNet weights not materialised under newer
# transformers. Pinned transformers 4.46.3 and force a full (non-meta) load; check the output range first.
import layerd.matting.birefnet as _lb, layerd.models.matting.birefnet_matting as _lm
from transformers import AutoModelForImageSegmentation as _AM
def _build(card="cyberagent/layerd-birefnet", **kw):
    card = os.environ.get("BIREF_CARD", card)   # v11: swap in generic BiRefNet
    m, info = _AM.from_pretrained(card, trust_remote_code=True, low_cpu_mem_usage=False, output_loading_info=True)
    print("birefnet load info:", {{k: len(v) for k, v in info.items()}}, {{k: v[:3] for k, v in info.items() if v}}, flush=True)
    return m
_lb.build_birefnet = _build; _lm.build_birefnet = _build
import transformers; print("transformers", transformers.__version__, flush=True)
OUT = '/kaggle/working/layerd_v2'; os.makedirs(OUT, exist_ok=True)
IMAGES = {{k: Image.open(io.BytesIO(base64.b64decode(v))).convert('RGB') for k, v in {json.dumps({k: b64(v) for k, v in IMAGES.items()})}.items()}}
from PIL import ImageDraw
_d = Image.new('RGB', (768, 768), (244, 226, 190)); _g = ImageDraw.Draw(_d)   # flat graphic design: LayerD's training domain
_g.ellipse((200, 120, 560, 480), fill=(220, 70, 50)); _g.rectangle((80, 560, 690, 680), fill=(40, 70, 120))
_g.text((120, 590), "STORY TIME", fill=(255, 255, 255), font_size=72)
IMAGES['design_flat'] = _d
VARIANTS = {{
    'default': dict(kw={{}}, th=None, iters=3),
    'generic_birefnet': dict(kw={{}}, th=None, iters=4, card='ZhengPeng7/BiRefNet'),
    'generic_birefnet_norefine': dict(kw=dict(fg_refine=False, bg_refine=False), th=None, iters=4, card='ZhengPeng7/BiRefNet'),
}}
import torch
def _diag(card):
    mm = _build(card).eval()
    ps = list(mm.parameters()); print(card, 'param dtype', ps[0].dtype, 'n', len(ps), 'abs-mean of first 3', [round(float(q.abs().mean()), 4) for q in ps[:3]], flush=True)
    im = list(IMAGES.values())[0].resize((1024, 1024))
    from torchvision import transforms as T
    x = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(im).unsqueeze(0).to(ps[0].dtype)
    with torch.no_grad(): out = mm(x)
    def walk(o, pre='out'):
        if torch.is_tensor(o): print(' ', pre, tuple(o.shape), o.dtype, 'min', float(o.min()), 'max', float(o.max()), 'nan', int(o.isnan().sum()), flush=True)
        elif isinstance(o, (list, tuple)):
            for i, e in enumerate(o): walk(e, f'{{pre}}[{{i}}]')
        else: print(' ', pre, type(o), flush=True)
    walk(out)
for _c in ():
    try: _diag(_c)
    except Exception as e: print('diag failed', _c, repr(e)[:300], flush=True)
stats = {{}}
for vname, v in VARIANTS.items():
    os.environ['BIREF_CARD'] = v.get('card', 'cyberagent/layerd-birefnet')
    t = time.time(); m = LayerD(matting_hf_card='cyberagent/layerd-birefnet', **v['kw']).to('cpu')
    if v['th'] is not None: m._th_alpha = v['th']
    cov = []
    class Spy:   # assigning __call__ on an instance is ignored by Python; wrap the object instead
        def __init__(self, inner): self.inner = inner
        def __getattr__(self, k): return getattr(self.inner, k)
        def __call__(self, img):
            a = np.asarray(self.inner(img), dtype=np.float64)
            cov.append(dict(above_th=round(float((a > m._th_alpha).mean()), 3), mean=round(float(a.mean()), 3), max=round(float(a.max()), 3)))
            return a
    m.matting_model = Spy(m.matting_model)
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
