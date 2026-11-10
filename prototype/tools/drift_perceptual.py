r"""CPU: LPIPS and DINOv2 drift outside the edited region for arms A / B / C of the fox drift experiment.

    fastsdcpu/env/Scripts/python.exe prototype/tools/drift_perceptual.py

Uses the saved per-turn outputs in prototype/kaggle/results_drift_{A,B,C}/drift_{A,B,C}; the edited region is the union of the
old and new fox alpha over all turns so far (from arm A's layers), same as drift_analyse.py. Writes
prototype/kaggle/results_drift/perceptual.json and prints a table. LPIPS: higher = more drift; DINO: lower = more drift.
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
sys.path.insert(0, PROTO)
from layerproto import drift_eval as D  # noqa: E402

K = os.path.join(PROTO, "kaggle")
A, B, C = (os.path.join(K, f"results_drift_{x}", f"drift_{x}") for x in "ABC")
E = json.load(open(os.path.join(HERE, "drift_edits.json")))
T = E["target_layer"]
nl = len(glob.glob(f"{A}/t0_layer*.png"))
layers = lambda d, t: [Image.open(f"{d}/t{t}_layer{i}.png").convert("RGBA") for i in range(nl)]
img0 = D.composite(layers(A, 0))
size = img0.size
P = D.Perceptual()
print("LPIPS available:", P.lp is not None, "| DINO available:", P.dino is not None)

arms = {
    "A_layered_relocate": lambda t: D.composite(layers(A, t)),
    "B_flat_qwen21": lambda t: Image.open(f"{B}/t{t}_composite.png").convert("RGB").resize(size),
    "C_redecompose_every_turn": lambda t: D.composite(layers(C, t)),
}
turns = {k: max(int(os.path.basename(f).split("_")[0][1:]) for f in glob.glob(f"{d}/t*_composite.png"))
         for k, d in (("A_layered_relocate", A), ("B_flat_qwen21", B), ("C_redecompose_every_turn", C))}
res = {}
for name, get in arms.items():
    edited = np.zeros((size[1], size[0]), bool)
    rows = []
    for t in range(1, turns[name] + 1):
        edited |= D.alpha_mask(layers(A, t - 1)[T]) | D.alpha_mask(layers(A, t)[T])
        rows.append(dict(turn=t, **P(img0, get(t), edited)))
    res[name] = rows
json.dump(res, open(os.path.join(K, "results_drift", "perceptual.json"), "w"), indent=1)
print("turn | " + " | ".join(f"{n[:1]} lpips  dino" for n in res))
for t in range(1, max(turns.values()) + 1):
    cells = []
    for n, rows in res.items():
        r = rows[t - 1] if t <= len(rows) else None
        cells.append(f"{r.get('lpips_out', float('nan')):.3f} {r.get('dino_out', float('nan')):.3f}" if r else "  -     -  ")
    print(f"{t:4d} | " + " | ".join(cells))
for n, rows in res.items():
    print(n, "mean LPIPS", round(float(np.mean([r.get("lpips_out", np.nan) for r in rows])), 3),
          "mean DINO", round(float(np.mean([r.get("dino_out", np.nan) for r in rows])), 3))
