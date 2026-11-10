r"""Arm C of the drift experiment on GPU: like arm A (six successive fox moves with relocate()), but the composite is
RE-DECOMPOSED every turn (Grounding DINO + SAM2 by noun + LaMa) before the next edit.

    python prototype/tools/fox_drift_armc.py --layers-dir layers/fox_cottage --edits prototype/tools/drift_edits.json --out out/
Saves layers + composite after each turn (t{n}_layer{i}.png, t{n}_composite.png) in the same layout as arm A, so
drift_analyse.py can compare them.
"""
import argparse
import glob
import json
import os
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from layerproto.backends.decomposer import DecomposeParams  # noqa: E402
from layerproto.backends.layerdiffuse_chain import LayerDiffuseChain  # noqa: E402
from layerproto.backends.sam2_decomposer import Sam2Decomposer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--layers-dir", required=True)
ap.add_argument("--edits", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--prompt", default="a red fox standing, storybook illustration")
ap.add_argument("--strength", type=float, default=0.3)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
E = json.load(open(a.edits))
S = 512
T = E["target_layer"]
NAMES = ["village houses", "cottage", "stone gate arch", "big tree", "red fox"]
layers = [Image.open(f).convert("RGBA").resize((S, S)) for f in sorted(glob.glob(os.path.join(a.layers_dir, "layer_*.*")))]


def comp(ls):
    c = ls[0].copy()
    for lay in ls[1:]:
        c.alpha_composite(lay)
    return c


def others(ls):
    return comp([lay for i, lay in enumerate(ls) if i != T])


def save(t):
    for i, lay in enumerate(layers):
        lay.save(f"{a.out}/t{t}_layer{i}.png")
    comp(layers).convert("RGB").save(f"{a.out}/t{t}_composite.png")


save(0)
x0, y0, x1, y1 = layers[T].getchannel("A").point(lambda v: 255 if v > 20 else 0).getbbox()
home = ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)
dec = Sam2Decomposer(device="cuda")
chain = LayerDiffuseChain(model="stabilityai/stable-diffusion-xl-base-1.0")
for t, e in enumerate(E["turns"], 1):
    tt = time.time()
    if t > 1:                                           # re-decompose the previous composite (the arm-C cost)
        prev = Image.open(f"{a.out}/t{t - 1}_composite.png").convert("RGB")
        try:
            new_layers = dec(prev, DecomposeParams(names=NAMES, method="sam2-lama"))
            if len(new_layers) == len(layers):
                layers = [lay.resize((S, S)) for lay in new_layers]
            else:
                print(f"turn {t}: decomposer found {len(new_layers) - 1} of {len(NAMES)} objects, keeping previous layers", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"turn {t}: decomposition failed ({ex!r}), keeping previous layers", flush=True)
    cx, cy = home[0] + e["dx"] * S, home[1] + e["dy"] * S
    w, h = home[2] * e["scale"], home[3] * e["scale"]
    box = (int(cx - w / 2), int(cy - h / 2), int(w), int(h))
    _naive, _harm, new = chain.relocate(others(layers), layers[T], box, a.prompt, seed=7, strength=a.strength, w=S, h=S, steps=20,
                                        guidance=7.0, extract="mask", free_last=2, band=9, paste_back=0.8)
    layers[T] = new
    save(t)
    print(f"turn {t}: box {box}, {time.time() - tt:.1f} s", flush=True)
