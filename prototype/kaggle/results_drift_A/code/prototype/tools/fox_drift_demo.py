r"""Arm A of the drift experiment on GPU: the fox is moved turn after turn with relocate() on RevealLayer's layers;
every other layer stays untouched. Saves layers + composite after each turn.

    python prototype/tools/fox_drift_demo.py --layers-dir layers/fox_cottage --edits prototype/tools/drift_edits.json --out out/
"""
import argparse, glob, json, os, sys, time
import torch
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from layerproto.backends.layerdiffuse_chain import LayerDiffuseChain  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--layers-dir", required=True); ap.add_argument("--edits", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--prompt", default="a red fox standing, storybook illustration"); ap.add_argument("--strength", type=float, default=0.3)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
E = json.load(open(a.edits)); S = 512; T = E["target_layer"]
layers = [Image.open(f).convert("RGBA").resize((S, S)) for f in sorted(glob.glob(os.path.join(a.layers_dir, "layer_*.*")))]
def comp(ls):
    c = ls[0].copy()
    for l in ls[1:]: c.alpha_composite(l)
    return c
def others(ls):
    return comp([l for i, l in enumerate(ls) if i != T])
for i, l in enumerate(layers): l.save(f"{a.out}/t0_layer{i}.png")
comp(layers).convert("RGB").save(f"{a.out}/t0_composite.png")
x0, y0, x1, y1 = layers[T].getchannel("A").point(lambda v: 255 if v > 20 else 0).getbbox()
home = ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)       # original centre and size
chain = LayerDiffuseChain(model="stabilityai/stable-diffusion-xl-base-1.0")
for t, e in enumerate(E["turns"], 1):
    tt = time.time()
    cx, cy = home[0] + e["dx"] * S, home[1] + e["dy"] * S
    w, h = home[2] * e["scale"], home[3] * e["scale"]
    box = (int(cx - w / 2), int(cy - h / 2), int(w), int(h))
    scene = others(layers)
    _naive, _harm, new = chain.relocate(scene, layers[T], box, a.prompt, seed=7, strength=a.strength, w=S, h=S, steps=20, guidance=7.0,
                                        extract="mask", free_last=2, band=9, paste_back=0.8)
    layers[T] = new
    for i, l in enumerate(layers): l.save(f"{a.out}/t{t}_layer{i}.png")
    comp(layers).convert("RGB").save(f"{a.out}/t{t}_composite.png")
    print(f"turn {t}: box {box}, {time.time() - tt:.1f} s", flush=True)
