r"""One GPU session: Sam2Decomposer amodal fill, old recipe vs new (neutral grey base + 'a complete X' prompt + negatives).

    python prototype/tools/amodal_v2_demo.py --image prototype/test_scene/fox_cottage.png --out out/
"""
import argparse
import json
import os
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
from layerproto.backends.decomposer import DecomposeParams, recomposite  # noqa: E402
from layerproto.backends.sam2_decomposer import Sam2Decomposer  # noqa: E402

BOXES = [[95, 100, 475, 330], [62, 245, 150, 322], [180, 0, 512, 352], [205, 325, 375, 452]]
NAMES = ["cottage", "stone gate arch", "big tree", "red fox"]
img = Image.open(a.image).convert("RGB")
t = time.time()
dec = Sam2Decomposer(device="cuda")
print("load", round(time.time() - t, 1), "s", flush=True)
for tag, neutral in (("neutral", True), ("oldbase", False)):
    dec.amodal_neutral = neutral
    t = time.time()
    layers = dec(img, DecomposeParams(method="sam2-lama", boxes=BOXES, names=NAMES, amodal=True))
    print(tag, round(time.time() - t, 1), "s", flush=True)
    for i, L in enumerate(layers):
        L.save(f"{a.out}/{tag}_{i}.png")
    recomposite(layers).convert("RGB").save(f"{a.out}/{tag}_recomposite.png")
