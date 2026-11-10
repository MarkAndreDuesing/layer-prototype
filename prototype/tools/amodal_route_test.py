"""Calls the live /decompose route (tunnel from prototype/config.json) with amodal on and off; saves the layers.

    fastsdcpu/env/Scripts/python.exe prototype/tools/amodal_route_test.py
"""
import json
import os
import sys
import time

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
sys.path.insert(0, PROTO)
from layerproto.backends.decomposer import DecomposeParams, RemoteDecomposer, recomposite  # noqa: E402

cfg = json.load(open(os.path.join(PROTO, "config.json"), encoding="utf-8"))
dec = RemoteDecomposer(cfg["remote_url"], cfg["remote_token"], timeout=1200)
img = Image.open(os.path.join(PROTO, "test_scene", "fox_cottage.png")).convert("RGB")
out = os.path.join(PROTO, "kaggle", "results_amodal_route")
os.makedirs(out, exist_ok=True)
BOXES = [[95, 100, 475, 330], [62, 245, 150, 322], [180, 0, 512, 352], [205, 325, 375, 452]]
NAMES = ["cottage", "stone gate arch", "big tree", "red fox"]
for tag, amodal in (("plain", False), ("amodal", True)):
    t = time.time()
    layers = dec.decompose(img, DecomposeParams(method="sam2-lama", boxes=BOXES, names=NAMES, amodal=amodal))
    print(tag, round(time.time() - t, 1), "s,", len(layers), "layers", flush=True)
    for i, L in enumerate(layers):
        L.save(os.path.join(out, f"{tag}_{i}_{'background' if i == 0 else NAMES[i - 1].replace(' ', '_')}.png"))
    recomposite(layers).convert("RGB").save(os.path.join(out, f"{tag}_recomposite.png"))
