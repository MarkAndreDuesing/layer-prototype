r"""CPU: compare shadow variants on the moved fox (RevealLayer split). Crops around the fox, 2.5x.

    fastsdcpu/env/Scripts/python.exe prototype/tools/shadow_variants_demo.py
Tiles: no shadow | black (the old default) | multiply 0.30 | multiply 0.45 + stronger opacity | multiply 0.30 with the light from the right.
"""
import os
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
ROOT = os.path.dirname(PROTO)
sys.path.insert(0, PROTO)
from layerproto.shadow import shadow_layer  # noqa: E402

RL = os.path.join(PROTO, "kaggle", "results_reveallayer_fox_v15", "reveal_out", "gs1.0_st12", "fox_cottage")
S = 512
bg = Image.open(f"{RL}/bg_rgba.png").convert("RGBA").resize((S, S))
objs = [Image.open(f"{RL}/layer_{i}_rgba.png").convert("RGBA").resize((S, S)) for i in range(5)]
fox = objs[4]
bbox = fox.getchannel("A").point(lambda a: 255 if a > 40 else 0).getbbox()
crop = fox.crop(bbox)
crop = crop.resize((int(crop.width * 1.15), int(crop.height * 1.15)), Image.LANCZOS)
moved = Image.new("RGBA", (S, S), (0, 0, 0, 0))
moved.alpha_composite(crop, (int(bbox[0] - 0.28 * S), int(bbox[1] - 0.04 * S)))
below = bg.copy()
for L in objs[:4]:
    below.alpha_composite(L)


def comp(sh=None):
    c = below.copy()
    if sh is not None:
        c.alpha_composite(sh)
    c.alpha_composite(moved)
    return c.convert("RGB")


variants = [
    ("no shadow", comp()),
    ("black (old default)", comp(shadow_layer(moved))),
    ("multiply 0.30", comp(shadow_layer(moved, background=below, multiply=0.30))),
    ("multiply 0.45, opacity 0.75", comp(shadow_layer(moved, background=below, multiply=0.45, opacity=0.75, ao=0.5))),
    ("multiply 0.30, light from the right", comp(shadow_layer(moved, background=below, multiply=0.30, light_dx=-0.6))),
]
mb = moved.getchannel("A").getbbox()
pad = 60
box = (max(0, mb[0] - pad), max(0, mb[1] - pad), min(S, mb[2] + pad + 40), min(S, mb[3] + pad))
Z = 2.5
tiles = [im.crop(box).resize((int((box[2] - box[0]) * Z), int((box[3] - box[1]) * Z)), Image.LANCZOS) for _, im in variants]
W = sum(t.width for t in tiles)
out = Image.new("RGB", (W, tiles[0].height), (255, 255, 255))
x = 0
for t in tiles:
    out.paste(t, (x, 0))
    x += t.width
path = os.path.join(ROOT, "docs", "img", "shadow_variants.jpg")
out.save(path, quality=90)
print("wrote", path, out.size, [n for n, _ in variants])
