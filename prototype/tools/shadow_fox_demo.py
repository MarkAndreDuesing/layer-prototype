r"""CPU demo: procedural cast shadows on the fox & cottage layers (RevealLayer split), before / after, plus the fox moved with its shadow.

    fastsdcpu/env/Scripts/python.exe prototype/tools/shadow_fox_demo.py
Writes docs/img/fox_shadow.jpg (original composite | + shadows | fox moved without shadow | fox moved with shadow) and prints timings.
"""
import os
import sys
import time

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
ROOT = os.path.dirname(PROTO)
sys.path.insert(0, PROTO)
from layerproto.shadow import shadow_layer  # noqa: E402

RL = os.path.join(PROTO, "kaggle", "results_reveallayer_fox_v15", "reveal_out", "gs3.5_st16", "fox_cottage")
bg = Image.open(f"{RL}/bg_rgba.png").convert("RGBA")
objs = [Image.open(f"{RL}/layer_{i}_rgba.png").convert("RGBA").resize(bg.size) for i in range(5)]   # houses, cottage, arch, tree, fox


def stack(layers):
    c = bg.copy()
    for L in layers:
        c.alpha_composite(L)
    return c


t = time.time()
shadows = [shadow_layer(o) for o in objs]
print("shadows for 5 layers:", round(time.time() - t, 2), "s")
plain = stack(objs)
with_sh = bg.copy()
for o, s in zip(objs, shadows):
    with_sh.alpha_composite(s)
    with_sh.alpha_composite(o)
# move the fox: 28 % left, 4 % up, 1.15x (same as the move loop in concepts/editing.md)
fox = objs[4]
bbox = fox.getchannel("A").point(lambda a: 255 if a > 40 else 0).getbbox()
crop = fox.crop(bbox)
w, h = int(crop.width * 1.15), int(crop.height * 1.15)
crop = crop.resize((w, h), Image.LANCZOS)
x = int(bbox[0] - 0.28 * bg.width)
y = int(bbox[1] - 0.04 * bg.height)
moved = Image.new("RGBA", bg.size, (0, 0, 0, 0))
moved.alpha_composite(crop, (max(0, x), max(0, y)))
base = stack(objs[:4])
no_sh = base.copy()
no_sh.alpha_composite(moved)
sh_m = shadow_layer(moved)
yes_sh = base.copy()
yes_sh.alpha_composite(sh_m)
yes_sh.alpha_composite(moved)
T = 300
S = Image.new("RGB", (T * 4, T), (255, 255, 255))
for i, im in enumerate((plain, with_sh, no_sh, yes_sh)):
    S.paste(im.convert("RGB").resize((T, T), Image.LANCZOS), (T * i, 0))
out = os.path.join(ROOT, "docs", "img", "fox_shadow.jpg")
S.save(out, quality=88)
print("wrote", out)
