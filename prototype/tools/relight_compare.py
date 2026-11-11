r"""CPU: object relighting side by side on the same layers and the same Qwen-2.1 background: unchanged | colour grade | IC-Light (pre-graded, B) | IC-Light (graded + lightness match, D) | Qwen-2.1 per-layer relight.

    fastsdcpu/env/Scripts/python.exe prototype/tools/relight_compare.py night   (or rain)
Needs results_iclight_tuning/<ev>_layer*_{grade,B,D}.png (iclight_night_variants.py <ev>) and results_perlayer_v3 (Qwen-2.1 relight of the 5 RevealLayer layers, 512 px).
"""
import os, sys
import numpy as np
from PIL import Image
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EV = sys.argv[1]
S = 384
K = os.path.join(P, "kaggle")
bg = Image.open(f"{K}/results_qwen21_bgevent/bg_events/bg_{EV}.png").convert("RGBA").resize((S, S))
RL = f"{K}/results_reveallayer_fox_v15/reveal_out/gs1.0_st12/fox_cottage"
def comp(layers):
    c = bg.copy()
    for L in layers: c.alpha_composite(L.convert("RGBA").resize((S, S)))
    return c.convert("RGB")
unchanged = comp([Image.open(f"{RL}/layer_{i}_rgba.png") for i in range(5)])
var = {v: comp([Image.open(f"{K}/results_iclight_tuning/{EV}_layer{i}_{v}.png") for i in range(5)]) for v in ("grade", "B", "D")}
q = comp([Image.open(f"{K}/results_perlayer_v3/perlayer3/{EV}_layer{i}.png") for i in range(1, 6)])
tiles = [("unchanged", unchanged), ("colour grade", var["grade"]), ("IC-Light B", var["B"]), ("IC-Light D", var["D"]), ("Qwen-2.1 per layer", q)]
lum = lambda im: round(float(np.asarray(im.convert("L"), np.float32).mean()), 1)
print({n: lum(im) for n, im in tiles}, "background alone", lum(bg))
W = Image.new("RGB", (S * 5, S))
for i, (_, im) in enumerate(tiles): W.paste(im, (S * i, 0))
out = os.path.join(P, "..", "docs", "img", f"relight_compare_{EV}.jpg"); W.save(out, quality=90); print(out)
