r"""CPU: can IC-Light be made to relight the objects of the night scene properly? (it left them pale blue-grey and not dark, docs/models/relight-ic-light.md)

    fastsdcpu/env/Scripts/python.exe prototype/tools/iclight_night_variants.py

Input: RevealLayer's split (filled background + 5 objects) and the Qwen-Image-2.1 edited NIGHT background (results_qwen21_bgevent). The objects are
relit back to front against the scene below. Variants (IC-Light v1 fbc, 8 steps, 384 px, float32 on the CPU, about 45 s per layer):
  grade     colour grade only (events.grade 'night'), no IC-Light  (reference)
  A         IC-Light as before
  B         pre-grade the object to night, then IC-Light
  C         IC-Light as before, then scale the lightness so the object's mean lightness equals the graded night version ('low-key fix')
  D         pre-grade, IC-Light, then the same lightness match
Output: docs/img/iclight_night_variants.jpg and prototype/kaggle/results_iclight_tuning/.
"""
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from skimage import color

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
ROOT = os.path.dirname(PROTO)
sys.path.insert(0, PROTO)
from layerproto.events import grade  # noqa: E402
from layerproto.relight import ICLightFBC  # noqa: E402

S = 384
RL = os.path.join(PROTO, "kaggle", "results_reveallayer_fox_v15", "reveal_out", "gs1.0_st12", "fox_cottage")
OUT = os.path.join(PROTO, "kaggle", "results_iclight_tuning")
os.makedirs(OUT, exist_ok=True)
objs = [Image.open(f"{RL}/layer_{i}_rgba.png").convert("RGBA").resize((S, S)) for i in range(5)]
bg_day = Image.open(f"{RL}/bg_rgba.png").convert("RGBA").resize((S, S))
night_bg = Image.open(os.path.join(PROTO, "kaggle", "results_qwen21_bgevent", "bg_events", "bg_night.png")).convert("RGB").resize((S, S))
PROMPT = "dark night scene, full moon, soft cool moonlight, storybook illustration"


def match_lightness(relit, target, gain_cap=(0.3, 1.5)):
    """Scale the relit layer's Lab lightness so its alpha-weighted mean equals the target layer's (keeps IC-Light's shading and colour)."""
    a = np.asarray(relit.getchannel("A"), np.float32) / 255
    r = color.rgb2lab(np.asarray(relit.convert("RGB"), np.float32) / 255)
    t = color.rgb2lab(np.asarray(target.convert("RGB").resize(relit.size), np.float32) / 255)
    w = a.sum() + 1e-6
    lr, lt = (r[..., 0] * a).sum() / w, (t[..., 0] * a).sum() / w
    g = float(np.clip(lt / max(lr, 1e-3), *gain_cap))
    r[..., 0] = np.clip(r[..., 0] * g, 0, 100)
    out = Image.fromarray((np.clip(color.lab2rgb(r), 0, 1) * 255).astype(np.uint8)).convert("RGBA")
    out.putalpha(relit.getchannel("A"))
    return out


rl = ICLightFBC(device="cpu", dtype=torch.float32)
VARIANTS = ["grade", "A", "B", "C", "D"]
scenes = {v: night_bg.convert("RGBA") for v in VARIANTS}
day = bg_day.copy()
for L in objs:
    day.alpha_composite(L)
t0 = time.time()
for i, L in enumerate(objs):
    graded = grade(L, "night")
    lit_a = rl.relight(L, scenes["A"].convert("RGB"), PROMPT, steps=8, size=S)
    lit_b = rl.relight(graded, scenes["B"].convert("RGB"), PROMPT, steps=8, size=S)
    new = {"grade": graded, "A": lit_a, "B": lit_b, "C": match_lightness(rl_c := rl.relight(L, scenes["C"].convert("RGB"), PROMPT, steps=8, size=S), graded),
           "D": match_lightness(rl.relight(graded, scenes["D"].convert("RGB"), PROMPT, steps=8, size=S), graded)}
    for v in VARIANTS:
        scenes[v].alpha_composite(new[v])
        new[v].save(f"{OUT}/night_layer{i}_{v}.png")
    print(f"layer {i} done, {time.time() - t0:.0f} s", flush=True)
naive = night_bg.convert("RGBA")
for L in objs:
    naive.alpha_composite(L)
tiles = [("day", day), ("Qwen-2.1 night background", night_bg.convert("RGBA")), ("objects unchanged", naive)] + [(v, scenes[v]) for v in VARIANTS]
lum = lambda im: round(float(np.asarray(im.convert("L"), np.float32).mean()), 1)
for n, im in tiles:
    print(f"{n:28s} mean grey {lum(im)}")
T = 256
sheet = Image.new("RGB", (T * 4, T * 2), (255, 255, 255))
for k, (n, im) in enumerate(tiles):
    sheet.paste(im.convert("RGB").resize((T, T), Image.LANCZOS), (T * (k % 4), T * (k // 4)))
sheet.save(os.path.join(ROOT, "docs", "img", "iclight_night_variants.jpg"), quality=90)
print("tiles:", [n for n, _ in tiles])
