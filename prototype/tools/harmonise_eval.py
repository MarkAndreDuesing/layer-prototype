"""CPU: contact sheet + drift numbers for results_qwen21_harmonise (Qwen-2.1 as the harmonise step)."""
import json, math, os
import numpy as np
from PIL import Image
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = os.path.join(P, "kaggle", "results_qwen21_harmonise", "harm")
ld = lambda n, m="RGB": Image.open(f"{D}/{n}.png").convert(m).resize((512, 512))
fox1 = np.asarray(ld("input_fox1", "RGBA"))[..., 3] > 40
fox2 = np.asarray(ld("input_fox2", "RGBA"))[..., 3] > 40
orig = np.asarray(ld("input_original"), float)
fox_old = np.abs(np.asarray(ld("input_naive"), float) - orig).sum(-1) > 30  # moved fox area + old place
def psnr(a, b, keep):
    d = (a - b)[keep]; m = (d ** 2).mean(); return 10 * math.log10(255 ** 2 / m) if m else 99
rows = []
def region(fox):  # dilate the moved fox + the old place
    from scipy.ndimage import binary_dilation
    old = np.asarray(Image.open(f"{D}/input_original.png").convert("RGB").resize((512, 512)))
    return binary_dilation(fox, iterations=25)
scene_nofox = ld("input_scene_nofox")
for name, naive_n, fox in (("control", "naive", fox1), ("harmonise_s42", "naive", fox1), ("harmonise_s7", "naive", fox1), ("shadow_only", "naive", fox1),
                           ("refs", "naive", fox1), ("layer_relight", "naive", fox1), ("harder_control", "naive2", fox2), ("harder_harmonise", "naive2", fox2)):
    out = ld(name, "RGBA")
    if name == "layer_relight":  # RGBA fox layer over the scene without the fox
        c = scene_nofox.convert("RGBA"); c.alpha_composite(out); out = c
    out = np.asarray(out.convert("RGB"), float)
    naive = np.asarray(ld("input_" + naive_n), float)
    near = region(fox)
    # the old fox place also counts as edited
    keep = ~(near | (np.abs(orig - np.asarray(ld("input_scene_nofox"), float)).sum(-1) > 30))
    rows.append((name, round(psnr(out, naive, keep), 1), round(float(np.abs(out - naive)[keep].mean()), 1), round(float(np.abs(out - naive)[fox].mean()), 1)))
for r in rows: print("%-18s PSNR outside the fox %5.1f dB | mean abs diff outside %5.1f | inside the fox %5.1f" % r)
tiles = ["input_naive", "control", "harmonise_s42", "harmonise_s7", "shadow_only", "refs", "layer_relight", "input_naive2", "harder_control", "harder_harmonise"]
T = 320; S = Image.new("RGB", (T * 5, T * 2))
for i, n in enumerate(tiles):
    im = ld(n, "RGBA")
    if n == "layer_relight":
        c = scene_nofox.convert("RGBA"); c.alpha_composite(im); im = c
    S.paste(im.convert("RGB").resize((T, T), Image.LANCZOS), (T * (i % 5), T * (i // 5)))
S.save(os.path.join(P, "..", "docs", "img", "qwen21_harmonise.jpg"), quality=90)
