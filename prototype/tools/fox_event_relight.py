r"""Story-event strategy "bg_relight": a new background (from an edit of the background layer) + IC-Light relighting of every object
layer against the scene below it, back to front. Compares: (a) objects pasted unchanged on the new background, (b) relit objects.

    python prototype/tools/fox_event_relight.py --layers-dir layers/fox_cottage --bg-dir bg_events --out out/
"""
import argparse, glob, os, sys, time
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from layerproto.relight import ICLightFBC  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--layers-dir", required=True); ap.add_argument("--bg-dir", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); S = 512
layers = [Image.open(f).convert("RGBA").resize((S, S)) for f in sorted(glob.glob(os.path.join(a.layers_dir, "layer_*.*")))]
objs = layers[1:]
day = layers[0].copy()
for L in objs: day.alpha_composite(L)
PROMPTS = {"night": "dark night scene, full moon, soft cool moonlight, storybook illustration",
           "rain": "heavy rain, dark grey overcast light, wet scene, storybook illustration"}
rl = ICLightFBC(device="cuda", dtype=torch.float16)
lum = lambda im: round(float(np.asarray(im.convert("L"), np.float32).mean()), 1)
for ev, prompt in PROMPTS.items():
    nb = Image.open(os.path.join(a.bg_dir, f"bg_{ev}.png")).convert("RGB").resize((S, S))
    naive = nb.convert("RGBA")
    for L in objs: naive.alpha_composite(L)
    scene = nb.convert("RGBA"); t0 = time.time()
    for i, L in enumerate(objs):
        lit = rl.relight(L, scene.convert("RGB"), prompt, steps=20, size=S)
        lit.save(f"{a.out}/{ev}_layer{i + 1}_relit.png"); scene.alpha_composite(lit)
    secs = time.time() - t0
    strip = Image.new("RGB", (S * 4, S))
    for k, im in enumerate((day, nb, naive, scene)): strip.paste(im.convert("RGB"), (S * k, 0))
    strip.save(f"{a.out}/{ev}_strip.jpg", quality=90); scene.convert("RGB").save(f"{a.out}/{ev}_relit_composite.png")
    print(f"{ev}: relight {len(objs)} layers {secs:.1f} s; luminance day {lum(day)}, new bg {lum(nb)}, naive {lum(naive)}, relit {lum(scene)}", flush=True)
