r"""Harmonise ART+ object layers into another background: per object (back to front) relocate() in place at strength s against the
composite so far (style / light pulled towards the scene), optionally followed by IC-Light. Saves strips: pasted | relocate | relocate + IC-Light.

    python prototype/tools/harmonise_artplus_demo.py --layers-dir layers/artplus --out out/
layer_0 = background, layer_1.. = objects back to front.
"""
import argparse, glob, os, sys, time
import torch
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from layerproto.backends.layerdiffuse_chain import LayerDiffuseChain  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--layers-dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--strengths", default="0.4,0.6"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); S = 512
layers = [Image.open(f).convert("RGBA").resize((S, S)) for f in sorted(glob.glob(os.path.join(a.layers_dir, "layer_*.*")))]
bg, objs = layers[0], layers[1:]
PROMPTS = ["small village houses", "a thatched stone cottage", "a small red car", "a leafy tree", "a red fox"]
pasted = bg.copy()
for L in objs: pasted.alpha_composite(L)
chain = LayerDiffuseChain(model="stabilityai/stable-diffusion-xl-base-1.0")
res = {}
for st in [float(x) for x in a.strengths.split(",")]:
    scene = bg.copy(); t0 = time.time(); outs = []
    for i, L in enumerate(objs):
        x0, y0, x1, y1 = L.getchannel("A").point(lambda v: 255 if v > 20 else 0).getbbox()
        _naive, _harm, new = chain.relocate(scene, L, (x0, y0, x1 - x0, y1 - y0), PROMPTS[i % len(PROMPTS)] + ", storybook illustration", seed=7, strength=st,
                                            w=S, h=S, steps=20, guidance=7.0, extract="mask", free_last=2, band=9, paste_back=0.5)
        outs.append(new); scene.alpha_composite(new)
    res[st] = (scene, outs); print(f"strength {st}: {time.time() - t0:.1f} s", flush=True)
del chain; torch.cuda.empty_cache()
from layerproto.relight import ICLightFBC  # noqa: E402
rl = ICLightFBC(device="cuda", dtype=torch.float16)
for st, (scene_r, outs) in res.items():
    scene = bg.copy()
    for i, L in enumerate(outs):
        lit = rl.relight(L, scene.convert("RGB"), "soft daylight, storybook illustration, village", steps=20, size=S); scene.alpha_composite(lit)
    strip = Image.new("RGB", (S * 4, S))
    for k, im in enumerate((bg, pasted, scene_r, scene)): strip.paste(im.convert("RGB"), (S * k, 0))
    strip.save(f"{a.out}/harmonise_s{int(st * 100)}_strip.jpg", quality=90)
    print(f"strength {st}: strip saved", flush=True)
