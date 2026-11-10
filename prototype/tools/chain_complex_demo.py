r"""LayerDiffuse SDXL chain on a COMPLEX scene (7 elements, several overlapping): background, then each element drawn into the
composite so far (bg2ble) and extracted (bgble2fg). Two settings: the engine defaults (512 px, 20 steps, CFG 7) and 1024 px.

    python prototype/tools/chain_complex_demo.py --out out/
Saves per setting: composite after every element, each extracted layer, a strip, per-layer transparency and timings.
"""
import argparse, json, os, sys, time
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from layerproto.backends.layerdiffuse_chain import LayerDiffuseChain  # noqa: E402
from layerproto.backends.chain_base import ChainParams  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
STYLE = "storybook illustration"
BG = f"{STYLE}, a village lawn at the edge of a pine forest, blue sky, soft clouds"
ELEMENTS = [   # back to front; later ones overlap earlier ones
    "a large old oak tree on the right",
    "a stone cottage with a thatched roof on the left",
    "a wooden fence in front of the cottage",
    "a small blue wagon next to the fence",
    "a campfire with a ring of stones in the middle",
    "a red fox sitting in front of the campfire",
    "a tall iron lantern post at the front left",
]
chain = LayerDiffuseChain(model="stabilityai/stable-diffusion-xl-base-1.0")
stats = {}
for S in (512, 1024):
    t0 = time.time(); res = []
    base = chain.background(ChainParams(prompt=BG, seed=11, width=S, height=S))
    scene = base.convert("RGBA"); tiles = [scene.convert("RGB")]; layers = []
    for i, el in enumerate(ELEMENTS):
        t = time.time(); p = ChainParams(prompt=f"{el}, {STYLE}", seed=20 + i, width=S, height=S)
        blended = chain.blend(scene, p); layer = chain.extract(scene, blended, p).convert("RGBA")
        hist = layer.split()[-1].histogram(); tr = round(100 * sum(hist[:10]) / max(1, sum(hist)), 1)
        scene = scene.copy(); scene.alpha_composite(layer)
        layer.save(f"{a.out}/s{S}_layer{i}.png"); scene.convert("RGB").save(f"{a.out}/s{S}_comp{i}.png")
        tiles.append(scene.convert("RGB")); layers.append(layer)
        res.append(dict(element=el, seconds=round(time.time() - t, 1), transparent_pct=tr)); print(S, res[-1], flush=True)
    tw = 256; strip = Image.new("RGB", (tw * (len(tiles) + len(layers)), tw), (40, 160, 60))
    for k, im in enumerate(tiles): strip.paste(im.resize((tw, tw)), (k * tw, 0))
    for k, L in enumerate(layers):
        g = Image.new("RGBA", L.size, (40, 160, 60, 255)); g.alpha_composite(L); strip.paste(g.convert("RGB").resize((tw, tw)), ((len(tiles) + k) * tw, 0))
    strip.save(f"{a.out}/s{S}_strip.jpg", quality=90)
    stats[S] = dict(total_s=round(time.time() - t0, 1), elements=res)
json.dump(stats, open(f"{a.out}/stats.json", "w"), indent=1)
