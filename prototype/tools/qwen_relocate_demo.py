"""Full-loop demo: Qwen-Image-Layered layers -> move/resize one layer -> re-harmonise (LayerDiffuseChain.relocate).

    python prototype/tools/qwen_relocate_demo.py --layers-dir <dir with layer_0.png ...> --index 2 --prompt "a campfire"

The "scene" for the edit is all other layers composited (background first); the chosen layer is moved by
(--dx, --dy) as fractions of the image size and scaled by --scale around its centre. Saves a strip per strength:
original composite | naive move | re-harmonised | new layer on green | final recomposite.
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from PIL import Image  # noqa: E402

from layerproto.backends.layerdiffuse_chain import LayerDiffuseChain  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--layers-dir", required=True)
ap.add_argument("--index", type=int, required=True)
ap.add_argument("--prompt", required=True)
ap.add_argument("--dx", type=float, default=-0.25)
ap.add_argument("--dy", type=float, default=0.0)
ap.add_argument("--scale", type=float, default=1.3)
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--guidance", type=float, default=7.0)
ap.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
S = a.size
layers = [Image.open(f).convert("RGBA").resize((S, S)) for f in sorted(glob.glob(os.path.join(a.layers_dir, "layer_*.png")))]
target = layers[a.index]
scene = layers[0].copy()
for i, L in enumerate(layers[1:], 1):
    if i != a.index:
        scene.alpha_composite(L)
orig = scene.copy()
orig.alpha_composite(target)
x0, y0, x1, y1 = target.getchannel("A").point(lambda v: 255 if v > 20 else 0).getbbox()
bw, bh = (x1 - x0) * a.scale, (y1 - y0) * a.scale
cx, cy = (x0 + x1) / 2 + a.dx * S, (y0 + y1) / 2 + a.dy * S
box = (int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh))
chain = LayerDiffuseChain(model=a.model)
for strength in (0.3, 0.4):
    t = time.time()
    naive, harm, new = chain.relocate(scene, target, box, a.prompt, seed=7, strength=strength, w=S, h=S,
                                      steps=a.steps, guidance=a.guidance, extract="mask")
    final = scene.copy()
    final.alpha_composite(new)
    green = Image.new("RGBA", (S, S), (40, 160, 60, 255))
    green.alpha_composite(new)
    strip = Image.new("RGB", (S * 5, S))
    for i, im in enumerate((orig, naive, harm, green, final)):
        strip.paste(im.convert("RGB"), (S * i, 0))
    name = f"{os.path.basename(os.path.normpath(a.layers_dir))}_L{a.index}_s{int(strength * 100)}"
    strip.save(os.path.join(a.out, name + "_strip.png"))
    print(name, "edit done in", round(time.time() - t, 1), "s", flush=True)
