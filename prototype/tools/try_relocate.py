"""Try the training-free move/resize re-harmonisation (LayerDiffuseChain.relocate) on a saved chain scene.

    fastsdcpu\\env\\Scripts\\python.exe prototype\\tools\\try_relocate.py [--model stabilityai/sdxl-turbo --steps 4 --guidance 1]

Uses prototype/scenes/_chain/base20_0_bg.png (background) and base20_2_layer.png (the extracted element).
Writes prototype/scenes/_relocate/<tag>_*.png and a strip: naive paste | re-harmonised | new layer | recomposite.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vendor", "diffuser_layerdiffuse"))
from PIL import Image  # noqa: E402

from layerproto.backends.layerdiffuse_chain import LayerDiffuseChain  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--guidance", type=float, default=7.0)
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--scene-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scenes", "_chain"))
ap.add_argument("--prompt", default="a skeleton")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scenes", "_relocate"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
S = a.size
bg = Image.open(os.path.join(a.scene_dir, "base20_0_bg.png"))
layer = Image.open(os.path.join(a.scene_dir, "base20_2_layer.png"))
chain = LayerDiffuseChain(model=a.model)
# the edits: move left and shrink, move right and enlarge
EDITS = {"left_small": (int(S * 0.05), int(S * 0.40), int(S * 0.35), int(S * 0.55)),
         "right_big": (int(S * 0.45), int(S * 0.15), int(S * 0.50), int(S * 0.82))}
for tag, box in EDITS.items():
    for strength in (0.4, 0.6):
        t = time.time()
        naive, harm, new = chain.relocate(bg, layer, box, a.prompt, seed=7, strength=strength, w=S, h=S,
                                          steps=a.steps, guidance=a.guidance)
        recomp = bg.convert("RGBA").resize((S, S))
        recomp.alpha_composite(new)
        green = Image.new("RGBA", (S, S), (40, 160, 60, 255))
        green.alpha_composite(new)
        strip = Image.new("RGB", (S * 4, S))
        for i, im in enumerate((naive, harm, green, recomp)):
            strip.paste(im.convert("RGB"), (S * i, 0))
        name = f"{tag}_s{int(strength * 100)}"
        strip.save(os.path.join(a.out, name + "_strip.png"))
        new.save(os.path.join(a.out, name + "_layer.png"))
        print(name, "done in", round(time.time() - t, 1), "s", flush=True)
