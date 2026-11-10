r"""Full move loop on GPU: layers -> move/resize one layer -> relocate() -> + shadow layer -> + IC-Light relight.

    python prototype/tools/fox_move_demo.py --layers-dir <dir with layer_0.png (background) ...> --index 5 \
        --prompt "a red fox, storybook illustration" --dx -0.25 --scale 1.2 --out out/

Per strength it saves a strip: original | naive paste | relocate | relocate + shadow | relocate + shadow + IC-Light,
and prints timings. Uses layerproto/backends/layerdiffuse_chain.py, layerproto/shadow.py, layerproto/relight.py.
"""
import argparse
import glob
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from layerproto.backends.layerdiffuse_chain import LayerDiffuseChain  # noqa: E402
from layerproto.shadow import shadow_layer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--layers-dir", required=True)
ap.add_argument("--index", type=int, required=True)
ap.add_argument("--prompt", required=True)
ap.add_argument("--light-prompt", default="soft daylight from the left")
ap.add_argument("--dx", type=float, default=-0.25)
ap.add_argument("--dy", type=float, default=0.0)
ap.add_argument("--scale", type=float, default=1.2)
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--strengths", default="0.3,0.45")
ap.add_argument("--paste-back", type=float, default=0.8)
ap.add_argument("--light-dx", type=float, default=0.7)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
S = a.size
layers = [Image.open(f).convert("RGBA").resize((S, S)) for f in sorted(glob.glob(os.path.join(a.layers_dir, "layer_*.*")))]
target = layers[a.index]
scene = layers[0].copy()
for i, L in enumerate(layers[1:], 1):
    if i != a.index:
        scene.alpha_composite(L)
orig = scene.copy(); orig.alpha_composite(target)
x0, y0, x1, y1 = target.getchannel("A").point(lambda v: 255 if v > 20 else 0).getbbox()
bw, bh = (x1 - x0) * a.scale, (y1 - y0) * a.scale
cx, cy = (x0 + x1) / 2 + a.dx * S, (y0 + y1) / 2 + a.dy * S
box = (int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh))

t = time.time(); chain = LayerDiffuseChain(model="stabilityai/stable-diffusion-xl-base-1.0"); print("sdxl load", round(time.time() - t, 1), "s", flush=True)
results = []
for strength in [float(v) for v in a.strengths.split(",")]:
    t = time.time()
    naive, harm, new = chain.relocate(scene, target, box, a.prompt, seed=7, strength=strength, w=S, h=S, steps=20, guidance=7.0,
                                      extract="mask", free_last=2, band=9, paste_back=a.paste_back)
    t_rel = time.time() - t
    t = time.time(); sh = shadow_layer(new, light_dx=a.light_dx, squash=0.4, opacity=0.55); t_sh = time.time() - t
    results.append((strength, naive, new, sh, t_rel, t_sh))
    print(f"strength {strength}: relocate {t_rel:.1f} s, shadow {t_sh * 1000:.0f} ms", flush=True)

# free SDXL before loading IC-Light (SD1.5)
del chain; torch.cuda.empty_cache()
from layerproto.relight import ICLightFBC  # noqa: E402
t = time.time(); rl = ICLightFBC(device="cuda", dtype=torch.float16); print("ic-light load", round(time.time() - t, 1), "s", flush=True)
for strength, naive, new, sh, t_rel, t_sh in results:
    t = time.time(); lit = rl.relight(new, scene.convert("RGB"), f"{a.prompt}, {a.light_prompt}", steps=20, size=S); t_rl = time.time() - t
    with_shadow = scene.copy(); with_shadow.alpha_composite(sh); with_shadow.alpha_composite(new)
    with_light = scene.copy(); with_light.alpha_composite(sh); with_light.alpha_composite(lit)
    plain = scene.copy(); plain.alpha_composite(new)
    strip = Image.new("RGB", (S * 5, S))
    for i, im in enumerate((orig, naive, plain, with_shadow, with_light)):
        strip.paste(im.convert("RGB"), (S * i, 0))
    name = f"move_L{a.index}_s{int(strength * 100)}_pb{int(a.paste_back * 100)}"
    strip.save(os.path.join(a.out, name + "_strip.png")); lit.save(os.path.join(a.out, name + "_relit_layer.png")); sh.save(os.path.join(a.out, name + "_shadow_layer.png"))
    print(f"{name}: relocate {t_rel:.1f} s, shadow {t_sh * 1000:.0f} ms, ic-light {t_rl:.1f} s", flush=True)
