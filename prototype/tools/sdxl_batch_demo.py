r"""One GPU session, two short SDXL jobs (one notebook, models loaded one after the other):
  A. amodal completion of the cottage: SDXL-inpaint fills the part hidden by the tree / gate / fox, so the cottage layer becomes whole
     (masks precomputed on the CPU in make_sdxl_batch_trial.py and shipped with the notebook);
  B. LayerDiffuse effect layers (fog, lantern glow, snowfall) drawn into the fox & cottage scene with bg2ble -> bgble2fg at 1024 px.

    python prototype/tools/sdxl_batch_demo.py --data data/ --out out/
"""
import argparse
import gc
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
S = 1024
ld = lambda n, mode="RGBA": Image.open(os.path.join(a.data, n)).convert(mode).resize((S, S), Image.LANCZOS)

# ---------------------------------------------------------------- A. amodal completion
t0 = time.time()
from diffusers import AutoencoderKL, StableDiffusionXLInpaintPipeline  # noqa: E402

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLInpaintPipeline.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", vae=vae,
                                                        torch_dtype=torch.float16, variant="fp16").to("cuda")
print("inpaint load", round(time.time() - t0, 1), "s", flush=True)
bg, cottage, tree = ld("bg.png"), ld("cottage.png"), ld("tree.png")
hidden = ld("hidden_mask.png", "L")
base = bg.copy()
base.alpha_composite(cottage)
PROMPTS = {
    "a": "a thatched stone cottage with a brown thatched roof, cream walls, small windows and a wooden door, storybook illustration",
    "b": "the left half of a thatched stone cottage, roof and walls continuing behind, storybook illustration, no tree",
}
for tag, prompt in PROMPTS.items():
    t = time.time()
    out = pipe(prompt=prompt, negative_prompt="tree, trunk, fox, animal, person, text, watermark", image=base.convert("RGB"), mask_image=hidden,
               num_inference_steps=30, strength=0.99, guidance_scale=7.0, generator=torch.Generator("cuda").manual_seed(5)).images[0]
    out.save(f"{a.out}/amodal_{tag}_inpainted.png")
    # completed cottage layer: its own pixels plus the filled hidden region
    h = np.asarray(hidden, np.float32) / 255
    alpha = np.maximum(np.asarray(cottage.getchannel("A"), np.float32) / 255, h)
    rgb = np.where(h[..., None] > 0.5, np.asarray(out.convert("RGB"), np.float32), np.asarray(cottage.convert("RGB"), np.float32))
    done = Image.fromarray(np.dstack([rgb, alpha * 255]).astype(np.uint8), "RGBA")
    done.save(f"{a.out}/amodal_{tag}_cottage_layer.png")
    print("amodal", tag, round(time.time() - t, 1), "s", flush=True)
del pipe, vae
gc.collect()
torch.cuda.empty_cache()

# ---------------------------------------------------------------- B. effect layers with the LayerDiffuse chain
from layerproto.backends.chain_base import ChainParams  # noqa: E402
from layerproto.backends.layerdiffuse_chain import LayerDiffuseChain  # noqa: E402

t0 = time.time()
chain = LayerDiffuseChain(model="stabilityai/stable-diffusion-xl-base-1.0")
print("chain load", round(time.time() - t0, 1), "s", flush=True)
scene = ld("scene.png")
EFFECTS = {
    "fog": "thick white fog drifting low over the ground and the path, soft mist, storybook illustration",
    "glow": "warm golden glow of a lantern with soft light rays, storybook illustration",
    "snow": "heavy snowfall, many white snowflakes in the air, storybook illustration",
}
for name, prompt in EFFECTS.items():
    t = time.time()
    p = ChainParams(prompt=prompt, seed=31, width=S, height=S, steps=20, guidance=7.0, negative_prompt="bad, ugly")
    blended = chain.blend(scene, p)
    layer = chain.extract(scene, blended, p).convert("RGBA")
    layer.save(f"{a.out}/effect_{name}_layer.png")
    comp = scene.copy()
    comp.alpha_composite(layer)
    comp.convert("RGB").save(f"{a.out}/effect_{name}_composite.png")
    hist = layer.split()[-1].histogram()
    print(f"effect {name}: {time.time() - t:.1f} s, transparent {100 * sum(hist[:10]) / max(1, sum(hist)):.0f}%", flush=True)
