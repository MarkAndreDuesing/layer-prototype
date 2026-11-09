"""Local dry run of layerproto/qwen_layered.py with TINY random models (same architectures, same pipeline code).

Verifies the whole call path (pipeline construction, prompt encoding + masks, text-encoder unload,
transformer registration, decomposition call, output handling, recomposition) in seconds on CPU,
without downloading the 30 GB of real weights. Only the real repo's tokenizer/processor/scheduler/configs are fetched.
Output quality is meaningless here; only "does every step run" is tested.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.stdout.reconfigure(encoding="utf-8")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from layerproto import qwen_layered as Q  # noqa: E402

torch.manual_seed(0)
DT = torch.float32  # CPU; on Kaggle the same code runs in float16
CAPTION = "A forest clearing at night with a campfire, a chair, pine trees and a full moon."
img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experiments",
    "blkE__P2__textonly__sdxlturbo-steps4-g1.0__seed11__impressionist.png")
img = Image.open(img_path).convert("RGB")
t_all = time.time()

# ---- tiny text encoder: real Qwen2.5-VL config, shrunk
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration  # noqa: E402

cfg = Qwen2_5_VLConfig.from_pretrained(Q.REPO, subfolder="text_encoder")
H = 64
for c in [cfg] + [getattr(cfg, "text_config", None)]:
    if c is None:
        continue
    for k, v in dict(hidden_size=H, intermediate_size=128, num_hidden_layers=1, num_attention_heads=4,
                     num_key_value_heads=2).items():
        if hasattr(c, k):
            setattr(c, k, v)
    if getattr(c, "rope_scaling", None):
        c.rope_scaling = dict(c.rope_scaling, mrope_section=[2, 3, 3])  # sums to head_dim/2 = 8
    if hasattr(c, "rope_parameters") and isinstance(getattr(c, "rope_parameters"), dict):
        c.rope_parameters = dict(c.rope_parameters, mrope_section=[2, 3, 3])
vc = cfg.vision_config
for k, v in dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=H,
                 fullatt_block_indexes=[0]).items():
    setattr(vc, k, v)
te = Qwen2_5_VLForConditionalGeneration(cfg).to(DT).eval()

# ---- tiny VAE and transformer: real configs, shrunk (z_dim / channels kept so shapes match the real model)
from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel  # noqa: E402

vcfg = AutoencoderKLQwenImage.load_config(Q.REPO, subfolder="vae")
vcfg.update(base_dim=8, num_res_blocks=1)
vae = AutoencoderKLQwenImage.from_config(vcfg).to(DT).eval()
tcfg = QwenImageTransformer2DModel.load_config(Q.REPO, subfolder="transformer")
tcfg.update(num_layers=1, num_attention_heads=2, attention_head_dim=16, axes_dims_rope=[4, 6, 6], joint_attention_dim=H)
tr = QwenImageTransformer2DModel.from_config(tcfg).to(DT).eval()
print(f"tiny models built ({time.time() - t_all:.1f}s)")

# ---- exactly the code path the Kaggle notebook uses
pipe = Q.make_pipeline(te, vae)
emb = Q.encode(pipe, CAPTION, device="cpu")
print("embeds:", [tuple(e.shape) for e in emb], "masks given:", emb[1] is not None and emb[3] is not None)
Q.unload_text_encoder(pipe, DT)
layers, stats = Q.decompose(pipe, tr, img, CAPTION, emb, steps=2, layers=4, resolution=640,
                            on_step=lambda i: print("  step", i))
canvas, info = Q.recompose(layers, img.size)
print("stats:", stats)
print("layers:", len(layers), info)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scenes", "_qwen_dryrun.png")
Q.strip([img, canvas] + layers).save(out)
print(f"DRY RUN OK in {time.time() - t_all:.1f}s -> {os.path.abspath(out)}")
