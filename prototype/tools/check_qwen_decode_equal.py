"""Checks that layerproto.qwen_layered.decompose (grab latents -> free transformer -> decode per layer) gives the
same layers as the stock QwenImageLayeredPipeline decode, using tiny random models and the same seed."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel  # noqa: E402
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration  # noqa: E402

from layerproto import qwen_layered as Q  # noqa: E402

DT, H = torch.float32, 64
torch.manual_seed(0)
cfg = Qwen2_5_VLConfig.from_pretrained(Q.REPO, subfolder="text_encoder")
for c in [cfg] + [getattr(cfg, "text_config", None)]:
    if c is None:
        continue
    for k, v in dict(hidden_size=H, intermediate_size=128, num_hidden_layers=1, num_attention_heads=4,
                     num_key_value_heads=2).items():
        if hasattr(c, k):
            setattr(c, k, v)
    if getattr(c, "rope_scaling", None):
        c.rope_scaling = dict(c.rope_scaling, mrope_section=[2, 3, 3])
    if hasattr(c, "rope_parameters") and isinstance(getattr(c, "rope_parameters"), dict):
        c.rope_parameters = dict(c.rope_parameters, mrope_section=[2, 3, 3])
for k, v in dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=H,
                 fullatt_block_indexes=[0]).items():
    setattr(cfg.vision_config, k, v)
te = Qwen2_5_VLForConditionalGeneration(cfg).to(DT).eval()
vcfg = AutoencoderKLQwenImage.load_config(Q.REPO, subfolder="vae"); vcfg.update(base_dim=8, num_res_blocks=1)
vae = AutoencoderKLQwenImage.from_config(vcfg).to(DT).eval()
tcfg = QwenImageTransformer2DModel.load_config(Q.REPO, subfolder="transformer")
tcfg.update(num_layers=1, num_attention_heads=2, attention_head_dim=16, axes_dims_rope=[4, 6, 6], joint_attention_dim=H)
tr = QwenImageTransformer2DModel.from_config(tcfg).to(DT).eval()
img = Image.new("RGB", (512, 384), (120, 80, 40))  # non-square on purpose (tests calculate_dimensions)
CAP = "a campfire in a forest at night"

pipe = Q.make_pipeline(te, vae)
emb = Q.encode(pipe, CAP, device="cpu")
pe, pm, ne, nm = emb
# stock pipeline decode
pipe.register_modules(transformer=tr)
with torch.no_grad():
    ref = pipe(image=img.convert("RGBA"), prompt=CAP, prompt_embeds=pe, prompt_embeds_mask=pm, negative_prompt_embeds=ne,
               negative_prompt_embeds_mask=nm, true_cfg_scale=4.0, layers=4, resolution=640, num_inference_steps=3,
               cfg_normalize=True, generator=torch.Generator("cpu").manual_seed(777)).images[0]
ref = [r.convert("RGBA") for r in ref]
# our path
Q.unload_text_encoder(pipe, DT)
mine, st = Q.decompose(pipe, tr, img, CAP, emb, steps=3, layers=4, resolution=640, seed=777)
diffs = [float(np.abs(np.array(a, dtype=int) - np.array(b, dtype=int)).max()) for a, b in zip(ref, mine)]
print("sizes ref/mine:", [r.size for r in ref], [m.size for m in mine])
print("max abs pixel diff per layer:", diffs)
print("EQUAL" if len(ref) == len(mine) and max(diffs) <= 1 else "DIFFERENT")
