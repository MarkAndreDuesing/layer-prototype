"""Local reproduction of the GGUF path: tiny Qwen-Image-Layered transformer written as a real GGUF file
(linears Q8_0, the embedding as BF16 like unsloth's files), loaded with diffusers' from_single_file + GGUFQuantizationConfig,
then run through layerproto.qwen_layered.decompose. Without fix_gguf_non_linear it must fail with the same
'3072 vs 6144'-style size error as on Kaggle (here 64 vs 128); with the fix it must run.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.stdout.reconfigure(encoding="utf-8")

import gguf  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from layerproto import qwen_layered as Q  # noqa: E402

torch.manual_seed(0)
DT = torch.float32
H = 64
tmp = tempfile.mkdtemp()

from diffusers import AutoencoderKLQwenImage, GGUFQuantizationConfig, QwenImageTransformer2DModel  # noqa: E402

tcfg = QwenImageTransformer2DModel.load_config(Q.REPO, subfolder="transformer")
tcfg.update(num_layers=1, num_attention_heads=2, attention_head_dim=16, axes_dims_rope=[4, 6, 6], joint_attention_dim=H)
tiny = QwenImageTransformer2DModel.from_config(tcfg)
cfg_dir = os.path.join(tmp, "cfg", "transformer")
tiny.save_config(cfg_dir)

# write the GGUF file
path = os.path.join(tmp, "tiny.gguf")
w = gguf.GGUFWriter(path, arch="qwen_image")
linear_names = {n + ".weight" for n, m in tiny.named_modules() if isinstance(m, torch.nn.Linear)}
for name, t in tiny.state_dict().items():
    a = t.detach().float().numpy()
    if name in linear_names and a.ndim == 2 and a.shape[-1] % 32 == 0:
        q = gguf.quants.quantize(a, gguf.GGMLQuantizationType.Q8_0)
        w.add_tensor(name, q, raw_shape=q.shape, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    elif name.endswith("addition_t_embedding.weight"):
        q = gguf.quants.quantize(a, gguf.GGMLQuantizationType.BF16)  # unsloth stores small tensors as BF16
        w.add_tensor(name, q, raw_shape=q.shape, raw_dtype=gguf.GGMLQuantizationType.BF16)
    elif name.endswith(".bias") and a.shape[-1] % 32 == 0 and "BF16_BIAS" in os.environ:
        q = gguf.quants.quantize(a, gguf.GGMLQuantizationType.BF16)  # stress case: linear biases as BF16 too
        w.add_tensor(name, q, raw_shape=q.shape, raw_dtype=gguf.GGMLQuantizationType.BF16)
    else:
        w.add_tensor(name, a.astype(np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("tiny gguf written:", os.path.getsize(path), "bytes")


def load():
    return QwenImageTransformer2DModel.from_single_file(
        path, quantization_config=GGUFQuantizationConfig(compute_dtype=DT), torch_dtype=DT,
        config=os.path.join(tmp, "cfg"), subfolder="transformer")


# tiny text encoder + VAE exactly as in dryrun_qwen_layered.py
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration  # noqa: E402

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
vcfg = AutoencoderKLQwenImage.load_config(Q.REPO, subfolder="vae")
vcfg.update(base_dim=8, num_res_blocks=1)
vae = AutoencoderKLQwenImage.from_config(vcfg).to(DT).eval()
img = Image.new("RGB", (512, 512), (120, 80, 40))
CAP = "a campfire in a forest at night"


def run(apply_fix):
    pipe = Q.make_pipeline(te, vae)
    emb = Q.encode(pipe, CAP, device="cpu")
    object.__setattr__(pipe, "text_encoder", type("S", (), {"dtype": DT})())
    tr = load()
    if apply_fix:
        print("  fixed params:", Q.fix_gguf_non_linear(tr, DT))
    layers, st = Q.decompose(pipe, tr, img, CAP, emb, steps=2, layers=4, resolution=640)
    return len(layers), st


try:
    run(apply_fix=False)
    print("WITHOUT FIX: ran (unexpected - bug not reproduced)")
except RuntimeError as e:
    print("WITHOUT FIX: reproduced ->", str(e)[:120])
n, st = run(apply_fix=True)
print(f"WITH FIX: OK, {n} layers, {st}")
