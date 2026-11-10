"""Profile `qwen21`: Qwen-Image-2.1-Turbo (7B, RGBA) on the two T4s of a Kaggle session; serves `qwen21-extract` on /decompose
and layer edits with the alpha retry.

    PIP = packages to install first        load() -> loaded pipeline (about 240 s: downloads + text encoder split over both GPUs)

Moved out of tools/make_qwen21_*_trial.py (the code that produced the measured runs, unchanged in behaviour):
  * transformer: unsloth GGUF Q4_K_S on GPU 0, its non-Linear GGUF parameters dequantised, `fp16_mixed` (Qwen models overflow plain fp16)
  * Qwen3-VL text encoder in fp16 split over both GPUs (device_map)
  * VAE: madebyollin/texture-fix-vae-for-qwen-image-2.1 (the stock VAE gives NaNs in fp16 on transparent pixels)
Qwen Research Licence (non-commercial) applies.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

PIP = ["transformers>=5.17", "gguf>=0.10", "accelerate", "git+https://github.com/huggingface/diffusers"]
REPO = "Qwen/Qwen-Image-2.1-Turbo"
GGUF_REPO = "unsloth/Qwen-Image-2.1-Turbo-GGUF"
GGUF_FILE = "qwen-image-2.1-turbo-Q4_K_S.gguf"
VAE_REPO = "madebyollin/texture-fix-vae-for-qwen-image-2.1"
RGBA_PROMPT = "This is an RGBA image with transparency. {}. The image has alpha channel and the background is transparent."


def pip_install(packages=PIP):
    cmd = [sys.executable, "-m", "pip", "-q", "install"] + list(packages)
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load(log=print):
    """Return the QwenImage21Pipeline with everything placed (transformer + VAE on cuda:0, text encoder split over both GPUs)."""
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download

    from ..fp16_mixed import apply_fp16_mixed

    t0 = time.time()
    from diffusers import (AutoencoderKLQwenImage21, GGUFQuantizationConfig, QwenImage21Pipeline,
                           QwenImage21Transformer2DModel)
    from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter, dequantize_gguf_tensor
    from transformers import AutoModelForImageTextToText

    d0, dt = torch.device("cuda:0"), torch.float16
    base = snapshot_download(REPO, ignore_patterns=["transformer/*.safetensors", "vae/*.safetensors", "assets/*"])
    gguf = hf_hub_download(GGUF_REPO, GGUF_FILE)
    log(f"downloads done {time.time() - t0:.0f} s")
    t = QwenImage21Transformer2DModel.from_single_file(gguf, config=base, subfolder="transformer",
                                                       quantization_config=GGUFQuantizationConfig(compute_dtype=dt), torch_dtype=dt)
    fixed = 0                                         # diffusers' GGUF loader only dequantises nn.Linear
    for m in t.modules():
        for n, p in list(m.named_parameters(recurse=False)):
            if isinstance(m, GGUFLinear) and n == "weight":
                continue
            if isinstance(p, GGUFParameter):
                setattr(m, n, torch.nn.Parameter(dequantize_gguf_tensor(p).to(dt), requires_grad=False))
                fixed += 1
    log(f"dequantised non-linear GGUF params: {fixed}")
    apply_fp16_mixed(t)
    t.to(d0).eval()
    te = AutoModelForImageTextToText.from_pretrained(f"{base}/text_encoder", torch_dtype=dt, device_map="auto",
                                                     max_memory={0: "5GiB", 1: "14GiB", "cpu": "24GiB"})
    vae = AutoencoderKLQwenImage21.from_pretrained(VAE_REPO, torch_dtype=dt)
    pipe = QwenImage21Pipeline.from_pretrained(base, transformer=t, text_encoder=te, vae=vae, torch_dtype=dt)
    pipe.vae.to(d0)
    type(pipe)._execution_device = property(lambda self: d0)
    log(f"loaded {time.time() - t0:.0f} s; GiB {[round(torch.cuda.memory_allocated(i) / 2**30, 1) for i in range(2)]}")
    return pipe


def edit_layer(pipe, layer, prompt: str, seed: int = 42, size: int = 512):
    """One RGBA layer edit (relight / reprompt) for layer_edit.edit_layer_keep_alpha: editor(layer, seed) -> layer."""
    import torch

    with torch.inference_mode():
        return pipe(prompt=RGBA_PROMPT.format(prompt), image=[layer], width=size, height=size, output_resolution=size,
                    use_kv_cache=True, generator=torch.Generator(pipe._execution_device).manual_seed(seed)).images[0]


def make(log=print):
    """Factory used by the profile registry: a MethodRouter serving `qwen21-extract` on the loaded pipeline."""
    from ..backends.decomposer import MethodRouter
    from ..backends.qwen21_decomposer import Qwen21Decomposer
    return MethodRouter({"qwen21-extract": Qwen21Decomposer(load(log))})
