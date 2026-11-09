r"""Kaggle 2x T4 control run: plain FLUX.1-dev GGUF Q4_K_S, no layer LoRA, to separate "FLUX on a T4 is broken" from
"RevealLayer's own code is broken" (RevealLayer run 5 gave pure noise).

    fastsdcpu\env\Scripts\python.exe prototype\tools\make_flux_control_trial.py
    cd prototype\kaggle\upload_flux_control_trial && python -m kaggle.cli kernels push -p .

Two variants, same prompt and seed, 512 px, 16 steps (non-finite block outputs are counted):
  A  plain fp16 (diffusers' own clamp in the single blocks only)
  B  fp16_mixed (ours: fp32 residual stream, scaled fp16 linears)
Encoders and VAE from ungated ostris/Flex.1-alpha. FLUX.1-dev Non-Commercial licence applies.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
OUT = os.path.join(PROTO, "kaggle", "upload_flux_control_trial")
MIXED = open(os.path.join(PROTO, "layerproto", "fp16_mixed.py"), encoding="utf-8").read()

src = r'''import signal, time as _t
_T0 = _t.time()
def _cap(*a): raise TimeoutError("25 min cap hit after %.0fs" % (_t.time() - _T0))
signal.signal(signal.SIGALRM, _cap); signal.alarm(25 * 60)
import subprocess, os, json, time
def sh(c): print(c, flush=True); subprocess.run(c, shell=True, check=True)
sh("pip -q install 'transformers==4.57.1' diffusers==0.35.2 gguf>=0.10 sentencepiece protobuf")
import torch, numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig, FlowMatchEulerDiscreteScheduler
gguf = hf_hub_download("city96/FLUX.1-dev-gguf", "flux1-dev-Q4_K_S.gguf")
BASE = snapshot_download("ostris/Flex.1-alpha", allow_patterns=["model_index.json","text_encoder/*","text_encoder_2/*","tokenizer/*","tokenizer_2/*","vae/*","scheduler/*"])
CFG = "/kaggle/working/flux_dev_cfg"; os.makedirs(f"{CFG}/transformer", exist_ok=True)
json.dump({"_class_name":"FluxTransformer2DModel","attention_head_dim":128,"axes_dims_rope":[16,56,56],"guidance_embeds":True,
           "in_channels":64,"joint_attention_dim":4096,"num_attention_heads":24,"num_layers":19,"num_single_layers":38,
           "patch_size":1,"pooled_projection_dim":768}, open(f"{CFG}/transformer/config.json","w"))
sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True, base_shift=0.5,
                                        max_shift=1.15, base_image_seq_len=256, max_image_seq_len=4096)
D0, D1, DT = "cuda:0", "cuda:1", torch.float16
PROMPT = "an impressionist oil painting of a campfire in a pine forest at night, full moon"
tp = FluxPipeline.from_pretrained(BASE, transformer=None, vae=None, torch_dtype=DT).to(D1)
with torch.inference_mode():
    pe, ppe, _ = tp.encode_prompt(PROMPT, prompt_2=None, device=D1, max_sequence_length=256)
print("prompt embeds finite:", bool(torch.isfinite(pe).all()), bool(torch.isfinite(ppe).all()), flush=True)
pe, ppe = pe.to(D0), ppe.to(D0); del tp; torch.cuda.empty_cache()
os.makedirs("/kaggle/working/flux_out", exist_ok=True)

def load():
    t = FluxTransformer2DModel.from_single_file(gguf, config=CFG, subfolder="transformer",
            quantization_config=GGUFQuantizationConfig(compute_dtype=DT), torch_dtype=DT)
    return t
nonfinite = {"n": 0}
def watch(m, i, o):
    x = o[0] if isinstance(o, tuple) else o
    if torch.is_tensor(x) and not torch.isfinite(x).all(): nonfinite["n"] += 1

def run(name, t):
    pipe = FluxPipeline.from_pretrained(BASE, transformer=t, text_encoder=None, text_encoder_2=None, tokenizer=None,
                                        tokenizer_2=None, scheduler=sched, torch_dtype=DT).to(D0)
    pipe.vae.to(torch.float32)
    hs = [b.register_forward_hook(watch) for b in list(t.transformer_blocks) + list(t.single_transformer_blocks)]
    nonfinite["n"] = 0; torch.cuda.reset_peak_memory_stats(0); t0 = time.time()
    try:
        with torch.inference_mode():
            lat = pipe(prompt_embeds=pe, pooled_prompt_embeds=ppe, height=512, width=512, num_inference_steps=16,
                       guidance_scale=3.5, output_type="latent", generator=torch.Generator(D0).manual_seed(7)).images
            lat = pipe._unpack_latents(lat, 512, 512, pipe.vae_scale_factor).float()
            lat = lat / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
            img = pipe.vae.decode(lat).sample
        arr = ((img[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
        Image.fromarray(arr).save(f"/kaggle/working/flux_out/{name}.png")
        print(f"{name}: {time.time()-t0:.0f}s, non-finite block outputs {nonfinite['n']}, pixel std {arr.std():.1f}, "
              f"peak GiB {torch.cuda.max_memory_allocated(0)/2**30:.1f}", flush=True)
    except BaseException as e:
        import traceback; traceback.print_exc(); print("FAILED", name, flush=True)
        if isinstance(e, TimeoutError): raise
    finally:
        for h in hs: h.remove()
    del pipe; torch.cuda.empty_cache()

t = load().to(D0)
run("A_plain_fp16", t)
exec(compile(__MIXED__, "fp16_mixed", "exec"))
apply_fp16_mixed(t)
run("B_fp16_mixed", t)
print("total", round(time.time() - _T0), "s", flush=True)
'''
src = src.replace("__MIXED__", repr(MIXED))
compile(src, "cell", "exec")
os.makedirs(OUT, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": src.strip("\n").splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(OUT, "flux-control-trial.ipynb"), "w", encoding="utf-8"), indent=1)
json.dump({"id": "markandreduesing/flux-control-trial", "title": "flux-control-trial", "code_file": "flux-control-trial.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true",
           "enable_tpu": "false", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
           "competition_sources": [], "kernel_sources": [], "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("wrote", OUT)
