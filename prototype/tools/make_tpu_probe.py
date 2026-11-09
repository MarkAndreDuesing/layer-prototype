r"""Kaggle TPU probe for Qwen-Image-Layered (measure, don't estimate). Separate 20 h/week TPU quota.

    fastsdcpu\env\Scripts\python.exe prototype\tools\make_tpu_probe.py
    cd prototype\kaggle\upload_tpu_probe && python -m kaggle.cli kernels push -p .

Stages (each prints a result line "STAGE n ..." and continues on failure; 55-min cap):
  1 environment: chip type/count, HBM, torch / torch_xla / jax / diffusers versions
  2 bf16 matmul throughput on one chip and sharded over all chips (SPMD)
  3 load the Qwen-Image-Layered transformer in bf16 (40 GB) and shard every Linear weight across the chips
    (template: diffusers examples/research_projects/pytorch_xla/inference/flux/flux_inference_spmd.py)
  4 one decomposition: 384 px, 4 layers, 4 steps, CFG 1 (our T4 setting, 27-35 s there); text encoder on CPU;
    first call includes XLA compile, second call is the steady-state time
TPU runs bf16 natively, so fp16_mixed is not needed here.
"""
import base64
import io
import json
import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
ROOT = os.path.dirname(PROTO)
OUT = os.path.join(PROTO, "kaggle", "upload_tpu_probe")
SCENE = "experiments/blkE__P2__textonly__sdxlturbo-steps4-g1.0__seed11__impressionist.png"
buf = io.BytesIO()
Image.open(os.path.join(ROOT, SCENE)).convert("RGB").resize((384, 384)).save(buf, "JPEG", quality=90)

src = r'''import signal, time as _t
_T0 = _t.time()
def _cap(*a): raise TimeoutError("55 min cap hit after %.0fs" % (_t.time() - _T0))
signal.signal(signal.SIGALRM, _cap); signal.alarm(55 * 60)
import subprocess, os, sys, time, json, base64, io, traceback
def sh(c, check=False):
    print("$", c, flush=True); r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print((r.stdout + r.stderr)[-3000:], flush=True); return r.returncode
R = {}
def stage(n, fn):
    t0 = time.time()
    try:
        out = fn(); R[n] = {"ok": True, "s": round(time.time() - t0, 1), **(out or {})}
    except BaseException as e:
        traceback.print_exc(); R[n] = {"ok": False, "s": round(time.time() - t0, 1), "error": repr(e)[:400]}
        if isinstance(e, TimeoutError): raise
    finally:
        print(f"STAGE {n}", json.dumps(R.get(n)), flush=True); json.dump(R, open("/kaggle/working/tpu_probe.json", "w"), indent=1)

# ---------------- 0 chips via the preinstalled JAX (Kaggle TPU image: torch CPU build + JAX, no torch_xla)
def s0():
    import jax, jax.numpy as jnp
    d = jax.devices(); out = {"jax": jax.__version__, "n": len(d), "kind": d[0].device_kind}
    try:
        out["hbm_GiB"] = round(d[0].memory_stats()["bytes_limit"] / 2**30, 1)
    except Exception:
        pass
    n = 8192; a = jnp.ones((n, n), jnp.bfloat16); f = jax.jit(lambda x: x @ x)
    f(a).block_until_ready(); t = time.time()
    for _ in range(10): r = f(a)
    r.block_until_ready(); out["one_chip_bf16_TFLOPS"] = round(10 * 2 * n**3 / (time.time() - t) / 1e12, 1)
    return out
stage(0, s0)

# ---------------- 1 environment (install torch_xla matching the image's torch)
def s1():
    import torch
    v = torch.__version__.split("+")[0]; mm = ".".join(v.split(".")[:2])
    # pip has torch_xla up to 2.9 (probe v2); Kaggle's image ships torch 2.11 -> pin torch 2.9 to match
    sh("pip -q install 'torch==2.9.0' 'torchvision==0.24.0' 'torch_xla[tpu]==2.9.0' -f https://storage.googleapis.com/libtpu-releases/index.html")
    sh("pip list 2>/dev/null | grep -iE '^(torch|torch.xla|torch_xla|jax|jaxlib|libtpu|diffusers|transformers|accelerate) '")
    sh("env | grep -iE 'TPU|XLA' | head -20")
    import torch, torch_xla, torch_xla.core.xla_model as xm, torch_xla.runtime as xr
    dev = xm.xla_device()
    info = {"torch": torch.__version__, "torch_xla": torch_xla.__version__, "devices": xr.global_runtime_device_count(),
            "device_kind": str(xm.xla_device_hw(dev))}
    try:
        info["mem"] = {k: round(v / 2**30, 1) for k, v in xm.get_memory_info(dev).items() if isinstance(v, (int, float))}
    except Exception as e:
        info["mem"] = repr(e)[:100]
    return info
stage(1, s1)

# ---------------- 2 matmul throughput, single chip and SPMD over all chips
def s2():
    import torch, torch_xla.core.xla_model as xm, torch_xla.runtime as xr
    import torch_xla.distributed.spmd as xs, numpy as np
    dev = xm.xla_device(); n = 8192; out = {}
    a = torch.randn(n, n, dtype=torch.bfloat16, device=dev); b = torch.randn(n, n, dtype=torch.bfloat16, device=dev)
    for _ in range(2): c = a @ b; xm.mark_step()
    xm.wait_device_ops(); t = time.time()
    for _ in range(10): c = a @ b; xm.mark_step()
    xm.wait_device_ops(); out["one_chip_TFLOPS"] = round(10 * 2 * n**3 / (time.time() - t) / 1e12, 1)
    return out
stage(2, s2)

# ---------------- 3 sharded load of the 20B transformer
pipe = None
def s3():
    global pipe
    sh("pip -q install -U 'diffusers>=0.36' 'transformers>=4.57,<5' accelerate")
    import torch, torch_xla.core.xla_model as xm, torch_xla.runtime as xr
    import torch_xla.distributed.spmd as xs, numpy as np
    xr.use_spmd()
    from diffusers import QwenImageLayeredPipeline
    n = xr.global_runtime_device_count()
    mesh = xs.Mesh(np.arange(n), (n, 1), ("x", "y")); xs.set_global_mesh(mesh)
    t = time.time()
    pipe = QwenImageLayeredPipeline.from_pretrained("Qwen/Qwen-Image-Layered", torch_dtype=torch.bfloat16)
    dl = round(time.time() - t, 1)
    dev = xm.xla_device()
    tr = pipe.transformer.to(dev)
    sharded = 0
    for name, p in tr.named_parameters():
        if p.dim() == 2 and p.shape[0] % n == 0:
            xs.mark_sharding(p, mesh, ("x", None)); sharded += 1
    pipe.vae.to(dev)
    xm.mark_step(); xm.wait_device_ops()
    mem = {}
    try:
        mem = {k: round(v / 2**30, 1) for k, v in xm.get_memory_info(dev).items() if isinstance(v, (int, float))}
    except Exception:
        pass
    return {"download_load_s": dl, "sharded_params": sharded, "mem_after": mem}
stage(3, s3)

# ---------------- 4 one decomposition (twice: compile + steady state)
def s4():
    import torch, torch_xla.core.xla_model as xm
    from PIL import Image
    img = Image.open(io.BytesIO(base64.b64decode("__SCENE__"))).convert("RGBA")
    dev = xm.xla_device()
    t = time.time()
    with torch.no_grad():   # text encoder on CPU, embeddings moved to the TPU
        pe, pm = pipe.encode_prompt(prompt="", device="cpu", num_images_per_prompt=1, max_sequence_length=256)
    enc = round(time.time() - t, 1)
    pe = pe.to(dev); pm = (pm if pm is not None else torch.ones(pe.shape[:2], dtype=torch.long)).to(dev)
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_layered as _pl
    _orig = _pl.calculate_dimensions
    _pl.calculate_dimensions = lambda area, ratio: _orig(384 * 384, ratio)
    times = []
    for k in range(2):
        t = time.time()
        with torch.no_grad():
            out = pipe(image=img, prompt_embeds=pe, prompt_embeds_mask=pm, negative_prompt_embeds=pe, negative_prompt_embeds_mask=pm,
                       true_cfg_scale=1.0, layers=4, resolution=640, num_inference_steps=4, cfg_normalize=False,
                       generator=torch.Generator().manual_seed(777))
        xm.mark_step(); xm.wait_device_ops(); times.append(round(time.time() - t, 1))
        print(f"split {k}: {times[-1]} s", flush=True)
    layers = out.images[0] if isinstance(out.images[0], list) else out.images
    os.makedirs("/kaggle/working/tpu_layers", exist_ok=True)
    for i, l in enumerate(layers): l.save(f"/kaggle/working/tpu_layers/layer_{i}.png")
    return {"encode_cpu_s": enc, "split_first_s": times[0], "split_second_s": times[1], "n_layers": len(layers)}
if pipe is not None:
    stage(4, s4)
print("TOTAL", round(time.time() - _T0), "s", json.dumps(R), flush=True)
'''
src = src.replace("__SCENE__", base64.b64encode(buf.getvalue()).decode())
compile(src, "cell", "exec")
os.makedirs(OUT, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": src.strip("\n").splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(OUT, "tpu-probe.ipynb"), "w", encoding="utf-8"), indent=1)
json.dump({"id": "markandreduesing/tpu-probe", "title": "tpu-probe", "code_file": "tpu-probe.ipynb", "language": "python",
           "kernel_type": "notebook", "is_private": "true", "enable_gpu": "false", "enable_tpu": "true",
           "enable_internet": "true", "machine_shape": "TpuV5E8", "dataset_sources": [], "competition_sources": [],
           "kernel_sources": [], "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("wrote", OUT)
