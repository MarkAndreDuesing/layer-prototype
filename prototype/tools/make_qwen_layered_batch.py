"""Lean batch version of the Qwen-Image-Layered feasibility notebook (Kaggle "Save & Run All", no editor needed).

    fastsdcpu\\env\\Scripts\\python.exe prototype\\tools\\make_qwen_layered_test.py
    fastsdcpu\\env\\Scripts\\python.exe prototype\\tools\\make_qwen_layered_batch.py

Keeps: setup, read check, stage 1, transformer load, helper definitions, two-GPU split, mixed-precision
smoke test (2 steps, saved as mixed_smoke_strip.png), the 50-step run and the results.
Drops: the failed fp16 experiments and the single-GPU smoke run (both only produce NaNs).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(HERE), "kaggle", "upload_qwen_test")
OUT = os.path.join(os.path.dirname(HERE), "kaggle", "upload_qwen_batch")
nb = json.load(open(os.path.join(SRC_DIR, "qwen-layered-feasibility.ipynb"), encoding="utf-8"))


def src(c):
    return "".join(c["source"])


def setsrc(c, s):
    c["source"] = s.splitlines(True)


PROBE = """
stats['mixed_smoke'] = _st7; save()
if _BAD:   # find the first module (execution order) whose output is non-finite, 1 step, then stop the run
    class _Found(Exception):
        pass
    def _amax(ts):
        return [round(t.detach().float().nan_to_num(0, 0, 0).abs().max().item(), 1) for t in ts if torch.is_tensor(t) and t.is_floating_point()]
    def _mk(name):
        def _hook(m, i, o):
            for t in (o if isinstance(o, tuple) else (o,)):
                if torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all():
                    raise _Found(f'{name} ({type(m).__name__}) in {_amax(i)} out {_amax(o if isinstance(o, tuple) else (o,))} dtype {t.dtype}')
        return _hook
    _hs = [m.register_forward_hook(_mk(n)) for n, m in tr.named_modules() if n]
    try:
        split_across_gpus(tr); Q.decompose(pipe, tr, img, CAPTION, emb, steps=1, layers=4, resolution=640)
        stats['first_nonfinite'] = 'none inside the transformer (NaN arises outside it)'
    except _Found as e:
        stats['first_nonfinite'] = str(e)
    finally:
        for h in _hs:
            h.remove()
    print('FIRST NON-FINITE:', stats['first_nonfinite'], flush=True); save()
assert not _BAD, 'NaN/inf still present - stopping before the 50-step run'
"""
SWEEP = """
# Run v9: (1) Stable-Layers with Heun via Q.HeunVelocity (the pipeline's Euler step becomes an exact Heun step, so the
# custom-sigma error of FlowMatchHeunDiscreteScheduler in v8 is avoided); (2) 2-layer splits (fewer frames -> faster);
# base model for comparison. All 384 px, CFG 1, both T4s.
import numpy as np, json as _json
from huggingface_hub import hf_hub_download
from layerproto.fp16_mixed import add_lora, remove_lora
cfg = _json.load(open(hf_hub_download('StabilityLabs/Stable-Layers', 'model/adapter_config.json')))
lp = hf_hub_download('StabilityLabs/Stable-Layers', 'model/adapter_model.safetensors')
lscale = cfg.get('lora_alpha', cfg.get('r', 1)) / cfg.get('r', 1)
IMGS = {'forest': (img, emb), **{k: (EXTRA[k][0], emb_x[k]) for k in EXTRA}}
import math
from huggingface_hub import hf_hub_download
from diffusers import FlowMatchEulerDiscreteScheduler
LIGHT_SCHED = dict(base_image_seq_len=256, base_shift=math.log(3), invert_sigmas=False, max_image_seq_len=8192,
                   max_shift=math.log(3), num_train_timesteps=1000, shift=1.0, shift_terminal=None,
                   stochastic_sampling=False, time_shift_type='exponential', use_beta_sigmas=False,
                   use_dynamic_shifting=True, use_exponential_sigmas=False, use_karras_sigmas=False)
orig_sched = pipe.scheduler
lightp = hf_hub_download('lightx2v/Qwen-Image-Lightning', 'Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors')
CONFIGS = [  # (variant, sampler, image, resolution, layers, steps)
    # v11: which text-to-image generator gives the most separable scene? FLUX.1-dev (plain control run), Qwen-Image-2.1 (t2i), SDXL-Turbo (fox, forest)
    ('stable_layers', 'heun', 'flux_campfire', 384, 4, 4), ('base', 'euler', 'flux_campfire', 384, 4, 4), ('base', 'euler', 'flux_campfire', 384, 6, 4),
    ('stable_layers', 'heun', 'q21_t2i', 384, 5, 4), ('base', 'euler', 'q21_t2i', 384, 5, 4),
    ('stable_layers', 'heun', 'fox_cottage', 384, 5, 4), ('base', 'euler', 'fox_cottage', 384, 5, 4),
    ('stable_layers', 'heun', 'forest', 384, 4, 4), ('base', 'euler', 'forest', 384, 4, 4),
]
stats['sweep'] = []
split_across_gpus(tr)
for variant, sampler, key, res, nl, steps in CONFIGS:
    remove_lora(tr)
    pipe.scheduler = orig_sched
    if variant.startswith('lightning'):
        hit, miss = add_lora(tr, lightp, scale=float(variant[len('lightning'):]))
        stats.setdefault('light_match', [hit, len(miss)])
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(LIGHT_SCHED)
    if variant == 'stable_layers':
        hit, miss = add_lora(tr, lp, scale=lscale)
        stats.setdefault('lora_match', [hit, len(miss)])
    im, e = IMGS[key]
    torch.cuda.synchronize(); t = time.time()
    try:
        out, st = Q.decompose(pipe, tr, im, CAPTION, e, steps=steps, layers=nl, resolution=res, true_cfg=1.0, offload=False, sampler=sampler)
    except Exception as ex:
        print('FAILED', variant, sampler, key, repr(ex)[:300], flush=True); stats['sweep'].append(dict(variant=variant, sampler=sampler, image=key, error=repr(ex)[:300])); save(); continue
    total = round(time.time() - t, 1)
    canvas, info = Q.recompose(out, im.size)
    tag = f'{variant}_{sampler}_{key}_{res}px_{nl}L_{steps}st'
    row = dict(variant=variant, sampler=sampler, image=key, res=res, layers=nl, steps=steps, total_s=total, **st,
               transparent_pct=[i['transparent_pct'] for i in info],
               recomposite_diff=round(float(np.abs(np.array(canvas.convert('RGB').resize(im.size), dtype=int) - np.array(im, dtype=int)).mean()), 2))
    stats['sweep'].append(row); save(); print(row, flush=True)
    Q.strip([im, canvas] + out).save(f'{OUT}/v11_{tag}.png')
remove_lora(tr); pipe.scheduler = orig_sched
"""
SWEEP_MODE = True   # False: the original single 50-step run at 640 px
keep = []
for c in nb["cells"]:
    s = src(c)
    if c["cell_type"] == "markdown":
        if any(k in s for k in ("# Feasibility", "## Stage 1", "## Results", "Mixed precision")):
            keep.append(c)
        continue
    if SWEEP_MODE and ("steps=50" in s or "Q.recompose(result" in s):
        if "steps=50" in s:
            setsrc(c, SWEEP); keep.append(c)
        continue
    if "fp16 overflow probe" in s or "fp16 fix: the MLP" in s or "_ln_fp32" in s or "TFLOPS" in s:
        continue                                          # failed experiments / measurements already done
    if "from_single_file" in s:                           # transformer load without the single-GPU smoke run
        s = s.split("# smoke test:")[0]
    elif "def _report" in s:                              # keep the NaN-reporting helpers, skip their run
        s = s.split("try:")[0].replace("    x = x.detach().float()\n",
              "    global _BAD\n    x = x.detach().float()\n    _BAD = _BAD or not torch.isfinite(x).all().item()\n")
        s = "_BAD = False\n" + s
    elif "fp32 residual stream + fp16" in s and SWEEP_MODE:     # cheaper NaN check at 384 px
        s = s.replace("resolution=640)", "resolution=384, true_cfg=1.0)") + PROBE
    elif "# Every linear: input scaled" in s:             # stop before the 50-step run if NaNs remain
        s += PROBE
    elif "def split_across_gpus" in s:                    # keep the split helpers, skip their run
        s = s.split("tr.to('cpu'); gc.collect()")[0]
    setsrc(c, s)
    keep.append(c)
nb["cells"] = keep
for c in keep:
    if c["cell_type"] == "code":
        compile(chr(10).join("#" if l.startswith("!") else l for l in src(c).splitlines()), "cell", "exec")
os.makedirs(OUT, exist_ok=True)
json.dump(nb, open(os.path.join(OUT, "qwen-layered-batch.ipynb"), "w", encoding="utf-8"), indent=1)
meta = json.load(open(os.path.join(SRC_DIR, "kernel-metadata.json")))
meta.update(is_private="false", id="markandreduesing/qwen-layered-batch", title="qwen-layered-batch", code_file="qwen-layered-batch.ipynb",
            machine_shape="NvidiaTeslaT4")
json.dump(meta, open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("cells:", [src(c).strip().splitlines()[0][:60] for c in keep])
print("wrote", OUT)
