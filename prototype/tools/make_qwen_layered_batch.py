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
keep = []
for c in nb["cells"]:
    s = src(c)
    if c["cell_type"] == "markdown":
        if any(k in s for k in ("# Feasibility", "## Stage 1", "## Results", "Mixed precision")):
            keep.append(c)
        continue
    if "fp16 overflow probe" in s or "fp16 fix: the MLP" in s or "_ln_fp32" in s or "TFLOPS" in s:
        continue                                          # failed experiments / measurements already done
    if "from_single_file" in s:                           # transformer load without the single-GPU smoke run
        s = s.split("# smoke test:")[0]
    elif "def _report" in s:                              # keep the NaN-reporting helpers, skip their run
        s = s.split("try:")[0].replace("    x = x.detach().float()\n",
              "    global _BAD\n    x = x.detach().float()\n    _BAD = _BAD or not torch.isfinite(x).all().item()\n")
        s = "_BAD = False\n" + s
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
meta.update(id="markandreduesing/qwen-layered-batch", title="qwen-layered-batch", code_file="qwen-layered-batch.ipynb",
            machine_shape="NvidiaTeslaT4")
json.dump(meta, open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("cells:", [src(c).strip().splitlines()[0][:60] for c in keep])
print("wrote", OUT)
