r"""Kaggle T4 notebook: move the fox on fox & cottage with relocate() + shadow layer + IC-Light (tools/fox_event_relight.py).

    fastsdcpu\env\Scripts\python.exe prototype\tools\make_fox_move_trial.py
    cd prototype\kaggle\upload_fox_event_trial && python -m kaggle.cli kernels push -p .

Input layers: RevealLayer's fox & cottage split (results_reveallayer_fox_v15, 12 steps): its background has the
objects removed and filled, so moving the fox leaves no hole. layer_0 = background, layer_1..5 = RevealLayer layers
(village houses, cottage, gate arch, tree, fox) -> the fox is index 5.
"""
import base64
import io
import json
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
OUT = os.path.join(PROTO, "kaggle", "upload_fox_event_trial")
RL = os.path.join(PROTO, "kaggle", "results_reveallayer_fox_v15", "reveal_out", "gs1.0_st12", "fox_cottage")

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for top in ("layerproto", os.path.join("vendor", "diffuser_layerdiffuse", "layer_diffuse")):
        for root, dirs, files in os.walk(os.path.join(PROTO, top)):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn in files:
                if fn.endswith(".py"):
                    full = os.path.join(root, fn)
                    z.write(full, os.path.join("prototype", os.path.relpath(full, PROTO)))
    z.write(os.path.join(HERE, "fox_event_relight.py"), "prototype/tools/fox_event_relight.py")
    from PIL import Image
    def webp(src):   # keep the notebook small (Kaggle rejected 1.1 MB): lossy WebP with alpha
        b = io.BytesIO(); Image.open(src).convert("RGBA").resize((512, 512)).save(b, "WEBP", quality=90, alpha_quality=100); return b.getvalue()
    import glob as _g
    for _f in sorted(_g.glob(os.path.join(PROTO, "kaggle", "results_qwen21_bgevent", "bg_events", "bg_*.png"))):
        if "bg_day" not in _f: z.writestr("bg_events/" + os.path.basename(_f), open(_f, "rb").read())
    z.writestr("layers/fox_cottage/layer_0.webp", webp(os.path.join(RL, "bg_rgba.png")))
    for i in range(5):
        z.writestr(f"layers/fox_cottage/layer_{i + 1}.webp", webp(os.path.join(RL, f"layer_{i}_rgba.png")))
code_b64 = base64.b64encode(buf.getvalue()).decode()

SRC = f'''
import signal, time as _t
_T0 = _t.time()
def _cap(*a): raise TimeoutError("25 min cap")
signal.signal(signal.SIGALRM, _cap); signal.alarm(25 * 60)
import base64, io, zipfile, subprocess, sys, os
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'diffusers==0.37.0'], check=False)  # same as the working relocate notebooks
os.makedirs('/kaggle/working/code', exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode("{code_b64}"))).extractall('/kaggle/working/code')
os.chdir('/kaggle/working/code')
cmds = [
    ["prototype/tools/fox_event_relight.py", "--layers-dir", "layers/fox_cottage", "--bg-dir", "bg_events", "--out", "/kaggle/working/event_relight"],
]
for c in cmds:
    r = subprocess.run([sys.executable] + c, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
'''
compile(SRC, "cell", "exec")
os.makedirs(OUT, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": SRC.strip("\n").splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(OUT, "fox-event-trial.ipynb"), "w", encoding="utf-8"), indent=1)
json.dump({"id": "markandreduesing/fox-event-trial", "title": "fox-event-trial", "code_file": "fox-event-trial.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true",
           "enable_tpu": "false", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
           "competition_sources": [], "kernel_sources": [], "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("wrote", OUT, round(os.path.getsize(os.path.join(OUT, "fox-event-trial.ipynb")) / 1e6, 2), "MB")
