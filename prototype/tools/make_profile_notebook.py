r"""Builds the Kaggle notebook for one server profile (layerproto/serve/profiles.py).

    fastsdcpu/env/Scripts/python.exe prototype/tools/make_profile_notebook.py --profile qwen-layered --mode bench
    cd prototype/kaggle/upload_profile_qwen-layered_bench && python -m kaggle.cli kernels push -p .      # runs it as a batch job

    --mode bench : load the profile once, run its fox & cottage cases (paths A and B) and save strips + results.json
                   (output: /kaggle/working/bench_<profile>/)
    --mode serve : load the profile, then serve /decompose behind a Cloudflare quick tunnel (import the .ipynb in the editor;
                   it is NOT pushed, pushing would start a run) and print the remote_url / remote_token lines for prototype/config.json

One profile per session: the two T4s hold one big model at a time and loading takes minutes (see docs/models/TEMPLATE.md).
"""
import argparse
import base64
import io
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
sys.path.insert(0, PROTO)
from layerproto.serve.profiles import PROFILES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--profile", required=True, choices=sorted(PROFILES))
ap.add_argument("--mode", default="bench", choices=["bench", "serve"])
a = ap.parse_args()
prof = PROFILES[a.profile]
OUT = os.path.join(PROTO, "kaggle", f"upload_profile_{prof.name}_{a.mode}")
os.makedirs(OUT, exist_ok=True)

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(os.path.join(PROTO, "layerproto")):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            if fn.endswith(".py"):
                full = os.path.join(root, fn)
                z.write(full, os.path.join("prototype", os.path.relpath(full, PROTO)))
    for rel, src in prof.files.items():          # extra images: 512 px JPEG data (the Kaggle API rejects notebooks of about 1.1 MB and more)
        from PIL import Image
        b = io.BytesIO()
        Image.open(os.path.join(os.path.dirname(PROTO), src)).convert("RGB").resize((512, 512)).save(b, "JPEG", quality=88)
        z.writestr(f"prototype/{rel}", b.getvalue())
    for fn in ("fox_cottage.png", "fox_cottage.json"):
        z.write(os.path.join(PROTO, "test_scene", fn), f"prototype/test_scene/{fn}")
code_b64 = base64.b64encode(buf.getvalue()).decode()

BENCH = f'''
from layerproto.serve import benchmark
res = benchmark.run_cases(dec, prof.cases, "/kaggle/working/bench_{prof.name}")
print("BENCH DONE", round(time.time() - _T0), "s since start", flush=True)
'''
SERVE = '''
import json, re, secrets, threading, urllib.request, uvicorn
from layerproto.remote_server import build_app
TOKEN = secrets.token_urlsafe(16)
stub = type("NoChain", (), {"info": lambda self: {"backend": "decompose-only", "profile": prof.name}})()
server = uvicorn.Server(uvicorn.Config(build_app(stub, TOKEN, decomposer=dec, profile=prof.name), host="127.0.0.1", port=8765, log_level="warning"))
threading.Thread(target=server.run, daemon=True).start()
if not os.path.exists("/kaggle/working/cloudflared"):
    urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "/kaggle/working/cloudflared")
    os.chmod("/kaggle/working/cloudflared", 0o755)
tunnel = subprocess.Popen(["/kaggle/working/cloudflared", "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:8765"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
url = None
for line in tunnel.stdout:
    m = re.search(r"https://[a-z0-9-]+[.]trycloudflare[.]com", line)
    if m:
        url = m.group(0); break
print("Put this into prototype/config.json on the laptop:")
print(json.dumps({"remote_url": url, "remote_token": TOKEN}, indent=1))
'''

SRC = f'''
import signal, time as _t
_T0 = _t.time()
def _cap(*a): raise TimeoutError("55 min cap hit after %.0fs" % (_t.time() - _T0))
signal.signal(signal.SIGALRM, _cap); signal.alarm(55 * 60)
import base64, io, zipfile, subprocess, sys, os, time, importlib
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.update({json.dumps(prof.env)})
PIP = {json.dumps(prof.pip)}
os.environ["SAM2_BUILD_CUDA"] = "0"
for _grp in PIP:
    _r = subprocess.run([sys.executable, "-m", "pip", "-q", "install"] + _grp.split(), capture_output=True, text=True)
    print("pip", _grp, "rc", _r.returncode, flush=True)
os.makedirs("/kaggle/working/code", exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode("{code_b64}"))).extractall("/kaggle/working/code")
sys.path.insert(0, "/kaggle/working/code/prototype")
from layerproto.serve.profiles import PROFILES
prof = PROFILES["{prof.name}"]
_mod, _fn = prof.factory.split(":")
t_load = time.time()
dec = getattr(importlib.import_module(_mod), _fn)()
print("PROFILE LOADED", prof.name, round(time.time() - t_load), "s", flush=True)
{BENCH if a.mode == "bench" else SERVE}
'''
compile(SRC, "cell", "exec")
cells = [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": SRC.strip("\n").splitlines(True)}]
if a.mode == "serve":
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": ["# keep the session alive; stop this cell or the notebook to shut the backend down\n", "while True:\n", "    time.sleep(60)\n"]})
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
name = f"profile-{prof.name}-{a.mode}"
json.dump(nb, open(os.path.join(OUT, f"{name}.ipynb"), "w", encoding="utf-8"), indent=1)
json.dump({"id": f"markandreduesing/{name}", "title": name, "code_file": f"{name}.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true", "machine_shape": prof.gpu,
           "dataset_sources": [], "competition_sources": [], "kernel_sources": prof.kernel_sources, "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("wrote", OUT, round(os.path.getsize(os.path.join(OUT, f"{name}.ipynb")) / 1e6, 2), "MB")
