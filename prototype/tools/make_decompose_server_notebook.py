r"""Builds a Kaggle notebook that serves the app's /decompose route from a T4 through a Cloudflare quick tunnel.

    fastsdcpu/env/Scripts/python.exe prototype/tools/make_decompose_server_notebook.py

Output: prototype/kaggle/decompose_server/decompose-server.ipynb  (NOT pushed by this script: pushing would start a run).
Use: import it in the Kaggle editor (File > Import notebook), Accelerator = GPU T4, Internet = On, Save Version > Save & Run (or run
cells in the editor); it prints two lines for prototype/config.json (remote_url, remote_token). The laptop app's "Decompose an image
(GPU)" tab then calls it. The tunnel lives as long as the session (max 12 h; every request needs the token). Personal, non-commercial use.
Models: Grounding DINO tiny + SAM2 small + LaMa (about 45 s to load, 5-12 s per image).
"""
import base64
import io
import json
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
OUT = os.path.join(PROTO, "kaggle", "decompose_server")
os.makedirs(OUT, exist_ok=True)

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(os.path.join(PROTO, "layerproto")):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            if fn.endswith(".py"):
                full = os.path.join(root, fn)
                z.write(full, os.path.join("prototype", os.path.relpath(full, PROTO)))
code_b64 = base64.b64encode(buf.getvalue()).decode()


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)}


cells = [
    md("# Decompose server (Grounding DINO + SAM2 + LaMa) behind a Cloudflare quick tunnel\n"
       "Settings: **Accelerator = GPU T4**, **Internet = On**. Run the cells in order; the last one keeps the session alive. "
       "Stop the notebook to shut the backend down."),
    code("""
import os, subprocess, sys
os.environ['SAM2_BUILD_CUDA'] = '0'
for pkg in (['git+https://github.com/facebookresearch/sam2.git'], ['--no-deps', 'simple-lama-inpainting'], ['fire'], ['fastapi', 'uvicorn']):
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + pkg, capture_output=True, text=True)
    print(pkg, 'rc', r.returncode)
"""),
    code(f"""
import base64, io, zipfile
os.makedirs('/kaggle/working/code', exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode("{code_b64}"))).extractall('/kaggle/working/code')
sys.path.insert(0, '/kaggle/working/code/prototype')
from layerproto.backends.sam2_decomposer import Sam2Decomposer
dec = Sam2Decomposer(device='cuda')
print('models loaded')
"""),
    code("""
import json, re, secrets, threading, time, urllib.request, uvicorn
from layerproto.remote_server import build_app

TOKEN = secrets.token_urlsafe(16)
stub = type('NoChain', (), {'info': lambda self: {'backend': 'decompose-only'}})()   # chain routes are not served here
server = uvicorn.Server(uvicorn.Config(build_app(stub, TOKEN, decomposer=dec), host='127.0.0.1', port=8765, log_level='warning'))
threading.Thread(target=server.run, daemon=True).start()
if not os.path.exists('/kaggle/working/cloudflared'):
    urllib.request.urlretrieve('https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64', '/kaggle/working/cloudflared')
    os.chmod('/kaggle/working/cloudflared', 0o755)
tunnel = subprocess.Popen(['/kaggle/working/cloudflared', 'tunnel', '--no-autoupdate', '--url', 'http://127.0.0.1:8765'],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
url = None
for line in tunnel.stdout:
    m = re.search(r'https://[a-z0-9-]+[.]trycloudflare[.]com', line)
    if m:
        url = m.group(0); break
print('Put this into prototype/config.json on the laptop (keep image_backend as it is):')
print(json.dumps({'remote_url': url, 'remote_token': TOKEN}, indent=1))
"""),
    code("""
# keep the session busy; stop this cell or the notebook to shut the backend down
while True:
    time.sleep(60)
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for c in cells:
    if c["cell_type"] == "code":
        compile("".join(c["source"]), "cell", "exec")
json.dump(nb, open(os.path.join(OUT, "decompose-server.ipynb"), "w", encoding="utf-8"), indent=1)
print("wrote", OUT, round(os.path.getsize(os.path.join(OUT, "decompose-server.ipynb")) / 1e6, 2), "MB")
