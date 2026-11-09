"""Builds prototype/kaggle/layerproto_chain.ipynb and prototype/kaggle/layerproto_code.zip.

    fastsdcpu\\env\\Scripts\\python.exe prototype\\tools\\make_kaggle_notebook.py

Upload layerproto_code.zip as a private Kaggle Dataset named "layerproto-code", then import the notebook,
attach the dataset, set Accelerator = GPU T4 x2 and Internet = On.
The notebook runs the engine inside the notebook; section 6 can optionally serve the GPU to the local UI
through a Cloudflare quick tunnel (personal, non-commercial use).
"""
import json
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
OUT = os.path.join(PROTO, "kaggle")
os.makedirs(OUT, exist_ok=True)


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(True)}


cells = [
    md("""
# Layer scene prototype: LayerDiffuse SDXL scene-to-layer chain on a Kaggle T4

Runs the prototype's engine **inside this notebook** (section 6 can also serve it to the local UI):

1. background layer from its prompt
2. every element is drawn **into the scene below it** (`bg2ble`) and extracted as its own aligned RGBA layer (`bgble2fg`)
3. re-prompt one layer: regenerated into the layers below it, nothing else changes
4. story event: a new element generated into the current scene as a new top layer
5. move / resize layers (canvas transform, no model call), then the final stacked image

Settings: **Accelerator = GPU T4 x2**, **Internet = On**, dataset **layerproto-code** attached.
"""),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install "diffusers==0.37.0" "transformers>=4.51" accelerate safetensors peft einops opencv-python-headless
"""),
    code("""
import glob, json, os, shutil, sys, zipfile
src = glob.glob('/kaggle/input/**/layerproto_code.zip', recursive=True) + glob.glob('/kaggle/input/**/prototype/app.py', recursive=True)
assert src, 'Attach the layerproto-code dataset (layerproto_code.zip) first'
if src[0].endswith('.zip'):
    zipfile.ZipFile(src[0]).extractall('/kaggle/working')
else:
    shutil.copytree(os.path.dirname(src[0]), '/kaggle/working/prototype', dirs_exist_ok=True)
sys.path.insert(0, '/kaggle/working/prototype')
os.environ['HF_HUB_DISABLE_XET'] = '1'
print(os.listdir('/kaggle/working/prototype'))
"""),
    code("""
from layerproto.config import Settings
from layerproto.engine import Engine

S = Settings(
    image_backend='local', device='cuda', sdxl_model='stabilityai/stable-diffusion-xl-base-1.0',
    width=1024, height=1024, steps=20, guidance=7.0, alpha_quality='best',
    # add-ons off: no Ollama on Kaggle by default -> comma split, no captions
    step_refine=False, step_split=False, step_words=False, step_captions=False, step_word_check=False,
    variants_per_layer=1, scenes_dir='/kaggle/working/scenes')
eng = Engine(S)
"""),
    code("""
from IPython.display import display
from PIL import Image

def show(scene, title=''):
    folder = scene.folder(S.scenes_dir)
    tiles = [eng.compose(scene)] + [Image.open(f'{folder}/{l.current().file}') for l in eng._ordered(scene) if l.current()]
    W = 256
    grid = Image.new('RGBA', (W * len(tiles), W), (40, 160, 60, 255))
    for i, t in enumerate(tiles):
        tile = Image.new('RGBA', (W, W), (40, 160, 60, 255)); tile.alpha_composite(t.convert('RGBA').resize((W, W)))
        grid.paste(tile, (i * W, 0))
    print(title, '| final image, then each layer bottom -> top (green = transparent)')
    display(grid)
"""),
    md("## 1. Describe the scene: first comma part(s) with scenery words = background, every other part = one element"),
    code("""
scene = eng.plan_scene('pixel art, night forest of tall pine trees, full moon and stars, '
                       'a large campfire with a stone ring in the center, a skeleton dancing on the right side of the fire')
for l in scene.layers: print(l.depth, l.role, '|', l.prompt)
print('style:', scene.style)
"""),
    md("## 2. Generate: background, then each element drawn into the scene and extracted"),
    code("""
import time
t = time.time()
eng.generate_all(scene, progress=print)
print(f'{time.time() - t:.0f} s')
show(scene, 'generated')
"""),
    md("## 3. Re-prompt one layer (regenerated into the layers below it; the rest stays)"),
    code("""
target = eng.object_order(scene)[-1]
eng.update_layer(scene, target.id, prompt='a skeleton playing a fiddle on the right side of the fire', progress=print)
show(scene, 're-prompted ' + target.name)
"""),
    md("## 4. Story event: a new element generated into the current scene"),
    code("""
eng.add_event_layer(scene, 'forest fire', 'the pine trees on the left are on fire, bright flames and smoke', progress=print)
show(scene, 'after story event')
"""),
    md("## 5. Move / resize a layer (no regeneration) and reorder"),
    code("""
el = eng.object_order(scene)[0]
eng.transform_layer(scene, el.id, dx=-0.1, dy=0.05, scale=1.2)
show(scene, f'moved and resized {el.name}')
"""),
    code("""
shutil.make_archive('/kaggle/working/scenes_out', 'zip', S.scenes_dir)
print('Download /kaggle/working/scenes_out.zip from the Output panel; unzip into prototype/scenes to open it in the local UI')
"""),
    md("""
## 6. Optional: use this GPU from the local UI (live backend)

Run this right after the **Engine** cell (step 4 above; the sections in between are optional).
It serves the already-loaded chain over HTTP and opens a Cloudflare quick tunnel.
Copy the printed `config.json` lines into `prototype/config.json` on the laptop, then restart the local app.
The tunnel lives as long as this notebook session; every request needs the token.
"""),
    code("""
import secrets, subprocess, threading, re, time, urllib.request, uvicorn
from layerproto.remote_server import build_app

TOKEN = secrets.token_urlsafe(16)
server = uvicorn.Server(uvicorn.Config(build_app(eng.chain, TOKEN), host='127.0.0.1', port=8765, log_level='warning'))
threading.Thread(target=server.run, daemon=True).start()

if not os.path.exists('/kaggle/working/cloudflared'):
    urllib.request.urlretrieve('https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64',
                               '/kaggle/working/cloudflared')
    os.chmod('/kaggle/working/cloudflared', 0o755)
tunnel = subprocess.Popen(['/kaggle/working/cloudflared', 'tunnel', '--no-autoupdate', '--url', 'http://127.0.0.1:8765'],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
url = None
for line in tunnel.stdout:
    m = re.search(r'https://[a-z0-9-]+\\.trycloudflare\\.com', line)
    if m:
        url = m.group(0); break
print('Put this into prototype/config.json on the laptop:')
print(json.dumps({'image_backend': 'remote', 'remote_url': url, 'remote_token': TOKEN,
                  'width': 1024, 'height': 1024}, indent=1))
"""),
    code("""
# keep the session busy while you play; stop this cell (or the notebook) to shut the backend down
while True:
    time.sleep(60)
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
with open(os.path.join(OUT, "layerproto_chain.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

skip_dirs = {"scenes", "__pycache__", "kaggle", ".git", "assets", "tests"}
with zipfile.ZipFile(os.path.join(OUT, "layerproto_code.zip"), "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(PROTO):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fn in files:
            if fn.endswith((".pyc", ".log")) or fn == "config.json":
                continue
            full = os.path.join(root, fn)
            z.write(full, os.path.join("prototype", os.path.relpath(full, PROTO)))
print("wrote", OUT)
