r"""Kaggle T4 notebook: SDXL-Turbo as the fast first step (text to image, image to image) on the fox & cottage scene.

    fastsdcpu/env/Scripts/python.exe prototype/tools/make_sdxl_turbo_bench.py
    cd prototype/kaggle/upload_sdxl_turbo_bench && python -m kaggle.cli kernels push -p .

Measured (model load not counted, first call is a warm-up): text to image at 512 px with 1, 2 and 4 steps (the prompt that made the shared scene, seeds 13 and 5),
image to image from the shared scene with a night and a rain prompt at strength 0.5 / 0.75 (4 steps), and 768 px text to image. Outputs are saved for inspection.
"""
import base64
import io
import json
import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
OUT = os.path.join(PROTO, "kaggle", "upload_sdxl_turbo_bench")
b = io.BytesIO()
Image.open(os.path.join(PROTO, "test_scene", "fox_cottage.png")).convert("RGB").resize((512, 512)).save(b, "JPEG", quality=92)
SCENE = base64.b64encode(b.getvalue()).decode()
BASE = ("storybook illustration, a large red fox standing on open green lawn in the right foreground, "
        "a small stone cottage with an orange thatched roof in the middle distance, a low dry-stone wall with a "
        "wooden gate to the left of the cottage, a cobblestone path winding from the cottage into the foreground, "
        "a big leafy tree behind the cottage, a few small village houses on the rolling hills in the background, "
        "blue sky with soft clouds, the fox stands alone on the lawn away from the wall and the path, clearly separated objects, soft daylight, painterly")
SRC = f'''
import signal, time as _t
_T0 = _t.time()
def _cap(*a): raise TimeoutError("20 min cap")
signal.signal(signal.SIGALRM, _cap); signal.alarm(20 * 60)
import base64, io, subprocess, sys, os, time, json, statistics
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "diffusers>=0.30", "accelerate"], check=False)
import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
t_load = time.time()
t2i = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda:0")
i2i = AutoPipelineForImage2Image.from_pipe(t2i)
print("LOAD", round(time.time() - t_load, 1), "s", flush=True)
scene = Image.open(io.BytesIO(base64.b64decode("{SCENE}"))).convert("RGB")
BASE = {json.dumps(BASE)}
os.makedirs("/kaggle/working/turbo", exist_ok=True)
rows = []
def timed(name, fn, reps=3):
    fn()                                              # warm-up
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.time(); im = fn(); torch.cuda.synchronize(); ts.append(time.time() - t)
    im.save(f"/kaggle/working/turbo/{{name}}.png")
    row = dict(case=name, median_s=round(statistics.median(ts), 2), runs=[round(x, 2) for x in ts], peak_gib=round(torch.cuda.max_memory_allocated(0) / 2**30, 1))
    rows.append(row); print(json.dumps(row), flush=True); json.dump(rows, open("/kaggle/working/turbo/results.json", "w"), indent=1)
for seed in (13, 5):
    for steps in (1, 2, 4):
        timed(f"t2i_512_s{{seed}}_{{steps}}st", lambda: t2i(prompt=BASE, num_inference_steps=steps, guidance_scale=0.0, width=512, height=512,
                                                         generator=torch.Generator("cuda:0").manual_seed(seed)).images[0])
timed("t2i_768_s13_4st", lambda: t2i(prompt=BASE, num_inference_steps=4, guidance_scale=0.0, width=768, height=768, generator=torch.Generator("cuda:0").manual_seed(13)).images[0])
EDITS = {{"night": "the same storybook scene at night, a full moon in a dark blue sky, warm light in the cottage windows, painterly",
          "rain": "the same storybook scene in heavy rain, dark grey storm clouds, rain streaks, wet cobblestones, painterly"}}
for ev, prompt in EDITS.items():
    for strength in (0.5, 0.75):
        timed(f"i2i_{{ev}}_str{{strength}}", lambda: i2i(prompt=prompt, image=scene, strength=strength, num_inference_steps=4, guidance_scale=0.0,
                                                    generator=torch.Generator("cuda:0").manual_seed(13)).images[0])
print("BENCH DONE", round(time.time() - _T0), "s", flush=True)
'''
compile(SRC, "cell", "exec")
os.makedirs(OUT, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": SRC.strip("\n").splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(OUT, "sdxl-turbo-bench.ipynb"), "w", encoding="utf-8"), indent=1)
json.dump({"id": "markandreduesing/sdxl-turbo-bench", "title": "sdxl-turbo-bench", "code_file": "sdxl-turbo-bench.ipynb", "language": "python",
           "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
           "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("wrote", OUT)
