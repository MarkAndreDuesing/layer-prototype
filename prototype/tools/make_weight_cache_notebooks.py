"""Builds two CPU-only Kaggle notebooks that download Qwen-Image-Layered weights into their saved output, so later
GPU notebooks attach them via kernel_sources (/kaggle/input/<slug>/...) instead of re-downloading ~29 GB.

    fastsdcpu\\env\\Scripts\\python.exe prototype\\tools\\make_weight_cache_notebooks.py

Each output stays under Kaggle's ~20 GB per-notebook output limit. No accelerator -> no GPU quota used.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "kaggle")

NOTEBOOKS = {
    "qwen-layered-weights-transformer": """
import os, shutil, time
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import hf_hub_download
t = time.time()
p = hf_hub_download('unsloth/Qwen-Image-Layered-GGUF', 'qwen-image-layered-Q4_K_S.gguf', local_dir='/kaggle/working')
print(p, round(os.path.getsize(p) / 1e9, 2), 'GB in', round(time.time() - t), 's')
shutil.rmtree('/kaggle/working/.cache', ignore_errors=True)   # hf_hub_download metadata, not needed
print(os.listdir('/kaggle/working'))
""",
    "qwen-layered-weights-base": """
import os, shutil, time
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import snapshot_download
t = time.time()
d = snapshot_download('Qwen/Qwen-Image-Layered', local_dir='/kaggle/working/Qwen-Image-Layered',
                      allow_patterns=['model_index.json', 'text_encoder/*', 'vae/*', 'tokenizer/*', 'processor/*',
                                      'scheduler/*', 'transformer/config.json'])
shutil.rmtree('/kaggle/working/Qwen-Image-Layered/.cache', ignore_errors=True)
total = 0
for root, _, files in os.walk(d):
    for f in files:
        total += os.path.getsize(os.path.join(root, f))
print(d, round(total / 1e9, 2), 'GB in', round(time.time() - t), 's')
for sub in sorted(os.listdir(d)):
    print(' ', sub)
""",
}

for slug, src in NOTEBOOKS.items():
    folder = os.path.join(OUT, "upload_" + slug)
    os.makedirs(folder, exist_ok=True)
    compile(src, slug, "exec")
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": src.strip("\n").splitlines(True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(os.path.join(folder, slug + ".ipynb"), "w", encoding="utf-8"), indent=1)
    json.dump({"id": "markandreduesing/" + slug, "title": slug, "code_file": slug + ".ipynb", "language": "python",
               "kernel_type": "notebook", "is_private": "true", "enable_gpu": "false", "enable_tpu": "false",
               "enable_internet": "true", "machine_shape": "", "dataset_sources": [], "competition_sources": [],
               "kernel_sources": [], "model_sources": []},
              open(os.path.join(folder, "kernel-metadata.json"), "w"), indent=1)
    print("wrote", folder)
