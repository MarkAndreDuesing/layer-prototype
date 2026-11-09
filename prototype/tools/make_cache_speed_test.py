"""CPU-only Kaggle notebook: how fast can the cached weight outputs be read? (no accelerator -> no GPU quota)

    fastsdcpu\\env\\Scripts\\python.exe prototype\\tools\\make_cache_speed_test.py

Measures, with timestamps:
  A  session start -> first cell (input attach time, from the kernel log)
  B  reading one text-encoder shard directly from /kaggle/input with safetensors (what transformers does), first 40 tensors
  C  one-pass sequential copy of the same shard to local disk (/tmp), then loading it from there
  D  sequential read throughput of the 12.4 GB GGUF (first 3 GB)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "kaggle", "upload_cache_speed_test")
SRC = r'''
import glob, os, shutil, time, json
T0 = time.time()
def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
R = {}
shards = sorted(glob.glob('/kaggle/input/**/Qwen-Image-Layered/text_encoder/*.safetensors', recursive=True))
gguf = glob.glob('/kaggle/input/**/qwen-image-layered-Q4_K_S.gguf', recursive=True)
log('shards', [(os.path.basename(s), round(os.path.getsize(s) / 1e9, 2)) for s in shards])
log('gguf', gguf)
os.system('df -h /tmp /kaggle/working | tail -n 2; mount | grep -i kaggle/input | head -3')
shard = min(shards, key=os.path.getsize)          # smallest shard (~1.7 GB) keeps the test short
size_gb = os.path.getsize(shard) / 1e9

# B: direct safetensors reads from the attached input (as transformers does), 40 tensors max, 5 min cap
from safetensors import safe_open
t = time.time(); n = 0; nbytes = 0
with safe_open(shard, 'pt') as f:
    for k in list(f.keys()):
        x = f.get_tensor(k); n += 1; nbytes += x.numel() * x.element_size()
        if n >= 40 or time.time() - t > 300:
            break
R['B_direct_tensors'] = n; R['B_direct_s'] = round(time.time() - t, 1); R['B_direct_MBps'] = round(nbytes / 1e6 / max(time.time() - t, 1e-3), 1)
log('B direct read:', R['B_direct_tensors'], 'tensors in', R['B_direct_s'], 's =', R['B_direct_MBps'], 'MB/s')

# C: sequential copy to local disk, then load everything from the local copy
t = time.time(); dst = '/tmp/' + os.path.basename(shard); shutil.copyfile(shard, dst)
R['C_copy_s'] = round(time.time() - t, 1); R['C_copy_MBps'] = round(size_gb * 1e3 / max(time.time() - t, 1e-3), 1)
log('C copy', round(size_gb, 2), 'GB in', R['C_copy_s'], 's =', R['C_copy_MBps'], 'MB/s')
t = time.time(); n = 0
with safe_open(dst, 'pt') as f:
    for k in f.keys():
        f.get_tensor(k); n += 1
R['C_local_load_tensors'] = n; R['C_local_load_s'] = round(time.time() - t, 1)
log('C local load: all', n, 'tensors in', R['C_local_load_s'], 's')
os.remove(dst)

# D: sequential read throughput of the GGUF (first 3 GB)
if gguf:
    t = time.time(); read = 0
    with open(gguf[0], 'rb', buffering=0) as fh:
        while read < 3e9:
            b = fh.read(64 * 1024 * 1024)
            if not b:
                break
            read += len(b)
    R['D_gguf_seq_MBps'] = round(read / 1e6 / max(time.time() - t, 1e-3), 1)
    log('D gguf sequential read:', round(read / 1e9, 2), 'GB at', R['D_gguf_seq_MBps'], 'MB/s')
R['total_s'] = round(time.time() - T0, 1)
log(json.dumps(R))
json.dump(R, open('/kaggle/working/cache_speed.json', 'w'), indent=1)
'''
compile(SRC, "cell", "exec")
os.makedirs(OUT, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": SRC.strip("\n").splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(OUT, "cache-speed-test.ipynb"), "w", encoding="utf-8"), indent=1)
json.dump({"id": "markandreduesing/cache-speed-test", "title": "cache-speed-test", "code_file": "cache-speed-test.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "false",
           "enable_tpu": "false", "enable_internet": "false", "machine_shape": "", "dataset_sources": [],
           "competition_sources": [],
           "kernel_sources": ["markandreduesing/qwen-layered-weights-transformer",
                              "markandreduesing/qwen-layered-weights-base"], "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("wrote", OUT)
