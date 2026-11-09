r"""Kaggle T4 notebook: RevealLayer (ICML 2026, image + boxes -> RGBA layers incl. hidden parts) at reduced settings.

    fastsdcpu\Scripts\python.exe prototype\tools\make_reveallayer_trial.py
    cd prototype\kaggle\upload_reveallayer_trial && python -m kaggle.cli kernels push -p .

Takes the untested adaptation cell from research_notes/Layered methods deep dive/flux_models_t4_adaptation.md (GGUF Q4_K_S
FLUX.1-dev from city96, ungated encoders/VAE from ostris/Flex.1-alpha, fp16, 512 px) and swaps the demo image for our
two test scenes with hand-drawn boxes. FLUX.1-dev Non-Commercial licence applies.
"""
import base64
import io
import json
import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
ROOT = os.path.dirname(PROTO)
OUT = os.path.join(PROTO, "kaggle", "upload_reveallayer_trial")
MD = os.path.join(ROOT, "research_notes", "Layered methods deep dive", "flux_models_t4_adaptation.md")
SCENES = {  # image, boxes (x0, y0, x1, y1) in original pixels, back to front
    "forest": ("experiments/blkE__P2__textonly__sdxlturbo-steps4-g1.0__seed11__impressionist.png",
               [[0, 0, 260, 420], [300, 0, 512, 420], [100, 320, 340, 460]]),
    "sketch_skeleton": ("experiments/blkB__P2__raw__str80-steps4-g1.0__seed11__hyperreal.png",
                        [[110, 240, 280, 350], [250, 100, 400, 370], [170, 270, 340, 420]]),
}

md = open(MD, encoding="utf-8").read()
cell = md.split("### 2a. RevealLayer")[1].split("```python", 1)[1].split("```", 1)[0]
head = cell.split("# ---- one example")[0]
# RevealLayer bundles diffusers 0.34, which imports FLAX_WEIGHTS_NAME from transformers 4.x (removed in transformers 5)
head = head.replace('pip -q install gguf>=0.10', 'pip -q install transformers==4.57.1 gguf>=0.10', 1)
assert "transformers==4.57.1" in head
# the transparent VAE hard-codes the authors' local path to torchvision's ImageNet ViT-B/32 weights
VIT_PATCH = r"""
os.makedirs('/kaggle/working/vit32', exist_ok=True)
sh('wget -q -nc -O /kaggle/working/vit32/vit_b_32-d86f8d99.pth https://download.pytorch.org/models/vit_b_32-d86f8d99.pth')
import glob as _g, re as _re
sh('mkdir -p /home/jovyan/boomcheng-work-shcdt/zhaoshihao/pretrained_model/vit32 && cp /kaggle/working/vit32/vit_b_32-d86f8d99.pth /home/jovyan/boomcheng-work-shcdt/zhaoshihao/pretrained_model/vit32/')
for _f in _g.glob(RL + '/**/*.py', recursive=True):
    _s = open(_f, encoding='utf-8', errors='ignore').read()
    if 'vit32' in _s and '/home/jovyan' in _s:
        _s = _re.sub(r"/home/jovyan/[^'\"]*?vit32", "/kaggle/working/vit32", _s)
        open(_f, 'w', encoding='utf-8').write(_s); print('vit path patched in', _f)
"""
# fp16 hazards found in RevealLayer's own code (research_notes/Layered loop open paths/flux_layered_t4_and_weights.md):
# -1e9 ROI-mask fill is -inf in fp16 (fully masked rows -> NaN), unguarded refiner residual, fp16 VAE decode
FP16_PATCH = r"""
patch("models/custom_model_mmdit.py", "torch.full((t_total, t_total), -1e9, dtype=dtype", "torch.full((t_total, t_total), -1e4, dtype=dtype")
patch("models/custom_model_mmdit.py", "hidden_states_part2 = hidden_states[:, l0_len:, :] + delta",
      "hidden_states_part2 = (hidden_states[:, l0_len:, :].float() + torch.nan_to_num(delta.float(), nan=0.0, posinf=6e4, neginf=-6e4)).clamp(-6e4, 6e4).to(hidden_states.dtype)")
patch("models/custom_pipeline.py", "image_segs = [self.vae.decode(latents_seg, return_dict=False)[0] for latents_seg in latents_segs]",
      "image_segs = [self.vae.float().decode(latents_seg.float(), return_dict=False)[0] for latents_seg in latents_segs]")
"""
head = head.replace("# ---- use bundled diffusers", VIT_PATCH + FP16_PATCH + "# ---- use bundled diffusers", 1)
assert "-1e4, dtype=dtype" in head
assert "vit_b_32-d86f8d99" in head
# FLUX overflows fp16 in the single blocks (guard fired every step -> pure noise); reuse our Qwen fix
MIXED = open(os.path.join(PROTO, "layerproto", "fp16_mixed.py"), encoding="utf-8").read()
APPLY = ("\n# ---- fp16_mixed (layerproto/fp16_mixed.py, inlined) ----\nexec(compile(" + repr(MIXED) + ", 'fp16_mixed', 'exec'))\n"
         "apply_fp16_mixed(t); print('fp16_mixed applied', flush=True)\n")
head = head.replace("# ---- R7: fp16 overflow guard", APPLY + "# ---- R7: fp16 overflow guard", 1)
assert "apply_fp16_mixed(t)" in head
imgs = {}
for k, (f, boxes) in SCENES.items():
    buf = io.BytesIO()
    Image.open(os.path.join(ROOT, f)).convert("RGB").save(buf, "PNG")
    imgs[k] = (base64.b64encode(buf.getvalue()).decode(), boxes)
tail = f'''
# ---- our scenes (replaces the demo example) ----
import base64, io, time
from types import SimpleNamespace
from PIL import Image
SCENES = {json.dumps(imgs)}
for name, (b, boxes) in list(SCENES.items())[:1]:
    p = f"/kaggle/working/{{name}}.png"; Image.open(io.BytesIO(base64.b64decode(b))).save(p)
    sample = {{"index": name, "layout": boxes, "full_image_path": p}}
    for true_gs, steps in ((1.0, 12),):
        print(f"start {{name}} at {{time.time()-_T0:.0f}}s since cell start", flush=True)
        args = SimpleNamespace(seed=43, cfg=1.0, steps=steps, true_gs=true_gs, output_dir=f"/kaggle/working/reveal_out/gs{{true_gs}}_st{{steps}}")
        torch.cuda.reset_peak_memory_stats(0); torch.cuda.reset_peak_memory_stats(1); t0 = time.time()
        try:
            with torch.inference_mode():
                infer.test_one_sample(args, sample, pipe, transp_vae, D0, shuffle=False)
            print(f"{{name}} true_gs={{true_gs}} steps={{steps}}: {{time.time()-t0:.0f}}s; guard hits={{_hit['n']}}; peak GiB",
                  [round(torch.cuda.max_memory_allocated(i)/2**30, 1) for i in range(2)], flush=True)
        except BaseException as e:
            import traceback; traceback.print_exc(); print("FAILED", name, true_gs, steps, flush=True)
sh("find /kaggle/working/reveal_out -name '*.png' | head -50")
'''
CAP = """import signal, time as _t
_T0 = _t.time()
def _cap(*a): raise TimeoutError("25 min cap hit after %.0fs" % (_t.time() - _T0))
signal.signal(signal.SIGALRM, _cap); signal.alarm(25 * 60)
"""
src = CAP + head + tail
compile(src, "cell", "exec")
os.makedirs(OUT, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": src.strip("\n").splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(OUT, "reveallayer-trial.ipynb"), "w", encoding="utf-8"), indent=1)
json.dump({"id": "markandreduesing/reveallayer-trial", "title": "reveallayer-trial", "code_file": "reveallayer-trial.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true",
           "enable_tpu": "false", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
           "competition_sources": [], "kernel_sources": [], "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("wrote", OUT, round(os.path.getsize(os.path.join(OUT, "reveallayer-trial.ipynb")) / 1e6, 2), "MB")
