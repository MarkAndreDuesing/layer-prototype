r"""Kaggle notebook: LayerDiffuse-Flux on the fox & cottage objects (paths A and B), built from the proven trial notebook.

    fastsdcpu/env/Scripts/python.exe prototype/tools/make_ldflux_fox_bench.py
    cd prototype/kaggle/upload_ldflux_fox_bench && python -m kaggle.cli kernels push -p .

LayerDiffuse-Flux makes ONE transparent object per call from text (it does not split a given image), so the benchmark generates the five
objects of the shared scene one by one: A = 28 steps (best case), B = 6 steps (live-demo budget, about 30 s). Compute = the denoise +
transparent decode of one object; the prompt encoding is measured separately and is not part of it (model load is not either).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
SRC_NB = os.path.join(PROTO, "kaggle", "upload_layerdiffuse_flux_trial", "layerdiffuse-flux-trial.ipynb")
OUT = os.path.join(PROTO, "kaggle", "upload_ldflux_fox_bench")
src = "".join(json.load(open(SRC_NB, encoding="utf-8"))["cells"][0]["source"])
head = src.split("import time\nPROMPTS")[0]
assert head.count("trans_vae.to(D0)") == 1
head = head.replace("25 * 60", "55 * 60").replace("25 min cap", "55 min cap")
STYLE = "storybook illustration, painterly, soft colours, high quality"
tail = f'''
import time
PROMPTS = {{
    "village_houses": "small village houses with orange tiled roofs, {STYLE}",
    "cottage": "a thatched stone cottage with a brown thatched roof, cream stone walls, small windows and a wooden door, {STYLE}",
    "gate_arch": "a stone gate arch with a wooden door, {STYLE}",
    "big_tree": "a large leafy oak tree with a thick trunk, {STYLE}",
    "red_fox": "a red fox standing, side view, {STYLE}",
}}
SETTINGS = [("A", 28), ("B", 6)]
os.makedirs("/kaggle/working/ld_out", exist_ok=True)
tp = FluxPipeline.from_pretrained(BASE, transformer=None, vae=None, torch_dtype=DT).to(D1)
results = []
for name, prompt in PROMPTS.items():
    t_enc = time.time()
    with torch.inference_mode():
        pe, ppe, _ = tp.encode_prompt(prompt, prompt_2=None, device=D1, max_sequence_length=256)
    pe, ppe = pe.to(D0), ppe.to(D0)
    enc_s = round(time.time() - t_enc, 1)
    for path, steps in SETTINGS:
        for rep in range(2 if name == "red_fox" else 1):                    # warm-up: the first generated object also pays the kernels
            try:
                torch.cuda.reset_peak_memory_stats(0); t0 = time.time()
                with torch.inference_mode():
                    lat = pipe(prompt_embeds=pe, pooled_prompt_embeds=ppe, height=H, width=W, num_inference_steps=steps,
                               guidance_scale=GUID, output_type="latent", generator=torch.Generator(D0).manual_seed(SEED)).images
                    lat = pipe._unpack_latents(lat, H, W, pipe.vae_scale_factor)
                    lat = lat / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
                    _, x = trans_vae.decode(lat)
                x = x.clamp(0, 1).permute(0, 2, 3, 1)
                im = Image.fromarray((x*255).float().cpu().numpy().astype(np.uint8)[0])
                im.save(f"/kaggle/working/ld_out/{{path}}_{{name}}.png")
                a = np.asarray(im)[..., 3]
                row = dict(object=name, path=path, steps=steps, compute_s=round(time.time() - t0, 1), encode_s=enc_s,
                           peak_gib=round(torch.cuda.max_memory_allocated(0)/2**30, 1),
                           transparent_pct=round(float((a < 16).mean()*100), 1), opaque_pct=round(float((a > 240).mean()*100), 1))
                results.append(row); print(json.dumps(row), flush=True)
                json.dump(results, open("/kaggle/working/ld_out/results.json", "w"), indent=1)
            except BaseException as e:
                import traceback; traceback.print_exc(); print("FAILED", name, path, flush=True)
                if isinstance(e, TimeoutError): raise
print("BENCH DONE", round(time.time() - _T0), "s since start", flush=True)
'''
SRC = head + tail
compile(SRC, "cell", "exec")
os.makedirs(OUT, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": SRC.splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(OUT, "ldflux-fox-bench.ipynb"), "w", encoding="utf-8"), indent=1)
json.dump({"id": "markandreduesing/ldflux-fox-bench", "title": "ldflux-fox-bench", "code_file": "ldflux-fox-bench.ipynb", "language": "python",
           "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
           "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("wrote", OUT, round(os.path.getsize(os.path.join(OUT, "ldflux-fox-bench.ipynb")) / 1e6, 2), "MB")
