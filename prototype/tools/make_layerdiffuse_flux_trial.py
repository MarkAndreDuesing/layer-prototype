r"""Kaggle 2x T4 notebook: LayerDiffuse-Flux (text -> transparent RGBA) at reduced settings.

    fastsdcpu\env\Scripts\python.exe prototype\tools\make_layerdiffuse_flux_trial.py
    cd prototype\kaggle\upload_layerdiffuse_flux_trial && python -m kaggle.cli kernels push -p .

Takes cell 2b of research_notes/Layered methods deep dive/flux_models_t4_adaptation.md and changes:
  * pins transformers 4.57.1 (diffusers 0.34 imports names removed in transformers 5)
  * replaces the NaN guard + peft LoRA with our fp16_mixed (FLUX overflows fp16, as RevealLayer run 5 showed)
    and fp16_mixed.add_lora (LoRA kept unmerged next to the GGUF weights)
  * 25 min SIGALRM cap so a stall still ends with a saved log
  * a few prompts from our scenes instead of the demo prompt
FLUX.1-dev Non-Commercial licence applies.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
ROOT = os.path.dirname(PROTO)
OUT = os.path.join(PROTO, "kaggle", "upload_layerdiffuse_flux_trial")
MD = os.path.join(ROOT, "research_notes", "Layered methods deep dive", "flux_models_t4_adaptation.md")

md = open(MD, encoding="utf-8").read()
cell = md.split("### 2b. LayerDiffuse-Flux")[1].split("```python", 1)[1].split("```", 1)[0]

cell = cell.replace('sh("pip -q install diffusers==0.34.0', 'sh("pip -q install transformers==4.57.1 diffusers==0.34.0', 1)
# drop peft LoRA load + guard; apply fp16_mixed + our LoRA loader instead
a = cell.index("pipe.load_lora_weights(lora)")
b = cell.index("trans_vae = TransparentVAE")
MIXED = open(os.path.join(PROTO, "layerproto", "fp16_mixed.py"), encoding="utf-8").read()
cell = (cell[:a]
        + "exec(compile(" + repr(MIXED) + ", 'fp16_mixed', 'exec'))\n"
        + "apply_fp16_mixed(pipe.transformer)\n"
        + "from safetensors.torch import load_file as _lf\n"
        + "from diffusers.loaders.lora_conversion_utils import _convert_kohya_flux_lora_to_diffusers as _conv\n"
        + "_sd = _lf(lora)\n"
        + "if any(k.startswith('lora_unet_') for k in _sd): _sd = _conv(_sd)  # BFL/kohya names, fused qkv -> diffusers\n"
        + "_hit, _miss = add_lora(pipe.transformer, _sd, 1.0); print('lora matched', _hit, 'missing', len(_miss), _miss[:3], flush=True)\n"
        + "assert _hit > 0, 'LoRA did not match'\n"
        + "hits = {'n': 0}\n\n"
        + cell[b:])
# generation part -> loop over prompts
g = cell.index("import time; t0 = time.time()")
gen = '''
import time
PROMPTS = {"campfire": "a small campfire with logs, painterly, high quality",
           "deer": "a single deer standing, side view, painterly, high quality",
           "bottle": "glass bottle, high quality"}
os.makedirs("/kaggle/working/ld_out", exist_ok=True)
for name, prompt in PROMPTS.items():
    try:
        torch.cuda.reset_peak_memory_stats(0)
        tp = FluxPipeline.from_pretrained(BASE, transformer=None, vae=None, torch_dtype=DT).to(D1)
        with torch.inference_mode():
            pe, ppe, _ = tp.encode_prompt(prompt, prompt_2=None, device=D1, max_sequence_length=256)
        pe, ppe = pe.to(D0), ppe.to(D0); del tp; torch.cuda.empty_cache()
        t0 = time.time()
        with torch.inference_mode():
            lat = pipe(prompt_embeds=pe, pooled_prompt_embeds=ppe, height=H, width=W, num_inference_steps=STEPS,
                       guidance_scale=GUID, output_type="latent", generator=torch.Generator(D0).manual_seed(SEED)).images
            lat = pipe._unpack_latents(lat, H, W, pipe.vae_scale_factor)
            lat = lat / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
            _, x = trans_vae.decode(lat)
        x = x.clamp(0, 1).permute(0, 2, 3, 1)
        Image.fromarray((x*255).float().cpu().numpy().astype(np.uint8)[0]).save(f"/kaggle/working/ld_out/{name}.png")
        print(f"{name}: {time.time()-t0:.0f}s, peak cuda:0 {torch.cuda.max_memory_allocated(0)/2**30:.1f} GiB", flush=True)
    except BaseException as e:
        import traceback; traceback.print_exc(); print("FAILED", name, flush=True)
        if isinstance(e, TimeoutError): break
'''
cell = cell[:g] + gen
cell = cell.replace("PROMPT, H, W, STEPS, GUID, SEED = \"glass bottle, high quality\", 512, 512, 20, 3.5, 11111",
                    "PROMPT, H, W, STEPS, GUID, SEED = \"glass bottle, high quality\", 512, 512, 16, 3.5, 11111")
CAP = '''import signal, time as _t
_T0 = _t.time()
def _cap(*a): raise TimeoutError("25 min cap hit after %.0fs" % (_t.time() - _T0))
signal.signal(signal.SIGALRM, _cap); signal.alarm(25 * 60)
'''
src = CAP + cell
assert "apply_fp16_mixed(pipe.transformer)" in src and "transformers==4.57.1" in src and "STEPS, GUID" in src
compile(src, "cell", "exec")

os.makedirs(OUT, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": src.strip("\n").splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(OUT, "layerdiffuse-flux-trial.ipynb"), "w", encoding="utf-8"), indent=1)
json.dump({"id": "markandreduesing/layerdiffuse-flux-trial", "title": "layerdiffuse-flux-trial",
           "code_file": "layerdiffuse-flux-trial.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
           "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [],
           "kernel_sources": [], "model_sources": []},
          open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=1)
print("wrote", OUT)
