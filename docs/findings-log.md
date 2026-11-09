# Findings log

All findings and decisions so far, in date order. Numbers are measured unless marked **[est]**. The reports in [`../reports/`](../reports/) hold the sourced research behind each decision.

## Phase 0: models, hosting and costs research (2026-11-05 → 07)

Full report: [Image generation models hosting and costs.md](../reports/Image%20generation%20models%20hosting%20and%20costs.md).

- **Quality bar:** Z-Image Turbo's level (arena Elo 936) is enough. Cost, accessibility and sketch input decide the model, not leaderboard rank.
- **Recurring free GPUs:** Kaggle (about 30 GPU-h/week), the LMU Informatik CIP pool (RTX 2060 SUPER 8 GB through Slurm, with a CIP account) and Hugging Face Spaces (5 GPU-min/day). The LMU Abaki servers (A5000 24 GB) are for theses only; LRZ needs a research project.
- **Ruled out as recurring options:** SageMaker Studio Lab (closed to new users), Lightning AI and Comfy Cloud (one-time trials), Colab web UIs (forbidden on the free tier).
- **APIs:** DeepSeek and Kimi have no image generation. Open models cost about $0.001–0.015 per image on fal, WaveSpeed, Runware or Together. OpenRouter adds a 5.5% top-up fee ($0.80 minimum).
- **Price-to-performance:** Pareto frontier over 101 arena models; the knee is at about $10 per 1,000 images (Muse Image). FLUX 3 Image Klein is the cheapest open-weights model on the frontier ($9 per 1,000). Script: [../research_notes/Image generation models hosting and costs/](../research_notes/Image%20generation%20models%20hosting%20and%20costs/).
- **Sketch input** needs img2img, a ControlNet or reference editing. Z-Image Turbo (Apache-2.0, fits 16 GB) and SDXL have scribble ControlNets; FLUX.2/3 Klein only reference editing.
- **CPU-only on the laptop:** only small distilled models (SD-Turbo, SDXL-Turbo, SDXS, SD 1.5 + LCM-LoRA) were expected to stay under 3 minutes. Phase 1 tested three of them. The research-time speed estimates (4–8 s per image) were about 4–10× too optimistic; measured: about 20 s and 58 s. They scaled published timings that used OpenVINO, the tiny decoder and 1 step, none of which the tested setup used. Speed-up options and a test order: [research_cpu_speedups.md](../reports/research_cpu_speedups.md).

## Phase 1: sketch-to-image on a CPU laptop (FastSD CPU)

**Setup:** Windows 11 laptop, 15.7 GB RAM, no GPU. Current guides (two paths, offline): [fastsd-cpu/INSTALL.md](fastsd-cpu/INSTALL.md) and [fastsd-cpu/USAGE.md](fastsd-cpu/USAGE.md). Results: [../experiments/CONFIRMATION_RESULTS.md](../experiments/CONFIRMATION_RESULTS.md). How the session went and why each decision was made: [session-summary-sketch-to-image.md](session-summary-sketch-to-image.md).

| Path | Speed and memory | Verdict |
|---|---|---|
| SD-Turbo img2img | About 20 s per 512 px image, about half the RAM of SDXL-Turbo | Fast, with strong impressionist and pixel-art styles. It keeps the sketch at Strength 0.7, but the result is barely stylised. At 0.8 and above the subject deforms. |
| SDXL-Turbo img2img | About 58 s per image, about 11 GB RAM | **Best overall.** Strength 0.8, 4 steps, Guidance 1, on a filled (coloured-background) sketch. All of the team's favourites came from SDXL-Turbo. |
| SD 1.5 + LCM-LoRA + scribble ControlNet | — | Follows the layout best. With `dreamshaper-8` every style looked the same; the neutral SD 1.5 base responds to style but quality is rough. |

**Lessons:**
- Strength 0.8 and 0.9 are identical at 4 steps, because `int(steps × strength)` is 3 for both. They give pixel-identical output.
- Blurring the input sketch only adds defocus.
- One seed in three produced a duplicated subject, so generate several seeds and pick.
- A filled sketch keeps the layout but constrains the style. Text-only shows the style but ignores the layout. A raw line sketch gives pale daytime scenes.
- The user's selection pass kept 35 of 188 images. The confirmation test kept the most from SDXL-Turbo text-only (6/12) and from filled or raw img2img (5/12 each).

**Laptop gotchas:**
- Smart App Control blocks unsigned native libraries, so scikit-learn was removed.
- Close Chrome, IDEs and VMs before running SDXL.
- Set `HF_HUB_DISABLE_XET=1` for downloads.
- FastSD CPU needs a float32 patch for SDXL-Turbo (see [../fastsdcpu_local/](../fastsdcpu_local/)).

**Final decisions (2026-11-07):**
- **Path 3 (ControlNet) dropped:** slower, and the fewest kept images (2/12 with a sketch, 0/12 text-only). The guides keep Path 1 (SD-Turbo, fast) and Path 2 (SDXL-Turbo, best).
- **Settings for both paths:** LCM mode, 4 steps, 512 × 512, Guidance 1, no negative prompt, Strength 0.8 for Image to Image, no fixed seed. Generate at least 3 images and pick one.
- **The user's 8 favourites** (all SDXL-Turbo: filled img2img, raw img2img and text-only) are in [../experiments/favourites/](../experiments/favourites/).

**Offline use (verified 2026-11-07):**
- A cache filled by the app's own automatic download **fails offline** with `IncompleteSnapshotError`: `huggingface_hub` compares the cache against the full repo listing.
- Fix: download complete model folders with `snapshot_download(local_dir='models/<name>', ignore_patterns=[...])` and list `models/<name>` in `configs/lcm-models.txt`. The ignore patterns cut the repos from 13 GB / 55.5 GB to 4.9 GB / 13 GB.
- `use_offline_model: true` in `configs/settings.yaml` persists across restarts; the UI checkbox does not. The `--use_offline_model` flag only affects CLI mode.
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `GRADIO_ANALYTICS_ENABLED=False` block all network calls ([../fastsdcpu_local/start-webui-offline.bat](../fastsdcpu_local/start-webui-offline.bat)).
- Both models worked in both modes with the network blocked: about 25 s (SD-Turbo) and about 70 s (SDXL-Turbo) per image. A physically disconnected run has not been done yet. Images: [../experiments/offline_test/](../experiments/offline_test/).

## Phase 2: layered scenes, local prototype (2026-11-07)

1. **First version:** the LLM splits the scene into nouns, and each layer is generated on its own with LayerDiffuse SDXL-Turbo, wrapped as "a single X, solo, centered", then placed in a box. About 80 s per layer on CPU, about 7 min for a 3-layer scene with captions.
2. **Audit:** that version is a sticker collage, not coherent layered generation. Each layer is an independent text-to-image call that shares only a style suffix. See [Coherent layered generation pipeline.md](../reports/Coherent%20layered%20generation%20pipeline.md). **Decision:** coherence has to come from one shared image or scene. Generate the scene first and then split it (Qwen-Image-Layered), or build it element by element on the existing scene (LayerDiffuse chain). A layer edit stays coherent only if it is regenerated through the composite.
3. **LayerDiffuse scene-to-layer chain** (`bg2ble → bgble2fg`, in `prototype/layerproto/backends/layerdiffuse_chain.py`):
   - On SDXL-Turbo, the extract step fails (0–5% transparent).
   - On SDXL base (20 steps, CFG 7, 512 px) it works: the scene is kept, the element is added in place, and the extracted layer is 63% transparent. That takes about 11 min per element on CPU.
   - On a Kaggle T4: about 18 s per pass at 1024 px / 20 steps. Re-prompting a single layer and story events work.
4. **Prompt-writing lessons:**
   - Object prompts need "a single X, solo, centered".
   - "No background" or "isolated object" makes LayerDiffuse paint an opaque backdrop.
   - The vendored LayerDiffuse VAE needs keyword arguments for diffusers 0.37 (see [prototype/vendor/](../prototype/vendor/)).
5. **Hosting decisions:**
   - Free Colab's terms forbid serving endpoints, so it was ruled out.
   - Comfy Cloud was rejected for the app as a proprietary paid UI.
   - Kaggle is the free GPU route. Modal ($30/month credit, needs a card) is the fallback.

## Phase 3: Qwen-Image-Layered on free Kaggle 2× T4 (2026-11-07 → 09)

[Qwen-Image-Layered](https://huggingface.co/Qwen/Qwen-Image-Layered) (20B, Apache-2.0) splits one image into RGBA layers. It is central to the coherent pipeline.

1. **Memory:**
   - The 4-bit GGUF (Q4_K_S, 12.67 GB) fits a T4.
   - The VAE decode ran out of memory while the transformer was still on the GPU. Fix: grab the latents in a callback, move the transformer to CPU, and decode one layer at a time.
   - Splitting the transformer across both T4s (blocks 0–29 and 30–59) uses 7.2 / 6.2 GB, peaking at 8.4 / 7.3 GB.
2. **Weight caching:**
   - The weights are stored as Kaggle notebook outputs (`qwen-layered-weights-transformer` 12.4 GB, `-base` 16.9 GB), now public.
   - transformers 5 loads tensors with parallel threads, which stalled at 0/729 tensors when reading from the `/kaggle/input` mount. Copying the base weights to `/tmp` first (about 100 s) fixes it.
   - The 8-bit bitsandbytes load was not the cause. The text encoder now runs in fp16 split across both GPUs.
3. **The NaN problem:** Qwen-Image is built for bf16, and the T4 has no bf16.
   - In fp16 the hidden values exceed 65504, so the latents come out all NaN.
   - Patching single layers (scaled linears, fp32 LayerNorm) only moved the failure to the next operation.
   - ComfyUI also lists Qwen-Image as bf16/fp32 only.
   - Measured T4 speeds: fp16 19 TFLOPS, bf16 2 (emulated), fp32 3.7.
4. **The fix** ([fp16_mixed.py](../prototype/layerproto/fp16_mixed.py)):
   - Every linear layer's input is scaled to absmax 32, multiplied in fp16 and rescaled in fp32. Scaling to absmax 1024 still gave NaNs.
   - Attention runs in fp16, with the values input scaled the same way.
   - The residual stream, norms and gates stay in fp32.
   - **Result:** no NaNs; 640 px, 4 layers, 50 steps in 35 min (42 s per step); 4 clean layers that recomposite with a mean pixel difference of 4.1 out of 255. See [prototype/kaggle/results_qwen_batch_v2/](../prototype/kaggle/results_qwen_batch_v2/).
5. **Kaggle workflow lessons:**
   - Run unattended batch notebooks with the CLI (`kernels push`, `machine_shape: NvidiaTeslaT4`).
   - In the editor, re-import the notebook (File → Import) to keep the kernel alive; the toolbar's circular-arrow button restarts the whole session.
   - Smoke-test with 2 steps before any full run, and stop automatically on NaN.

## Phase 4: the speed target (≤ 40 s per layer split), 2026-11-09

- **Why the T4 is slow:** one step at 640 px with 4 layers processes about 8,000 image tokens through 20B parameters, twice with CFG. That is about 6.6 × 10¹⁴ FLOPs per step, so about 35 s at the T4's 19 TFLOPS. That matches the 42 s measured, so no software trick reaches 40 s at the default settings.
- **Comfy Cloud** takes seconds because it runs on RTX PRO 6000 Blackwell GPUs (96 GB, native bf16).
- **Official settings:** the model card and the ComfyUI template both use 50 steps, CFG 4 and 640 px. fal defaults to 28 steps. No Lightning or few-step LoRA exists for Layered.
- **Options under test:**
  - A lower resolution (512 / 384 px; the pipeline was patched to accept sizes below 640), fewer steps, no CFG (one pass per step) and 3 layers. This mirrors the SDXL-Turbo settings that ran in under 60 s on the laptop.
  - Kaggle TPU v5e-8, which the account offers: 8 chips × 16 GB, bf16. **[est]** about 40 s at about 30 steps, but it needs a torch_xla port of the pipeline (2–5 days) and nobody has published Qwen-Image on TPU.
  - Hosted APIs: fal about $0.05 per run; WaveSpeed $0.025 per layer. Comfy Cloud plans start at $20/month.

## Working agreements

- Instructions are written as numbered steps with links and code.
- No arbitrary "best" picks: build neutral, complete outputs (all images, neutral comparison sheets) and let the user choose.
- Check facts before stating them, and mark estimates.
- Don't invent features. Follow the plan and the references.
- GPU quota is limited: verify locally (`prototype/tools/dryrun_qwen_*.py` with tiny models), smoke-test 2 steps, and avoid full runs per bug.
