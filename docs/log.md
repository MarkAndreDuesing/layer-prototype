# Log

> **Role: history (append-only).** How we got here, phase by phase, with what was tried and measured. The current state is in [status.md](status.md), per-model facts in [models/](models/README.md), open work in [todo.md](todo.md).

All findings and decisions so far, in date order. Numbers are measured unless marked **[est]**. The sourced research behind each decision is in [research/](research/) and the dated snapshots in [archive/](archive/MOVED.md).

## Phase 0: models, hosting and costs research (2026-11-05 → 07)

Full report: Image generation models hosting and costs.md (removed 2026-11-10; in git history before commit 35dc02a).

- **Quality bar:** Z-Image Turbo's level (arena Elo 936) is enough. Cost, accessibility and sketch input decide the model, not leaderboard rank.
- **Recurring free GPUs:** Kaggle (about 30 GPU-h/week), the LMU Informatik CIP pool (RTX 2060 SUPER 8 GB through Slurm, with a CIP account) and Hugging Face Spaces (5 GPU-min/day). The LMU Abaki servers (A5000 24 GB) are for theses only; LRZ needs a research project.
- **Ruled out as recurring options:** SageMaker Studio Lab (closed to new users), Lightning AI and Comfy Cloud (one-time trials), Colab web UIs (forbidden on the free tier).
- **APIs:** DeepSeek and Kimi have no image generation. Open models cost about $0.001–0.015 per image on fal, WaveSpeed, Runware or Together. OpenRouter adds a 5.5% top-up fee ($0.80 minimum).
- **Price-to-performance:** Pareto frontier over 101 arena models; the knee is at about $10 per 1,000 images (Muse Image). FLUX 3 Image Klein is the cheapest open-weights model on the frontier ($9 per 1,000). Script: [../research_notes/Image generation models hosting and costs/](../research_notes/Image%20generation%20models%20hosting%20and%20costs/).
- **Sketch input** needs img2img, a ControlNet or reference editing. Z-Image Turbo (Apache-2.0, fits 16 GB) and SDXL have scribble ControlNets; FLUX.2/3 Klein only reference editing.
- **CPU-only on the laptop:** only small distilled models (SD-Turbo, SDXL-Turbo, SDXS, SD 1.5 + LCM-LoRA) were expected to stay under 3 minutes. Phase 1 tested three of them. The research-time speed estimates (4–8 s per image) were about 4–10× too optimistic; measured: about 20 s and 58 s. They scaled published timings that used OpenVINO, the tiny decoder and 1 step, none of which the tested setup used. Speed-up options and a test order: [research_cpu_speedups.md](research/cpu-speedups.md).

## Phase 1: sketch-to-image on a CPU laptop (FastSD CPU)

**Setup:** Windows 11 laptop, 15.7 GB RAM, no GPU. Current guides (two paths, offline): [fastsd-cpu/INSTALL.md](guides/fastsd-cpu/INSTALL.md) and [fastsd-cpu/USAGE.md](guides/fastsd-cpu/USAGE.md). Results: [../experiments/CONFIRMATION_RESULTS.md](../experiments/CONFIRMATION_RESULTS.md). How the session went and why each decision was made: session-summary-sketch-to-image.md (removed 2026-11-10; in git history before commit 35dc02a).

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
2. **Audit:** that version is a sticker collage, not coherent layered generation. Each layer is an independent text-to-image call that shares only a style suffix. See Coherent layered generation pipeline.md (removed 2026-11-10; in git history before commit 35dc02a). **Decision:** coherence has to come from one shared image or scene. Generate the scene first and then split it (Qwen-Image-Layered), or build it element by element on the existing scene (LayerDiffuse chain). A layer edit stays coherent only if it is regenerated through the composite.
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
  - Kaggle TPU v5e-8, which the account offers: 8 chips × 16 GB, bf16. **[est]** about 40 s at about 30 steps, but it needs a torch_xla port of the pipeline and nobody has published Qwen-Image on TPU. (The "2–5 days" effort figure first written here was an unverified AI guess; a measuring probe followed, see Phase 6.)
  - Hosted APIs: fal about $0.05 per run; WaveSpeed $0.025 per layer. Comfy Cloud plans start at $20/month.

### Result: about 33–41 s per split on the free T4s (sweeps v3 and v4)

All runs use the forest test image on Kaggle 2× T4 with `fp16_mixed` and the Q4_K_S GGUF. "Total" is the wall time for one full split (denoise, VAE decode and recomposite), not counting model loading. The images are in [prototype/kaggle/results_qwen_sweep_v3/](../prototype/kaggle/results_qwen_sweep_v3/) and [results_qwen_sweep_v4/](../prototype/kaggle/results_qwen_sweep_v4/).

| Resolution | Layers | Steps | CFG | s / step | Total | Layers usable? |
|---|---|---|---|---|---|---|
| 640 | 4 | 50 | 4 | 42 | 35 min | yes (reference) |
| 512 | 4 | 20 | 4 | 25.9 | 525 s | yes |
| 512 | 3 | 12 | 4 | 22.1 | 272 s | yes |
| 512 | 3 | 12 | 1 | 11.1 | 139 s | yes, same as CFG 4 |
| 384 | 3 | 8 | 1 | 7.3 | 63 s | yes |
| 512 | 3 | 5 | 1 | 10.9 | 60 s | yes |
| 448 | 3 | 5 | 1 | 8.7 | 48 s | yes |
| 384 | 4 | 5 | 1 | 8.1 | 45 s | yes: trees, pines, fire+chair, stump |
| **384** | **3** | **5** | **1** | **6.8** | **41 s** | **yes** |
| **384** | **3** | **4** | **1** | **6.9** | **33 s** | **yes, slightly softer edges** |

**Findings:**
- **CFG 1 instead of 4 halves the time** (one pass per step instead of two) with no visible loss.
- **Steps can drop from 50 to 4–5.** The layer split stays sensible (background, tree group, campfire group). Edges get a little softer, and the reconstruction error rises from 4.1 to about 7 out of 255.
- **Time per step scales with pixel count:** about 7 s at 384 px, 9 s at 448 and 11 s at 512.
- Each total included about 5 s of moving the transformer between GPU and CPU. Sweep v5 removes that (`offload=False`) and tests other images.

### Sweep v5: transformer kept on the GPUs, 3 different images

**What was done and why:**
- Until v4, `decompose()` moved the 20B transformer from the GPUs back to CPU RAM after every split. That was needed at 640 px, where the VAE decode ran out of memory while the transformer was still on the GPU, and it cost about 5 s per split.
- v5 checked that at 384–512 px the split transformer leaves enough room for the decode, so it can stay on the GPUs between splits (`offload=False`). Peak memory is 8.0 / 7.6 GB on the two T4s, so it fits.
- v5 also tested whether the fast settings carry over beyond the forest painting: the same model on a pixel-art camp and a pale pencil sketch, at 3 or 4 layers, 3 to 5 steps and 384 or 512 px, always with CFG 1.
- **Reading the table:** "Total" is the wall time for one split (denoise + VAE decode of every layer + recomposite), with the model already loaded. "Split" describes what ended up in each layer, judged from the saved layer strips.

Images are in [prototype/kaggle/results_qwen_sweep_v5/](../prototype/kaggle/results_qwen_sweep_v5/).

| Image | Resolution | Layers | Steps | Total | Split |
|---|---|---|---|---|---|
| impressionist forest | 384 | 3 | 3 | **20 s** | background / bare trees / pines + campfire; usable, slightly softer |
| impressionist forest | 384 | 3 | 4 | **27 s** | same split |
| impressionist forest | 384 | 4 | 4 | 30 s | adds a stump and log layer |
| impressionist forest | 512 | 3 | 3 | 31 s | sharper |
| pixel-art camp | 384 | 3 | 4 | 27 s | — |
| pixel-art camp | 384 | 4 | 5 | 38 s | sky + moon / ground + campfire / tree group / skeleton + green tree; good |
| pixel-art camp | 512 | 4 | 4 | 48 s | — |
| pale sketch, skeleton | 384 | 3 | 4 | 27 s | campfire, table and logs separated, but the **skeleton stays half in the background** |
| pale sketch, skeleton | 384 | 4 | 5 | 38 s | — |

**Conclusion:**
- **On the free T4s, 384 px with 3–4 layers and 4 steps takes 27–30 s**, well under the 40 s target. 512 px with 3 steps (31 s) is the sharper alternative.
- The split carries over to pixel-art and line-art styles.
- Weak spot: thin or low-contrast subjects (the pale skeleton) may stay in the background layer at 3 layers.
- Why 384 px / 4 layers / 4 steps / CFG 1 became the recommended setting: it is the fastest setting that kept a sensible split on all three styles. 3 steps also works on the forest (20 s) but edges get softer; 512 px is sharper but about 1.5× slower.
### Sweep v6: Qwen-Image-Lightning 4-step LoRA on Layered (untested combination)

**What was done and why:**
- Qwen-Image-Lightning (lightx2v) is a speed LoRA distilled for the plain Qwen-Image text-to-image model so it works at 4–8 steps. Nobody had tried it on Qwen-Image-Layered, which shares the base architecture but was fine-tuned for decomposition. The question was whether it makes the 4-step split better or faster.
- Variants: the Lightning scheduler config alone (no LoRA), then the LoRA at strength 0.5 and at 1.0, each at 384 px, 4 layers, 4 steps, on the 3 test images.
- **"Recomposite diff"** = mean absolute pixel difference (0–255) between the input image and the alpha-composite of all output layers. Low means the layers together still reproduce the input; it does not measure how well objects are separated.

**Setup:**
- `fp16_mixed.add_lora` attaches the LoRA without merging it into the GGUF weights. All 720 LoRA modules matched Layered's layers.
- The Lightning scheduler config is from the lightx2v model card.
- Images are in [prototype/kaggle/results_qwen_sweep_v6_lightning/](../prototype/kaggle/results_qwen_sweep_v6_lightning/).

| Variant (384 px, 4 layers, 4 steps) | Total | Recomposite diff | Observation |
|---|---|---|---|
| base model (v5) | 30 s | 7.1 | reference |
| Lightning scheduler only | 35 s | 8.6 | about the same as the base |
| Lightning LoRA at 0.5 | 39 s | 10.3 | in between |
| Lightning LoRA at 1.0 | 37–39 s | 11–13.5 | sketch: **the skeleton now separates cleanly** (base: half stuck in the background). But colours and texture drift: noisier sky, the image no longer matches the input as closely. |

**First-attempt result (the user decides verdicts):** at these settings it gives no speed-up (the base model already works at 4 steps), adds about 5 s of LoRA overhead, and lowers how faithfully the layers match the input. Its plus is cleaner separation of thin subjects.
**Next attempts:** 2-step runs (where a distillation LoRA should matter most), a finer scale sweep (0.25–0.75), and using it only for hard, low-contrast scenes.

- GPU time used for the Qwen sweeps so far: about 2.5 h (Kaggle quota shows 5.4 h this week, including earlier editor sessions).

## Working agreements

- Instructions are written as numbered steps with links and code.
- No arbitrary "best" picks: build neutral, complete outputs (all images, neutral comparison sheets) and let the user choose.
- Check facts before stating them, and mark estimates.
- Don't invent features. Follow the plan and the references.
- GPU quota is limited: verify locally (`prototype/tools/dryrun_qwen_*.py` with tiny models), smoke-test 2 steps, and avoid full runs per bug.

## Phase 5: other decomposition routes and move/resize re-harmonisation (2026-11-09)

### LayerD (released, Apache-2.0) on our scenes: first attempt returned one layer

- **Setup:** Kaggle CPU, no GPU quota. It needs `transformers<5`, because BiRefNet's remote code breaks on transformers 5.
- **Result:** about 60 s per image, but on all 3 scenes it returned **1 layer, the unchanged image**. It was trained on graphic designs and finds nothing to peel off in painted scenes.
- Results: [prototype/kaggle/results_layerd_trial/](../prototype/kaggle/results_layerd_trial/).
- **Next attempts:** lower the matting/peel thresholds and allow more recursion depth; test on the new fox & cottage scene and on a flatter, graphic-style image closer to its training domain.

### Generate → SAM2 → LaMa (segment + inpaint route): first attempt fast, automatic grouping poor

- **Pipeline** ([tools/make_segment_inpaint_trial.py](../prototype/tools/make_segment_inpaint_trial.py)): SAM2 hiera-small automatic masks, then the 4 largest object masks (each under 35% of the image) as layers, then LaMa fills the background.
- **Speed and fidelity:** **3–7 s per image** on a T4. The recomposite matches almost exactly (diff 0.2–2.4), because the layers are cut-outs of the original pixels.
- **Grouping is poor.** SAM picks fragments, not story objects:
  - Forest: moon, stump, flame, logs. The trees and chair are not separated.
  - Sketch: one tree, campfire, log, plus a "layer" that is the white background. The **skeleton is not separated**.
- **No amodal completion:** hidden parts of objects stay missing.
- **First-attempt result:** with automatic "largest K masks" selection, Qwen-Image-Layered gave far more meaningful layers.
- **Next attempts:** prompt SAM2 with points or boxes per noun from the LLM breakdown (Grounded-SAM style) instead of "largest K"; merge masks per object; compare LaMa with SDXL inpainting for the hidden parts; add amodal completion (pix2gestalt) for occluded objects.
- Results: [prototype/kaggle/results_segment_inpaint/](../prototype/kaggle/results_segment_inpaint/).

### Move/resize with re-harmonisation on the LayerDiffuse chain (training-free, our proposal)

`LayerDiffuseChain.relocate()` in [layerdiffuse_chain.py](../prototype/layerproto/backends/layerdiffuse_chain.py) does four things:
1. Warps the layer into its new box and pastes it over the background.
2. Runs an SDXL img2img pass, conditioned on the background through LayerDiffuse `bg2ble`, at strength s.
3. Pins the object's interior to its re-noised self for the first 40% of steps.
4. Re-extracts the layer.

The idea follows DreamLayer's harmonisation step. Nothing is trained.

**Step by step, in more detail:**
1. **Warp in pixel space.** The layer's RGBA is scaled and shifted into the target box with PIL and alpha-composited over the background (the composite of all other layers). This is the "moved but not yet integrated" image: right position, but old lighting and no contact with the new spot.
2. **Encode and re-noise.** That image is VAE-encoded and noised to strength s (s = 0.3 means the last 30% of the denoising schedule is redone). Low s keeps the object; high s lets the model redraw more.
3. **Denoise with the scene as condition.** SDXL base with the LayerDiffuse `bg2ble` weights denoises it, conditioned on the background latent, so lighting, glow and style come from the scene.
4. **Pin the interior.** In the early steps the latents *inside* the object (its alpha, eroded by a few pixels) are replaced by the original object's latents, re-noised to the current step. Only the rim and the surroundings can change (contact, glow, edges), which keeps the object's identity.
5. **Re-extract the layer.** v1 used LayerDiffuse's `bgble2fg` model to extract the moved object as a new RGBA layer. v2 uses a mask instead: the object's own warped alpha plus every pixel that changed noticeably within a ring around it (where new glow or contact appeared).

**What strength does:** s = 0.3 keeps identity and adds light and glow from the new surroundings; s = 0.5 also redraws details such as faces; s = 0.6 with v1's sampler broke down.

**Tests (SDXL base, 512 px, 20 steps, CFG 7, Kaggle T4):**

| Version | Result |
|---|---|
| v1: DPM++ SDE scheduler, `bgble2fg` extraction | About 11 s per edit. At s = 0.6 the image fills with artifacts: a stochastic sampler adds fresh noise every step, which fights the pinned latents, so pinned and free regions disagree. The extracted layer is translucent or the wrong object. Replaced by v2's deterministic Euler sampler and mask extraction. |
| **v2: Euler scheduler, mask extraction** (the object's alpha plus wherever the image changed, within a ring around it) | **About 4–5 s per edit.** The moved object takes on the new surroundings' light and glow, and the extracted layer is clean. **s = 0.3 keeps identity**; s = 0.5 re-draws details such as the face. `bgble2fg` extraction (about 9.5 s) is still worse than the mask. |

Results: [prototype/kaggle/results_relocate_v2/](../prototype/kaggle/results_relocate_v2/); v1 in `results_relocate_smoke_turbo/` and the notebook history.

**Open:**
- Contact shadows are weak.
- The old position isn't inpainted when the background itself contains the object (here the background was clean).
- Only one object and one scene were tested. A proper evaluation is research question RQ4 in [Research gap and questions.md](research/research-gap.md).

### Full loop on free T4s: Qwen-Image-Layered split → move a layer → re-harmonise

**Pipeline** ([tools/qwen_relocate_demo.py](../prototype/tools/qwen_relocate_demo.py) and [tools/make_qwen_relocate_trial.py](../prototype/tools/make_qwen_relocate_trial.py)):
1. Qwen-Image-Layered run v7: 384 px, 4 layers, 4 steps, **about 32 s**, every RGBA layer saved. See [results_qwen_v7_layers/](../prototype/kaggle/results_qwen_v7_layers/).
2. Pick one layer, move and resize it, then re-harmonise it against the composite of all the other layers with `relocate()` (SDXL base, 512 px, mask extraction). That takes **about 3.5 s per edit** once SDXL is loaded; the first edit pays about 20 s of loading.

**How the two parts connect:**
- The v7 notebook saved each scene's 4 RGBA layers as PNGs; they were uploaded as a private Kaggle dataset (`qwen-v7-layers`). The relocate notebook reads them, so the 20B Qwen model and SDXL never have to share the GPUs.
- For each scene one layer was picked by hand (the one with the clearest object). The other three were alpha-composited back to front to form the background condition, so the moved object is harmonised against the actual decomposed scene, not the original image.
- Each result strip shows: original composite, the move without harmonisation (plain paste), and the harmonised result at each strength.

| Scene | Edit | Result (strength 0.3, pin 40%) |
|---|---|---|
| impressionist forest | campfire + chair: move right, scale 1.2 | It fits the new spot, with the same painterly style and fire glow. But the log pile is redrawn flatter and the chair changes shape. |
| pale sketch | skeleton: move left | It sits plausibly behind the campfire, but is re-rendered darker and more detailed than the original pale style. |
| pixel art | skeleton + green tree: move left | Placement and style hold. The tree came along because Qwen put it in the same layer. Some ghosting remains at the old spot. |

**Takeaways:**
- The loop is technically complete and fast enough for interaction on free hardware: about 32 s per split and about 4 s per move.
- The remaining problem at that point was **identity drift**: re-harmonising redraws the object. **Run v2 pinned the interior for 70% of the steps instead of 40%: almost no change** (explained later: because of a bug the pin mask was nearly empty; fixed in v5 below) (the forest is nearly identical, the skeleton slightly closer to the original). The drift happens in the unpinned final steps. The next candidates are identity conditioning (IP-Adapter on the original object, or reference-only attention) or pinning with a softer mask through every step ([results_qwen_relocate_v2_pin70/](../prototype/kaggle/results_qwen_relocate_v2_pin70/)).
- Qwen sometimes groups unrelated objects into one layer (skeleton + tree). Using more layers, or splitting the grouped layer again, is the obvious fix.
- Results: [prototype/kaggle/results_qwen_relocate_v1/](../prototype/kaggle/results_qwen_relocate_v1/).

### Identity drift in `relocate()`: two bugs found, fixed in v5

- **v3/v4 bug:** the strips were byte-identical whatever the pin settings, so pinning was a no-op. The pin mask used alpha > 128, then eroded 12 px per side. Qwen layers have soft alpha, so this covered only about 42 of 4,096 latent pixels for the campfire.
  - A separate wrong timestep lookup (img2img keeps the full timestep list) turned out not to matter: the Euler scheduler's `add_noise` uses its own step index.
- **v5:** the mask now uses alpha > 40 with a 4 px band. The interior is pinned on every step except the last 2. Optional **paste-back** blends the original object pixels back inside its alpha after decoding (several papers do this; see `spatial_edit_survey_verified_part*.md`).
- **Result:**
  - The pinned interior now holds (forest log pile and flame keep their shape).
  - With paste-back 0.8, the moved skeleton keeps its look in the new place, while the surroundings are still re-harmonised.
  - About 4 s per edit.
  - Remaining problems: thin parts at the edge (chair legs) still change, and contact shadows stay weak.
- Results: [prototype/kaggle/results_qwen_relocate_v5_fixedpin/](../prototype/kaggle/results_qwen_relocate_v5_fixedpin/).

### Stable-Layers (Stability AI) at reduced settings on the T4 (run v8)

- **Setup:** its 330 MB PEFT LoRA attached via `fp16_mixed.add_lora`; all 420 modules matched. Euler sampler, CFG 1, 384 px, 4 layers.
- **Speed:** 4 steps about 34 s; 8 steps about 71 s.
- **Result:**
  - Cleaner semantic split than the base model: on the sketch the background is truly empty paper and the fire gets its own layer.
  - But colours and contrast shift, so the recomposite is about 2× further from the input (difference 15–18 vs about 7).
- **Heun sampler** (what Stable-Layers recommends) failed: the diffusers pipeline passes custom sigmas, which `FlowMatchHeunDiscreteScheduler` rejects. **Since then:** `qwen_layered.HeunVelocity` turns the pipeline's own Euler step into an exact Heun step (checked on a toy ODE: error 0.05 vs 0.37 for Euler at 8 steps); the v9 sweep with it waits for Kaggle.
- **Next attempts:** Heun at 4–8 steps, more steps at 384 px, and a colour-shift correction (match the recomposite's colour statistics to the input).
- Results: [prototype/kaggle/results_stable_layers_v8/](../prototype/kaggle/results_stable_layers_v8/).

### FLUX-class trials on 2× T4 and the Kaggle outage (2026-11-09 afternoon)

- **RevealLayer** ([tools/make_reveallayer_trial.py](../prototype/tools/make_reveallayer_trial.py)): GGUF Q4_K_S FLUX.1-dev, with the encoders and VAE from ungated `ostris/Flex.1-alpha`.
  - Runs 1–4 failed in turn on:
    1. transformers 5 (pinned 4.57.1)
    2. the authors' hard-coded ViT-B/32 path (download it and also copy it to the original path)
    3. the same path again
    4. a cancel with no log
  - **Run 5 runs end to end:** model load about 4 min, then **69 s for 12 steps** at 512 px, peak 10.5 / 11.2 GB. **But every layer is pure noise.** The fp16 guard fired on all 12 steps in `FluxSingleTransformerBlock`, the same overflow as Qwen-Image. Results: [results_reveallayer_v5/](../prototype/kaggle/results_reveallayer_v5/).
  - Runs 6 and 7 apply `fp16_mixed` to the transformer. They are still untested because of the outage.
  - Since then, three fp16 hazards were found in RevealLayer's own code and patched in the notebook (run 8): the attention mask is filled with −1e9 (−inf in fp16, so fully masked rows give NaN), an unguarded refiner residual, and an fp16 VAE decode the authors themselves flagged ("odd purple output"). A plain-FLUX control notebook separates "FLUX on a T4" from "RevealLayer's code".
- **LayerDiffuse-Flux** ([tools/make_layerdiffuse_flux_trial.py](../prototype/tools/make_layerdiffuse_flux_trial.py)): uses `fp16_mixed` and `add_lora`. Pushed, but queued throughout the outage.
- **Qwen-Image-2.1-Turbo RGBA** ([tools/make_qwen21_rgba_trial.py](../prototype/tools/make_qwen21_rgba_trial.py)): written, not pushed yet.
- **Kaggle outage:** from about 14:00, runs stayed QUEUED or went to CANCEL_REQUESTED with no log. The editor showed "Failed to Authorize Session". A Kaggle banner confirmed a platform issue. Quota (6:50 of 30 h) and phone verification were fine.
  - **Lesson:** every batch trial now gets a 25-minute SIGALRM cap so that a stall still ends with a log. Check for a Kaggle banner before re-pushing.

## Phase 6: loop modules on CPU, research round, TPU probe (2026-11-09 evening)

- **Story-event router** ([layerproto/events.py](../prototype/layerproto/events.py)): Ollama `qwen3:4b` turns an event into per-layer actions (keep / relight / reprompt / remove) plus effect overlays. 5 test events routed sensibly at about 30 s each on CPU; instant CPU preview with colour grading and procedural rain, snow, fog, smoke and glow. Results: [results_local_events/](../prototype/kaggle/results_local_events/).
- **Procedural shadow layer** ([layerproto/shadow.py](../prototype/layerproto/shadow.py)): EasyOmnimatte-style recipe, one contact line per object, capped length; about 0.6 s per layer. Results: [results_local_shadow/](../prototype/kaggle/results_local_shadow/).
- **IC-Light relighting** ([layerproto/relight.py](../prototype/layerproto/relight.py)): the SD1.5 fbc model relights a layer against a new background, keeping the layer's alpha and detail; 44 s on CPU (8 steps, 384 px). The fire also loses brightness (the model has no notion of light sources). Results: [results_local_relight/](../prototype/kaggle/results_local_relight/).
- **Drift harness** ([layerproto/drift_eval.py](../prototype/layerproto/drift_eval.py)): 3 arms (layered / flat / re-decompose), per-turn metrics outside the edited regions. Self-test with stand-in editors only, not a result about real models.
- **Qwen levers:** `cached_encode()` (prompt embeddings cached on disk) and `HeunVelocity`; v9 sweep generated.
- **Notebooks waiting for Kaggle GPUs:** Qwen-Image-2.1 RGBA (with the fp16-safe texture-fix VAE), FLUX control, RevealLayer run 8, ART+ (base LoRA hash-identical to the official CLD file), LayerDiffuse-Flux.
- **TPU probe** ([tools/make_tpu_probe.py](../prototype/tools/make_tpu_probe.py)): three CLI runs never got a TPU. Kaggle stores `machine_shape: TpuV5E8`, but the batch runs used the standard CPU image (JAX saw only a CPU). Also found: pip's newest `torch_xla` is 2.9 while the image ships torch 2.11, so torch has to be pinned to 2.9. Next: run the probe as an interactive TPU session in the editor.
- **Test image:** from now on one consistent scene built for layer splitting (fox & cottage, see [prototype/kaggle/test_scene_candidates/](../prototype/kaggle/test_scene_candidates/)) is the main test input; the old forest, sketch and pixel-art scenes stay as style checks.
- Research write-up: Layered loop open paths.md (removed 2026-11-10; in git history before commit 35dc02a).

## Phase 7: Kaggle back, FLUX-class models running (2026-11-09 night)

- **Kaggle:** GPUs recovered around 21:55. Kaggle allows **at most 2 batch GPU sessions at once** ("Maximum batch GPU session count of 2 reached"); a small queue script pushes the next notebook when a slot frees.
- **Plain FLUX.1-dev control** (GGUF Q4_K_S, 512 px, 16 steps, one T4): plain fp16 **41 s**, `fp16_mixed` 53 s, **no non-finite values** in either, 7.5 GB peak. So FLUX itself is fine on a T4; RevealLayer's noise came from its own code. Results: `prototype/kaggle/results_flux_control_v1/`.
- **Qwen-Image-2.1-Turbo** (GGUF Q4_K_S, 512 px, 8 steps, texture-fix VAE):
  - Run 1: loaded (249 s, 8.0 / 12.7 GB) but every call failed (`4096 vs 8192`): the same diffusers GGUF loader gap as for Qwen-Image-Layered (a non-Linear parameter kept its packed bytes). Fixed by dequantising it after loading.
  - **Run 2 works: 17–23 s per image**, peak 9.6–10.7 GB. Transparent campfire from text (37% transparent; dark smoky halo), RGBA edit "blue flames" keeps the alpha, **extracting the campfire from the forest scene gives a clean in-place layer**, scene "rainy night" edit keeps composition but is subtle. KV cache on/off gives the same image. Results: `results_qwen21_v2/`.
- **RevealLayer:**
  - Run 8 stopped at a patch anchor: the adaptation cell had already replaced −1e9 by `torch.finfo(dtype).min` (−65504), which still overflows to −inf once a negative score is added.
  - Run 9 failed importing a Qwen module from `fp16_mixed` inside RevealLayer's older bundled diffusers; the Qwen attention patch is now optional.
  - Run 10: **denoising finished** for the first time; failed in the transparent decoder (fp16 inputs, fp32 weights).
  - **Run 11 works:** forest + 3 boxes → background with the hidden areas filled in (trees and campfire removed, forest completed) + 3 clean RGBA layers (pines, bare trees, campfire + chair). **134 s for 12 steps**, peak 11.4 / 11.2 GB, no overflow. Results: `results_reveallayer_v11/`.
- **LayerDiffuse-Flux** run 2 ran (55 s per image, 7.6 GB) but **0 LoRA modules matched**: its LoRA uses BFL/kohya key names with fused qkv. Run 4 converts the keys with diffusers' kohya converter first.
- **ART+** run 2: all three LoRAs matched completely (pre_fuse 190, ART base from CLD 304, ART+ 304; weights from the official Drive folders), then stopped loading the transparent-decoder checkpoint (`weights_only` default in torch ≥ 2.6). Run 3 loads it with `weights_only=False` (official `thuteam/CLD` file).
- **Qwen v9 sweep** failed on every config (`unexpected keyword 'sampler'`): the batch notebook embeds `qwen_layered.py` via the test-notebook generator, which had not been re-run. Regenerated and re-queued.
- **LayerDiffuse-Flux run 4 works:** all 494 LoRA modules matched after key conversion; campfire, deer and glass bottle as clean transparent objects (75–82% transparent, the bottle semi-transparent), 57–63 s each at 512 px / 16 steps, 8.9 GB on one T4.
- **ART+ run 3:** loaded in 295 s (7.0 / 11.0 GB), decoder matched; generation failed on fp16 inputs to fp32 VAEs → run 4 adds input-casting hooks.
- **Qwen v9 sweep (re-run) works:** exact Heun via `HeunVelocity` runs (2 calls per step: 58–61 s at 4 steps, 384 px). Stable-Layers + Heun gives a very clean forest background (trees and campfire removed and filled) with the trees in clean layers; on the sketch skeleton, fire and logs stay together; colour shift persists (diff 15–18). Base + Heun diff 7.0. 2-layer splits take 25.7 s (bg + one fg). Results: `results_qwen_v9b/`.
- **Test scene:** the user chose the fox & cottage variant with the cottage, a stone wall with gate, a path, a fox, a tree and background houses; the prompt was tuned (larger free-standing fox) and the user picked seed 13: `prototype/test_scene/fox_cottage.png`, with layer names and boxes in `fox_cottage.json`.
- **Docs restructured** into status / todo / log / glossary / concepts / models / research / guides / archive (old → new paths in [archive/MOVED.md](archive/MOVED.md)).

## Phase 8: fox & cottage runs, ART+ end to end, docs consolidated (2026-11-10)

| Run | Result |
|---|---|
| Qwen-Image-Layered on fox & cottage (384 px, 4 layers, 4 steps) | 33 s; background / house roof / cottage + tree / fox; 6 layers 41 s; 5 layers at 512 px 62 s |
| Stable-Layers on fox & cottage (5 layers) | 39 s Euler, 67 s Heun; cleanest split so far: sky / meadow / cottage + tree / path / fox; colour shift (diff ≈ 21) |
| ART+ run 4 | denoising finished, OOM in the fp32 VAE decode with the transformer still on GPU 0 |
| ART+ run 5 | end to end (345 s / 285 s, 10 GB) after offloading the transformer before decoding; output wrong: small image in one corner, opaque rectangles, case 2 blank |
| RevealLayer runs 12–13 (fox & cottage) | failed: fp16 inputs to the scene VAE that our patch had switched to fp32 (the forest test ran only one case); run 14 casts the inputs |

Docs: the six concept pages became three ([pipeline](concepts/pipeline.md) with a workflow diagram and routes A–F, [editing](concepts/editing.md), [fp16-on-t4](concepts/fp16-on-t4.md)); LayerFusion and the unreleased methods merged into one methods table ([papers-survey](research/papers-survey.md)); the two Kaggle guides merged ([guides/kaggle.md](guides/kaggle.md)); missing archive tables (APIs, price-performance, base models, LMU access, TPU routes, tool licences) moved into [research/hosting.md](research/hosting.md); research-note facts moved into the factsheets as tables. Where each snapshot's content lives: [archive/MOVED.md](archive/MOVED.md).

## Phase 9: ART+ size bug, decoder fixes, re-tries on fox & cottage (2026-11-10)

| Run | Result |
|---|---|
| ART+ run 6 | cause of the "small image in a corner": PrismLayers was written for diffusers 0.31, where FLUX's `vae_scale_factor` was 16; in 0.32+ it is 8, so latents were built for a 1024 px image with boxes in one quarter (also why run 4 ran out of memory). With 16 restored: **full-size coherent scenes** (330 s / 282 s, 8.2 GB). Layers are still opaque crops: CLD's transparent decoder was used; LoRA headers have no alpha keys, so scale 1.0 was right. Run 7 loads ART+'s own decoder. |
| RevealLayer runs 14–15 (fox & cottage) | 14: out of memory in the fp32 decode (5 boxes → 7 frames); **15 works** with the transformer offloaded before decoding: 140 s (12 steps) / 334 s (16 steps, true_gs 3.5), filled background + cottage / gate / tree / fox |
| Docs | research_notes de-duplicated into hubs; superseded drafts, the HCI note and the docs archive snapshots deleted (content merged first; originals in git history before 35dc02a) |
| SAM2 + LaMa v2 | **by noun works:** hand boxes and Grounding DINO boxes from the layer names both give village house / cottage / gate arch / tree / fox as clean cut-outs in 7–8 s (DINO missed only "village houses"); automatic masks still give fragments; LaMa fill blurry for large holes |
| LayerD v2 | CPU sweep pushed: lower alpha threshold, no palette refinement, 1024 px matting, 5 iterations; logs alpha coverage per step |
| Qwen-Image-2.1 | fox & cottage notebook written (extraction, fox → wolf RGBA edit, night / heavy-rain edits, scene from caption) |

## Phase 10: Qwen-Image-2.1 on fox & cottage, LayerD root cause (2026-11-10)

| Run | Result |
|---|---|
| Qwen-Image-2.1 on fox & cottage | about 20 s per call: fox extracted cleanly in place (ComfyUI background-removal prompt; the RGBA-wrapper prompt gave an empty image); cottage extraction kept the fox; fox → grey wolf RGBA edit keeps alpha; **night and heavy-rain scene edits are strong and keep the composition** (explicit "keep every object, the composition and the style"); scene from caption and a lantern prop good |
| LayerD v4 diagnostics | the matting network outputs exactly 0 on every pixel (max 0.0) for all variants and scenes → the "1 layer" results never tested LayerD; weights not materialised. v6 with transformers 4.46.3 failed (checkpoint has no safetensors metadata); v7 keeps transformers < 5 and forces a full load (`low_cpu_mem_usage=False`) with a load report |
| ART+ v7 | own transparent decoder loaded fully; output unchanged → whole and background frames identical, so the per-layer mechanism is not taking effect; v8 logs `layer_pe` and per-frame latents |

## Appendix: sketch-to-image phase summary (2026-11-05 → 07)

Merged from the former `session-summary-sketch-to-image.md` (original kept in archive/ (removed 2026-11-10; in git history before commit 35dc02a)).


A summary of the first working session, which became Phase 1 of this repo. Numbers and results are in [findings-log.md](log.md) (Phase 1) and [../experiments/](../experiments/). This page records **what was done, in what order, and why each decision was made**.

### Starting point

- **Goal:** prototype a sketch-to-image tool that runs on a CPU-only Windows laptop (Intel Core 7 150U, 15.7 GB RAM).
  - The user draws a rough sketch and describes a scene. The tool redraws it in a requested style.
- **Constraint set by the user:** the style varies per request (pixel art, impressionist, hyperrealistic, or none), so nothing may be tuned for one style.
- **Test scene:** one skeleton dancing by a campfire in a night forest. Sketches are in [../experiments/inputs/](../experiments/inputs/) (`sketch_original.png`, `sketch_filled.png`).

### Timeline

| Date | Step | Outcome |
|---|---|---|
| 11-05 | Checked the [FastSD CPU](https://github.com/rupeshs/fastsdcpu) requirements, installed Python 3.11.9 and uv, cloned the repo and ran `install.bat` | Works after removing scikit-learn, which Smart App Control blocked |
| 11-05 | Chose three paths to compare: SD-Turbo img2img, SDXL-Turbo img2img, and a ControlNet path. Set them up in the web UI through Claude in Chrome | SDXL-Turbo + ControlNet is not offered by FastSD CPU, so Path 3 became SD 1.5 + LCM-LoRA + scribble ControlNet |
| 11-05 → 06 | Fixed runtime errors | SDXL-Turbo "Half and Float" error under transformers 5. Fixed with a float32 cast patch ([../fastsdcpu_local/](../fastsdcpu_local/)) |
| 11-06 | Rewrote the user's prompt into a style-swappable structure: style prefix + scene text | Scene text used in every later run (see [../experiments/README.md](../experiments/README.md)) |
| 11-06 | Batch 1: plain vs blurred sketch, then a filled (coloured) sketch, 4 styles per run | Filled sketch gives night scenes; blurring only adds defocus |
| 11-06 | Round 2: tuned each path over several seeds (Strength, steps, ControlNet base and scale, negative prompt) | Strength 0.8 = 0.9 at 4 steps. `dreamshaper-8` flattens every style; the neutral SD 1.5 base responds to style |
| 11-06 | The user objected to "best" picks made by Claude. All 188 images were compiled into one folder with neutral comparison sheets instead | The user filtered the images by hand, keeping 35 of 188 |
| 11-06 → 07 | Pre-registered confirmation test: 120 images on new seeds (11, 42, 99), with text-only counterparts added | [../experiments/CONFIRMATION_TEST_PLAN.md](../experiments/CONFIRMATION_TEST_PLAN.md) |
| 11-07 | The user filtered the confirmation images (27 kept, 21 parked as "interesting but not what was asked") and named 8 favourites. All favourites are SDXL-Turbo | [../experiments/CONFIRMATION_RESULTS.md](../experiments/CONFIRMATION_RESULTS.md), [../experiments/favourites/](../experiments/favourites/) |
| 11-07 | Final ranking delivered as a sheet, not prose | [../experiments/MODEL_RANKING_SHEET.xlsx](../experiments/MODEL_RANKING_SHEET.xlsx) |
| 11-07 | Wrote the setup and usage guide, then split it into two editable docs | First version archived: archive/SETUP_AND_USAGE_GUIDE_3_paths.md (removed 2026-11-10; in git history before commit 35dc02a) |
| 11-07 | **Dropped Path 3** (slower, fewest kept). Generalised both docs for teammates: upstream install steps, no machine-specific workarounds, no test prompt or sketch, fully offline after setup | [fastsd-cpu/INSTALL.md](guides/fastsd-cpu/INSTALL.md), [fastsd-cpu/USAGE.md](guides/fastsd-cpu/USAGE.md) |
| 11-07 | Tested offline use; found and fixed the cache problem (below). Added Step 0, a Requirements section, fully independent setup per path, and no fixed seed | Both models verified offline, in both modes |

### Decisions and why

1. **Recommended path: SDXL-Turbo, 4 steps, Guidance 1, 512 px, Strength 0.8 for Image to Image.**
   - It had the most kept images and all 8 favourites.
   - SD-Turbo is kept as the fast option (about 20 s vs 58 s).
2. **No negative prompt.** At Guidance 1 it is ignored. At Guidance 1.5 it added little and cost about 1.5× the time.
3. **No fixed seed in the guides.** The guides do not reproduce one test case. Generating at least three images and picking one handles the one-in-three duplicated subject.
4. **Path 3 removed.** It follows the layout best, but it was slower, rougher and kept least (2 of 12 with a sketch, 0 of 12 text-only).
5. **Offline by local model folders, not the Hugging Face cache.**
   - The app's automatic download caches only the files it used.
   - With `HF_HUB_OFFLINE=1`, `huggingface_hub` checks the cache against the full repo file list and raises `IncompleteSnapshotError`.
   - Downloading with `snapshot_download(local_dir=..., ignore_patterns=[...])` and listing `models/<name>` in `configs/lcm-models.txt` fixes this.
   - The ignore patterns matter because the full repos are 13 GB (sd-turbo) and 55.5 GB (sdxl-turbo).

### Offline facts found along the way

- `use_offline_model: true` in `configs/settings.yaml` persists. The UI checkbox alone resets on restart.
- The `--use_offline_model` command-line flag applies only to CLI mode, not to the web UI (`app.py -w`).
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `GRADIO_ANALYTICS_ENABLED=False` stop all network calls. They are set in [../fastsdcpu_local/start-webui-offline.bat](../fastsdcpu_local/start-webui-offline.bat).
- Verified through the running UI's API with the network blocked by a dead proxy plus those variables. Results: both models, Text to Image and Image to Image, about 25 s per image (SD-Turbo) and about 70 s (SDXL-Turbo). Output: [../experiments/offline_test/](../experiments/offline_test/).
- **Not yet verified:** a run with the network physically disconnected, and the download commands run from an empty folder.

### Working agreements from this session

- Processes are written as numbered steps with links and code snippets.
- No arbitrary "best" picks by Claude. Build neutral, complete outputs (all images, neutral sheets) and let the user choose.
- Results go into sheets and tables, not long write-ups, when a ranking is asked for.

### Open items

- A second, different sketch to check that the findings generalise. Every test used one scene.
- True disconnected-network verification, and a from-scratch run of the download commands.
- The ranking sheet's keep-rate formulas have no cached values. Open it in Excel and save once to compute them.
