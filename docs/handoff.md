# Hand-off: first draft of the layer prototype (2026-11-11)

> **Role: start here if you are new to the project.** What the prototype is, what works today and what does not, how to run it, where things are in the code, what the evidence says about the models, and what to build next. Detail lives in the linked pages; this page does not repeat numbers it can link.

## 1. What this is

A free, non-commercial prototype for a storytelling game: **text prompt → coherent layered scene → edit single layers (re-prompt, move, resize) → recomposite → story events change the scene**. Hardware: a Windows laptop (CPU) and Kaggle (2× T4, fp16 only, 30 GPU-hours a week, at most two GPU sessions at once). No user study is planned; the prototype and its measurements are the result ([research/research-gap.md](research/research-gap.md)). Everything in this repo (code, experiments, research notes, decisions) is on GitHub: `github.com/MarkAndreDuesing/layer-prototype`.

## 2. State of the draft

| Part | State | Where |
|---|---|---|
| Local app (Gradio, `prototype/app.py`) | **runs** (tested: starts, serves http://127.0.0.1:7870); LLM planner (Ollama), legacy layer-by-layer generation, layer cards, move / resize sliders, fit to planned box, recomposite, add an event layer | [prototype/README.md](../prototype/README.md) |
| Decompose an existing image on a Kaggle GPU | **works end to end** through a Cloudflare tunnel (SAM2 route proven from the laptop app; Qwen-2.1 route proven in a session). Methods and presets come from the server (`/methods`) | [guides/kaggle.md §3b](guides/kaggle.md) |
| Server profiles (one model family per Kaggle session) | **built and benchmarked** for `qwen-layered` (+ Stable-Layers, Lightning), `qwen21`, `sam2`, `reveallayer`, `cld`, `artplus`; the benchmark notebooks ran; the tunnel (`serve`) notebooks are generated and build, but **none has been run live yet** (the earlier SAM2-only `decompose_server` notebook was, and the profiles run the same code in the benchmarks) | `prototype/layerproto/serve/`, `prototype/tools/make_profile_notebook.py` |
| Move / resize with re-harmonising (`relocate()`) | works (3.4 s); runs on LayerDiffuse SDXL; app button **Fit + re-harmonise (GPU)** is tested with a fake chain only | [models/relocate-harmonise.md](models/relocate-harmonise.md) |
| Shadow layers, relight (IC-Light, Qwen-2.1), alpha check-and-retry, event glue | **code and tests exist** (`shadow.py`, `relight.py`, `layer_edit.py`, `event_apply.py`); **not wired into the app UI** | [models/shadow-layers.md](models/shadow-layers.md), [models/relight-ic-light.md](models/relight-ic-light.md) |
| Story-event router (LLM → per-layer actions) | built (`events.py`), previews on the CPU; **not called by the app** | [concepts/editing.md](concepts/editing.md) |
| SDXL-Turbo as the fast first image | measured (1.1-1.5 s on a T4); **no server route yet** | [models/sdxl-turbo.md](models/sdxl-turbo.md) |
| Progressive loading of layers (see 6) | **not built** | below |

All 9 CPU tests in `prototype/tools/test_*.py` pass; the doc link check (`prototype/tools/check_doc_links.py`) reports no broken links.

## 3. The recommended core path (from the evidence so far)

1. **SDXL-Turbo** makes the first picture, or a first picture from a sketch (seconds on a GPU, about a minute on the CPU).
2. **Qwen-Image-2.1** splits it (one call per object plus one for the background) and modifies it (scene edits for story events, per-layer edits and relights, add / remove an object, hidden parts).
3. **Stable-Layers** (or CLD) are optional splitters: Stable-Layers split the night and rain edits cleanly where Qwen-2.1 extraction did not; the base Qwen-Image-Layered is inconsistent but kept in contention.
4. **IC-Light, shadow layers, `relocate()`** act on single layers: IC-Light is kept pending more tests (Qwen-2.1's per-layer relight was better on night and rain objects); the procedural shadow is cheap and used; `relocate()` stays as the re-harmonise step until a replacement is found.
5. **RevealLayer** is kept for inpainting (filled backgrounds), **not** for splitting. ART+, LayerDiffuse SDXL (as a generator and its effect layers) and LayerDiffuse-Flux are not pursued.

The reasoning, per model, with the images: [models/pipeline-plan.md](models/pipeline-plan.md). **Read [models/provenance.md](models/provenance.md) before trusting any picture in the model pages:** many strips chain models (SDXL-Turbo makes the scene, Qwen-2.1 edits it, a splitter splits it); it says which model made which part. The verdicts are the user's; the pages report measurements.

## 4. Run it

### 4.1 The local app (laptop)

1. Install Ollama and pull the two models the app uses (`qwen3:4b`, `qwen2.5vl:3b`): [prototype/README.md](../prototype/README.md).
2. Get the LayerDiffuse code the legacy generator needs: [prototype/vendor/README.md](../prototype/vendor/README.md).
3. Start the app from the repository folder (the Python environment is `fastsdcpu\env`, see [guides/fastsd-cpu/INSTALL.md](guides/fastsd-cpu/INSTALL.md)):
   ```bash
   fastsdcpu\env\Scripts\python.exe prototype\app.py
   ```
   or double-click `prototype\run_prototype.bat`. Open http://127.0.0.1:7870.
4. About 11 GB of free RAM is needed for SDXL on the CPU.

### 4.2 A model on Kaggle behind the app (one profile per session)

1. Build the notebook for a profile (names: `qwen-layered`, `qwen21`, `sam2`, `reveallayer`, `cld`, `artplus`):
   ```bash
   fastsdcpu\env\Scripts\python.exe prototype\tools\make_profile_notebook.py --profile qwen21 --mode serve
   ```
   The file is `prototype/kaggle/upload_profile_qwen21_serve/profile-qwen21-serve.ipynb`. **Do not push it with the CLI** (a push starts a batch run and spends GPU quota); import it in the Kaggle editor instead.
2. Kaggle → New Notebook → File → Import notebook → that file. Accelerator **GPU T4 x2** (the only GPU option the editor offers), Internet **On**. For `qwen-layered`, use **Add Input** to attach the two public notebook outputs `markandreduesing/qwen-layered-weights-transformer` and `...-base` (the pre-downloaded weights; without them the weights come from Hugging Face, which is slower); an imported notebook does not carry that setting.
3. Run all cells. Loading takes 1 to 7 minutes depending on the profile. The tunnel cell prints a `remote_url` and a `remote_token`.
4. Put both into `prototype/config.json` (it is git-ignored; never commit it):
   ```json
   {"remote_url": "https://....trycloudflare.com", "remote_token": "..."}
   ```
   Do not set `image_backend` to `remote` unless the profile serves the chain routes (only the LayerDiffuse chain does).
5. Start the app, open **Decompose an image (GPU)**, press **Ask the server which methods it offers**, choose the method and a preset (A = best case, B = live-demo budget), upload an image, give names and boxes where the method needs them, press **Decompose**.
6. **Stop the Kaggle session when done.** The tunnel URL changes every session.

### 4.3 Benchmarks, tests, docs

- A benchmark session for one profile (paths A and B on the fox & cottage scene): `make_profile_notebook.py --profile NAME --mode bench`, then push it with `python -m kaggle.cli kernels push -p .` from the notebook folder; the Kaggle CLI setup and rules are in [guides/kaggle.md](guides/kaggle.md). `prototype/tools/wait_and_fetch.sh` waits and downloads the outputs.
- CPU tests: every `prototype/tools/test_*.py`; layer measurements: `prototype/tools/layer_report.py`.
- After a measured run: add a row to `prototype/kaggle/benchmarks.csv`, run `python prototype/tools/build_benchmarks.py`, then `python prototype/tools/check_doc_links.py` before pushing. Add the image chain to [models/provenance.md](models/provenance.md).

## 5. Where things are in the code

| Path | What |
|---|---|
| `prototype/app.py` | the Gradio UI (Scene, Decompose, Settings tabs) |
| `prototype/layerproto/engine.py` | the loop: plan, generate, decompose, edit layer, transform, relocate, fit to box, compose |
| `prototype/layerproto/backends/` | contracts and clients: `chain_base`, `decomposer` (`DecomposeParams`, `RemoteDecomposer`, `MethodRouter`), `remote_chain`, `sam2_decomposer`, `qwen21_decomposer`, `layerdiffuse_chain` |
| `prototype/layerproto/serve/` | one module per Kaggle profile (`qwen_layered`, `qwen21`, `sam2`, `reveallayer`, `cld`, `artplus`), `profiles.py` (registry and the benchmark cases), `benchmark.py` |
| `prototype/layerproto/remote_server.py` | FastAPI: `/health`, `/methods`, `/decompose`, `/chain/*` incl. `/chain/relocate` |
| `prototype/layerproto/{shadow,relight,layer_edit,matte_fix,events,event_apply,fp16_mixed}.py` | shadow layers, IC-Light, alpha retry, halo fix, story events, the fp16 fix for T4 |
| `prototype/tools/` | notebook generators (`make_*`), tests, measurement and comparison scripts |
| `prototype/kaggle/` | generated notebooks (`upload_*`) and every measured result (`results_*`, table in `benchmarks.csv`) |
| `docs/` | state, plan, models, concepts, guides, history ([../README.md](../README.md)) |

## 6. What to build next (suggested order)

1. **Progressive loading of layers (the user's design).** The user should be able to start working on the fast SDXL-Turbo baseline, or on the first Qwen-2.1 layer, while the other layers and objects keep arriving. Qwen-2.1 costs about 24 s per object (118 s for four objects plus background; 139-184 s on the story composites), which is acceptable if layers stream in. Sketch: an async decompose route (job id, `GET /jobs/<id>` returns the layers finished so far) or server-sent events; a Gradio generator function that adds layer cards as they arrive; the background and the biggest object first. The decomposer contract currently returns all layers at once (`Decomposer.decompose`).
2. **SDXL-Turbo route** (`/txt2img`, `/img2img`) on a server profile, and a "first image" step in the app; a prompt style that gives separable layouts (the draft images often contain a second fox).
3. **Qwen-2.1 edit route** (`/edit`: scene edit, per-layer edit with the alpha retry via `layer_edit.py`) and wire `event_apply.py`, the event router and the shadow layers into the app.
4. **A tunnel notebook that serves two profiles** is not possible on two T4s with both Qwen models loaded (the Qwen-2.1 profile alone loads about 21 GiB); decide: swap profiles, a bigger GPU, or the hosted API for demos ([research/hosting.md](research/hosting.md)).
5. The open experiments are in [todo.md](todo.md) (harmonise with a lighting mismatch, IC-Light tuning, Qwen-Image-Layered on more scenes, Stable-Layers at fewer steps, SAM2 + Qwen-2.1 fill, the combined run).

## 7. Known problems and cautions

- **Qwen-Image-Layered is inconsistent**: good on the forest scene, no object layers on the fox & cottage scene at 384, 512 or 640 px; the Stable-Layers LoRA on the same weights split it cleanly at 640 px ([models/qwen-image-layered.md](models/qwen-image-layered.md), [models/stable-layers.md](models/stable-layers.md)).
- **Qwen-2.1 extraction** leaves objects in the background layer on night and rain pictures, can invent a misplaced object (a stone dome instead of the small arch) and ignores boxes for position ([models/qwen-image-2.1.md](models/qwen-image-2.1.md)). Every call re-renders pixels slightly; a whole-scene edit redraws the whole picture (about 31 dB against the input even when told to change nothing).
- **Time:** only SAM2 meets the 30 s live-demo target by itself; Qwen-2.1 is about 24 s per call, Stable-Layers about 100 s at 640 px.
- **Licences:** FLUX.1-dev (RevealLayer, CLD, ART+, LayerDiffuse-Flux) and Qwen-Image-2.1 are non-commercial; Stable-Layers has Stability's community licence. This is a personal non-commercial project; check before any other use.
- **Secrets:** the tunnel token and the Kaggle token (`~/.kaggle/access_token`) never go into the repo; `prototype/config.json` is git-ignored.
- **Kaggle:** the weekly quota is shown in the editor; a session stuck on "Starting" can still be running (stop it from the Active Events panel); notebooks of about 1.1 MB or more are rejected by the API; every trial has a 25-55 minute time cap ([guides/kaggle.md](guides/kaggle.md)).
- **Windows paths:** the scripts assume the folder layout of this repository and the Python environment in `fastsdcpu\env`.
