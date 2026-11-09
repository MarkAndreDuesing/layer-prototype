# Session summary: sketch-to-image with FastSD CPU (2026-11-05 → 07)

A summary of the first working session, which became Phase 1 of this repo. Numbers and results are in [findings-log.md](findings-log.md) (Phase 1) and [../experiments/](../experiments/). This page records **what was done, in what order, and why each decision was made**.

## Starting point

- **Goal:** prototype a sketch-to-image tool that runs on a CPU-only Windows laptop (Intel Core 7 150U, 15.7 GB RAM).
  - The user draws a rough sketch and describes a scene. The tool redraws it in a requested style.
- **Constraint set by the user:** the style varies per request (pixel art, impressionist, hyperrealistic, or none), so nothing may be tuned for one style.
- **Test scene:** one skeleton dancing by a campfire in a night forest. Sketches are in [../experiments/inputs/](../experiments/inputs/) (`sketch_original.png`, `sketch_filled.png`).

## Timeline

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
| 11-07 | Wrote the setup and usage guide, then split it into two editable docs | First version archived: [archive/SETUP_AND_USAGE_GUIDE_3_paths.md](archive/SETUP_AND_USAGE_GUIDE_3_paths.md) |
| 11-07 | **Dropped Path 3** (slower, fewest kept). Generalised both docs for teammates: upstream install steps, no machine-specific workarounds, no test prompt or sketch, fully offline after setup | [fastsd-cpu/INSTALL.md](fastsd-cpu/INSTALL.md), [fastsd-cpu/USAGE.md](fastsd-cpu/USAGE.md) |
| 11-07 | Tested offline use; found and fixed the cache problem (below). Added Step 0, a Requirements section, fully independent setup per path, and no fixed seed | Both models verified offline, in both modes |

## Decisions and why

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

## Offline facts found along the way

- `use_offline_model: true` in `configs/settings.yaml` persists. The UI checkbox alone resets on restart.
- The `--use_offline_model` command-line flag applies only to CLI mode, not to the web UI (`app.py -w`).
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `GRADIO_ANALYTICS_ENABLED=False` stop all network calls. They are set in [../fastsdcpu_local/start-webui-offline.bat](../fastsdcpu_local/start-webui-offline.bat).
- Verified through the running UI's API with the network blocked by a dead proxy plus those variables. Results: both models, Text to Image and Image to Image, about 25 s per image (SD-Turbo) and about 70 s (SDXL-Turbo). Output: [../experiments/offline_test/](../experiments/offline_test/).
- **Not yet verified:** a run with the network physically disconnected, and the download commands run from an empty folder.

## Working agreements from this session

- Processes are written as numbered steps with links and code snippets.
- No arbitrary "best" picks by Claude. Build neutral, complete outputs (all images, neutral sheets) and let the user choose.
- Results go into sheets and tables, not long write-ups, when a ranking is asked for.

## Open items

- A second, different sketch to check that the findings generalise. Every test used one scene.
- True disconnected-network verification, and a from-scratch run of the download commands.
- The ranking sheet's keep-rate formulas have no cached values. Open it in Excel and save once to compute them.
