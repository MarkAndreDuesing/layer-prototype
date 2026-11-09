# Experiment output guide

Phase 1 of the repo: sketch-to-image on a CPU laptop with FastSD CPU. The summary is in [../docs/findings-log.md](../docs/log.md); the session story is in [../docs/session-summary-sketch-to-image.md](../docs/archive/2026-11-07%20Session%20summary%20sketch-to-image.md).

**What is in git:** docs, scripts, logs, manifests, `inputs/`, the comparison sheets (`sheets/`, `confirm_sheets/`), `favourites/`, `offline_test/`, the 3 `final/` sheets and the parked images in this folder. The raw image folders (`all_images/`, `confirm_images/`, `p1…`, `r2_…`, `confirm/`) stay local because of their size (about 180 MB).

## Quick index

| File / folder | What |
|---|---|
| `MODEL_RANKING_SHEET.xlsx` | Final ranking of models, settings, styles and seeds (all 3 paths, before Path 3 was dropped) |
| `CONFIRMATION_TEST_PLAN.md` / `CONFIRMATION_RESULTS.md` | Pre-registered 120-image test and its results |
| `confirm_sheets/` | One sheet per path and seed for the confirmation test. Black cells were removed by the user. |
| `favourites/` | The user's 8 named favourites from the confirmation test, all SDXL-Turbo. The first 5 come closest to the original pixel-art intent; the other 3 are favourites in other styles. |
| `*.png` in this folder | Images the user parked as "conceptually interesting, but not what was asked" |
| `offline_test/` | Proof images from the offline test (both models, Text to Image and Image to Image) |
| `run_*.py`, `compile_*.py`, `build_sheets_filtered.py` | Scripts that generated the images and sheets, through the FastSD CPU web UI API |

Every generated PNG is in `all_images/` (188 files, no selection made).
`all_images_manifest.csv` has one row per image with all settings and the original source path.
`sheets/` has one comparison sheet per path, input image and seed (14 sheets).

## File names in `all_images/`

`<path>__ph<phase>__<input>__<settings>__seed<seed>__<style>.png`

| Part | Meaning |
|---|---|
| path | P1 = SD-Turbo img2img, P2 = SDXL-Turbo img2img, P3 = SD 1.5 + LCM-LoRA + scribble ControlNet |
| ph1 | batch1: plain sketch (`raw`) and blurred sketch (`blur`), one seed (123123) |
| ph2 | filled1: filled-in sketch, one seed (123123) |
| ph3 | round2: filled sketch (P1, P2) or neutral-base ControlNet (P3), several seeds |
| ph4 | round2-extra: P2 with 5 steps, P3 with 6 steps |
| input | `raw` plain sketch, `blur` blurred plain sketch, `filled` filled-in sketch; P3 uses the control image: `canny-raw`, `scribble-raw`, `canny-filled` |
| settings | `strNN` img2img Strength (NN/100), `scaleNN` ControlNet conditioning scale, `stepsN`, `gX` Guidance Scale (`neg` = negative prompt used), `dreamshaper8` or `neutral` = P3 base model |
| style | `pixelart`, `impressionist`, `hyperreal`, `nostyle` (style words put in front of the same scene text) |

Scene text (same in every run): "one skeleton dancing next to a large campfire with a stone ring,
campfire in the center, skeleton on the right side, night forest of tall pine trees, full moon and stars".
Style prefixes: pixel art, retro 16-bit game scene / impressionist oil painting, visible brushstrokes, soft light /
hyperrealistic photograph, ultra detailed, cinematic lighting / none.
All runs: 512x512, CPU, Guidance 1.0 except P3 `g1.5neg` (negative prompt: blurry, low quality, multiple skeletons,
extra people, text, watermark).

## Sheets in `sheets/`

Names: `<path>_input-<input>_seed<seed>.png`. Columns: input image, then the 4 styles.
Rows: every run for that path, input and seed, in run order (phase, then Strength/scale, steps, guidance).
Each row label shows phase, configuration name, settings and source folder.

## Confirmation test file names (`confirm_images/`, `favourites/`)

`blk<block>__<path>__<input>__<settings>__seed<seed>__<style>.png`. Blocks: A = P2 filled (Strength 0.7/0.8/0.9), B = P2 raw sketch, C = P1 filled, D = P3 Canny ControlNet, E = P2 text-only, F = P1 text-only, G = P3 text-only. Seeds 11, 42, 99. See `CONFIRMATION_TEST_PLAN.md`.

## Other folders

- `inputs/`: the sketches and control images used.
- `p1 p2 p3 p1_filled p2_filled p3_filled r2_p1 r2_p2 r2_p3`: original run outputs and their own sheets.
- `final/`: an earlier hand-picked selection (kept for reference, not used by the files above).
