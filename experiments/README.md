# Experiment output guide

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

## Other folders

- `inputs/`: the sketches and control images used.
- `p1 p2 p3 p1_filled p2_filled p3_filled r2_p1 r2_p2 r2_p3`: original run outputs and their own sheets.
- `final/`: an earlier hand-picked selection (kept for reference, not used by the files above).
