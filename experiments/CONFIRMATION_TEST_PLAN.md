# Confirmation test plan (written before any images were generated)

Goal: check whether the patterns seen in the user's 35 kept images hold on NEW seeds.

## Fixed for every run
- Scene text, the four style prefixes, 512x512, Guidance 1.0 (img2img paths), CPU.
- Seeds: 11, 42, 99 (none used before).
- Same sketch as before (`inputs/sketch_original.png`, `inputs/sketch_filled.png`).
  A second, different sketch is NOT included; see "Not covered".

## Runs (4 styles each, 3 seeds each)
| Block | Path | Input | Settings | Images |
|---|---|---|---|---|
| A | P2 SDXL-Turbo img2img | filled | Strength 0.7, 0.8, 0.9; 4 steps | 36 |
| B | P2 SDXL-Turbo img2img | raw | Strength 0.8; 4 steps | 12 |
| C | P1 SD-Turbo img2img | filled | Strength 0.7, 0.8; 4 steps | 24 |
| D | P3 neutral SD1.5 + LCM-LoRA + ControlNet | canny-raw | scale 0.7; 6 steps; Guidance 1.0 | 12 |

Total 84 images, roughly 65 minutes. Output: `confirm/<block>/seed<seed>/<config>__<style>.png`.

## Hypotheses (decided in advance)
- H1: P2 at Strength 0.8 is kept more often than at 0.7 or 0.9.
- H2: Filled input gives night scenes with a visible moon; raw input gives pale daytime scenes (descriptive check).
- H3: P2 at 0.8 gives exactly one skeleton in at least 3 of 4 styles on at least 2 of the 3 seeds.
- H4: P1 loses the skeleton more often than P2 at the same Strength.
- H5: P3 images are rarely kept (below about 10%).

## How it is judged
1. The user filters the images by hand, as before. Keep rate per block and per Strength is computed from what remains.
2. Separately, skeleton count and presence are read by eye from the images and reported as a plain table, labelled as an
   unaided reading, not a rating.

## Not covered
- A different sketch or scene (needs a second sketch from the user).
- Seeds beyond three per configuration; small counts mean differences of one or two images are not meaningful.

## Amendment 1 (added after block A had started, before blocks C and D or any text-only image existed)
- Prompts: all blocks use the restructured, style-swappable prompt (style prefix + the optimised scene text
  "one skeleton dancing next to a large campfire with a stone ring, campfire in the center, skeleton on the right side,
  night forest of tall pine trees, full moon and stars"), not the user's original wording. Unchanged since block A began.
- New text-only counterparts (no input image, no ControlNet), same prompts and seeds 11, 42, 99, 4 styles each:
  | Block | Model | Settings | Counterpart of | Images |
  |---|---|---|---|---|
  | E | SDXL-Turbo text-to-image | 4 steps, Guidance 1.0 | A and B | 12 |
  | F | SD-Turbo text-to-image | 4 steps, Guidance 1.0 | C | 12 |
  | G | neutral SD1.5 + LCM-LoRA text-to-image, no ControlNet | 6 steps, Guidance 1.0 | D | 12 |
  Text-only runs have no Strength or input, so blocks A and B share one counterpart (E).
- Total is now 120 images. New hypothesis H6 (decided before generating): text-only outputs follow the sketch layout
  (campfire central, skeleton on the right) less often than the image-guided counterparts at Strength 0.7-0.8.
