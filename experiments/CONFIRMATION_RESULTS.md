# Confirmation test results (120 images, seeds 11/42/99)

Sources: your final filtering of `confirm_images` (27 kept), the 21 `blk*` images you parked in `experiments/`
(conceptually interesting, not what was asked), and my unaided reading of all 120 images (marked "my reading").

## 1. Where every image ended up
| Block | Setup | Kept | Parked | Removed | Total |
|---|---|---|---|---|---|
| A | P2 SDXL-Turbo img2img, filled, 0.7 / 0.8 / 0.9 | 7 (0.7: 2, 0.8: 0, 0.9: 5) | 0 | 29 | 36 |
| B | P2 SDXL-Turbo img2img, raw sketch, 0.8 | 5 | 2 | 5 | 12 |
| C | P1 SD-Turbo img2img, filled, 0.7 / 0.8 | 4 (0.7: 1, 0.8: 3) | 1 | 19 | 24 |
| D | P3 SD1.5 + LCM-LoRA + Canny ControlNet | 2 | 2 | 8 | 12 |
| E | P2 SDXL-Turbo text-only | 6 | 2 | 4 | 12 |
| F | P1 SD-Turbo text-only | 3 | 3 | 6 | 12 |
| G | P3 LCM-LoRA text-only | 0 | 11 | 1 | 12 |

Keep rate by style: hyperreal 0/30, impressionist 8/30, pixelart 8/30, nostyle 11/30.
Keep rate by seed: seed 11 7/40, seed 42 14/40, seed 99 6/40.

## 2. Important data fact
Strength 0.8 and 0.9 at 4 steps give pixel-identical images (12 of 12 pairs checked). Both run 3 real steps.
So "kept at 0.8: 0" vs "kept at 0.9: 5" is the same set of images, and only the 0.9 file names were kept.
Treat them as ONE condition: 5 of 12 kept (0.7: 2 of 12).

## 3. Hypotheses (pre-registered in CONFIRMATION_TEST_PLAN.md)
| H | Result |
|---|---|
| H1 P2 at 0.8 kept more than 0.7 or 0.9 | Not testable as written (0.8 = 0.9). Combined 0.8/0.9: 5/12 vs 0.7: 2/12, small counts. |
| H2 filled gives night + moon, raw gives pale daytime | Held (my reading): all 36 A, 24 C, 12 D are night with a moon; all 12 B have a pale off-white background with the moon as a circle or arc. |
| H3 P2 0.8 gives exactly one skeleton in 3 of 4 styles on 2 of 3 seeds | Held on all 3 seeds (my reading). Seed 99 impressionist is a small reclined skeleton, ambiguous. |
| H4 P1 loses the skeleton more than P2 at same strength | Presence: not supported (skeleton visible in nearly all). Clarity: supported. At 0.7 the campfire turns into a red blob in 8 of 12 images and seed 99 adds a second person. At 0.8 the skeleton is small or ghostlike and sometimes left of the fire. |
| H5 P3 kept below about 10% | Block D alone 2/12 = 17% (not met). D and G together 2/24 = 8%. |
| H6 text-only follows layout less than image-guided | Supported (my reading) for E: skeleton and moon positions vary, and fire is not always central. F and G lose the skeleton more often (see below). |

## 4. Unaided reading (not a rating)
| Block | Skeleton visible | Exactly one | Campfire central, skeleton right |
|---|---|---|---|
| A 0.7 | 12/12 | 12/12 | 12/12 |
| A 0.8 = 0.9 | 12/12 | 12/12 | 12/12 |
| B raw 0.8 | 12/12 | 11/12 (seed 99 pixelart has two) | mostly |
| C 0.7 | 12/12 | 8/12 (seed 99: extra figure) | fire altered in 8/12 |
| C 0.8 | about 11/12, small or faint | about 11/12 | skeleton left in seed 99 |
| D | 12/12 | 12/12 | 12/12 (trees have stick legs, odd chairs) |
| E | 11/12 | 11/12 | varies |
| F | about 9/12, mostly outlines or sitting | n/a | varies |
| G | 3/12 | n/a | no |

## 5. Observations that may matter for the prototype
1. Text-only SDXL-Turbo (E) has the highest keep rate (50%), tied with the 0.8/0.9 filled and raw conditions (about 42%), but it does not follow the sketch layout.
2. Style response differs by input: with the filled sketch (A) hyperreal and nostyle look near-identical and impressionist is modest, while text-only (E) shows clearly different styles (painterly vs photographic). The sketch palette seems to constrain style.
3. Raw sketch (B) gives the strongest impressionist results (3/3 kept), but daytime/pale rather than night.
4. Hyperrealistic style was never kept (0/30), including text-only E images that look photographic. This is your selection, not a judgement by me.
5. Seed 42 accounts for 14 of 27 kept images and 5 of your 8 named favourites.
6. G (no sketch, SD1.5) gives pleasant forest scenes but mostly without a skeleton, which fits your 11 of 12 parked.
7. Your 8 named favourites: A (3), B (2), E (3). None come from P1 or P3.

## 6. Limits
Three seeds, one sketch, one scene. Differences of one or two images are not meaningful.
