# Drift over long edit sequences (layered vs flat)

> **Role: research synthesis (current).** Summarises the sources; details and citations stay in the linked snapshots and research notes.

**Question (RQ1, gap G1):** do persistent RGBA layer stacks keep a scene stable over many story edits better than flat-image editing? No benchmark tests a layered editor, and nobody has measured the error of repeated decompose → recompose cycles. Full notes: [../../research_notes/Layered loop open paths/multi_turn_drift_evaluation.md](../../research_notes/Layered%20loop%20open%20paths/multi_turn_drift_evaluation.md).

## What is known for flat editors
- **EdiVal-Agent (ICLR 2026, 3 turns):** consistency drops about 3–4 points per turn for every editor (FLUX.1-Kontext-dev 95.3 → 90.2; Qwen-Image-Edit 84.2 → 78.0); flow editors get brighter each turn; VLM judges miss gradual quality loss.
- **MTEdit-Bench (8 turns):** at turn 8 some open editors fall to near zero consistency.
- **TailSteer (10 turns, edit inside a box, measure outside):** outside-box PSNR 11–19 dB after 10 turns.
- **ImageDirector (UIST '26):** compositing instead of regenerating kept background SSIM at 0.998 vs 0.922 over 3 edits, but lighting and cohesion were not better.

## Our protocol
- **Arms:** (A) layered: edit only the target layer, recomposite; (B) flat: an editor chaining its own outputs; (C) layered but re-decomposed every turn (new: isolates decompose/recompose error).
- **Metrics per turn vs turn 0:** PSNR / SSIM (optionally LPIPS, DINO) outside the union of all edited regions; per-layer alpha IoU and alpha-weighted L1; mean luminance shift; edit success; area under the curve and the first turn below 25 dB.
- **Scale:** about 20 scenes × 10 turns (5 scenes × 20); EdiVal-Agent's harness can host our editors (`generate_single_edit`).

## Code
Harness: [`../../prototype/layerproto/drift_eval.py`](../../prototype/layerproto/drift_eval.py); self-test with stand-in editors: [`../../prototype/tools/drift_harness_selftest.py`](../../prototype/tools/drift_harness_selftest.py) (tests the logic, not real editors). Soft alpha edges mean even the layered arm is not exactly unchanged.

## First result with real editors (fox & cottage, 6 successive fox moves, 2026-11-10)

Task: move the red fox six times (left, right, far right, back left, back home), the same ground-truth edits in both arms. Arm A (layered): `relocate()` on RevealLayer's layers, every other layer untouched. Arm B (flat): Qwen-Image-2.1 edits the whole composite each turn from the previous output ("move the red fox to …, keep everything else the same"). Metrics: PSNR / SSIM against turn 0 **outside** the union of old and new fox areas (`drift_eval.py`).

| Turn | A layered: PSNR out / SSIM out | B flat: PSNR out / SSIM out | B luminance shift |
|---|---|---|---|
| 1 | 59.2 dB / 0.992 | 19.6 dB / 0.569 | −2.5 |
| 2 | 60.9 / 0.995 | 18.6 / 0.517 | −5.1 |
| 3 | 60.8 / 0.995 | 17.1 / 0.425 | −6.6 |
| 4 | 60.8 / 0.995 | 15.9 / 0.351 | −7.3 |
| 5 | 60.5 / 0.996 | 15.0 / 0.295 | −9.5 |
| 6 | 61.9 / 0.997 | 14.3 / 0.252 | −11.6 |

Top row = arm A, bottom row = arm B (turn 0 … 6):

![drift strips](../img/drift_fox_strip.jpg)

- **Flat editor:** falls below 25 dB at turn 1, darkens by 11.6 grey levels over 6 turns, loses fine texture, and from turn 5 draws a second fox (the edit "move" turns into "add"). Cost about 23 s per turn.
- **Layered arm:** the background is exact by construction (about 60 dB; the remainder is soft alpha edges), which is the expected strength of layers and not a surprise. The moved object itself is not free: against turn 0 its colour L1 grows 0.22 → 0.30 over six moves (crop-normalised), and its size does not return exactly (back at its home position it covers 84 % of the original area, and the first move came out 1.7× the area instead of the intended 1.3×).
- **Not yet run:** arm C (re-decompose every turn), more scenes and edit types (re-prompt, relight), LPIPS / DINO. One scene, one seed: a demonstration of the harness with real editors, not a benchmark.

Files: `prototype/tools/fox_drift_demo.py` (arm A), `make_qwen21_drift_trial.py` (arm B), `drift_analyse.py`, results in `prototype/kaggle/results_drift*/`.

## Next
Arm C and more scenes ([../todo.md](../todo.md)).
