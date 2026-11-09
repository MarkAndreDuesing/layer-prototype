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

## Next
Run with real editors on the fox & cottage scene ([../todo.md](../todo.md)).
