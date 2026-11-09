# LayerFusion (idea only)
> **Role: current factsheet.** Template: What it does · Licence & weights · How we run it (happy path) · Benchmarks · Best features · Weak spots · Status · Next attempts · Sources. Verdicts are the user's.

## What it does

Joint foreground + background + blend generation on SDXL + LayerDiffuse, sharing attention between branches (training-free).

## Licence & weights

No code released; method is training-free on our stack.

## How we run it (happy path)

Re-implementation needed: masks from the fg branch's self-attention and `<EOS>` cross-attention at t = 0.8T; Eq. 1 `a'_Blended = a_FG·m_soft + a_Blended·(1 − m_soft)`, Eq. 2 `a'_FG = a'_Blended·m_hard + a_FG·(1 − m_hard)`. Its paper has **no** move/resize editing.

## Benchmarks

—

## Best features

Harmonised joint generation of a foreground with its background, on our own stack.

## Weak spots

Not implemented; one foreground in the paper (N foregrounds would be our extension).

## Status

Not started.

## Next attempts

Prototype on SDXL + LayerDiffuse if joint generation becomes a priority.

## Sources

[`../../research_notes/Layered methods deep dive/spatial_edit_mechanics.md`](../../research_notes/Layered%20methods%20deep%20dive/spatial_edit_mechanics.md).
