# Research notes (raw sources)

> **Role: source layer.** Detailed notes with citations from each research round. They are not kept up to date; current conclusions live in `docs/` (status, factsheets, concepts, research). Some notes carry a dated update line at the top.

| Folder | Round | Main contents | Summarised in |
|---|---|---|---|
| `Image generation models hosting and costs/` | 2026-11-05 | price-performance script + output | [../docs/research/hosting.md](../docs/research/hosting.md) |
| `Layered image generation models/` | 2026-11-07 | first survey: image→layers, prompt→layers, modular pipelines | [../docs/research/papers-survey.md](../docs/research/papers-survey.md) |
| `Free hostable layered models/` | 2026-11-07 | hosting, matting, licences, LayerDiffuse and research models, platform terms | [../docs/research/hosting.md](../docs/research/hosting.md) §2, §7 |
| `Coherent layered generation pipeline/` | 2026-11-07 | Qwen-Image-Layered + LayerDiffuse papers, other methods, pipelines audit | [../docs/concepts/pipeline.md](../docs/concepts/pipeline.md), [../docs/research/papers-survey.md](../docs/research/papers-survey.md) |
| `Layered methods deep dive/` | 2026-11-09 | joint text→layers, PSD methods, release re-check, spatial-edit survey, heavy-model plans, FLUX on T4 | [../docs/research/papers-survey.md](../docs/research/papers-survey.md), [../docs/concepts/relocate.md](../docs/concepts/editing.md#1-move--resize-with-re-harmonisation-relocate), factsheets |
| `Layered loop open paths/` | 2026-11-09 | Qwen-Image-2.1 RGBA, FLUX layered on T4 + weights, harmonisation, story events, drift evaluation, HCI (context only) | factsheets, [../docs/concepts/editing.md](../docs/concepts/editing.md), [../docs/research/drift-evaluation.md](../docs/research/drift-evaluation.md) |

`spatial_edit_survey.md` is the first, unverified pass; `spatial_edit_survey_verified_part1–3.md` supersede it.
