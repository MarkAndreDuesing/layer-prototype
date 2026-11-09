# Research notes (raw sources)

> **Role: source layer.** Detailed notes with citations from each research round. They are not kept up to date; current conclusions live in `docs/` (status, factsheets, concepts, research). Some notes carry a dated update line at the top.

| Folder | Round | Main contents | Summarised in |
|---|---|---|---|
| `Image generation models hosting and costs/` | 2026-11-05 | price-performance script + output | [../docs/research/hosting.md](../docs/research/hosting.md) |
| `Layered image generation models/` | 2026-11-07 | first survey: image→layers, prompt→layers, modular pipelines | [../docs/research/papers-survey.md](../docs/research/papers-survey.md) |
| `Free hostable layered models/` | 2026-11-07 | hosting, matting, licences, LayerDiffuse and research models, platform terms | [../docs/research/hosting.md](../docs/research/hosting.md) §2, §7 |
| `Coherent layered generation pipeline/` | 2026-11-07 | Qwen-Image-Layered + LayerDiffuse papers, other methods, pipelines audit | [../docs/concepts/pipeline.md](../docs/concepts/pipeline.md), [../docs/research/papers-survey.md](../docs/research/papers-survey.md) |
| `Layered methods deep dive/` | 2026-11-09 | methods hub (joint_text_to_layers: all methods, release status, paper facts), PSD / decomposition methods, spatial-edit survey (verified parts 1–3, mechanics) | [../docs/research/papers-survey.md](../docs/research/papers-survey.md), [../docs/concepts/relocate.md](../docs/concepts/editing.md#1-move--resize-with-re-harmonisation-relocate), factsheets |
| `Layered loop open paths/` | 2026-11-09 | Qwen-Image-2.1 RGBA, FLUX layered on T4 + weights, harmonisation, story events, drift evaluation | factsheets, [../docs/concepts/editing.md](../docs/concepts/editing.md), [../docs/research/drift-evaluation.md](../docs/research/drift-evaluation.md) |


## De-duplication (2026-11-10)

Overlapping notes were merged by moving sections **verbatim** (citations kept) into one hub per topic; the source notes keep a pointer.

| Topic | Hub | Merged in |
|---|---|---|
| All layered-generation methods: release status, paper facts, code and weights | `Layered methods deep dive/joint_text_to_layers.md` | `release_status_recheck.md` (whole), `other_layered_methods.md` Q1, `prompt_to_layers_generation.md` multi-layer section, `layerdiffuse_and_research_models.md` §3 |
| Hosting platforms, terms, licences, notebooks | `Free hostable layered models/platform_terms.md` | `hosting_and_matting.md` §A (whole file removed), `open_items_closeout.md` (whole) |
| Matting, segmentation, inpainting | `Layered image generation models/modular_pipelines_and_hci.md` Q6 | `hosting_and_matting.md` §B |

Superseded drafts (first spatial-edit survey, heavy-model plans and re-check, untested FLUX cells, 11-07 image-to-layers note) and the HCI note were deleted on 2026-11-10; they are in git history before commit 35dc02a.
