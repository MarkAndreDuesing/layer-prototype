# Removed snapshots and where their content lives

> **Role: history pointer.** The dated reports, plans and checklists from 2026-11-05 to 11-09 were merged into the current docs and then deleted (2026-11-10). Their useful, current content lives in the files below; the original prose is in git history (e.g. `git show 35dc02a:"docs/archive/<name>.md"`). Research-note drafts superseded by later rounds were deleted the same way.


| Snapshot | Useful content now in |
|---|---|
| 2026-11-05 Image generation models hosting and costs | [../research/hosting.md](../research/hosting.md) §2, §6 (APIs, price-performance, sketch methods, models, test links, browser generators); CPU-only section in [../research/cpu-speedups.md](../research/cpu-speedups.md); Phase 1 results in [../log.md](../log.md) |
| 2026-11-07 Coherent layered generation pipeline | [../concepts/pipeline.md](../concepts/pipeline.md) (routes A–F table, re-prompt recipe) |
| 2026-11-07 Free hostable layered models | [../research/hosting.md](../research/hosting.md) §7 (tool table), factsheets |
| 2026-11-07 Kaggle migration plan | superseded by measurements: [../guides/kaggle.md](../guides/kaggle.md), [../status.md](../status.md) |
| 2026-11-07 Layer based architecture models and hosting | [../research/hosting.md](../research/hosting.md) §7 (ruled-out list), [../concepts/pipeline.md](../concepts/pipeline.md) |
| 2026-11-07 Layered image generation models | [../research/papers-survey.md](../research/papers-survey.md), [../research/hosting.md](../research/hosting.md) §6 (base models) |
| 2026-11-07 Qwen layered multi model pipeline plan | [../concepts/pipeline.md](../concepts/pipeline.md) (roles, data model, build status) |
| 2026-11-07 Session summary sketch-to-image | [../log.md](../log.md) appendix |
| 2026-11-07 Setup and usage guide 3 paths | superseded by [../guides/fastsd-cpu/](../guides/fastsd-cpu/INSTALL.md) (Path 3 dropped) |
| 2026-11-09 Investigation checklist, Open paths | [../todo.md](../todo.md) |
| 2026-11-09 Layered loop open paths, Layered methods - how to get them working | factsheets, [../research/papers-survey.md](../research/papers-survey.md), [../concepts/editing.md](../concepts/editing.md), [../research/research-gap.md](../research/research-gap.md) |
| 2026-11-09 Research block summary | [../status.md](../status.md), [../log.md](../log.md) |
| 2026-11-09 Speed and APIs, University GPU access LMU, TPU route | [../research/hosting.md](../research/hosting.md) §3–§5 |

Not carried over (duplicate or superseded): original prose and reasoning, estimates replaced by measurements, the 3-paths install steps (replaced by the FastSD CPU guides). Also removed: `research_notes/archive/` (superseded drafts) and the HCI note (no user study is planned).
