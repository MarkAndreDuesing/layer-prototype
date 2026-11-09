# Layer prototype: coherent layered image generation for a storytelling game

**text prompt → coherent layered scene** (layers that share style and placement, like a PSD) **→ edit single layers** (re-prompt, move, resize) **→ recomposite → story events change the scene**

Personal, non-commercial team prototype on free hardware (Kaggle 2× T4, a CPU laptop). This repo holds the code, experiments, research and decisions so far.

## Where to start

| You want… | Read |
|---|---|
| **Where are we now?** | [docs/status.md](docs/status.md) |
| What's left to do / open questions | [docs/todo.md](docs/todo.md) |
| How we got here (history, all runs) | [docs/log.md](docs/log.md) |
| Each model: happy path, benchmarks, best features | [docs/models/](docs/models/README.md) |
| How things work: the loop and routes ([pipeline](docs/concepts/pipeline.md)), move/resize, relighting, story events ([editing](docs/concepts/editing.md)), fp16 on T4 ([fp16-on-t4](docs/concepts/fp16-on-t4.md)) | [docs/concepts/](docs/concepts/) |
| All methods in one table | [docs/research/papers-survey.md](docs/research/papers-survey.md) |
| Research gap, hosting & costs & GPUs, drift evaluation, CPU speed-ups | [docs/research/](docs/research/) |
| Step-by-step guides (Kaggle, FastSD CPU) | [docs/guides/](docs/guides/) |
| Terms (split, recomposite diff, pinning, GGUF…) | [docs/glossary.md](docs/glossary.md) |
| Raw research notes with sources | [research_notes/](research_notes/README.md) |
| Older reports and plans (read-only snapshots) | [docs/archive/](docs/archive/) |

## Quick start
1. Local app: [prototype/README.md](prototype/README.md) (needs Ollama and the LayerDiffuse code from [prototype/vendor/README.md](prototype/vendor/README.md)).
2. Any model on Kaggle (incl. Qwen-Image-Layered): [docs/guides/kaggle.md](docs/guides/kaggle.md).
3. Offline CPU image generation with FastSD CPU (earlier sketch-to-image phase): [docs/guides/fastsd-cpu/INSTALL.md](docs/guides/fastsd-cpu/INSTALL.md), then [USAGE.md](docs/guides/fastsd-cpu/USAGE.md).

## Repository layout
- `prototype/`: the app (`app.py`, `layerproto/`), Kaggle notebook generators (`tools/make_*.py`), notebooks (`kaggle/upload_*`) and results (`kaggle/results_*`, measured runs in `kaggle/benchmarks.csv`).
- `docs/`: current documentation (see the table above).
- `research_notes/`: raw sources. `experiments/`: sketch-to-image tests. `fastsdcpu_local/`: our FastSD CPU patch.

## Two layers of documentation
- **For reading:** `README.md`, `docs/status.md`, `docs/todo.md`, `docs/log.md`, `docs/research/`, `docs/models/`, `docs/concepts/`, `docs/guides/`. Each fact lives in one of these.
- **For reproduction and sources (mostly for AI-assisted follow-up work):** `research_notes/` (cited raw notes) and `docs/archive/` (dated snapshots, unchanged).

## Keeping the docs tidy
- Each fact lives in one place: current state in `docs/status.md`, per-model facts in `docs/models/`, history only in `docs/log.md`; archive files are never edited.
- After a run: add a row to `prototype/kaggle/benchmarks.csv`, then `python prototype/tools/build_benchmarks.py`.
- Before pushing: `python prototype/tools/check_doc_links.py`.
- Files moved in the 2026-11-09 restructure (old → new paths): [docs/archive/MOVED.md](docs/archive/MOVED.md).
