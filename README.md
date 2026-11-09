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
| How things work (pipeline, LayerDiffuse modes, move/resize, fp16 on T4, story events, harmonisation) | [docs/concepts/](docs/concepts/) |
| Research: papers, research gap, drift evaluation, hosting & costs | [docs/research/](docs/research/) |
| Step-by-step guides | [docs/guides/](docs/guides/) |
| Terms (split, recomposite diff, pinning, GGUF…) | [docs/glossary.md](docs/glossary.md) |
| Raw research notes with sources | [research_notes/](research_notes/README.md) |
| Older reports and plans (read-only snapshots) | [docs/archive/](docs/archive/) |

## Quick start
1. Local app: [prototype/README.md](prototype/README.md) (needs Ollama and the LayerDiffuse code from [prototype/vendor/README.md](prototype/vendor/README.md)); summary in [docs/guides/prototype-local.md](docs/guides/prototype-local.md).
2. Qwen-Image-Layered on Kaggle: [docs/guides/qwen-on-kaggle.md](docs/guides/qwen-on-kaggle.md); any other trial: [docs/guides/kaggle-cli.md](docs/guides/kaggle-cli.md).
3. Offline CPU image generation with FastSD CPU (earlier sketch-to-image phase): [docs/guides/fastsd-cpu/INSTALL.md](docs/guides/fastsd-cpu/INSTALL.md), then [USAGE.md](docs/guides/fastsd-cpu/USAGE.md).

## Repository layout
- `prototype/`: the app (`app.py`, `layerproto/`), Kaggle notebook generators (`tools/make_*.py`), notebooks (`kaggle/upload_*`) and results (`kaggle/results_*`, measured runs in `kaggle/benchmarks.csv`).
- `docs/`: current documentation (see the table above).
- `research_notes/`: raw sources. `experiments/`: sketch-to-image tests. `fastsdcpu_local/`: our FastSD CPU patch.

## Keeping the docs tidy
- Each fact lives in one place: current state in `docs/status.md`, per-model facts in `docs/models/`, history only in `docs/log.md`; archive files are never edited.
- After a run: add a row to `prototype/kaggle/benchmarks.csv`, then `python prototype/tools/build_benchmarks.py`.
- Before pushing: `python prototype/tools/check_doc_links.py`.
- Files moved in the 2026-11-09 restructure (old → new paths): [docs/archive/MOVED.md](docs/archive/MOVED.md).
