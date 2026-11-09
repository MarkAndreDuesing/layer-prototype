# Running the prototype app locally

> **Role: guide.** The app's own README has the full steps; this page only points there.

- Start here: [../../prototype/README.md](../../prototype/README.md) (Ollama setup, `run_prototype.bat`, settings, pipeline switches).
- The app runs on the laptop CPU (Gradio, port 7870); heavy steps can move to a Kaggle GPU via `layerproto/remote_server.py`.
- Gradio 6 pitfalls hit so far: keep one `gr.Checkbox` per setting (CheckboxGroup synced in two places loops); avoid Gallery + per-card Sketchpads inside `gr.render` (froze the browser). Test copies on port 7871.
