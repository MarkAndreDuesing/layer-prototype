# Methods without released weights or code

> **Role: current.** Re-checked 2026-11-09 with web searches ([`../../research_notes/Layered methods deep dive/release_status_recheck.md`](../../research_notes/Layered%20methods%20deep%20dive/release_status_recheck.md)). Ideas we borrow are noted; details in [../research/papers-survey.md](../research/papers-survey.md).

| Method | What it does | Release status | What we borrow |
|---|---|---|---|
| PSDiffusion | joint multi-layer generation on SDXL | "code coming soon" | partial joint self-attention (training-free idea) |
| DreamLayer (ICCV 2025) | joint 2–4 layers on SD1.5 with move/resize and shadows | dataset only (`LL3RD/DreamLayer`) | its harmonisation step → our `relocate()` |
| OmniPSD | text/image → PSD (FLUX + Kontext) | code yes, **weights no** | — |
| LASAGNA (Adobe) | text → composite + background + foreground with shadows | not released | effects-with-layers idea |
| LayerFusion | joint fg/bg/blend generation | no code ([layerfusion.md](layerfusion.md)) | attention blending equations |
| LayeringDiff, LayerDecomp, DiffDecompose, Text2Layer, RefLayer, LiWi, PixelART | various decomposition / layered generation | not released | — |
| LayerCraft | LLM agents plan and edit layered images | code + LoRA released (object insertion, not a layer generator) | design reference for the planner and event router |
| Alfie, See-through | single transparent objects / anime PSD | code yes | not tried (Alfie: sprites; See-through: anime only) |
