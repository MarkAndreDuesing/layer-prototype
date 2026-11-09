# Story events across layers

> **Role: concept (stable explanation).** How it works and why. Current numbers live in [../status.md](../status.md) and the factsheets in [../models/](../models/README.md); history in [../log.md](../log.md).

**Problem:** an event such as "it starts to rain", "night falls" or "the campfire goes out" must change several layers coherently while characters keep their identity. No layered method does this (gap G2); literature: [../../research_notes/Layered loop open paths/story_events_global_edits.md](../../research_notes/Layered%20loop%20open%20paths/story_events_global_edits.md).

## Router ([`../../prototype/layerproto/events.py`](../../prototype/layerproto/events.py))
An LLM (Ollama `qwen3:4b`, JSON schema) gets the layer names and the event and returns:
- a **strategy:** `bg_relight` (weather, light, time of day; most events), `per_layer` (several objects change appearance), `recompose` (geometry changes across objects, e.g. snow piling up), `effects` (only an overlay);
- a **background prompt;**
- per layer an **action:** keep / relight / reprompt (with a new prompt) / remove;
- **effects:** rain, snow, fog, night, fire_glow, smoke.
Rules in the prompt: lighting changes apply to every visible layer; grouped layers ("campfire and chair") are re-prompted as a whole, never removed for one part. A keyword heuristic is the fallback when the LLM is unavailable.

## Strategies
| | Strategy | Generative step | Evidence |
|---|---|---|---|
| a | edit composite, then re-split | Qwen-Image-2.1 or SDXL img2img on the composite → Qwen-Image-Layered | fallback for geometry changes |
| b | same instruction per layer | Qwen-Image-2.1 RGBA edit per layer (alpha kept) | edit keeps alpha (tested) |
| c | **edit background, relight foregrounds** | background img2img + IC-Light per layer ([harmonisation.md](harmonisation.md)) | favoured by the literature |
| d | add effect layers | procedural rain/snow; LayerDiffuse fog/fire/glow on the background | — |

## Instant preview (CPU, milliseconds)
`preview(layers, plan)`: colour grading per layer (night, rain, snow, fog, fire glow) + procedural overlays (rain streaks, snowflakes, ground fog, glow, smoke). Shown immediately while the generative pass runs; ImageDirector (UIST '26) found users prefer an immediate proxy over waiting for harmonisation.

## Measured
5 test events routed sensibly at about 30 s each on CPU; previews in [`../../prototype/kaggle/results_local_events/`](../../prototype/kaggle/results_local_events/). Qwen-Image-2.1's scene-wide "rainy night" edit kept the composition but was subtle. GPU generative pass: open ([../todo.md](../todo.md)).
