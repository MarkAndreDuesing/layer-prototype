# Open paths and ideas still to research

These come from a team voice note (2026-11-08) on the possible routes to the prototype goal:

**prompt → coherent layered image → edit single layers → recomposite**

This page lists only what is **new, unconfirmed or still open**, updated to the state of 2026-11-09. Settled results are in [findings-log.md](findings-log.md).

## 1. Where do strong text-to-image models fit?

- **Idea:** the strong regular text-to-image models (the shortlist from the team's own testing) and small local models (about 30 s per image locally) can't split images into layers. They still belong in the pipeline, just not at every stage.
- **Most likely role:** generating the **base image**, which a layering step (Qwen-Image-Layered, or segmentation) then splits. Comfy Cloud's reference workflow does exactly this: Krea2-Turbo text-to-image → Qwen-Image-Layered (6 layers) → recomposite.
- **Open:**
  - Which of the shortlisted models give the best input for decomposition (clear, separable objects)?
  - Which of them run on Kaggle, and under which licences?

## 2. Image-to-image as the front end

- **Idea:** a small local image-to-image model (a sketch or image plus the prompt, giving one image) works well and fast. Its output could feed any layering route.
- **Known:** SDXL-Turbo img2img gives about 58 s per image on the laptop CPU (see [findings-log.md](findings-log.md), Phase 1).
- **Open:** the combination img2img → Qwen-Image-Layered or img2img → segmentation is **untested**. How it fits the LayerDiffuse route, which builds its own layers, is unclear.
- **Planned test:** feed generated base images (SDXL-Turbo locally, Qwen-Image-2.1 on Kaggle) into the Qwen-Image-Layered batch, using the shared fox & cottage test scene.

## 3. LayerDiffuse for complex scenes

- **Observation:** in the team's own tests, LayerDiffuse handles one central figure plus a background. In complex scenes the layers just stack up and need repositioning by hand.
- **Since then:** the prototype's **scene-to-layer chain** (`bg2ble → bgble2fg`) adds each element on top of the existing scene, in place. It works on SDXL base and on a Kaggle T4 (about 18 s per pass at 1024 px).
- **Open:**
  - Quality with many objects (5 or more) and overlapping objects is not evaluated.
  - The paper reports better results than the team saw; the gap may be the settings.

## 4. Continuous map development (story events)

- **Idea:** the scene changes as the story goes on, e.g. "the forest is now on fire". The change becomes a new layer stacked on the existing scene.
- **Since then:** a story-event step through the LayerDiffuse chain works on a Kaggle T4.
- **Since 2026-11-09 evening:** a story-event router ([events.py](../prototype/layerproto/events.py)) maps an event to per-layer actions (keep / relight / re-prompt / remove) plus effect overlays, with an instant CPU preview (procedural rain, snow, fog, smoke, glow). The research found no layered method that does scene-wide events (gap G2); four strategies are compared in [Layered loop open paths.md](../reports/Layered%20loop%20open%20paths.md).
- **Open:**
  - The generative part of scene-wide events on a GPU: background edit + IC-Light relight of each layer, LayerDiffuse effect layers, or edit-and-re-split.
  - Whether the same thing works on top of Qwen-Image-Layered output: re-generate a changed layer through the composite.

## 5. Prompt breakdown with AI-decided placement and size

- **Idea:** split the prompt by its words, sentiment and the relations between objects, give each noun its own prompt and generation, then stack the results. A further AI step decides each object's position and size from the context.
- **Status:** this was the first prototype version; the LLM already guesses boxes.
- **Known problem:** objects generated separately don't share lighting or style ("sticker collage", see [Coherent layered generation pipeline.md](../reports/Coherent%20layered%20generation%20pipeline.md)).
- **Open:** whether a strong model plus a harmonising pass can fix the coherence. One candidate is IC-Light relighting (Apache-2.0); see [Free hostable layered models.md](../reports/Free%20hostable%20layered%20models.md).
- **Since then:** an IC-Light relight module ([relight.py](../prototype/layerproto/relight.py)) and a procedural shadow layer ([shadow.py](../prototype/layerproto/shadow.py)) work on CPU; `relocate()` re-harmonises moved layers in about 4 s on a T4. A GPU test of all three together is pending.

## 6. Generate → segment → inpaint (first attempt done, more to try)

- **Idea:** generate one normal image, then cut out the objects with a segmentation model, one layer each. Inpaint the hole each object leaves in the layer below.
- **Candidates:**
  - **Matting** (clean cut-outs): BiRefNet (MIT) and BEN2 base (MIT), already researched.
  - **Segmentation:** a model like SAM (Segment Anything) to find the objects. The name from the voice note is unclear and still to be confirmed.
  - **Inpainting:** SDXL inpainting, or the chain's own backgrounds.
- **First attempt (2026-11-09):** SAM2 automatic masks + LaMa on a Kaggle T4: 3–7 s per image, near-exact recomposite, but "largest K masks" picked fragments instead of story objects, and hidden parts stay missing. Details in [findings-log.md](findings-log.md), Phase 5.
- **Open / next attempts:**
  - Segment by noun (points or boxes from the LLM breakdown, Grounded-SAM style) instead of "largest K".
  - LaMa vs SDXL inpainting for the background; amodal completion (pix2gestalt) for occluded objects.
  - Quality on the shared fox & cottage test scene.

## 7. Qwen-Image-Layered: speed and budget

- **Quality:** best of all routes. On Comfy Cloud it decomposed scenes into sensible layers.
- **Kaggle:** the bf16 problem is **solved** with [fp16_mixed.py](../prototype/layerproto/fp16_mixed.py). The default settings take 35 min, but **384 px, 4 layers, 4 steps, CFG 1 takes 27–35 s** on the free 2× T4 (sweeps v3–v7).
- **Open:**
  - Quality at the fast settings versus the defaults on more scenes (Stable-Layers with Heun, 2-layer splits: v9 sweep ready).
  - The Kaggle TPU route: estimated about 40 s ([research_tpu_route.md](../reports/research_tpu_route.md)); the effort estimate was a guess. A probe notebook exists, but CLI batch runs did not get a TPU, so it has to run as an interactive TPU session.
- **API budget:**
  - At about $0.05 per run (fal), €100 for the team is roughly 2,000 runs, before any other testing.
  - That suits final demos better than early prototyping ([research_speed_and_apis.md](../reports/research_speed_and_apis.md)).

## 8. Hosting

- **Kaggle:** free and reasonable on quota and licences, but not every model runs there.
- **Comfy Cloud:** needs a subscription; plans start at $20/month.
- **Paid APIs:** always possible, but too expensive for early prototyping.
- **Open:** which models from sections 1 and 6 run on Kaggle's T4s or TPU.
