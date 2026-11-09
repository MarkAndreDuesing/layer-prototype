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

## 3. LayerDiffuse for complex scenes

- **Observation:** in the team's own tests, LayerDiffuse handles one central figure plus a background. In complex scenes the layers just stack up and need repositioning by hand.
- **Since then:** the prototype's **scene-to-layer chain** (`bg2ble → bgble2fg`) adds each element on top of the existing scene, in place. It works on SDXL base and on a Kaggle T4 (about 18 s per pass at 1024 px).
- **Open:**
  - Quality with many objects (5 or more) and overlapping objects is not evaluated.
  - The paper reports better results than the team saw; the gap may be the settings.

## 4. Continuous map development (story events)

- **Idea:** the scene changes as the story goes on, e.g. "the forest is now on fire". The change becomes a new layer stacked on the existing scene.
- **Since then:** a story-event step through the LayerDiffuse chain works on a Kaggle T4.
- **Open:**
  - Effects that cover the whole scene (fire, rain, night) rather than single objects.
  - Whether the same thing works on top of Qwen-Image-Layered output: re-generate a changed layer through the composite.

## 5. Prompt breakdown with AI-decided placement and size

- **Idea:** split the prompt by its words, sentiment and the relations between objects, give each noun its own prompt and generation, then stack the results. A further AI step decides each object's position and size from the context.
- **Status:** this was the first prototype version; the LLM already guesses boxes.
- **Known problem:** objects generated separately don't share lighting or style ("sticker collage", see [Coherent layered generation pipeline.md](../reports/Coherent%20layered%20generation%20pipeline.md)).
- **Open:** whether a strong model plus a harmonising pass can fix the coherence. One candidate is IC-Light relighting (Apache-2.0); see [Free hostable layered models.md](../reports/Free%20hostable%20layered%20models.md).

## 6. Generate → segment → inpaint (untested)

- **Idea:** generate one normal image, then cut out the objects with a segmentation model, one layer each. Inpaint the hole each object leaves in the layer below.
- **Candidates:**
  - **Matting** (clean cut-outs): BiRefNet (MIT) and BEN2 base (MIT), already researched.
  - **Segmentation:** a model like SAM (Segment Anything) to find the objects. The name from the voice note is unclear and still to be confirmed.
  - **Inpainting:** SDXL inpainting, or the chain's own backgrounds.
- **Open:**
  - Speed of the three passes on Kaggle T4 or the laptop.
  - Quality of the inpainted background.
  - How objects get grouped into layers.

## 7. Qwen-Image-Layered: speed and budget

- **Quality:** best of all routes. On Comfy Cloud it decomposed scenes into sensible layers.
- **Kaggle:** the bf16 problem is **solved** with [fp16_mixed.py](../prototype/layerproto/fp16_mixed.py), but a run takes 35 min at the default settings.
- **Open:**
  - The speed sweep (512 / 384 px, fewer steps, no CFG, 3 layers) to see whether ≤ 40 s is reachable on a T4 and what quality is lost.
  - The Kaggle TPU v5e-8 port: an estimate of about 40 s, 2–5 days of work, unproven ([research_tpu_route.md](../reports/research_tpu_route.md)).
- **API budget:**
  - At about $0.05 per run (fal), €100 for the team is roughly 2,000 runs, before any other testing.
  - That suits final demos better than early prototyping ([research_speed_and_apis.md](../reports/research_speed_and_apis.md)).

## 8. Hosting

- **Kaggle:** free and reasonable on quota and licences, but not every model runs there.
- **Comfy Cloud:** needs a subscription; plans start at $20/month.
- **Paid APIs:** always possible, but too expensive for early prototyping.
- **Open:** which models from sections 1 and 6 run on Kaggle's T4s or TPU.
