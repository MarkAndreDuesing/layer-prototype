# Layer scene prototype (draft 1)

Describe a scene, then:
1. The local LLM refines the prompt and breaks it down into words.
2. Each layer is generated separately: transparent objects plus an opaque background.
3. You can update any single layer.
4. Each layer gets an image-to-text caption.

Layers are not merged back into one image in this draft.

| Part | Draft 1 (local) | Later |
|---|---|---|
| Prompt refiner + word breakdown | Ollama `qwen3:4b` (JSON schema forced) | `"llm_backend": "openai"` → OpenRouter, or a vLLM / llama.cpp server on Kaggle |
| Layer images | SDXL-Turbo + LayerDiffuse SDXL LoRA, CPU float32, 512 px, 4 steps | `"image_backend": "remote"` → `layerproto/remote_server.py` on a Kaggle / Modal GPU |
| Image-to-text per layer | Ollama `qwen2.5vl:3b` (or `moondream`, faster but garbled) | any vision model |

Background reading: [../reports/Layer based architecture models and hosting.md](../reports/Layer%20based%20architecture%20models%20and%20hosting.md)

## Run it

1. Close memory-heavy programs. SDXL needs about 11 GB of free RAM: check Task Manager → Performance → Memory → Available.
2. Make sure Ollama is running. It sits in the tray after install. To check:
   ```bash
   curl http://127.0.0.1:11434/api/version
   ```
3. Start the UI from the `StabeDiffusionCPU` folder by double-clicking `prototype\run_prototype.bat`, or run:
   ```bash
   fastsdcpu\env\Scripts\python.exe prototype\app.py
   ```
4. Open http://127.0.0.1:7870.
5. Type a scene. Under **How to split**, choose a mode:
   - **One layer per object:** one layer per noun, up to the **Layers** number.
   - **Depth bands:** background plus up to 3 bands (midground, foreground and front foreground). Each band can hold several objects.
   - **Exact number of layers:** exactly **Layers**. The LLM groups or splits objects to fit.
6. Set **Layers (incl. background)** from 2 to 8, then click **Refine + split**. This takes about 2–4 minutes on CPU; longer descriptions take longer.
7. To change the split without retyping, adjust the mode or number and click **Re-split with these settings**. Your edited refined prompt and style are kept.
8. Edit the **Layers** table:
   - Untick **generate?** to skip a layer.
   - Click **➕ Add layer**, then fill in its name and prompt.
   - Tick layers in **Select layers for Delete / Merge**, then click **🗑 Delete selected** or **🔗 Merge selected**. Merging combines the names, prompts and boxes. The background can't be merged.
9. Set **Variants per layer**. The **Estimate** line shows the expected time.
10. Click **Generate all layers**. If some layers are missing, for example after adding a row or after a crash, click **Generate missing only**.
11. In **5. Fix one layer**, each layer has its own card:
    1. Click a variant to select it. The caption, % transparent and words found or not found update.
    2. Edit the **Layer prompt**.
    3. Choose **How**:
       - `new`: fresh seeds.
       - `refine`: img2img on the selected variant.
       - `sketch`: img2img from the shared **Sketch pad** above the cards, where white counts as transparent.
    4. Set **Strength** and **Variants**.
    5. Click **Update '<layer>' only**. No other layer changes.

12. **4. Final image** stacks all layers into one picture, saved as `final.png` in the scene folder:
    - The background is always the bottom layer. Each object layer's selected variant goes on top, in layer order.
    - **Layer order** lists the layers top to bottom, like a Photoshop layers panel. Pick a layer in **Layer to move** and click **⬆ Move up** or **⬇ Move down**.
    - With **Place objects in their boxes** ticked, each object is cropped to its visible pixels and fitted into its `box x,y,w,h` from the table: centred, standing on the box's bottom edge. Untick it to stack every layer full-frame.
    - The image rebuilds automatically after generating, loading, updating a layer or selecting a variant. After editing boxes in the table, click **Recompose**.
    - An object with a failed cut-out (the ⚠️ on its card) covers everything below it, so pick or regenerate a variant with a clean cut-out.

Measured on this laptop (2026-11-07, 512 px, 4 steps):
- **Model load:** about 80 s, first time only.
- **Each layer:** about 80 s.
- **Refining one layer:** about 66–75 s.
- **Caption:** about 55 s with `qwen2.5vl:3b`, or about 14 s with `moondream`, which is often garbled.
- **Total:** a 3-layer scene with 1 variant each takes about 7 minutes, captions included.

Object layers come out transparent with "a single …, solo, centered" added to the prompt, which the engine does automatically. A ⚠️ next to **% transparent** means the cut-out failed, so try another seed.

If the app ever closes without an error, send `prototype/crash.log`.

Scenes are saved in `prototype/scenes/<id>/`: one `scene.json` (prompts, breakdown, seeds and captions) plus one RGBA PNG per variant.

## Pipeline steps (switch on/off)
Each step can be switched off to save time:
- **Quick toggles:** on the **Scene** tab, open **Pipeline steps**.
- **All settings:** open the **Settings** tab.
- **Apply** takes effect immediately; **Save as default** also writes `prototype/config.json`.

Switching a step off also hides its part of the Scene page, and the **Pipeline:** strip greys it out:

| Unticked step | Hidden on the Scene page |
|---|---|
| Prompt refinement | "Refined prompt" box |
| Layer split | "How to split" choice, layer-count slider and "Re-split" button |
| Word & relation breakdown | Word breakdown and relations section |
| Multiple variants | Variants sliders (scene and per layer) |
| Transparent objects | "% transparent" info on layer cards |
| Shared style | "Shared style" box |
| Image-to-text captions | Caption lines on layer cards |
| Word check | Found / not-found word lines |
| Object prompt wrapper, Keep models loaded | Nothing; these only change behaviour |

| Step | When off | Time saved (this laptop) |
|---|---|---|
| Prompt refinement (LLM) | Your text is used as is; the style is detected from keywords | Part of the LLM call |
| Layer split (LLM) | Split on commas: scenery words go to the background, every other chunk becomes an object layer | Part of the LLM call |
| Word & relation breakdown (LLM) | No word or relation tables | ~140 s → ~50 s for the LLM call |
| All three LLM steps off | No LLM call at all | ~1–4 min → 0 s |
| Multiple variants per layer | Always 1 variant | ~2.3 min per extra variant |
| Transparent objects (LayerDiffuse) | Objects come out as opaque images | Small |
| Object prompt wrapper | Object prompts are used without "a single …, solo, centered" | None (quality only) |
| Shared style on every layer | The style isn't appended | None (quality only) |
| Image-to-text captions | No captions | ~55 s per image |
| Word check against captions | No found / not-found words | None |
| Keep models loaded (off by default) | Turning this on keeps the LLM and caption model in RAM | Saves reload time, costs ~3–5 GB RAM |

The **Settings** tab also sets:
- the LLM model and backend (Ollama, or an OpenAI-compatible URL)
- the caption model
- the image backend (local, or a remote URL)
- alpha edge quality
- diffusion steps
- image size
- default variants
- the object prompt prefix and suffix

## Settings
Create `prototype/config.json` and include only the keys you want to override. All keys are in `layerproto/config.py`. For example:
```json
{"variants_per_layer": 1, "alpha_quality": "best", "llm_model": "qwen3:4b", "captioner": "none"}
```

## Moving the heavy parts off this laptop later
1. On a Kaggle notebook (T4, Internet on), copy the `prototype` folder in and start the server:
   ```bash
   pip install fastapi uvicorn diffusers transformers accelerate peft einops opencv-python-headless
   ```
   ```bash
   python -m layerproto.remote_server --port 8765 --token SECRET
   ```
2. Expose the port with a tunnel, such as cloudflared or Pinggy.
3. Locally, point `config.json` at the tunnel:
   ```json
   {"image_backend": "remote", "remote_url": "https://<tunnel>", "remote_token": "SECRET"}
   ```

Colab's free tier forbids serving endpoints from a notebook. Use Kaggle, or Modal ($30/month free credit), for a live backend. The LLM can move the same way: set `"llm_backend": "openai"` with a base URL and an API-key env-var name.

## Files
- `app.py`: Gradio UI.
- `layerproto/engine.py`: plans the scene, generates every layer, updates single layers, and adds captions.
- `layerproto/llm.py`: breakdown schema and prompt, the Ollama and OpenAI-compatible clients, the captioner, and the word check.
- `layerproto/backends/`: the `ImageBackend` interface, the local LayerDiffuse SDXL backend, and the remote client.
- `layerproto/remote_server.py`: FastAPI wrapper for a GPU host.
- `vendor/diffuser_layerdiffuse`: MIT diffusers port of LayerDiffuse (rootonchair).
- `tools/smoke_layerdiffuse.py`: a one-image speed and memory test.
