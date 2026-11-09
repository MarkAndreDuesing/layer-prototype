# Story events: global edits across a layered RGBA scene (Nov 2026, pass 2)

Scope: weather, time of day, fire, season and lighting changes applied consistently to background + object layers. Hardware target: Kaggle 2x T4 (fp16, 16 GB each).
"[unverified]" marks anything I did not confirm on a primary page (paper, README or model card) this session. Pass 2 read primary pages for LBM, IC-Light, StyleAligned, WeatherEdit, TokenFlow, Rerender, LASAGNA and Qwen-Image-Layered. The other entries rest on search snippets that cite the primary page.

## Consistent multi-image / video editing that propagates one edit

### Takeaway
For "same event on every layer", the most practical option is a training-free shared-attention method on SDXL. StyleAligned (Apache-2.0, SDXL notebooks, ControlNet-Depth support) is the drop-in candidate. Edicho (ICCV 2025) is the closest real-image method for applying the same edit across several images, but it is correspondence-based and built on an inpainting editor. TokenFlow and Rerender are video-frame propagators: they assume frames share structure and suit "same layer before and after", not "different layers".

### Cited Findings
- StyleAligned (CVPR 2024): every image in a batch shares attention with the first image, with AdaIN applied to the queries and keys. It needs no optimisation. — [CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Hertz_Style_Aligned_Image_Generation_via_Shared_Attention_CVPR_2024_paper.html)
- StyleAligned repo facts:
  - Licence and status: Apache-2.0, "not an officially supported Google product", archived and read-only since 3 March 2025.
  - Notebooks: SDXL, SDXL reference-style transfer, SDXL + ControlNet-Depth, and SD2 MultiDiffusion.
  - Tested with PyTorch 2.1 and Diffusers 0.16.
  - "StyleAligned from an input image" is still on the TODO list. An `inversion.py` exists but is undocumented.
  — [GitHub](https://github.com/google/style-aligned)
- The StyleAligned paper says the method excels at diverse structures with consistent style, in contrast to structure-preserving editing. In practice this means pairing it with img2img or ControlNet-Depth to keep layer geometry. — [CVPR PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Hertz_Style_Aligned_Image_Generation_via_Shared_Attention_CVPR_2024_paper.pdf)
- Edicho (ICCV 2025): training-free consistent editing of in-the-wild images using Correspondence-Guided Attention and Correspondence-Guided CFG. It builds on a pretrained single-image editor (for example BrushNet) plus an explicit correspondence extractor. It contrasts itself with self-attention-modifying methods such as MasaCtrl and Cross-Image-Attention. — [arXiv 2412.21079](https://arxiv.org/html/2412.21079v3); [ICCV](https://openaccess.thecvf.com/content/ICCV2025/papers/Bai_Edicho_Consistent_Image_Editing_in_the_Wild_ICCV_2025_paper.pdf)
- ConsiStory: training-free, shares internal activations for consistent subject generation. It is a generation method, not an editing method. — [arXiv 2402.03286](https://arxiv.org/pdf/2402.03286.pdf)
- TokenFlow (ICLR 2024, MIT):
  - Text-driven video editing that enforces "consistency in the diffusion feature space" by propagating features across frames through inter-frame correspondences.
  - Builds on Plug-and-Play, ControlNet or SDEdit editing, with the Stable Diffusion version selectable.
  - The README states no VRAM figure.
  — [GitHub](https://github.com/omerbt/TokenFlow)
- Rerender-A-Video (SIGGRAPH Asia 2023):
  - Zero-shot two-stage method: keyframes are edited with cross-frame constraints, then propagated with temporal-aware patch matching and blending.
  - Uses SD1.5 + ControlNet.
  - The README says "24GB VRAM is required", so it exceeds a T4 without the memory-reduction options.
  - The licence file was not read.
  — [GitHub](https://github.com/williamyang1991/Rerender_A_Video)
- WeatherEdit uses temporal-view attention to keep weather edits consistent across frames and views. — [arXiv 2505.20471](https://arxiv.org/html/2505.20471v3)
- DreamLayer's Layer-Shared Self-Attention concatenates the noisy latents of all layers along the sequence dimension, so attention runs across every layer at once. This is the same mechanism a joint layer edit would need. — [arXiv 2503.12838](https://arxiv.org/html/2503.12838v1)

### Inferences
- For strategy (b), the recipe is:
  1. Batch the background plus each object layer (over a neutral matte) for SDXL img2img with the same event prompt.
  2. Use StyleAligned shared attention with the background as the reference.
  3. Add ControlNet-Depth to keep geometry.
  4. Reapply each layer's alpha, or re-decode it with LayerDiffuse.
- DreamLayer-style concatenated-latent attention is the principled version of this. No released editing code was confirmed.
- Rerender at 24 GB is not viable on a T4. TokenFlow's memory needs are unknown.

### Gaps
- I found no method that runs shared-attention editing on RGBA layers with alpha awareness.
- I did not read the TokenFlow or Edicho VRAM figures, or the Edicho code or licence.

## Relighting / time-of-day / weather methods and alpha handling

### Takeaway
IC-Light v1 (Apache-2.0, SD1.5, includes an fbc model conditioned on foreground + background) and LBM relighting (CC BY-NC 4.0, one step, "relight a foreground object according to a provided background") are the T4-viable options. Both output RGB, so you keep or re-estimate alpha yourself. WeatherEdit does have code and pretrained weights, but it is a 3D Gaussian scene pipeline, not a 2D layer editor.

### Cited Findings
- LBM relighting:
  - Licence: "Creative Commons BY-NC 4.0".
  - Install: `pip install git+https://github.com/gojasper/LBM.git`.
  - Usage: `get_model("jasperai/LBM_relighting", torch_dtype=torch.bfloat16)` then `evaluate(model, source_image, num_sampling_steps=1)`.
  - Purpose: "trained to relight a foreground object according to a provided background".
  - The card does not say how the foreground and background are supplied (the example passes a single composite) and states no resolution.
  — [HF card](https://huggingface.co/jasperai/LBM_relighting); [GitHub](https://github.com/gojasper/LBM); [arXiv 2503.07535](https://arxiv.org/html/2503.07535v2)
- LBM is a latent Brownian-bridge model, conditioned on a light map for relighting, and runs in 1 NFE. It was an ICCV 2025 highlight. — [arXiv](https://arxiv.org/html/2503.07535v2)
- [unverified] The card's example uses bf16. T4s have no native bf16, so the model probably needs fp16, which is untested.
- IC-Light:
  - The repo is Apache-2.0.
  - Checkpoints: `iclight_sd15_fc` (text + foreground; the default), `iclight_sd15_fbc` (text + foreground + background) and `iclight_sd15_fcon` (offset noise, slightly worse).
  - The bundled BRIA RMBG 1.4 background remover "is for non-commercial use"; BiRefNet is suggested as an alternative.
  — [GitHub](https://github.com/lllyasviel/IC-Light)
- IC-Light V2 (Flux-based) has its own "IC-Light V2 License". — [ComfyUI wiki](https://comfyui-wiki.com/en/models/ic-light/ic-light-v2)
- WeatherEdit (AAAI-26):
  - Code is released in `General_Scene` and `Driving_Scene` folders, with "our provided pretrained model" on HF and Google Drive.
  - Builds on 3D Gaussian Splatting, img2img-turbo and OmniRe. The licence file was not read.
  - The 2D part is an all-in-one adapter that edits the background for snow, rain and fog from text plus segmentation.
  — [GitHub](https://github.com/Jumponthemoon/WeatherEdit); [arXiv](https://arxiv.org/html/2505.20471v3)
- [unverified] Because WeatherEdit builds on img2img-turbo, its 2D background-weather adapter might be extractable as a one-step SD-Turbo model, which would be T4-friendly. This is not checked.
- WeatherWeaver (NVIDIA): video diffusion that adds or removes rain, snow, fog and clouds without 3D. No code was found. — [project](https://research.nvidia.com/labs/toronto-ai/WeatherWeaver)
- SceneCrafter (CVPR 2025, Waymo): weather via CLIP text, time of day via positionally encoded sun angle. It was trained on pairs generated by a self-attention Prompt-to-Prompt variant. No public code was found. — [arXiv 2506.19488](https://arxiv.org/html/2506.19488v1)
- Third-party summary figures for SceneCrafter (not checked against the paper): time-of-day FID 37.2 vs 60.4 for SDEdit. — [Moonlight](https://www.themoonlight.io/review/scenecrafter-controllable-multi-view-driving-scene-editing)
- LightIt (CVPR 2024, TUM + Adobe):
  - A ControlNet-like controller conditioned on shading + normal maps, with single-bounce shading that includes cast shadows.
  - Also trains an identity-preserving relighting model that takes an image plus a target shading.
  - 95.57% user preference for lighting accuracy.
  - No code release was found.
  — [project](https://peter-kocsis.github.io/LightIt); [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Kocsis_LightIt_Illumination_Modeling_and_Control_for_Diffusion_Models_CVPR_2024_paper.html)
- Neural Gaffer (NeurIPS 2024): object relighting under an environment map. Code is at Haian-Jin/Neural_Gaffer, built on Zero123-HF, and training uses 8 GPUs. The code licence was not stated; CC BY-NC-SA appears to be the arXiv paper licence. — [arXiv 2406.07520](https://arxiv.org/abs/2406.07520); [NeurIPS PDF](https://papers.nips.cc/paper_files/paper/2024/file/ff7373914a96956f2a7cacbdf3b0b8d8-Paper-Conference.pdf)
- DiLightNet (SIGGRAPH 2024): radiance-hint ControlNet on SD 2.1, with code at iamNCJ/DiLightNet. The licence type was not confirmed. — [GitHub](https://github.com/iamNCJ/DiLightNet) [licence unverified]

### Inferences
- Alpha recipe for relighting: composite the layer over the new background, relight with IC-Light fbc or LBM, then multiply by the original alpha. Re-estimate soft edges with BiRefNet or the LayerDiffuse decoder.
- Neural Gaffer and DiLightNet are object-centric (and Zero123 is 256 px), which makes them a weaker fit for scene layers.

### Gaps
- I did not check whether LBM works in fp16, and did not read the DiLightNet or WeatherEdit licences.

## Strategies (a)–(d): evidence

### Takeaway
No source compares the strategies head to head. The architectural evidence (WeatherEdit separates background editing from the particle layer; LASAGNA and DreamLayer generate layers together with their shadows and effects) favours (c) edit the background and relight the foregrounds, plus (d) separate effect layers. Use (a) as the fallback for geometry-changing events.

### Cited Findings
- (a) Qwen-Image-Layered: "Our code and models are released" at QwenLM/Qwen-Image-Layered. Layers "can be independently manipulated without affecting other content". The abstract does not mention global or lighting edits. — [arXiv 2512.15603](https://arxiv.org/abs/2512.15603)
- (c) LBM and IC-Light fbc relight a foreground to match a given background. — [LBM card](https://huggingface.co/jasperai/LBM_relighting); [IC-Light](https://github.com/lllyasviel/IC-Light)
- (d) LayerDiffuse supports foreground- or background-conditioned layer generation, and its examples include semi-transparent fire. — [arXiv 2402.17113](https://arxiv.org/pdf/2402.17113v3)
- (d) The ComfyUI-layerdiffuse workflow includes "generating foreground based on an existing background", which is the natural mode for an effect layer over the scene. — [RunComfy](https://www.runcomfy.com/comfyui-workflows/generating-transparent-images-with-layerdiffuse-in-comfyui)
- (d) A Flux LayerDiffuse port exists (ComfyUI_FluxLayerDiffuse). — [RunComfy node](https://www.runcomfy.com/comfyui-nodes/ComfyUI_FluxLayerDiffuse)
- (d) I found no source showing LayerDiffuse rain, snow, fog or smoke overlays, and no ComfyUI particle-overlay workflow for them. — (two searches, empty)
- (d) Game engines handle rain and wetness with shaders and particles, for example Unity's Shader Graph Production Ready Weather sample (drops, drips, puddles, wetness). — [Unity docs](https://docs.unity3d.com/Packages/com.unity.shadergraph@14.0/manual/Shader-Graph-Sample-Production-Ready-Weather.html)
- LASAGNA: existing object layers "typically lack realistic visual effects such as shadows and reflections". It promises that the dataset and benchmark "will be publicly released"; code release is not stated. — [arXiv 2601.15507](https://arxiv.org/abs/2601.15507v1)

### Inferences
- A suggested event pipeline:
  1. Edit the background with SDXL img2img using the event prompt.
  2. Relight the object layers with IC-Light fbc or LBM, then reapply alpha.
  3. Generate fog, fire or glow layers with LayerDiffuse, conditioned on the background.
  4. Make rain and snow procedural (particles in the viewer).
  5. Optionally run a low-strength SDXL harmonise pass.

### Gaps
- I found no quantitative evidence for any of the four strategies.

## Layered/PSD papers: global edits, style or lighting across layers

### Takeaway
No layered paper found offers event-driven or global relight/weather edits across layers. Joint-generation papers (DreamLayer, LayerFusion, PSDiffusion, OmniPSD) offer cross-layer attention that could in principle be reused by regenerating with a changed global prompt. Only the anime workflow-aware decomposition explicitly targets relighting.

### Cited Findings
- DreamLayer (ICCV 2025 Highlight): Context-Aware Cross-Attention, Layer-Shared Self-Attention and Information Retained Harmonization. Generates foreground and background layers together with consistent occlusion and shadowing, trained on a 400k-sample dataset. — [arXiv 2503.12838](https://arxiv.org/html/2503.12838v1); [ICCV](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_DreamLayer_Simultaneous_Multi-Layer_Generation_via_Diffusion_Model_ICCV_2025_paper.html)
- LayerFusion (Adobe/Virginia Tech, Dec 2024): produces a foreground (RGBA), background and blended image through attention-level interaction between two denoisers. It is training-free and supports spatial edits. — [arXiv 2412.04460](https://arxiv.org/html/2412.04460v1); [project](https://layerfusion.github.io)
- OmniPSD (Dec 2025): diffusion-transformer text-to-PSD generation and image-to-PSD decomposition. — [arXiv 2512.09247](https://arxiv.org/html/2512.09247v1)
- LayerCraft (NeurIPS 2025): LLM agents orchestrate layered generation and editing. — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/bf2a5ce85aea9ff40d9bf8b2c2561cae-Abstract-Conference.html)
- ART (Anonymous Region Transformer): variable multi-layer transparent generation. — cited in [OmniPSD](https://arxiv.org/html/2512.09247v1)
- CLD (Nov 2025): decomposes an image into RGBA layers guided by bounding boxes, using an LD-DiT on FLUX.1-dev. — [arXiv 2511.16249](https://arxiv.org/html/2511.16249v2)
- Stable-Layers (arXiv 2605.30257): fine-tunes a decomposition model with VLM-judge feedback. — [PDF](https://stability-ai.github.io/stable-layers.github.io/Stable_Layers.pdf)
- PSDiffusion: generates layers jointly from a global prompt plus per-layer prompts. — [arXiv 2505.11468](https://arxiv.org/pdf/2505.11468v2)
- MagicScroll (arXiv 2312.10899): story-driven, non-standard aspect-ratio generation with multi-layered semantic-aware denoising, GPT layout and style control. It is not RGBA layers, and the code release was unconfirmed. — [arXiv](https://arxiv.org/html/2312.10899v1)
- Workflow-Aware Structured Layer Decomposition (anime, Mar 2026): separates line art, base colour and shading to enable layer-level relighting and recolouring. — [arXiv 2603.14925](https://arxiv.org/html/2603.14925)
- Searches found no explicit global-edit, weather or lighting claims for OmniPSD, LayerCraft, ART, CLD or Stable-Layers. Only abstracts or snippets were checked. — see the links above

### Inferences
- A shading layer (as in the anime paper) would make time-of-day edits cheap: recolour or replace only the shading. The open question is whether Qwen-Image-Layered produces such a layer.

### Gaps
- I did not read the full papers (sections on editing applications). Code and weights status for DreamLayer, LayerFusion and OmniPSD was not checked.

## Games / interactive fiction with generative scene-state changes

### Takeaway
No AIIDE, FDG or CHI PLAY paper on generative day/night or weather changes was found. The practice evidence comes from indie devlogs: lock the layout (sketch or ControlNet) and pre-generate a variant for each state.

### Cited Findings
- Visual-novel developer: keep day and night backgrounds geometrically consistent with a locked-layout sketch-to-image workflow. — [itch.io](https://itch.io/blog/1145381/solved-how-to-keep-ai-backgrounds-geometrically-consistent-day-vs-night)
- Procedural visual-novel story generation with an LLM + SDXL (JGGAG 2025). Generated uniqueness decreased on replay. — [JGGAG](https://journal.binus.ac.id/index.php/jggag/article/view/12743)
- VinA: open-source Ren'Py + Stable Diffusion WebUI visual-novel generator. Weather and time-of-day features were not mentioned. — [GitHub](https://www.github.com/vina-ai/vina)
- [unverified] A 2023 r/StableDiffusion solo-developer post (Tales of Syn) reportedly used instruct-pix2pix for a wet-look map driven by a Unity weather system. This is search-snippet only and the post was not located. — [reddit mirror](https://redlib.hbubli.cc/r/StableDiffusion/comments/11fb7oq/isometric_rpg_game_tales_of_syn_developed_with)
- GameNGen: diffusion as a real-time game engine conditioned on actions. — [HF](https://huggingface.co/papers/2408.14837.md)
- An ACM talk cites about 10 s per image as a barrier to live changes. — [ACM PDF](https://oc.acm.org/docs/TacklingInteractiveGenAI.pdf)

### Inferences
- Cache event variants per layer and switch between them on story state. Precompute likely events in the background.

### Gaps
- I did not search conference proceedings (AIIDE, FDG, CHI PLAY) directly.
