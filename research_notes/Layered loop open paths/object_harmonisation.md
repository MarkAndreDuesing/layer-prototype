# Object-layer harmonisation after place/move/resize (status Nov 2026)

> **Built since (2026-11-09):** `prototype/layerproto/relight.py` (IC-Light fbc relight keeping the layer's alpha, with detail transfer; CPU test 44 s at 8 steps / 384 px) and `prototype/layerproto/shadow.py` (procedural shadow layer from this note's EasyOmnimatte recipe; about 0.6 s per layer). Results in `prototype/kaggle/results_local_relight/` and `results_local_shadow/`.

Scope: methods to fix lighting, colour, shadows, contact, reflections and style on an RGBA object layer after `relocate()` (warp, then SDXL img2img with LayerDiffuse bg2ble, latent pinning, mask extraction, paste-back; ~4 s at 512 px on a T4). Target: Kaggle 2x T4, fp16, 16 GB each. This file has had two passes (about 40 tool calls). Items marked [unverified] come from snippets, mirrors or prior knowledge and were not checked against the primary page.

## IC-Light v1/v2 and relighting alternatives

### Takeaway
IC-Light v1 fbc (SD1.5, Apache-2.0) is the strongest background-conditioned relighter that runs on a T4. It is a 12-channel UNet whose fg and bg latents are concatenated to the noise. IC-Light v2 (FLUX) has never had downloadable weights, and as of the August 2025 comments only a Space and a fal.ai API existed. Open SDXL-class alternatives: LBM relighting (1 step, CC BY-NC) and libcom's LBM harmoniser. The DiT-class alternatives are the Qwen-Image-Edit-2509 Relight and Fusion LoRAs and libcom's FLUX-Kontext blending/harmonisation LoRAs, all too heavy for a few seconds on a T4.

### Cited Findings
**IC-Light v1 (primary sources)**
- Models: `iclight_sd15_fc` (text + fg), `fcon` (offset noise, slightly worse), `fbc` (text + fg + bg; "does not require careful prompting"). The background demo is `python gradio_demo_bg.py`. Licence is Apache-2.0. The default BRIA RMBG-1.4 matting model is non-commercial; the README suggests BiRefNet instead. The News section also links an "Alternative model" for stronger illumination changes (discussion #109). — [lllyasviel/IC-Light README](https://github.com/lllyasviel/IC-Light); [discussion #109](https://github.com/lllyasviel/IC-Light/discussions/109)
- Exact mechanism, from `gradio_demo_bg.py`:
  - Base model: `stablediffusionapi/realistic-vision-v51`.
  - The UNet `conv_in` is widened from 4 to 12 input channels. The new weights are zeroed, then the original 4-channel weights are copied in.
  - fbc offset weights are added to the base UNet: `sd_merged = {k: sd_origin[k] + sd_offset[k]}`, loaded with `strict=True`.
  - fg and bg are each VAE-encoded (`latent_dist.mode() * scaling_factor`) and concatenated into an 8-channel `c_concat`. A hooked UNet forward then concatenates that onto the noisy latent, giving `torch.cat([sample, c_concat], dim=1)`.
  - Scheduler: DPMSolverMultistep with `sde-dpmsolver++` and Karras sigmas.
  - Defaults: 20 steps, CFG 7, 512x640. Highres pass at 1.5x scale and 0.5 denoise, run as img2img with `steps/denoise` scheduled steps.
  - — [gradio_demo_bg.py](https://raw.githubusercontent.com/lllyasviel/IC-Light/main/gradio_demo_bg.py)
- ComfyUI: the `ICLightConditioning` node takes an optional bg latent that only works with fbc. Fill the transparent area of the fg with grey before encoding, or the bg darkens. The `DetailTransfer` node restores high-frequency detail from the original. — [RunComfy node doc](https://www.runcomfy.com/comfyui-nodes/ComfyUI-IC-Light/ICLightConditioning); [ComfyUI-IC-Light-Native](https://github.com/huchenlei/ComfyUI-IC-Light-Native); [ComfyUI-IC-Light](https://github.com/huchenlei/ComfyUI-IC-Light)

**IC-Light v2 status**
- The maintainer post (27 Oct 2024) describes FLUX-based models with a 16-channel VAE and native high resolution. Only the fg-conditioned detail-preserving model was shown. Three were planned but not released: one for large modifications, one conditioned on fg+bg, and one for HDRI. The post says "You can only use the hf space to test for now. More inference codes and weights may be released later", and that the v2 licence "will be non-commercial only". Comments in May 2025 report the Space returning 404, and in Aug 2025 a startup error. In Apr 2025 a community member published a ComfyUI node that calls v2 through fal.ai (an API, not weights). — [IC-Light discussion #98](https://github.com/lllyasviel/IC-Light/discussions/98)
- ComfyUI-Wiki pages contradict each other ("Open Source" vs "weights not open"). The discussion above supports "not open". `molidiss/iclight-v2` on HF is an unofficial repo [unverified content]. — [ComfyUI-Wiki EN](https://comfyui-wiki.com/en/models/ic-light/ic-light-v2); [ComfyUI-Wiki JA](https://comfyui-wiki.com/ja/models/ic-light); [HF molidiss/iclight-v2](https://huggingface.co/molidiss/iclight-v2)

**LBM (Latent Bridge Matching, Jasper AI)**
- The GitHub repo ships three HF checkpoints: normals, depth and relighting. Inference: `examples/inference/inference.py --model_name relighting --source_image ... --output_path ...`, plus a local Gradio demo. The input is a composite with the fg already pasted onto the target bg, translated in a single step. The licence text says CC BY-NC 4.0, but one badge links CC BY-ND 4.0, an inconsistency on the page. No resolution or VRAM figures are given. — [gojasper/LBM](https://github.com/gojasper/LBM)
- The HF card says it is a fine-tune of SDXL-base-1.0, trained "to relight a foreground object according to a provided background", with example `evaluate(model, source_image, num_sampling_steps=1)`. — [HF jasperai/LBM_relighting](https://huggingface.co/jasperai/LBM_relighting)
- libcom exposes LBM as one of its `ImageHarmonizationModel` methods, alongside PCTNet. — [bcmi/libcom](https://github.com/bcmi/libcom)

**Qwen-Image-Edit / Kontext relight and fusion LoRAs**
- `dx8152/Qwen-Image-Edit-2509-Relight` (Apache-2.0) must be used with a `lightx2v/Qwen-Image-Lightning` LoRA. A demo Space fuses it at scale 1.25 on a 4-step model. The README also links Kontext relight LoRAs it collected (CivitAI; `dong625/kontext_loras` [unverified]). — [HF Relight](https://huggingface.co/dx8152/Qwen-Image-Edit-2509-Relight); [Comfy-Org mirror](https://huggingface.co/comfy-org/qwen-image-edit_comfyui)
- `dx8152/Qwen-Image-Edit-2509-Fusion`: its trigger text asks the model to blend the image, correct the product's perspective and lighting, and merge it into the background. It is used with the Lightning LoRA. Related: `ostris/qwen_image_edit_inpainting` (used as a "fusion" adapter in a Space), the flymy-ai 2509 InScene LoRA, and the QIE-2511 Object-Adder, Studio-DeLight and 2509 Light-Migration LoRAs. — [HF Fusion](https://huggingface.co/dx8152/Qwen-Image-Edit-2509-Fusion); [flymy-ai InScene](https://huggingface.co/flymy-ai/qwen-image-edit-2509-inscene-lora); [QIE-2511 Object-Adder mirror](https://hug-proxy.pages.dev/prithivMLmods/Qwen-Image-Edit-2511-Object-Adder); [Studio-DeLight](https://huggingface.co/mario-d-robot/QIE-2511-Studio-DeLight); [Light-Migration](https://huggingface.co/mario-d-robot/Qwen-Edit-2509-Light-Migration)
- libcom trains LoRA adapters on FLUX.1 Kontext to make two models, image blending and image harmonisation (`KontextBlendingHarmonizationModel`). The docs page gives no VRAM or licence. — [libcom Kontext doc](https://libcom.readthedocs.io/en/latest/kontext.html); [bcmi/libcom](https://github.com/bcmi/libcom)
- SpotLight: training-free relighting of an inserted object, controlled by a user-specified coarse shadow. CC BY-NC-SA 4.0. — [lvsn/SpotLight](https://github.com/lvsn/SpotLight); [arXiv 2411.18665](https://api.semanticscholar.org/arXiv:2411.18665)

### Exact usage in diffusers (fg RGBA + bg)
SD1.5 IC-Light fbc. This follows the official demo's code path above; the glue code is my own and has not been run.
1. Load `stablediffusionapi/realistic-vision-v51` (StableDiffusionPipeline, fp16).
2. Widen `unet.conv_in` to 12 channels: zero the new weights, copy the old 4-channel weights, keep the bias.
3. Download `iclight_sd15_fbc.safetensors` (HF `lllyasviel/ic-light` [unverified repo id]) and add it to `unet.state_dict()` key by key, then call `load_state_dict(strict=True)`.
4. fg = RGBA composited over mid-grey (127) after warping to the new box, at full canvas size. bg = the clean scene without the object. Both are sized to multiples of 64.
5. `c_concat = cat([vae.encode(fg).mode(), vae.encode(bg).mode()], dim=1) * 0.18215`, for 8 channels.
6. Wrap `unet.forward`: `sample = cat([sample, c_concat.repeat(batch,1,1,1)], 1)` (CFG doubles the batch).
7. Run img2img on the existing composite at about 0.4-0.7 strength, 20-25 steps, DPM++ SDE Karras, prompt "<object>, natural lighting".
8. Output alpha = the warped source alpha (IC-Light outputs RGB only). Then do frequency-split paste-back: final = relit_lowpass + original_highpass inside the eroded mask.

SDXL: there is no official IC-Light SDXL. Use LBM relighting:
1. Build the composite (warped RGBA alpha-blended onto the scene).
2. `from lbm.inference import get_model, evaluate` [unverified module path, check the repo's `examples/inference/inference.py`], then `model = get_model("jasperai/LBM_relighting", torch_dtype=fp16)` and `out = evaluate(model, composite, num_sampling_steps=1)`.
3. Re-apply the original alpha and the paste-back as above. T4 note: SDXL fp16 is about 7 GB, and one UNet step at 1024 px takes about 1 s [unverified estimate].

### Inferences
- On a T4, fbc costs about 20 steps at 512 px, roughly 2-4 s [unverified], and runs separately from the SDXL loop on GPU 1. LBM is likely the fastest at 1 step, but it is NC and its relight strength isn't controllable.
- Qwen-Image-Edit (20B) and Kontext (12B) LoRAs need 4-bit plus offload on 16 GB, which means tens of seconds per edit. Keep them for an offline "final polish" pass [unverified].

### Gaps
- Real T4 timings and VRAM for fbc and LBM were not measured. The exact HF repo id for the fbc weights and the LBM Python API path were not read from source.
- I found nothing on Lumina-based relighting or a community "IC-Light FLUX" port.

## Diffusion harmonisation / insertion methods

### Takeaway
Training-free, SD-sized options with code: TF-ICON, PrimeComposer, FreeCompose, Diff-Harmonization. Trained SD-sized options with code and weights: Paint-by-Example (OpenRAIL-M), AnyDoor (MIT, SD2.1 + DINOv2), MimicBrush (Apache-2.0, two U-Nets), DiffHarmony, libcom models. FLUX and DiT options: Add-it (training-free), Insert Anything (LoRA on FLUX-Fill, 10-26 GB), PosBridge and LooseRoPE (training-free), Insert In Style (trained LoRA), libcom Kontext LoRAs, Qwen-Edit Fusion and InScene LoRAs. No public code: ObjectDrop, Thinking-outside-the-BBox, ObjectStitch (as far as I could confirm). I found no paper named "OmniHarmony".

### Cited Findings
| Method | Mechanism | Training-free? | Code / weights | Licence | VRAM / base |
|---|---|---|---|---|---|
| Paint-by-Example (CVPR'23) | SD inpainting conditioned on a CLIP embedding of the exemplar | trained | yes; diffusers pipeline (deprecated) — [repo](https://github.com/Fantasy-Studio/Paint-by-Example), [diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/paint_by_example) | CreativeML OpenRAIL-M — [HF card](https://www.huggingface.co/Fantasy-Studio/Paint-by-Example/blob/main/README.md) | SD1.x-size, fits a T4 [unverified] |
| ObjectStitch (CVPR'23) | conditional diffusion with content adaptor; does geometry, harmonisation and shadows in one model | trained | no official code found; libcom/BCMI links an ObjectStitch reimplementation [unverified] — [arXiv 2212.00932](https://arxiv.org/abs/2212.00932) | ? | ? |
| AnyDoor | SD2.1 + ControlNet-style control copy; DINOv2 ID tokens + high-frequency detail map | trained (zero-shot for new objects) | yes; ModelScope/HF — [ali-vilab/AnyDoor](https://github.com/ali-vilab/AnyDoor) | MIT | no figure given; training used 2xA100 |
| MimicBrush (NeurIPS'24) | reference U-Net + imitative U-Net; reference features injected into attention | trained | yes; HF `xichenhku/MimicBrush` — [repo](https://github.com/ali-vilab/MimicBrush) | Apache-2.0 | "model is big", two U-Nets; no figure |
| TF-ICON (ICCV'23) | exceptional-prompt inversion + composite self-attention maps | training-free | [Shilin-LU/TF-ICON](https://github.com/Shilin-LU/TF-ICON) (link from paper) — [paper](https://shilin-lu.github.io/tf-icon.github.io/tficon/tficon.pdf) | ? | SD2.1 [unverified] |
| PrimeComposer | attention steering across noise levels; faster than TF-ICON | training-free | [CodeGoat24/PrimeComposer](https://github.com/CodeGoat24/PrimeComposer) — [arXiv 2403.05053](https://arxiv.org/html/2403.05053v1) | ? | SD [unverified] |
| FreeCompose | pretrained diffusion as prior: removal, harmonisation, composition | training-free | [aim-uofa/FreeCompose](https://github.com/aim-uofa/FreeCompose) — [arXiv 2407.04947](https://arxiv.org/html/2407.04947v1) | ? | ? |
| DiffHarmony | SD-inpainting with composite+mask as conditions, empty text, plus refinement stage | trained | announced at nicecv/DiffHarmony [live status unverified] — [arXiv 2404.06139](https://arxiv.org/html/2404.06139v1) | ? | SD1.5-inpaint size |
| Diff-Harmonization | zero-shot harmonisation with frozen SD | training-free | [windvchen/diff-harmonization](https://github.com/windvchen/diff-harmonization) | Apache-2.0 | SD |
| R2R / RPHarmony (2025) | region-to-region transformation harmonisation | trained | anonymised repo — [arXiv 2508.09746](https://arxiv.org/abs/2508.09746) | ? | ? |
| Reference-aware harmonisation (Neural Networks 2026) | diffusion harmonisation guided by a reference | trained | GitHub link in paper — [PubMed](https://pubmed.ncbi.nlm.nih.gov/41386153/) | ? | ? |
| libcom (BCMI) | toolbox: PCTNet/LBM harmonisation, PHDiffusion painterly, GPSDiffusion shadows, RGDiffusion reflections, OSInsert insertion, Kontext blend/harm LoRAs, harmony and placement scorers | trained | [bcmi/libcom](https://github.com/bcmi/libcom) | Apache-2.0 (toolbox); per-model unchecked | no figures |
| Add-it (ICLR'25) | FLUX weighted extended attention over scene, prompt and generated image; subject-guided latent blending | training-free | yes — [NVIDIA project](https://research.nvidia.com/labs/par/project/addit.html), [arXiv](https://arxiv.org/html/2411.07232v2) | NVIDIA + FLUX-dev NC [unverified] | FLUX.1-dev, >16 GB without quantisation [unverified] |
| Insert Anything | in-context editing on FLUX.1-Fill-dev (reference diptych/triptych) + LoRA | trained LoRA | yes — [repo](https://github.com/paperwave/insert-anything) | MIT per mirror [unverified] + FLUX-dev NC | 26 GB standard; 10 GB with Nunchaku int4 |
| PosBridge | transplants positional embeddings from the reference; FLUX.1-Fill | training-free (optional LoRA) | ? — [BMVC'25 poster](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_348/poster.pdf) | ? | FLUX |
| LooseRoPE (2026) | content-aware RoPE/attention manipulation for semantic harmonisation of cut-and-paste | training-free | ? — [arXiv 2601.05127](https://arxiv.org/html/2601.05127v1) | ? | DiT |
| Insert In Style | zero-shot composition with LoRAs trained on frozen FLUX.1-dev | partly trained | ? — [arXiv 2508.17302](https://arxiv.org/html/2508.17302v1) | ? | FLUX |
| ObjectDrop (ECCV'24) | trained on counterfactual photos; bootstrapped insertion with shadows and reflections | trained | no public code found — [arXiv 2403.18818](https://arxiv.org/html/2403.18818v1) | — | — |
| Thinking outside the BBox (ECCV'24) | unconstrained compositing; generates effects outside the mask, auto-placement | trained | no official code confirmed — [arXiv 2409.04559](https://arxiv.org/html/2409.04559v2) | — | — |
| Qwen-Edit 2509 Fusion / InScene, libcom Kontext LoRAs | instruction-edit DiT + LoRA | trained LoRA | see section above | Apache-2.0 (Fusion); Kontext-dev NC | 12-20B params |

- I found no "OmniHarmony" paper or repo in two searches. — [search noted; nearest: R2R](https://arxiv.org/abs/2508.09746)
- I found no Qwen-Image-2.1-specific insertion or harmonisation method.

### Inferences
- For our loop, the useful mechanisms are: (a) a fbc-style bg-conditioned relight (cheap); (b) TF-ICON/PrimeComposer-style attention compositing, which is training-free on SD and conceptually close to our KV/pinning approach; (c) ObjectDrop/BBox's idea of letting the generated region extend beyond the object mask for shadows. All three can be done with SD1.5 or SDXL on a T4.
- AnyDoor and MimicBrush re-synthesise the object from a reference, so they can drift on identity and they lose our clean alpha. They are useful only as an "insert new object" path, not for re-harmonising an existing layer.

### Gaps
- Licences for TF-ICON, PrimeComposer, FreeCompose and DiffHarmony; VRAM for all SD-based methods; whether PosBridge and LooseRoPE released code.

## Shadow (and reflection) generation, incl. as a separate layer

### Takeaway
Released and usable: BCMI's SGDiffusion (DESOBAv2, CVPR'24) and GPSDiffusion (CVPR'25, geometry prior; SD1.5 version and SDXL version since 2 Jul 2025, both in libcom), plus BCMI's reflection model (DEROBA, arXiv 2604.02168). All output a composite with the shadow, not a separate layer. The shadow-as-layer idea appears in LayerDecomp (CVPR'25, which puts effects into the transparent fg layer, code not found), the Omnimatte line (video), and an Adobe-style patent. Practical approach: diff the model output against the input to extract a darkening matte, or build one procedurally.

### Cited Findings
- SGDiffusion: first diffusion-based shadow generator; DESOBAv2 has 21,575 images and 28,573 pairs; code, data and model released. — [CVPR'24 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Shadow_Generation_for_Composite_Image_Using_Diffusion_Model_CVPR_2024_paper.pdf); [bcmi DESOBAv2 repo](https://github.com/bcmi/Object-Shadow-Generation-Dataset-DESOBAv2)
- GPSDiffusion (Zhao et al., CVPR 2025): two stages, geometry-prior prediction then geometry-guided diffusion. Scripts are `infer_GPSDiffusion.py` and `post_processing.py`. The README warns performance is unstable, so you should generate several results and pick one. An SDXL version was released 2025/7/02. — [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_Shadow_Generation_Using_Diffusion_Model_with_Geometry_Prior_CVPR_2025_paper.html); [repo (mirror)](https://gh.ali360.cloudns.ch/bcmi/GPSDiffusion-Object-Shadow-Generation); [SDXL repo (mirror)](https://github.colasdn.workers.dev/bcmi/GPSDiffusion-Object-Shadow-Generation-SDXL)
- libcom's `ShadowGenerationModel` uses GPSDiffusion and its `ReflectionGenerationModel` uses RGDiffusion. — [bcmi/libcom](https://github.com/bcmi/libcom)
- Reflection generation for composite images (SD1.5-based, type-aware; a vertical-flip reflection uses a flipped object crop as reference). DEROBA dataset and model released. — [arXiv 2604.02168](https://arxiv.org/html/2604.02168v1); repo `bcmi/Object-Reflection-Generation-Dataset-DEROBA` per paper
- Newer shadow papers, code not confirmed: physics-grounded light direction with confidence (Hu et al. 2025); Keypoints Linear Model + Shadow Triangle ([arXiv 2511.08169](https://www.arxiv.org/abs/2511.08169)); MultiShadow for multiple objects ([arXiv 2603.02743](https://arxiv.org/html/2603.02743v3)).
- SpotLight takes a user shadow as the control input. — [lvsn/SpotLight](https://github.com/lvsn/SpotLight)
- Shadow as a layer:
  - LayerDecomp (Adobe/UCSC, CVPR'25) decomposes an image into a clean bg and a transparent fg layer that keeps shadows and reflections. No code link found. — [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Generative_Image_Layer_Decomposition_with_Visual_Effects_CVPR_2025_paper.pdf); [project](https://rayjryang.github.io/LayerDecomp)
  - Generative Omnimatte (video) produces layers with their effects. — [CVPR'25](https://mlanthology.org/cvpr/2025/lee2025cvpr-generative)
  - EasyOmnimatte (CVPR'26) synthesises training shadow mattes from the fg alpha via vertical compression, shear, semi-transparency and blur, which is exactly a procedural shadow recipe. — [supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Hu_EasyOmnimatte_Taming_Pretrained_CVPR_2026_supplemental.pdf)
  - A USPTO patent describes generating a shadow layer for an object rather than painting the shadow into the image. — [USPTO 12597186](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12597186)
- DreamLayer's IRH ablation: without harmonisation a chair "floats" with no shadow. IRH generates the shadow in the background layer, and the aesthetic score drops 0.1 without it. — [arXiv 2503.12838](https://arxiv.org/html/2503.12838v1)

### Inferences
- Shadow-layer recipe:
  1. Procedural matte: squash and shear the alpha along the light direction (EasyOmnimatte-style), blur it with distance from the contact line, and add an ambient-occlusion blob at the contact points.
  2. Optionally run GPSDiffusion-SDXL on the composite and take `matte = clip(1 - out_lum/in_lum, 0, 1)` outside the dilated object mask.
  3. Store the result as a black RGBA layer (or a multiply layer) between bg and object.
- Following DreamLayer's IRH, shadows "belong" to the background layer. That matches storing them as a separate layer that moves with the object.

### Gaps
- Licences and VRAM for GPSDiffusion and DEROBA; GitHub was only seen through mirrors. No released model outputs a shadow matte directly for still images.

## Identity preservation during re-harmonisation

### Takeaway
No paper found compares reference-only attention, KV injection, IP-Adapter, latent pinning and paste-back head-to-head. The move/resize editors each report their own metrics: GeoDiffuser uses CLIP similarity, warp error and mean distance; RegionDrag uses LPIPS and mean distance; a 2025 training-free geometric-editing paper uses DINO subject consistency and CLIP background consistency. The mechanisms with the best evidence are KV/self-attention sharing from the source path (RegionDrag, DragonDiffusion family, Add-it) and detail transfer (AnyDoor's high-frequency map, IC-Light DetailTransfer).

### Cited Findings
- GeoDiffuser (WACV'25): zero-shot, optimisation-based 2D/3D object edits. Metrics: Mean Distance, Warp Error, CLIP similarity. For 3D edits FreeDrag scores CLIP 0.977 vs GeoDiffuser 0.967. — [arXiv 2404.14403](https://arxiv.org/html/2404.14403v1); [WACV paper](https://openaccess.thecvf.com/content/WACV2025/papers/Sajnani_GeoDiffuser_Geometry-Based_Image_Editing_with_Diffusion_Models_WACV_2025_paper.pdf)
- A training-free geometric image editing paper (2025) uses DINO Subject Consistency, CLIP Background Consistency, and GeoDiffuser's warp error and mean distance. — [arXiv 2507.23300](https://arxiv.org/pdf/2507.23300)
- RegionDrag: identity is preserved through training-free mutual self-attention control, reusing K/V from the unedited latent path. Metrics are LPIPS and DIFT-based Mean Distance on DragBench-SR/DR. A secondary review says it has the lowest MD and LPIPS, with MD >30% lower than point-based methods on DR [verify against Table 1]. — [arXiv 2407.18247](https://arxiv.org/pdf/2407.18247); [review](https://www.themoonlight.io/en/review/regiondrag-fast-region-based-image-editing-with-diffusion-models)
- A 2025 survey (arXiv 2512.12287) argues that drag-editing evaluation is unreliable for lack of standard benchmarks and metrics. — [arXiv 2512.12287](https://arxiv.org/pdf/2512.12287.pdf)
- Magic Fixup (Adobe/UMD): takes a coarse user edit (move/resize) and synthesises a photoreal output while transferring fine details from the original. Trained on video frames. Weights are `HadiZayer/MagicFixup` (about 7.7 GB) under an Adobe research licence, plus an HF Space. An aggregator lists an ObjMove-A CLIP score of 89.62 [unverified]. — [project](https://magic-fixup.github.io); [arXiv 2403.13044](https://arxiv.org/html/2403.13044v2); [HF weights](https://huggingface.co/HadiZayer/MagicFixup/commits/main)
- DesignEdit (AAAI'25): SDXL + prompt-to-prompt, multi-layered latent decomposition and fusion for move/resize/remove. MIT licence, code since 2024-03-28. — [design-edit/DesignEdit (via ecosyste.ms)](https://awesome.ecosyste.ms/projects/github.com%2Fdesign-edit%2FDesignEdit)
- InstantDrag (SIGGRAPH Asia'24): official repo SNU-VGILab/InstantDrag; weights status not confirmed. — [ecosyste.ms entry](https://awesome.ecosyste.ms/projects/github.com%2FSNU-VGILab%2FInstantDrag)
- DreamLayer (ICCV'25): Context-Aware Cross-Attention, Layer-Shared Self-Attention, and Information-Retained Harmonizing (IRH), which fuses fg and bg latents during denoising. Its ablation shows shadows and detail appear with IRH. — [arXiv 2503.12838](https://arxiv.org/html/2503.12838v1); [ICCV paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_DreamLayer_Simultaneous_Multi-Layer_Generation_via_Diffusion_Model_ICCV_2025_paper.pdf)
- Add-it uses weighted extended attention and latent blending to keep the scene and object consistent. — [arXiv 2411.07232](https://arxiv.org/html/2411.07232v2)
- AnyDoor's identity comes from DINOv2 ID tokens plus a high-frequency detail map. — [ali-vilab/AnyDoor](https://github.com/ali-vilab/AnyDoor)
- DragonDiffusion, DiffEditor and Diffusion Self-Guidance: no identity metrics were retrieved in this pass. From prior knowledge, DragonDiffusion and DiffEditor use memory-bank KV injection and Self-Guidance uses attention-based guidance [unverified].

### Inferences
- For our `relocate()`:
  - Keep latent pinning, but pin only the eroded interior.
  - Add mutual self-attention K/V from a parallel "source" pass of the warped layer (RegionDrag/MasaCtrl-style). This is training-free and cheap, roughly one extra UNet pass per step.
  - Finish with frequency-split paste-back. Re-apply alpha from the warped source so it is always preserved exactly.
- To measure it ourselves: DINO-v2 cosine (subject consistency) and LPIPS inside the mask, plus CLIP/LPIPS on the background, as in arXiv 2507.23300 and RegionDrag.

### Gaps
- No head-to-head study of reference-only vs IP-Adapter vs pinning vs paste-back. Identity numbers for DragonDiffusion, DiffEditor, InstantDrag, DesignEdit and Self-Guidance were not retrieved.

## Best practical combination on T4 (a few seconds per edit)

### Takeaway
Pipeline (estimated 3-6 s at 512 px, not benchmarked):
1. `relocate()` warp, plus SDXL bg2ble with eroded-interior pinning and source-KV sharing (GPU 0).
2. Relight with IC-Light fbc (bg = the scene without the object; 0.4-0.6 strength) or LBM in 1 step (GPU 1).
3. Frequency-split paste-back and the original alpha.
4. A procedural shadow layer (EasyOmnimatte-style shear and blur plus contact AO), optionally refined with a GPSDiffusion run whose luminance-ratio matte becomes its own RGBA layer.
5. Optional SDXL inpaint over the dilated contact zone.

Keep the FLUX/Qwen paths (libcom Kontext harmonisation LoRA, Qwen-Edit Fusion/Relight, Add-it, Insert Anything) as an offline polish step.

### Cited Findings
- Component sources and licences: IC-Light Apache-2.0 ([repo](https://github.com/lllyasviel/IC-Light)); LBM CC BY-NC ([repo](https://github.com/gojasper/LBM)); GPSDiffusion and libcom ([libcom](https://github.com/bcmi/libcom)); KV-sharing evidence from RegionDrag ([arXiv 2407.18247](https://arxiv.org/pdf/2407.18247)); procedural shadow matte recipe from EasyOmnimatte ([supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Hu_EasyOmnimatte_Taming_Pretrained_CVPR_2026_supplemental.pdf)).

### Inferences
- All recommended components are fine for a personal non-commercial project. NC licences (LBM, SpotLight, Kontext-dev, IC-Light v2) only matter if the project ever goes commercial.

### Gaps
- No T4 measurements for any component.
