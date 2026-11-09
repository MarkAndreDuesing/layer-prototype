# Image-to-Layers Decomposition Models (as of Nov 2026)

## Qwen-Image-Layered: what it is and how to use it

### Takeaway
Qwen-Image-Layered (arXiv 2512.15603, 17 Dec 2025, CVPR 2026) is a 20B, Apache-2.0 diffusion model fine-tuned from Qwen-Image. It turns one RGB image into a variable number of RGBA layers and fills in the parts each layer hides. It is the open reference model for this task, but it is heavy and slow. The prompt only describes the whole image; it cannot name which objects go on which layer.

### Cited Findings
- **Paper:** "Qwen-Image-Layered: Towards Inherent Editability via Layer Decomposition", arXiv 2512.15603, submitted 17 Dec 2025, accepted to CVPR 2026. — [arXiv](https://arxiv.org/html/2512.15603v1); [CVPR open access](https://openaccess.thecvf.com/content/CVPR2026/html/Yin_Qwen-Image-Layered_Towards_Inherent_Editability_via_Layer_Decomposition_CVPR_2026_paper.html)
- **Authors:** Yin, Zhang, Tang, … Junyang Lin, Chenfei Wu (corresponding author), from HKUST(GZ), Alibaba and HKUST. — [arXiv](https://arxiv.org/html/2512.15603v1)
- **Base model and size:** finetune of Qwen/Qwen-Image; 20B parameters, BF16 safetensors; Apache 2.0 licence. — [HF model card](https://huggingface.co/Qwen/Qwen-Image-Layered)
- **RGBA-VAE:**
  - It widens the Qwen-Image VAE's first encoder convolution and last decoder convolution from 3 to 4 channels, following AlphaVAE.
  - The RGB weights are copied over. The alpha encoder weights start at zero, and the alpha decoder starts at weight 0 with bias 1.
  - RGB and RGBA images share one latent space.
  - Each layer is encoded separately, with no compression across layers.
  - Source: [arXiv](https://arxiv.org/html/2512.15603v1); AlphaVAE is [arXiv 2507.09308](https://arxiv.org/pdf/2507.09308).
- **VLD-MMDiT (Variable Layers Decomposition MMDiT):**
  - Text and image streams use separate parameters, and their token sequences are joined in attention.
  - Training uses flow matching.
  - "Layer3D RoPE" adds a layer-index axis: output layers are 0, 1, 2…, and the input image is −1. This is what lets the layer count vary.
  - Source: [arXiv](https://arxiv.org/html/2512.15603v1)
- **Training in three stages:**
  1. Text→RGB adapted to Text→RGBA for the new VAE (500K steps).
  2. Text→multi-RGBA, which produces the "T2L" model (400K steps).
  3. Image→multi-RGBA, which produces the "I2L" model (400K steps).
  - Up to 20 layers were used in training.
  - Source: [arXiv](https://arxiv.org/html/2512.15603v1)
- **Training data:** real Photoshop (PSD) files. Layers were pulled out with psd-tools; anomalous and non-contributing layers were filtered out, and non-overlapping layers were merged. Qwen2.5-VL wrote the captions. The paper gives no total PSD count. — [arXiv](https://arxiv.org/html/2512.15603v1)
- **How the text prompt is used:**
  - It is optional and is encoded by the multimodal language model (Qwen2.5-VL).
  - `use_en_prompt=True` auto-captions the image when no prompt is given. Example settings: `true_cfg_scale` 4.0, `cfg_normalize` on. — [HF](https://huggingface.co/Qwen/Qwen-Image-Layered)
  - ComfyUI's guide says the prompt should describe the whole image, including partly hidden content (e.g. text behind an object). It does not control what goes on each layer. — [ComfyUI Wiki](https://comfyui-wiki.com/en/tutorial/advanced/image/qwen/qwen-image-layered)
- **Layer count:**
  - Set with the `layers` argument (the example uses 4); 3 and 8 layers are demonstrated. — [HF](https://huggingface.co/Qwen/Qwen-Image-Layered)
  - Replicate allows 2–8 layers and suggests 3–4 for most images, 6–8 for complex scenes. Too many layers over-segments the image. — [Replicate](https://replicate.com/qwen/qwen-image-layered)
- **Recursive decomposition:** any output layer can be decomposed again, which the card calls "infinite decomposition". — [HF](https://huggingface.co/Qwen/Qwen-Image-Layered)
- **Resolution:** buckets of 640 and 1024; 640 is recommended for this version. — [HF](https://huggingface.co/Qwen/Qwen-Image-Layered)
- **Diffusers usage:** `QwenImageLayeredPipeline`. Needs transformers ≥4.51.3, diffusers installed from GitHub, and python-pptx (outputs can be exported to PPTX). — [HF](https://huggingface.co/Qwen/Qwen-Image-Layered)
- **Benchmark against LayerD (Crello dataset):**

  | Merges allowed | Qwen-Image-Layered RGB L1 | LayerD RGB L1 | Qwen-Image-Layered alpha soft-IoU | LayerD alpha soft-IoU |
  |---|---|---|---|---|
  | 0 | 0.0594 | 0.0709 | 0.8705 | 0.7520 |
  | 5 | 0.0363 | 0.0396 | 0.9160 | 0.8650 |

  Lower L1 and higher IoU are better. The authors needed to fine-tune on Crello because of a distribution gap. — [arXiv](https://arxiv.org/html/2512.15603v1)
- **ComfyUI support:** native, no custom nodes needed. Files:
  - `qwen_image_layered_bf16.safetensors`, or `qwen_image_layered_fp8mixed.safetensors` for low VRAM
  - `qwen_image_layered_vae.safetensors`
  - `qwen_2.5_vl_7b_fp8_scaled` text encoder
  - Source: [ComfyUI Wiki](https://comfyui-wiki.com/en/tutorial/advanced/image/qwen/qwen-image-layered)
- **VRAM and speed:**
  - Peak use was about 45 GB on a 96 GB RTX Pro 6000, with about 120 s per 1024px run (default 50 steps, CFG 4).
  - RTX 4090 (24 GB) users report it nearly fills VRAM; FP8 is advised.
  - Source: [ComfyUI Wiki](https://comfyui-wiki.com/en/tutorial/advanced/image/qwen/qwen-image-layered)
  - A RunPod write-up recommends a 48 GB GPU for BF16. — [lilting.ch](https://lilting.ch/en/articles/runpod-qwen-image-layered-lora)
- **GGUF / quantized versions:**
  - The HF card lists 7 quantized variants. — [HF](https://huggingface.co/Qwen/Qwen-Image-Layered)
  - The ComfyUI Wiki lists the GGUF workflow as "pending update". — [ComfyUI Wiki](https://comfyui-wiki.com/en/tutorial/advanced/image/qwen/qwen-image-layered)
  - A third-party guide claims a ~3.2 GB Q4_K_M file that suits 8–12 GB VRAM. Unverified, and its sizes conflict with other sources. — [dev.to](https://dev.to/gary_yan_86eb77d35e0070f5/how-to-use-qwen-image-layered-gguf-in-comfyui-complete-installation-and-usage-guide-2en3)
- **Hosted APIs:**
  - fal: `fal-ai/qwen-image-layered`, about $0.05 per image according to aggregators, not fal's own page. — [CostGoat](https://costgoat.com/pricing/qwen-image); [PricePerToken](https://pricepertoken.com/image/model/fal-ai-qwen-image-layered)
  - Replicate: `qwen/qwen-image-layered`; the price is not shown on the page. — [Replicate](https://replicate.com/qwen/qwen-image-layered)
- **Known weaknesses:**
  - Works best on clear, well-defined objects. — [Replicate](https://replicate.com/qwen/qwen-image-layered)
  - Decomposition is ambiguous: several splits can be equally plausible. — [arXiv](https://arxiv.org/html/2512.15603v1)
  - High inference latency because of the heavy diffusion design. — [arXiv 2605.15167](https://arxiv.org/pdf/2605.15167)
  - Slow and memory-hungry. — [ComfyUI Wiki](https://comfyui-wiki.com/en/tutorial/advanced/image/qwen/qwen-image-layered)

### Inferences
- For the game: generate the flat scene, run Qwen-Image-Layered (hosted, about $0.05 per call), and recurse on any layer that holds several characters. The model chooses the groupings itself, so the game must identify each layer afterwards, e.g. with a VLM or by matching masks.
- The T2L variant (text→multiple layers in one step) exists in the paper and may avoid the flat-then-split step altogether. I did not confirm whether its weights were released.

### Gaps
- No official VRAM figure; the community numbers conflict (13 GB vs 40 GB for BF16).
- fal's own price page was not verified.
- Release status of the T2L weights is unconfirmed.
- Exact Replicate price not found.

## Other decomposition methods

### Takeaway
Most other methods target graphic designs (LayerD, CLD, OmniPSD, design-decomposition papers) or a single foreground/background split (LayerDecomp). For natural scenes, the 2026 work is RevealLayer, PixelART and See-through (anime characters).

### Cited Findings
- **LayerD** (Sept 2025, arXiv 2509.25134): splits raster graphic designs into layers by repeatedly peeling off the top unoccluded layer. — [summary](https://fugumt.com/fugumt/paper_check/2509.25134v1_enmode)
- **LayerDecomp, "Generative Image Layer Decomposition with Visual Effects"** (arXiv 2411.17864, CVPR 2025, Adobe Research):
  - Produces a clean background plus one transparent foreground layer, and keeps shadows and reflections with the foreground. It is aimed at object removal and moving objects.
  - Trained on simulated data plus camera-captured images.
  - Source: [arXiv](https://arxiv.org/html/2411.17864v1); [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Generative_Image_Layer_Decomposition_with_Visual_Effects_CVPR_2025_paper.pdf)
- **MuLAn** (arXiv 2404.02790, CVPR 2024, Huawei Noah's Ark and Edinburgh):
  - A pipeline that splits RGB images into RGBA stacks, plus the dataset it produced: 44,862 images and 101,269 instances from COCO and LAION.
  - LayerDecomp notes that MuLAn often fails to keep visual effects (shadows, reflections) on the right layer.
  - Source: [arXiv](https://arxiv.org/pdf/2404.02790); [project](https://mulan-dataset.github.io)
- **CLD, "Controllable Layer Decomposition for Reversible Multi-Layer Image Generation"** (arXiv 2511.16249, 20 Nov 2025):
  - User bounding boxes decide the layers.
  - Built on FLUX.1-dev with two modules: LD-DiT and a Multi-Layer Conditional Adapter.
  - Trained on PrismLayersPro.
  - Authors report it beats LayerD, SAM2 and ZIM.
  - Source: [arXiv](https://arxiv.org/html/2511.16249v2); [HF papers](https://huggingface.co/papers/2511.16249)
- **OmniPSD** (arXiv 2512.09247, Dec 2025): Flux-based. Does both text→PSD and image→PSD using in-context learning. — [summary](https://arxiv.deeppaper.ai/papers/2512.09247v1)
- **DiffDecompose** (arXiv 2505.21541, CVPR 2026):
  - Decomposes semi-transparent and transparent alpha-composited layers using a DiT, with a new AlphaBlend dataset.
  - Sources conflict on whether the code is released.
  - Source: [CVPR poster](https://cvpr.thecvf.com/virtual/2026/poster/38101)
- **LayerPeeler:** a layer-wise image *vectorization* method; it is not a raster RGBA decomposer. — [summary](https://fugumt.com/fugumt/paper_check/2509.25134v1_enmode)
- **Referring Layer Decomposition** (arXiv 2602.19358, ICLR 2026):
  - Input is an image plus a prompt; output is the matching complete RGBA layer.
  - Comes with the RefLade dataset (~430K images) and a RefLayer diffusion baseline. Spatial prompts are colour-coded maps, and a parallel alpha decoder produces the layer.
  - Source: [arXiv](https://arxiv.org/html/2602.19358v1)
- **RevealLayer** (arXiv 2605.11818, 12 May 2026): image→multiple RGBA layers for *natural* images.
  - Has region-aware attention, an occlusion-guided adapter that fills in hidden content, and a sharp-alpha loss.
  - A search snippet said it takes bounding boxes, but the abstract does not mention any conditioning. Unresolved.
  - Code status is not stated.
  - Source: [arXiv](https://arxiv.org/abs/2605.11818)
- **PixelART** (arXiv 2609.22789, Sept 2026): a pixel-space rectified-flow transformer trained from scratch, with no RGBA-VAE and no text-to-image pretraining. It is a lighter-pipeline alternative. — [arXiv](https://arxiv.org/abs/2609.22789)
- **See-through** (arXiv 2602.03749, Feb 2026): single-image layer decomposition for anime characters. — [arXiv](https://arxiv.org/html/2602.03749v1)
- **Cycle-Consistent Tuning for Layered Image Decomposition** (arXiv 2602.20989, 2026). — [arXiv](https://arxiv.org/pdf/2602.20989)
- **"Does Synthetic Layered Design Data Benefit Layered Design Decomposition?"** (arXiv 2605.15167): a design-domain study with a useful related-work survey. — [arXiv](https://arxiv.org/pdf/2605.15167)
- A survey-style page names LayerD, CLD, Qwen-Image-Layered and "LaDe" as the leading methods. I did not locate a "LaDe" paper. — [EmergentMind](https://www.emergentmind.com/topics/generative-image-layer-decomposition)

### Inferences
- For illustrated game scenes with characters, Qwen-Image-Layered, RevealLayer and CLD are the most relevant. LayerD, OmniPSD and the design-decomposition paper are tuned for flat design and text.

### Gaps
- Did not verify the code or weights release, or the HF links, for RevealLayer, PixelART, Referring Layer Decomposition, CLD or OmniPSD.
- No head-to-head benchmark across all methods on natural scenes.
- "LaDe" not found.

## Decomposition by user-specified objects or prompts

### Takeaway
Qwen-Image-Layered cannot be told "tower, trees, Bob". The best fits are Referring Layer Decomposition (one prompt → one layer) and CLD (bounding boxes → layers). A practical pipeline is: detect each named object with a grounding detector, then run a box-conditioned decomposer.

### Cited Findings
- In Qwen-Image-Layered, the prompt describes the whole image and does not control what goes on each layer. — [ComfyUI Wiki](https://comfyui-wiki.com/en/tutorial/advanced/image/qwen/qwen-image-layered)
- Referring Layer Decomposition outputs the RGBA layer for a referred target (prompt or spatial prompt). — [arXiv](https://arxiv.org/html/2602.19358v1)
- CLD takes user bounding boxes and produces one layer per box. — [arXiv](https://arxiv.org/html/2511.16249v2)

### Inferences
- For "Bob": find Bob with a grounding detector to get a box, pass it to CLD or RefLayer, or crop Bob's layer out of Qwen-Image-Layered's output and run recursive decomposition on it.

### Gaps
- Whether RefLayer or CLD weights are public, and how well they handle painterly or natural scenes.

## Lighter / smaller options

### Takeaway
No small model matches Qwen-Image-Layered's quality on many layers. Cheaper routes are: FP8 or GGUF quantization, the hosted API (about $0.05 per image), LayerDecomp-style one-object-at-a-time splits, or a classic segment-then-inpaint pipeline. PixelART is the research direction toward lighter models.

### Cited Findings
- The FP8-mixed checkpoint "significantly reduces" VRAM. — [ComfyUI Wiki](https://comfyui-wiki.com/en/tutorial/advanced/image/qwen/qwen-image-layered)
- PixelART drops the VAE and text-to-image pretraining. — [arXiv](https://arxiv.org/abs/2609.22789)
- Later work criticises Qwen-Image-Layered's diffusion latency. — [arXiv 2605.15167](https://arxiv.org/pdf/2605.15167)

### Inferences
- On this project's CPU-only PC, local use is impractical. Use fal or Replicate.

### Gaps
- No parameter counts or quality numbers gathered for the smaller alternatives.
