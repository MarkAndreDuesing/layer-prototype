# Image generation: models, hosting and costs (researched 2026-11-05 → 07)

The research that came before Phase 1 (sketch-to-image with FastSD CPU) and fed the later hosting decisions. It covers where to run open image models for free or cheaply (including LMU's own GPUs), what hosted APIs cost, a fixed price-to-performance metric, which models accept a sketch, and what runs offline on the team's CPU laptop.

**[est]** = estimate, not a tested run. Prices, free tiers and Space availability change often, so re-check before relying on them. Section 9 compares the estimates here with what was measured later. The layered-model reports ([Layer based architecture models and hosting.md](2026-11-07%20Layer%20based%20architecture%20models%20and%20hosting.md) and others) cover RGBA and layer models; this report covers ordinary RGB generators.

The analysis script and its output are in [../research_notes/Image generation models hosting and costs/](../../research_notes/Image%20generation%20models%20hosting%20and%20costs/).

## 1. Summary

- **Quality bar:** Z-Image Turbo (arena Elo 936) is good enough for the prototype. Cost, accessibility and sketch input matter more than leaderboard rank.
- **Free GPU routes that recur:** Kaggle (about 30 GPU-h/week), the LMU Informatik CIP pool (RTX 2060 SUPER 8 GB via Slurm) and Hugging Face Spaces (5 GPU-min/day). Colab free only as Python code in cells.
- **Cheapest hosted:** Cloudflare Workers AI has a free daily allowance (FLUX.1 schnell only). Paid hosts run the open models for about $0.001–0.015 per image. OpenRouter adds a 5.5% top-up fee.
- **Sketch + prompt needs image conditioning:** plain img2img, a ControlNet, or reference-image editing (section 5).
- **CPU-only offline:** only the small distilled models (SD-Turbo, SDXL-Turbo, SDXS, SD 1.5 + LCM-LoRA) stay well under 3 minutes per image on the laptop. FLUX and Qwen-Image do not.
- **DeepSeek and Kimi APIs have no image generation.**

## 2. Where to run open models

Excluded: Colab's web-UI route (forbidden on the free tier), SageMaker Studio Lab (closed to new customers since 30 July 2026) and one-time trials (Lightning AI's one-time credits; Comfy Cloud, now 5 free runs then from $20/month).

| Option | Cost | GPU and limits | Fits which models | Sketch / ControlNet workflows | Main catch |
|---|---|---|---|---|---|
| **LMU Informatik CIP pool (Slurm)** | Free with a CIP account | RTX 2060 SUPER 8 GB (partition `NvidiaAll`); max 15 jobs at once | SDXL; FLUX 3 Klein (INT8, [est]); others only quantized | Yes (ComfyUI, diffusers) | Jobs pause when someone sits at the machine; weekly reboots; > 4 GB/day of downloads needs approval; a web UI over an SSH tunnel is unconfirmed (ask rbg@ifi.lmu.de) |
| **LMU Abaki servers** | Free | RTX A5000 24 GB | All models in section 6 | Yes | Bachelor/master thesis only, applied for by the supervisor |
| **LRZ AI Systems** | Free for research | H100 (94 GB) | All | Yes | Needs an LRZ account and an AI-enabled project through a chair |
| **Kaggle Notebooks** | Free, weekly | About 30 GPU-h/week (secondary sources); P100 or 2× T4 (16 GB each); 12 h sessions | SDXL, Z-Image Turbo, FLUX.2 Klein 4B, Qwen-Image-2.1 (tight) | Yes; community ComfyUI notebooks | Phone verification; models re-downloaded each session unless cached; NSFW use gets accounts banned; older GPUs are slow |
| **Google Colab (free)** | Free | Usually a T4 16 GB; limits unpublished | Same as Kaggle | Only as Python code (`diffusers`) in cells | Web UIs get terminated on the free tier; no guaranteed GPU |
| **Hugging Face Spaces (ZeroGPU)** | Free | 5 GPU-min/day with a free account (2 without) | Whatever public Spaces exist | Only if a Space offers it | A handful of images a day; Spaces break or pause |
| **Cloudflare Workers AI** | 10,000 neurons/day free, then $0.011 per 1,000 neurons | Managed | FLUX.1 schnell, SDXL base | Not verified | FLUX.1 schnell is near the bottom of the arena (Elo 740) |
| **Pay-per-image hosts** (WaveSpeed, fal, Runware, Together, Replicate) | About $0.001–0.015/image for section 6 models | Managed | Most section 6 models | On some endpoints (e.g. WaveSpeed Z-Image ControlNet $0.01) | Card and prepaid credit |
| **OpenRouter** | Provider price + 5.5% fee on credit top-ups ($0.80 minimum) | 50+ image models, `POST /api/v1/images` | FLUX.2 Klein 4B and many closed models | Reference editing on supported models | Small top-ups cost proportionally more (16% on $5); top up $15–20 |

### LMU details

1. **Who gets a CIP account:** students attending courses of the Institut für Informatik or CIS (from any university), and students of the Faculty of Mathematics, Informatics and Statistics. Apply through CipConf with the LMU Campus ID, and enable remote access there.
2. **Connect** to the gateway, then submit Slurm jobs from it. Never compute on the gateway machines:
   ```bash
   ssh <cip-account>@remote.cip.ifi.lmu.de
   ```
3. **Which CIP machines:**
   - "Gesteine A": RTX 2060 SUPER 8 GB. The useful ones.
   - "Edelsteine": GTX 1050 Ti 4 GB. SD 1.5 only.
   - "Gesteine B": AMD Radeon RX 5500 8 GB. Avoid: no CUDA. Partition `NvidiaAll` keeps jobs on NVIDIA machines.
4. **Rules that matter for image generation:**
   - More than 4 GB/day downloaded from outside the university network needs approval. One SDXL checkpoint is about 6.5 GB.
   - Keep models on the machines' local disks, not on the home-directory file servers.
   - Multi-day jobs outside Slurm must be course-related. Course-project image generation is clearly legitimate.
   - Nothing found covers running ComfyUI or Forge as a Slurm job behind an SSH tunnel. The institute's policy is to ask when in doubt: rbg@ifi.lmu.de.
5. **LRZ Compute Cloud** also has GPU nodes (2× V100 16 GB), but access is restricted and needs an LRZ project.

## 3. Hosted APIs

| Provider / model | Price per 1024 px image | Notes |
|---|---|---|
| DeepSeek API | n/a | Text-only models (V4-Pro, V4-Flash). Janus-Pro is self-host only |
| Kimi (Moonshot) API | n/a | Vision models only read images |
| Cloudflare Workers AI, FLUX.1 schnell | About $0.00063 (≈ 58 neurons per image [est]) | Free 10,000 neurons/day, so about 170 images/day by arithmetic; some guides say 20–50 |
| Together AI, FLUX.1 schnell | $0.0027 | OpenAI-style client |
| fal / Replicate, FLUX.1 schnell | $0.003 per megapixel | |
| Runware | From $0.0006 | Cheapest at volume |
| fal, Z-Image Turbo (text-to-image and image-to-image) | $0.005 per megapixel | Image-to-image has a `strength` setting |
| WaveSpeed, Z-Image Turbo | $0.005; $0.01 with ControlNet | |
| fal, FLUX 3 Image Klein | $0.009 per 1K image (generation and edit) | |
| OpenRouter, FLUX.2 Klein 4B | About $0.014 (measured by OpenRouter) | |
| OpenAI GPT Image 2 | $0.006 at low/default quality, $0.211 at high | OpenRouter's test: default billed the same as low |
| Alibaba Qwen Image 2.1 Turbo / ByteDance Seedream 5.0 Flash | $0.014–0.018 | Separate accounts and regions |
| Google Nano Banana (Gemini 2.5 Flash Image) | $0.039 ($0.0195 batch) | |

**Routers:**
- **LiteLLM** (MIT, free) gives one interface over OpenAI, Azure, Google AI Studio, Vertex AI, Bedrock, Black Forest Labs, Recraft, OpenRouter and any OpenAI-compatible endpoint. Worth it only when routing between several providers.
- **OpenRouter** passes model prices through and returns the exact billed amount in `usage.cost`. Sensible while comparing models; once one model is chosen, its provider's own key avoids the fee.

## 4. Price-to-performance metric

**Data:** Artificial Analysis text-to-image arena (blind pairwise votes, Elo), 101 models with a price, at 1024×1024 on the creator's API with default settings.

**Method:**
1. **Pareto frontier.** A model is efficient only if no other model is both cheaper and higher-rated. Anything off the frontier is beaten on both axes.
2. **Marginal cost along the frontier.** Rule of thumb: +50 Elo means the better model wins about 57% of blind votes. Scores carry about ±10–12 Elo of uncertainty, so steps under about 25 Elo are noise.
3. **Rejected:** Elo per dollar (it just rewards the cheapest; FLUX.1 schnell comes second), and residuals from a fitted price–quality line (price explains only 10% of Elo variance, R² = 0.10, so the residual mostly measures model age).

| Model | Elo | $/1,000 | Step from previous | Verdict |
|---|---|---|---|---|
| P-Image-Ideogram (Very Low) | 909 | 3.0 | — | Cheapest floor |
| FLUX.2 [dev] Flash | 988 | 5.0 | +79 for +$2 | Good step |
| FLUX.2 [dev] Turbo | 995 | 8.0 | +7 for +$3 | Within noise |
| FLUX 3 Image Klein | 1043 | 9.0 | +48 for +$1 | Good step; open weights |
| Muse Image | 1127 | 10.0 | +84 for +$1 | Knee of the curve |
| Nano Banana 2.1 | 1176 | 33.6 | +49 for +$23.6 (3.4× price) | Diminishing returns |
| GPT Image 2.5 Sunburst (max) | 1223 | 210.7 | +47 for +$177 (6.3× price) | Quality at any cost |

Reproduce with [price_performance.py](../../research_notes/Image%20generation%20models%20hosting%20and%20costs/price_performance.py) (output: [price_performance_output.txt](../../research_notes/Image%20generation%20models%20hosting%20and%20costs/price_performance_output.txt)).

**Limits:** arena Elo reflects general prompts, not sketches; prices are defaults at 1024 px; cheaper quality tiers are not benchmarked.

## 5. Sketch + prompt: three ways in

Text-to-image models take only a prompt, so a sketch needs one of these:

| Method | How the sketch is used | Behaviour |
|---|---|---|
| **Plain image-to-image** | The sketch is noised, then redrawn. One "strength" setting | Trades faithfulness against realism. Sparse black lines on white tend to come out as a pale sketch or get ignored |
| **ControlNet (scribble / lineart)** | Lines constrain structure through the whole generation; the prompt decides appearance | The established, most predictable method |
| **Reference-image editing** | The sketch is a reference image with an instruction ("turn this sketch into…") | Looser composition, often understands rough drawings better. Regenerates from scratch rather than partially redrawing |

## 6. Models: self-hostable or cheaply hosted

| Model / combination | Sketch method | Kaggle 16 GB / LMU 8 GB | Cheapest hosted | License | Try it |
|---|---|---|---|---|---|
| **SDXL + scribble ControlNet** | ControlNet (any line type and width); img2img; inpainting | Yes / yes | About $0.0013/image (base SDXL, fal) | CreativeML Open RAIL++-M; ControlNet Apache-2.0 | [Scribble SDXL Flash Space](https://huggingface.co/spaces/linoyts/scribble-sdxl-flash) · [ControlNet](https://huggingface.co/xinsir/controlnet-scribble-sdxl-1.0) · [Base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
| **Z-Image Turbo (6B) + union ControlNet** | ControlNet with scribble (since the 2601 release); img2img (`strength`, default 0.6); inpainting | Yes (developers: fits within 16 GB) / quantized only [est] | $0.005/image; $0.01 with ControlNet (WaveSpeed) | Apache-2.0 | [Official Space](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo) · [mrfakename Space](https://huggingface.co/spaces/mrfakename/Z-Image-Turbo) · [Model](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) · [ControlNet](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1) |
| **FLUX.2 Klein 4B** | Reference images / editing | Yes (card: about 13 GB) / borderline with FP8 [est] | About $0.014 (OpenRouter); $0.0006–0.002 (Runware) | Apache-2.0 | [Official Space](https://huggingface.co/spaces/black-forest-labs/FLUX.2-klein-4B) · [Model](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) |
| **FLUX 3 Image Klein (5.6B)** | Reference editing (up to 10 references, bounding boxes); sketch input untested | See section 10 / INT8 build runs on 8 GB RTX cards (BFL blog) | $0.009 per 1K image (fal) | FLUX Kommunity v1.0 (free under $5M revenue), gated | [Model](https://huggingface.co/black-forest-labs/flux-3-image-klein) · [fal edit](https://fal.ai/models/blackforestlabs/flux-3/klein/edit-image) |
| **Qwen-Image-2.1 + union ControlNet** | ControlNet with scribble and lineart; inpainting | Tight, with offloading / no | No official API | Qwen Research License (non-commercial); ControlNet research license | [Model](https://huggingface.co/Qwen/Qwen-Image-2.1) · [ControlNet](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union) |
| FLUX.1 schnell (drafts only) | None verified | n/a | Cloudflare free daily | Apache-2.0 | [Model](https://huggingface.co/black-forest-labs/FLUX.1-schnell) |

Dropped: Fibo 2.0 (CC BY-NC 4.0, about 16 GB for weights alone) and Ideogram 4.0 (non-commercial, needs a 24 GB GPU or heavy 4-bit offloading). Both are beaten by FLUX 3 Klein on price and quality, and neither has a sketch ControlNet.

## 7. CPU-only and offline on the team laptop

**Target:** Intel Core 7 150U (15 W, 10 cores / 12 threads, AVX2), 15.7 GB DDR5 on a single module, Intel Iris Xe integrated graphics (96 EUs, shared memory), no dedicated GPU, no NPU. Cinebench R23 multi-core: 8,883, against 19,714 for the desktop i7-12700 that FastSD CPU's published timings use, so **about 2.2× slower** before throttling.

**Tools** (both MIT, both offline once models are downloaded):
- [FastSD CPU](https://github.com/rupeshs/fastsdcpu): web UI, img2img, ControlNet (LCM-LoRA mode only), OpenVINO, optional Intel iGPU (`DEVICE=GPU`; reports are mixed and the first run compiles).
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp): C++ engine for CPU and Vulkan/OpenCL GPUs; SD 1.x/2.x/XL, Turbo models, FLUX.2 Klein, Qwen Image, Z-Image; ControlNet for SD 1.5 only. One Iris Xe data point: SD 1.5 at 20 steps took 321 s and needed `--diffusion-fa`.

| Model | Published on i7-12700 (OpenVINO, 512 px) | Laptop estimate at research time [est] | RAM | Sketch support | Link |
|---|---|---|---|---|---|
| SDXS-512-0.9 | 0.82 s, 1 step | 2–3 s | Low | img2img | [Model](https://huggingface.co/IDKiro/sdxs-512-0.9) |
| SD-Turbo | 1.7 s | 4–5 s | About 4 GB | img2img | [Model](https://huggingface.co/stabilityai/sd-turbo) |
| SDXL-Turbo | 2.5 s | 6–8 s | 8 GB minimum | img2img | [Model](https://huggingface.co/stabilityai/sdxl-turbo) |
| Hyper-SD SDXL | 6.3 s at 768 px | 15–20 s | 8 GB | img2img | [Model](https://huggingface.co/ByteDance/Hyper-SD) |
| SD 1.5 + LCM-LoRA + ControlNet | Not published | 20–60 s | 4 GB | ControlNet (LineArt, SoftEdge, scribble) | [Base](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) · [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) · [Scribble](https://huggingface.co/lllyasviel/control_v11p_sd15_scribble) · [LineArt](https://huggingface.co/lllyasviel/control_v11p_sd15_lineart) |
| Iris 3B | About 90 s, 1024 px, 4 steps | 3–4 min | 8.4 GB | Not documented | — |
| FLUX.2 Klein 4B | Not published | Minutes | 8 GB | Reference | [Space](https://huggingface.co/spaces/black-forest-labs/FLUX.2-klein-4B) |
| Z-Image Turbo | Not published | Several minutes | Tight on 16 GB | img2img | [Demo](https://huggingface.co/spaces/mrfakename/Z-Image-Turbo) |
| FLUX.1 schnell | About 4.5 min | — | About 30 GB | — | Ruled out (RAM) |

**Licences:** SD 1.5 and the ControlNets are open-weights with use restrictions. The Turbo models use Stability's more restrictive licence (FastSD CPU's README: research purposes). The strictly open-source (Apache-2.0) options are FLUX.2 Klein 4B and Z-Image Turbo, which are the ones unlikely to finish within 3 minutes on this laptop.

### Sketch methods on the laptop, as compared at research time

| | SD 1.5 + LCM-LoRA + ControlNet | SD-Turbo img2img | SDXL-Turbo img2img | SDXL-Turbo + scribble ControlNet |
|---|---|---|---|---|
| Composition fidelity | High | Low–medium | Low–medium | High |
| Rough, thin lines | Good | Poor | Poor | Good |
| Image quality | Lowest | About SD 1.5 | Best | Best |
| Time per 512 px image [est] | 20–60 s | 4–5 s | 6–8 s | Unknown; may exceed 3 min |
| RAM | About 4 GB | About 4 GB | 8 GB | Likely 10 GB+ |
| Built into FastSD CPU | Yes | Yes | Yes | No (ComfyUI or diffusers) |

Ranking of the three Turbo options at research time (1 = best): SDXL-Turbo img2img first overall on the laptop, SD-Turbo second, SDXL-Turbo + ControlNet last (first on a GPU). Phase 1 then tested the first three columns; see section 9.

## 8. Hosted test links: same input, all four candidate models

| Model | Text-to-image | Image input | Cost |
|---|---|---|---|
| **Z-Image Turbo** | [Official Space](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo) · [mrfakename Space](https://huggingface.co/spaces/mrfakename/Z-Image-Turbo) · [fal](https://fal.ai/models/fal-ai/z-image/turbo) | No free Space found; [fal image-to-image](https://fal.ai/models/fal-ai/z-image/turbo/image-to-image) | Spaces free; fal $0.005/MP |
| **FLUX.2 Klein 4B** | [Official Space](https://huggingface.co/spaces/black-forest-labs/FLUX.2-klein-4B) | Same Space (upload images to edit); [Runware](https://runware.ai/models/flux2-klein-4b) | Free (HF quota); Runware $0.0006–0.002 |
| **FLUX 3 Image Klein** | [Community Space](https://huggingface.co/spaces/zgj119/flux-3-image-klein-demo) · [fal](https://fal.ai/models/blackforestlabs/flux-3/klein/text-to-image) | Same Space (image-edit mode) · [fal edit](https://fal.ai/models/blackforestlabs/flux-3/klein/edit-image) | Space free; fal $0.009 per 1K image |
| **Qwen-Image-2.1** | [hugging-apps/qwen-image-2-1](https://huggingface.co/spaces/hugging-apps/qwen-image-2-1) · [Turbo](https://huggingface.co/spaces/hugging-apps/qwen-qwen-image-2-1-turbo) | Same Spaces (up to 10 references) | Free (HF quota) |

The Z-Image and FLUX.2 Klein Spaces are run by the model makers. The Qwen Spaces were built by a Hugging Face staff member and are linked from the model's discussion page. The FLUX 3 Klein Space is a community upload that does not state its model version; use fal for a guaranteed official run. Live status of the Spaces was not confirmed on 2026-11-07.

**Test protocol:**
1. Log in to Hugging Face first (5 GPU-min/day, roughly 10–30 images across these models [est]).
2. Use the same sketch, the same prompt, 1024×1024 and a fixed seed where offered.
3. Z-Image on fal does classic img2img: try strength 0.6, 0.75 and 0.9.
4. The other three do reference editing: phrase the prompt as an instruction, for example "Turn this pencil sketch into a [style] image of [scene], keeping the composition."

## 9. What later testing showed

Phase 1 ([../docs/findings-log.md](../log.md), [../experiments/CONFIRMATION_RESULTS.md](../../experiments/CONFIRMATION_RESULTS.md)) measured three of the section 7 options on the laptop.

| Item | Research-time estimate | Measured |
|---|---|---|
| SD-Turbo, 512 px, 4 steps | 4–5 s | About 20 s (about 25 s offline) |
| SDXL-Turbo, 512 px, 4 steps | 6–8 s, 8 GB RAM | About 58 s (about 70 s offline), about 11 GB RAM |
| SD 1.5 + LCM-LoRA + ControlNet | Best for sketch fidelity | Followed the layout best, but slower and rougher; kept 2/12 with a sketch, 0/12 text-only. **Dropped** |
| Best overall on the laptop | SDXL-Turbo img2img | SDXL-Turbo img2img (Strength 0.8, 4 steps, Guidance 1): all 8 favourites |

The speed estimates were about 4–10× too optimistic. They scaled FastSD CPU's published OpenVINO + tiny-decoder timings, while the tested setup used LCM mode with the full decoder (see [../docs/fastsd-cpu/USAGE.md](../guides/fastsd-cpu/USAGE.md)). OpenVINO and the tiny decoder were not measured on this laptop. Options to close the gap, with a test order: [research_cpu_speedups.md](../research/cpu-speedups.md).

## 10. Differences from the other reports

- **FLUX 3 Image Klein on a T4:** section 6 marked it as fitting Kaggle. [Layer based architecture models and hosting.md](2026-11-07%20Layer%20based%20architecture%20models%20and%20hosting.md) and [Kaggle migration plan.md](2026-11-07%20Kaggle%20migration%20plan.md) call it a poor or borderline fit, citing the card's 25–31 GB peak at 1024 px, and the T4 has no FP8 acceleration. The INT8 build plus text-encoder offload is the only route; treat the fit as unverified until measured.
- **Z-Image Turbo licence and VRAM:** listed as "not researched" in [Layer based architecture models and hosting.md](2026-11-07%20Layer%20based%20architecture%20models%20and%20hosting.md) §7. This research found Apache-2.0 and "fits within 16 GB" (developers' model card).
- **Lightning AI:** this report treats its free GPU hours as a one-time grant (5 credits at signup, 25 more after adding a card, expiring after 12 months), so it is excluded as a recurring option.

## 11. Open items

- Real speed of SD-Turbo / SDXL-Turbo on the laptop with OpenVINO and the tiny decoder, and on the Iris Xe iGPU.
- Whether Z-Image Turbo (quantized) runs on the LMU RTX 2060 SUPER 8 GB, and how fast.
- Whether the LMU CIP pool allows a web UI behind an SSH tunnel (ask rbg@ifi.lmu.de).
- Sketch-following of FLUX 3 Klein and FLUX.2 Klein reference editing on the Phase 1 sketches (section 8 protocol).
- Whether Comfy Cloud's free tier is 5 runs or 400 credits a month: its own pages disagreed on 2026-11-05.

## Appendix: free browser generators (no setup)

| Service | Free offer (2026-11-05) | Notes |
|---|---|---|
| Google Gemini | Most generous free tier | Good photorealism |
| ChatGPT | Limited free generations | Tight caps |
| Microsoft Copilot / Bing Image Creator | Free | Personal, non-commercial use only |
| Leonardo.ai | 150 tokens/day | Commercial use allowed, no card |
| Ideogram | About 10 images/day | Best text rendering |
| Adobe Firefly | Limited monthly credits | Free outputs are watermarked |
| Midjourney | Not free (from $10/month) | niji・journey app: 20 free generations, anime style |

## Sources

- Arena and costs: [Artificial Analysis text-to-image leaderboard](https://artificialanalysis.ai/text-to-image/arena/leaderboard-text) · [OpenRouter: image models compared](https://openrouter.ai/blog/insights/image-generation-models-compared/) · [LiteLLM image generation](https://docs.litellm.ai/docs/image_generation) · [Z-Image Turbo pricing (WaveSpeed)](https://wavespeed.ai/blog/posts/blog-z-image-turbo-pricing) · [fal Z-Image Turbo](https://fal.ai/models/fal-ai/z-image/turbo) · [fal Z-Image Turbo image-to-image](https://fal.ai/models/fal-ai/z-image/turbo/image-to-image) · [fal FLUX.3 Klein text-to-image](https://fal.ai/models/blackforestlabs/flux-3/klein/text-to-image) · [fal FLUX.3 Klein edit](https://fal.ai/models/blackforestlabs/flux-3/klein/edit-image) · [Runware FLUX.2 Klein 4B](https://runware.ai/models/flux2-klein-4b)
- Hosting: [Lightning AI pricing](https://lightning.ai/pricing) · [Comfy Cloud pricing](https://comfy.org/pricing) · [Colab FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html) · [Hugging Face ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)
- LMU: [IfI compute resources](https://www.rz.ifi.lmu.de/infos/compute_en.html) · [IfI Slurm](https://www.rz.ifi.lmu.de/infos/slurm_en.html) · [IfI compute job rules](https://www.rz.ifi.lmu.de/infos/rechenjobs_en.html) · [IfI accounts](https://www.rz.ifi.lmu.de/infos/kennung_de.html)
- Models: [FLUX 3 Image Klein (BFL)](https://bfl.ai/blog/flux-3-image-klein) · [FLUX.3 Klein in Diffusers](https://huggingface.co/blog/flux-3) · [FLUX 3 Klein in ComfyUI](https://docs.comfy.org/tutorials/flux/flux-3-klein) · [FLUX.2 Klein (BFL)](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence) · [Qwen-Image-2.1 (Unsloth)](https://unsloth.ai/docs/models/qwen-image-2.1) · [Qwen-Image-2.1 demo discussion](https://huggingface.co/Qwen/Qwen-Image-2.1/discussions/25) · [Z-Image Turbo card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) · [Z-Image Diffusers pipelines](https://huggingface.co/docs/diffusers/api/pipelines/z_image) · [Fibo 2.0 (OrcaRouter)](https://www.orcarouter.ai/blog/fibo-2-0-open-weights-image-editing-leader) · [Ideogram 4.0 open weights (Techsy)](https://techsy.io/da/blog/ideogram-4-open-weight-24gb-gpu) · [FluxKleinControlNetImg2Img node](https://comfy.icu/node/FluxKleinControlNetImg2Img)
- CPU: [Image generation without a GPU (LocalAIMaster)](https://localaimaster.com/blog/image-generation-without-gpu) · [FastSD CPU](https://github.com/rupeshs/fastsdcpu) · [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) · [stable-diffusion.cpp issue #1449](https://github.com/leejet/stable-diffusion.cpp/issues/1449) · [OpenVINO issue #26391](https://github.com/openvinotoolkit/openvino/issues/26391) · [Intel Core 7 150U](https://nanoreview.info/en/cpu/intel-core-7-150u) · [Intel Core i7-12700](https://nanoreview.info/en/cpu/intel-core-i7-12700)
