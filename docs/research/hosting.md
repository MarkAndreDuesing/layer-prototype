# Hosting, compute, costs and base models

> **Role: research synthesis (high).** Where things run, what they cost, how to get university GPUs, the TPU route, and which base text-to-image models / APIs exist. Merged from six reports (kept as snapshots in [../archive/](../archive/MOVED.md)); tables below are copied from them where they were not superseded. [U] = unverified, [est] = estimate.

## 1. What we use now
- **Kaggle (free):** 2× T4 16 GB (or 1× P100), about 30 GPU-h/week, sessions up to 12 h, about 29 GB RAM, internet after phone verification. **Max 2 batch GPU sessions at once.** No native bf16/fp8 on T4 → [../concepts/fp16-on-t4.md](../concepts/fp16-on-t4.md). Terms: personal, non-commercial use; no serving farms. Workflow: [../guides/kaggle.md](../guides/kaggle.md).
- **Laptop CPU (Core 7 150U):** LLM (Ollama), SDXL-Turbo (about 60–160 s per image), CPU-tested modules. Speed-ups: [cpu-speedups.md](cpu-speedups.md).

## 2. Free and cheap options (from the 11-05 research)
| Option | Cost | GPU | Catch |
|---|---|---|---|
| Kaggle notebooks | free | 2× T4 / P100 | weekly quota, 2 batch sessions |
| LMU CIP pool (Slurm) | free | RTX 2060 SUPER 8 GB | jobs pause when someone uses the machine |
| LMU abakus servers | free | stronger cards (A5000 24 GB per earlier research) | request by mail to the RBG with a good reason (thesis, publication, project) |
| LRZ AI Systems | free for research | P100 / V100 / A100 / H100 | needs an AI-enabled LRZ project via a chair |
| Google Colab free | free | usually T4 | web UIs / serving forbidden; code cells only |
| HF Spaces ZeroGPU | free | shared | about 5 GPU-min/day |
| Modal | $30/month credit | T4/L4/A10 | clean live HTTPS endpoint |
| Pay-per-image hosts (fal, WaveSpeed, Replicate, Together, Runware) | about $0.001–0.015 per image for small models | managed | card + prepaid credit |
| Comfy Cloud | from $20/month | RTX PRO 6000 (96 GB) | proprietary paid UI (the team prefers to avoid it); runs Qwen-Image-Layered in seconds |

## 3. Qwen-Image-Layered: speed and API cost
- Official settings: 640 px, 50 steps, CFG 4. No published per-GPU timings; no Lightning for Layered.
- On our free 2× T4: 35 min at defaults, **27–35 s** at 384 px / 4 layers / 4 steps / CFG 1 ([../models/qwen-image-layered.md](../models/qwen-image-layered.md)).
- Hosted: fal about **$0.05 per image** (28 steps default); WaveSpeed $0.025 base (scales with layers); 302.ai about $0.05. €100 ≈ 2,000 runs: suits final demos more than prototyping.

### Paid APIs vs university GPUs (for about 2,000 Qwen-Image-Layered runs)

| Option | Cost for 2,000 runs | Latency per run | Setup effort | Terms |
|---|---|---|---|---|
| fal.ai (~$0.05/run) [U price] | ~$100 | ~10 to 60 s [U] | Minutes: API key and Python client | Commercial ToS; your images go to a third party. Fine for a personal or non-commercial project. |
| WaveSpeed ($0.025/layer) [U price] | 4 layers ≈ $0.10/run, so ~$200 | similar [U] | Minutes | Same as fal.ai |
| RBG abakus | Free | Queue wait, then ~2 to 4 min if A100-class [U] | Email the RBG with an advisor confirmation; days | Institute rules; academic use only |
| LRZ AI Systems | Free | Queue (can be hours), then ~1.5 to 4 min | Needs a PI's AI-enabled project; days to weeks. Enroot container build. | Research-focus requirement |

Bottom line: for a course prototype with about 2,000 runs, a paid API costs $100 to $200 and works within the hour. The university GPUs are free but depend on a supervisor and take setup time. They become worth it for a recurring or larger workload, or for fine-tuning.

### Speed estimate per GPU (FLOPs-based, 640 px, 4 layers, CFG, 50 steps)

Assumptions: 6.6e14 FLOPs per step (640 px, 4 layers, with CFG) and 50 steps, giving 3.3e16 FLOPs per run.

| GPU | Peak | Assumed effective | s/step | s/run (50 steps) | 2,000 runs |
|---|---|---|---|---|---|
| T4 (measured) | 65 fp16 | 19 TFLOPS | ~35 | ~29 min | ~970 h |
| A100 bf16 | 312 | ~150 (about 50% MFU) | ~4.4 | ~3.7 min | ~122 h |
| H100 bf16 | ~990 | ~400 (about 40% MFU) | ~1.7 | ~1.4 min | ~46 h |

The T4 is also too small for bf16 (16 GB, compute capability 7.5), so on a T4 you need 4-bit plus offloading, which makes it slower still. An A100 40 GB is tight for bf16, so an 80 GB A100 or an H100 is preferable. Inference with fewer steps (for example 20 to 30) cuts these times proportionally.

## 4. LMU and LRZ access

| Route | Hardware | Access | Notes |
|---|---|---|---|
| RBG Slurm (institute) | CIP pool, RTX 2060 SUPER 8 GB (partition `NvidiaAll`) | all students and staff, no registration; 15 running / 30 submitted jobs | jobs pause when someone uses the machine |
| RBG abakus servers | stronger cards (A5000 24 GB per earlier research [U]) | mail rbg@ifi.lmu.de with a good reason + advisor confirmation; `srun -q abaki -p Abaki …`; default runtime 1 day | GPU models not listed publicly |
| LRZ AI Systems | P100 / V100 / A100 / H100 | valid LRZ account + AI-enabled project with research focus; master user adds you to `<projectID>-ai-c`; Slurm + Enroot | projects renewed yearly; BayernKI as alternative route |
| MCML [U] | DGX / A100 / H100 at LRZ | via an MCML PI | not verified on a public page |
| NHR@FAU (Alex) [U] | A100 / A40 | free NHR-Starter projects via a PI | better fit than Gauss for small jobs |
| Gauss Centre [U] | JUWELS, SuperMUC-NG | proposals for large projects | overkill |

### How to get access, step by step

1. **Try the RBG Slurm cluster now (no registration).** Log in to a CIP machine via SSH (see https://www.rz.ifi.lmu.de/) and run `sinfo -o "%P %G %N"` to see partitions and GPUs.
2. **Ask your course supervisor for a short confirmation** that this is an official course or research project.
3. **Email rbg@ifi.lmu.de asking for abakus access.** Include the advisor confirmation, the model's needs (40 GB in bf16 with compute capability 8.0 or higher, or 13 GB in 4-bit) and the expected volume (about 2,000 runs, about 100 GPU-hours).
4. **Test a job** with `srun -q abaki -p Abaki --gres=gpu:1 --pty bash`, then `nvidia-smi`.
5. **If abakus is insufficient, go to LRZ.** Ask your supervisor or chair whether they have an AI-enabled LRZ project. If they do, the Master User adds you to `<projectID>-ai-c` in the IDM portal.
6. **If they have no such project,** the supervisor (as PI) applies with the LRZ project PDF, writing "AI Systems" in the remarks, then files a Servicedesk request for "AI Systems Project Activation/Update". The BayernKI route is the alternative.
7. **On LRZ,** build an Enroot image from a PyTorch NGC or Docker image (`enroot import docker://nvcr.io#nvidia/pytorch:<tag>`). Store the model weights on project storage, not in your home directory.
8. **Run batch jobs** with `sbatch` on an A100 or H100 partition. Use an interactive `salloc` session only for debugging.
9. **As a fallback for deadlines,** keep a fal.ai or WaveSpeed key with a small budget cap.

### Questions to ask (RBG, supervisor, LRZ)

1. Which GPU models and how much VRAM do the abakus servers have (and the CIP machines, if they have GPUs)? Is anything 40 GB or larger, with compute capability 8.0 or higher?
2. Do CUDA drivers work on these Slurm nodes now (the 2024 news said there were no drivers in a test environment)? Which CUDA version?
3. Does a course project count as a "good reason" for abakus access? Who has to confirm it?
4. Are containers (Apptainer/Singularity, Enroot) available? If not, can we pip-install PyTorch into our home directory?
5. What are the home and scratch quotas? The weights are about 40 to 60 GB.
6. Is outbound internet access available on compute nodes, for downloading weights from Hugging Face?
7. Does our chair or supervisor have an AI-enabled LRZ project we can join? Who is its Master User?
8. On LRZ, what are the GPU job time limits and fair-share policy for small projects? Which partition do you recommend for a 20B diffusion inference job?
9. Does MCML or the chair have its own GPU pool that course students can use?
10. Are there rules against sending course data to commercial APIs such as fal.ai or WaveSpeed?

<a id="tpu"></a>
## 5. TPU
- Kaggle offers TPU v5e-8 (8 chips × 16 GB HBM, native bf16) on a separate 20 h/week quota. Template: diffusers' PyTorch/XLA FLUX SPMD example (tested on v5e-8). Estimated about 40 s per Qwen split at about 30 steps; the "2–5 days" effort figure was an unverified guess.
- **Probe ([`../../prototype/tools/make_tpu_probe.py`](../../prototype/tools/make_tpu_probe.py)), 2026-11-09:** three CLI batch runs never got a TPU (Kaggle stored `machine_shape: TpuV5E8`, the run used the standard CPU image). The image ships torch 2.11 (CPU) + JAX 0.11.1 and no torch_xla; pip's newest torch_xla is 2.9, so torch must be pinned to 2.9. **2026-11-10 interactive attempt:** the editor offers "TPU v5e-8" (TPU quota shown as 00:00 / 20 hrs) but the session sat in a queue: "TPUs are popular right now. You are #1510 in the queue". No TPU was allocated, so no probe stage ran. **Next:** retry at a quieter time (the queue position is shown in the editor), or skip the TPU route.

### TPU software routes (researched, not measured)

| Route | Status | Notes |
|---|---|---|
| PyTorch/XLA + diffusers SPMD | template exists (diffusers PR #13474, FLUX on v5e-8, ~1.76 s steady state [U]) | shard params with `xs.mark_sharding`; text encoders on CPU; ~195 s first compile; recompile issues reported |
| TorchTPU (Google, open-sourced 2026-10-20 [U]) | `pipe.to("tpu")`; 96.5% of diffusers tests pass [U] | not confirmed on Kaggle; Qwen-Image and multi-chip sharding unknown |
| JAX / MaxDiffusion | SD, SDXL, Flux, Wan on v5e/v6e | no Qwen-Image port; a JAX port would be a large effort |

**Blockers:** memory fits (40 GB / 8 chips ≈ 5 GB per chip on v5e); several minutes of compile per new shape (fix resolution, layer count and steps; pad text); torch_xla lags torch (seen: torch 2.11 vs torch_xla 2.9). **Speed estimate:** v5e-8 about 50–70 s at 50 steps or 30–40 s at 30 steps; v6e-8 about 12–20 s at 50 steps [est].

## 6. Base text-to-image models and image APIs

Used for the "whole scene first" step (route A in [../concepts/pipeline.md](../concepts/pipeline.md)) and the earlier sketch-to-image phase.

### Hosted image APIs (prices per 1024 px image, 2026-11-05)

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

### Price-to-performance (arena Elo vs price)

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

### Sketch + prompt: three ways in

Text-to-image models take only a prompt, so a sketch needs one of these:

| Method | How the sketch is used | Behaviour |
|---|---|---|
| **Plain image-to-image** | The sketch is noised, then redrawn. One "strength" setting | Trades faithfulness against realism. Sparse black lines on white tend to come out as a pale sketch or get ignored |
| **ControlNet (scribble / lineart)** | Lines constrain structure through the whole generation; the prompt decides appearance | The established, most predictable method |
| **Reference-image editing** | The sketch is a reference image with an instruction ("turn this sketch into…") | Looser composition, often understands rough drawings better. Regenerates from scratch rather than partially redrawing |

### Self-hostable or cheaply hosted models

| Model / combination | Sketch method | Kaggle 16 GB / LMU 8 GB | Cheapest hosted | License | Try it |
|---|---|---|---|---|---|
| **SDXL + scribble ControlNet** | ControlNet (any line type and width); img2img; inpainting | Yes / yes | About $0.0013/image (base SDXL, fal) | CreativeML Open RAIL++-M; ControlNet Apache-2.0 | [Scribble SDXL Flash Space](https://huggingface.co/spaces/linoyts/scribble-sdxl-flash) · [ControlNet](https://huggingface.co/xinsir/controlnet-scribble-sdxl-1.0) · [Base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
| **Z-Image Turbo (6B) + union ControlNet** | ControlNet with scribble (since the 2601 release); img2img (`strength`, default 0.6); inpainting | Yes (developers: fits within 16 GB) / quantized only [est] | $0.005/image; $0.01 with ControlNet (WaveSpeed) | Apache-2.0 | [Official Space](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo) · [mrfakename Space](https://huggingface.co/spaces/mrfakename/Z-Image-Turbo) · [Model](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) · [ControlNet](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1) |
| **FLUX.2 Klein 4B** | Reference images / editing | Yes (card: about 13 GB) / borderline with FP8 [est] | About $0.014 (OpenRouter); $0.0006–0.002 (Runware) | Apache-2.0 | [Official Space](https://huggingface.co/spaces/black-forest-labs/FLUX.2-klein-4B) · [Model](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) |
| **FLUX 3 Image Klein (5.6B)** | Reference editing (up to 10 references, bounding boxes); sketch input untested | INT8 build runs on 8 GB RTX cards (BFL blog) | $0.009 per 1K image (fal) | FLUX Kommunity v1.0 (free under $5M revenue), gated | [Model](https://huggingface.co/black-forest-labs/flux-3-image-klein) · [fal edit](https://fal.ai/models/blackforestlabs/flux-3/klein/edit-image) |
| **Qwen-Image-2.1 + union ControlNet** | ControlNet with scribble and lineart; inpainting | Tight, with offloading / no | No official API | Qwen Research License (non-commercial); ControlNet research license | [Model](https://huggingface.co/Qwen/Qwen-Image-2.1) · [ControlNet](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union) |
| FLUX.1 schnell (drafts only) | None verified | n/a | Cloudflare free daily | Apache-2.0 | [Model](https://huggingface.co/black-forest-labs/FLUX.1-schnell) |

Dropped: Fibo 2.0 (CC BY-NC 4.0, about 16 GB for weights alone) and Ideogram 4.0 (non-commercial, needs a 24 GB GPU or heavy 4-bit offloading). Both are beaten by FLUX 3 Klein on price and quality, and neither has a sketch ControlNet.

### Hosted test links (same input, four candidate models)

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

### Free browser generators (no setup)

| Service | Free offer (2026-11-05) | Notes |
|---|---|---|
| Google Gemini | Most generous free tier | Good photorealism |
| ChatGPT | Limited free generations | Tight caps |
| Microsoft Copilot / Bing Image Creator | Free | Personal, non-commercial use only |
| Leonardo.ai | 150 tokens/day | Commercial use allowed, no card |
| Ideogram | About 10 images/day | Best text rendering |
| Adobe Firefly | Limited monthly credits | Free outputs are watermarked |
| Midjourney | Not free (from $10/month) | niji・journey app: 20 free generations, anime style |

## 7. Matting, segmentation and helper tools

VRAM column: "estimate" means it was inferred from parameter counts and file sizes. None of these were run on a T4.

| Model | License | Free-hostable where | VRAM fit on T4 16 GB | Layer capability | HF link |
|---|---|---|---|---|---|
| LayerDiffuse SD1.5 / SDXL | Code Apache-2.0; weights CreativeML OpenRAIL-M; diffusers port MIT | Kaggle, Colab (notebook use), Modal | Yes. SD1.5 is about 4 GB and SDXL about 7 GB fp16; SDXL has a `--cpu-offload` flag (estimate) | Native text-to-RGBA foreground; SD1.5 also does joint fg+bg+blend and fg-given-bg | [layerdiffusion-v1](https://huggingface.co/LayerDiffusion/layerdiffusion-v1) |
| Qwen-Image-2.1 | Qwen Research License (non-commercial) | Kaggle, Colab, Modal | Likely, with Q8 GGUF denoiser (7.64 GB) plus Q4 text encoder (about 5 GB), or with offload (estimate) | Native text-to-RGBA, transparent layer editing, subject extraction | [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) |
| FLUX.2 Klein 4B | Apache-2.0 | Kaggle, Colab, Modal | Yes. The card says "as little as 13GB" | None; needs matting afterwards | [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) |
| FLUX 3 Image Klein (5.6B) | FLUX Kommunity License v1.0 (free commercial use under $5M revenue); gated | Kaggle, Modal | Hard. Card peak is about 25–31 GB at 1024 px; would need INT8 plus encoder offload, and T4 has no FP8/NVFP4 acceleration (estimate) | None; needs matting afterwards | [flux-3-image-klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) |
| Qwen-Image-Layered (20B) | Apache-2.0 | Kaggle only, maybe; Colab free fails on RAM | No at BF16 (about 40 GB). A 4-bit SDNQ/GGUF with offload is unproven | Image-to-layers only; no text-to-layers weights found | [Qwen/Qwen-Image-Layered](https://huggingface.co/Qwen/Qwen-Image-Layered) |
| LayerDiffuse-Flux | Repo shows no LICENSE; HF labels the weights afl-3.0; inherits FLUX.1-dev non-commercial | Kaggle (about 29 GB RAM) | Only with heavy quantization or offload (estimate) | Native RGBA on FLUX.1-dev | [RedAIGC/Flux-version-LayerDiffuse](https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse) |
| CLD | MIT code; FLUX.1-dev non-commercial base | Kaggle | Only with quantization or offload (estimate) | Layer decomposition | [thuteam/CLD](https://huggingface.co/thuteam/CLD) |
| LayerD | Apache-2.0 code (repo archived) | Anywhere, including CPU and ZeroGPU | Yes; BiRefNet-based (estimate) | Graphic-design image-to-layers | [cyberagent/layerd-birefnet](https://huggingface.co/cyberagent/layerd-birefnet) |
| BiRefNet | MIT | Anywhere, including CPU and ZeroGPU | Yes (0.2B) | Cut-out matting | [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) |
| BEN2 base | MIT (the Full model is API-only) | Anywhere, including CPU | Yes (94.6M) | Cut-out matting | [PramaLLC/BEN2](https://huggingface.co/PramaLLC/BEN2) |
| SAM 2.1 | Apache-2.0, not gated | Anywhere | Yes (39M–224M) | Prompted segmentation | [GitHub](https://github.com/facebookresearch/sam2) |
| Florence-2-large | MIT | Anywhere | Yes (0.77B) | Text grounding and detection to drive SAM | [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) |
| LaMa | Apache-2.0 | Anywhere, including CPU | Yes | Background fill behind removed objects | [smartywu/big-lama](https://huggingface.co/smartywu/big-lama) |
| RMBG-2.0 | CC BY-NC 4.0, gated | Anywhere | Yes | Matting (non-commercial) | [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) |
| SAM 3 | Custom Meta licence, gated | ZeroGPU, Kaggle | Likely (about 3.6 GB F32 weights) | Text-prompted segmentation | [facebook/sam3](https://huggingface.co/facebook/sam3) |

### Ruled out at the time (11-07) and why

- **ART** (Microsoft): weights and inference code removed 23 Jul 2025; HF repo returns 401. ART+ checkpoints are on Google Drive with no license.
- **OmniPSD**: code is CC BY-NC-SA 4.0, no weights, FLUX.1-dev/Kontext base; aimed at graphic design.
- **PSDiffusion**: best match on paper (per-layer prompts), but no released weights.
- **DreamLayer**: no verified weights; joint generation re-runs the whole stack.
- **LayerFusion**: paper only; joint generation.
- **RefLayer** (Referring Layer Decomposition): no released weights.
- **DiffDecompose**: no released weights.
- **LiWi**: no released weights.
- **LASAGNA**: no released weights.
- **LayerDecomp** (Adobe): no released weights.
- **RMBG-2.0**: CC BY-NC 4.0, gated. Use BiRefNet, which has the same architecture and an MIT license.
- **ObjectClear**: NTU S-Lab License, non-commercial.
- **OmniEraser**: card says Apache-2.0, but it is a FLUX.1-dev finetune, so effectively non-commercial. GitHub has no LICENSE file.
- **pix2gestalt**: amodal completion is not needed when objects are generated whole.
- **IC-Light v2**: weights unreleased, announced as non-commercial.
- **Colab tunnelling**: banned by the Colab FAQ on free runtimes.

(PrismLayersPro/Plus datasets are Apache-2.0 and usable for any future fine-tuning, but the model built on them, ART+, has no license.)

Since then: ART+ weights were found (ART base hash-identical to the Apache-2.0 `thuteam/CLD` file) and run on our T4s; Qwen-Image-Layered, LayerDiffuse-Flux and CLD turned out to fit on 2× T4 ([papers-survey.md](papers-survey.md)).

## 8. Licences that matter (non-commercial project)
FLUX.1-dev (non-commercial) for RevealLayer, ART+, CLD, LayerDiffuse-Flux; Qwen Research licence for Qwen-Image-2.1; Stability Community licence for Stable-Layers; Apache-2.0 for Qwen-Image-Layered, IC-Light v1, SAM2, LaMa; OpenRAIL-M for LayerDiffuse weights. All fine for this personal, non-commercial project.
