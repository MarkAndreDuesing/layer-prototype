# Free GPU hosting and cut-out/segmentation models for a layered-image prototype

All pages were checked on 2026-11-07 unless noted. "Unverified" means the primary page could not be read or did not state the fact.

## A. Free GPU hosting: can a notebook expose an HTTP API or ComfyUI to a local app?

### Takeaway
On the Colab free tier, the official FAQ forbids "bypassing the notebook UI to interact primarily via a web UI" and forbids remote proxies, so a ComfyUI or API tunnel there risks termination. Kaggle could not be verified from primary sources: its docs page is rendered with JavaScript, though community projects do tunnel ComfyUI from Kaggle. Hugging Face ZeroGPU does NOT require PRO to host: free accounts can host 2 ZeroGPU Spaces, but get only 5 minutes of GPU per day. Modal's $30/month free credit is the cleanest way to run a real HTTP API for free.

### Cited Findings
- **Colab free tier.** Free notebooks run up to 12 hours. GPU limits are not published and change over time. Idle sessions time out. — [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- **Colab, banned on all runtimes:** "file hosting, media serving, or other web service offerings not related to interactive compute with Colab" and "connecting to remote proxies". — [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- **Colab, also banned on free runtimes without a compute-unit balance** (these may be terminated without warning): "remote control such as SSH shells, remote desktops", "bypassing the notebook UI to interact primarily via a web UI", and "running distributed computing workers". — [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- **Colab in practice.** ComfyUI Colab notebooks have switched between ngrok, cloudflared and localtunnel because of reliability problems. — [stable-diffusion-art forum](https://stable-diffusion-art.com/forums/topic/colab-notebook-no-longer-has-ngrok-link/)
- **Kaggle quota.** Kaggle gives a floating weekly GPU quota, commonly about 30 hours (some users report 30 to 41), that resets Saturday at midnight UTC. Sessions last about 12 hours for CPU/GPU and 9 hours for TPU. These figures come from secondary sources only. — [gpuperhour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits), [iotbyhvm](https://iotbyhvm.ooo/best-free-cloud-gpu-platforms-in-2026-google-colab-kaggle-and-more)
- **Kaggle bonus for Colab subscribers.** Colab Pro and Pro+ users who link accounts reportedly get 15 or 30 extra Kaggle GPU hours per week. Secondary source only. — [gpuperhour](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)
- **Kaggle tunnelling works in practice.** A community repo runs ComfyUI on Kaggle through a Pinggy tunnel and requires Internet to be enabled in the notebook settings. — [comfyui-kaggle listing](https://awesome.ecosyste.ms/projects/github.com%2Fkayas881%2Fcomfyui-kaggle)
- **HF ZeroGPU, hosting.** Free personal accounts in good standing (verified email, account older than 30 days) can host up to 2 ZeroGPU Spaces. PRO accounts can host 10. Spaces must use the Gradio SDK. — [HF ZeroGPU docs](https://huggingface.co/docs/hub/spaces-zerogpu)
- **HF ZeroGPU, hardware and quota.** The default GPU is half an RTX Pro 6000 Blackwell (48 GB VRAM). Each call runs for 60 seconds by default and can be set longer. The daily GPU quota is 2 minutes unauthenticated, 5 minutes on a free account, and 40 minutes on PRO (extensible at $1 per 10 minutes). `torch.compile` is not supported; ahead-of-time compilation is. — [HF ZeroGPU docs](https://huggingface.co/docs/hub/spaces-zerogpu)
- **Modal Starter plan.** $0 base with $30 of free credits per month. GPU rates are about $0.59/h for a T4, $0.80/h for an L4 and $1.10/h for an A10, before a 1.15 to 1.75x region multiplier. — [Modal pricing](https://modal.com/pricing)
- **Lightning AI free plan.** Up to 30 credits: 5 at signup and 25 more after adding a card. The FAQ says this is about 50 GPU hours on interruptible machines. Credits expire after 12 months. Limits are 1 GPU per Studio, 4-hour session restarts, 10 GB of free persistent storage, and 1 free CPU Studio. — [Lightning pricing](https://lightning.ai/pricing)
- **Model caching.** On Kaggle, attached datasets and models appear read-only under `/kaggle/input`, and `/kaggle/working` is saved as notebook output. This comes from the fetch tool's general knowledge because the docs page did not render. — [Kaggle docs (did not render)](https://www.kaggle.com/docs/notebooks)

### Inferences
- **Colab:** a ComfyUI or API tunnel driven from a local app is exactly the "interact primarily via a web UI" / web-service case, so it is non-compliant on the free tier. It technically works, but a session can be killed at any time.
- **Kaggle:** the most practical free notebook option, with about 30 GPU hours per week and persistent model caching via private Datasets/Models. The URL changes every session and lasts at most about 12 hours.
- **ZeroGPU:** good for short, on-demand calls such as matting or segmentation through `gradio_client`. 5 minutes per day on a free account is enough for roughly dozens of small matting or SAM calls, but not for heavy diffusion.
- **Modal:** $30 per month is roughly 50 T4-hours or 37 L4-hours at base price. It gives a stable, ToS-clean HTTPS endpoint, with weights cached in a Modal Volume.

### Gaps
- **Kaggle:** the official docs and ToS pages render only through JavaScript and could not be read. No primary statement on tunnels or quotas was obtained; I tried a WebFetch of kaggle.com/docs/notebooks and two searches.
- **Paperspace / DigitalOcean Gradient free GPU tier:** not checked because of the call budget. Its status as of November 2026 is unverified.
- **Google Drive model caching on Colab:** not re-verified; it is standard practice via `drive.mount`.

## B. Matting, segmentation, inpainting and harmonization models

### Takeaway
For commercial-safe cut-outs, use BiRefNet (MIT) or the BEN2 base model (MIT). SAM 2.1 (Apache-2.0) and Florence-2 (MIT) cover prompted segmentation. LaMa (Apache-2.0) covers background fill. RMBG-2.0, ObjectClear and IC-Light v2 are non-commercial or unreleased. SAM 3 is gated under a custom licence.

### Cited Findings
- **BiRefNet** (`ZhengPeng7/BiRefNet`): MIT, 0.2B parameters, default 1024×1024 input with fp16 CUDA in the example. A HF collection holds 18 task- and scale-specific variants. — [HF](https://huggingface.co/ZhengPeng7/BiRefNet)
  - rembg ships these BiRefNet ONNX variants: general, general-fp16, general-lite, portrait, dis, hrsod, cod and massive. — [rembg](https://github.com/danielgatis/rembg)
- **BEN2** (`PramaLLC/BEN2`): MIT, 94.6M parameters, runs on CUDA or CPU. Only the Base model is open. The Full model (proprietary 22K-image dataset, better hair and 4K handling) is available only through the API or web demo; commercial access goes through sales. — [HF](https://huggingface.co/PramaLLC/BEN2)
- **RMBG-2.0** (`briaai/RMBG-2.0`): CC BY-NC 4.0 (bria-rmbg-2.0). Gated behind a form. Built on the BiRefNet architecture, 0.2B parameters, 1024 input. Commercial use requires a paid agreement. — [HF](https://huggingface.co/briaai/RMBG-2.0)
- **InSPyReNet** (via the `transparent-background` package): MIT. Modes are `base` (1024), `fast` (384), `base-nightly` and `custom`. CPU install instructions are provided. — [GitHub](https://github.com/plemeri/transparent-background)
- **rembg:** the code is MIT, but each weight file keeps its own licence. The default is `bria-rmbg`, which is non-commercial. Supported models: u2net, u2netp, u2net_human_seg, u2net_cloth_seg, silueta, isnet-general-use, isnet-anime, sam, the BiRefNet set above, and bria-rmbg. ViTMatte refiners are also listed. — [GitHub](https://github.com/danielgatis/rembg)
- **SAM 2 / 2.1:** Apache-2.0 and not gated. Checkpoint sizes are 38.9M (tiny), 46M (small), 80.8M (base+) and 224.4M (large). — [GitHub](https://github.com/facebookresearch/sam2)
- **SAM 3** (`facebook/sam3`): gated (you must share your HF info). The licence is listed as "other" (custom Meta). It takes text-phrase and exemplar prompts for images and video. About 0.9B parameters stored as F32, so roughly 3.6 GB of weights. — [HF](https://huggingface.co/facebook/sam3)
- **Grounded-SAM-2:** Apache-2.0 plus component licences (SAM 2, GroundingDINO, cctorch). Grounding DINO and Florence-2 run locally. Grounding DINO 1.5/1.6 and DINO-X need an API token. The README does not mention SAM 3. — [GitHub](https://github.com/IDEA-Research/Grounded-SAM-2)
- **Florence-2-large:** MIT, about 0.77B parameters. Tasks include captioning, phrase grounding, object detection, dense region captioning, OCR and segmentation. — [HF](https://huggingface.co/microsoft/Florence-2-large)
- **LaMa:** Apache-2.0. Weights: `smartywu/big-lama` on HF, or Google Drive. — [LICENSE](https://raw.githubusercontent.com/advimman/lama/main/LICENSE), [GitHub](https://github.com/advimman/lama)
- **ObjectClear:** NTU S-Lab License 1.0 (non-commercial). Built on SDXL-Inpainting and has an fp16 flag. Weights download through a repo script; there is a HF demo Space `jixin0101/ObjectClear`. — [GitHub](https://github.com/zjx0101/ObjectClear)
- **OmniEraser:** weights at `theSure/Omnieraser` and `theSure/Omnieraser_Controlnet_version`. FLUX-based (FLUX.1-dev ControlNet inpainting). The repo states no licence. — [GitHub](https://github.com/PRIS-CV/OmniEraser)
- **pix2gestalt:** amodal completion fine-tuned from Lambda Labs' image-conditioned SD. Weights are at `cvlab/pix2gestalt-weights`. The Gradio app needs 22 to 28 GB VRAM. The dataset is research-only. — [GitHub](https://github.com/cvlab-columbia/pix2gestalt)
- **IC-Light:**
  - The SD1.5 repo is Apache-2.0, with models `fc`, `fcon` and `fbc` (foreground plus background conditioning). Its default remover, RMBG-1.4, is non-commercial, and the repo suggests BiRefNet instead. — [GitHub](https://github.com/lllyasviel/IC-Light)
  - IC-Light v2 (FLUX): the weights are not released, and the maintainer said the licence "will be non-commercial only". It is reachable only through the fal.ai API, and the HF Space was reported as 404 or broken in 2025. — [Discussion #98](https://github.com/lllyasviel/IC-Light/discussions/98)

### Inferences
- **CPU-feasible:** BEN2 base, BiRefNet-lite, InSPyReNet fast, u2net/isnet, SAM 2.1 tiny/small, LaMa, and probably Florence-2-base.
- **GPU-only in practice:** SAM 3, ObjectClear (SDXL), OmniEraser (FLUX, likely 24 GB+), pix2gestalt (22 to 28 GB) and IC-Light SD1.5 (about 6 to 8 GB, typical for SD1.5).
- ZeroGPU's 48 GB slice can run all of these.

### Gaps
- The exact HF repo IDs for BiRefNet_lite, BiRefNet_HR and BiRefNet-matting were not opened individually; only the collection reference was seen on the main card.
- **No VRAM figures are published** for BiRefNet, SAM 3, ObjectClear, OmniEraser or IC-Light. The CPU/GPU split above is my estimate.
- **OmniEraser** has no licence in its repo, so treat it as all rights reserved.
- **The SAM 3 licence text** was not opened.
- **The pix2gestalt code licence** was not stated in the README.
