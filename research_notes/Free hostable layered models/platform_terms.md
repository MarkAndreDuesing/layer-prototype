# GPU hosting platforms: terms and capabilities (checked 2026-11-07)

VERIFIED means I read it on the official page today. UNVERIFIED means it is secondary or from memory and needs checking.

## Comparison
| | Kaggle | Modal | fal.ai | Colab free |
|---|---|---|---|---|
| Free compute | About 30 GPU-h/week (UNVERIFIED, secondary sources say it varies, about 30-41 h) | $30/month credits, no monthly fee (VERIFIED) | No free credits shown on pricing page (VERIFIED absent) | Variable |
| GPUs | T4x2, P100 (UNVERIFIED) | T4 $0.000164/s, L4 $0.000222/s, A10 $0.000306/s, A100-40 $0.000583/s, A100-80 $0.000694/s, L40S $0.000542/s, H100 $0.001097/s (VERIFIED; listed as "A10", not "A10G") | Serverless: H100 $4.50/h list (as low as $2.49), H200, B200, RTX PRO 6000 $4.00/h; no A100 (VERIFIED) | T4 |
| Session | 12 h GPU (UNVERIFIED; some sources say 9 h) | Serverless, scale-to-zero | Serverless | Hours, varies |
| HTTP server + tunnel | Not found in readable terms (UNVERIFIED, pages render with JavaScript and could not be fetched) | Native web endpoints (fastapi_endpoint / asgi_app give a *.modal.run URL; from docs knowledge, UNVERIFIED today) | Native API | Disallowed on free tier (see below) |
| Qwen-Image-Layered | Needs about 40 GB+; does not fit on T4 without heavy quantisation | A100-80/H100 | Hosted endpoint, $0.05/image (VERIFIED) | No |
| LayerDiffuse SDXL | Fits on T4 16 GB | Yes (custom) | No hosted endpoint found (Runware has one) | n/a |

## Colab free tier wording (VERIFIED, https://research.google.com/colaboratory/faq.html)
On free runtimes these are disallowed: "remote control such as SSH shells, remote desktops", "bypassing the notebook UI to interact primarily via a web UI", "chess training", "running distributed computing workers". "Connecting to remote proxies" appears in the list that applies to all managed runtimes.

## Kaggle
- https://www.kaggle.com/terms and https://www.kaggle.com/docs/notebooks render with JavaScript, so WebFetch and curl got no body text. Nothing about tunnels, crypto mining or hosting could be quoted. Read them in a browser before relying on Kaggle.
- Phone verification is needed for internet access and GPU (UNVERIFIED, widely known).
- Batch "Save & Run All" commit runs keep the internet setting, but a tunnel only lives while the session runs (UNVERIFIED).
- Limits on dataset and model size for caching weights: NOT VERIFIED.

## Modal (https://modal.com/pricing)
- Starter: "$30 / month free credits". The page does not say whether a card is required (UNVERIFIED).
- Cold start for SDXL is typically tens of seconds. Use Volumes to cache weights (docs, UNVERIFIED today).
- The acceptable use policy was not read (UNVERIFIED).

## fal.ai
- https://fal.ai/models/fal-ai/qwen-image-layered: "decomposing an image into multiple RGBA layers". $0.05 per image, output is an images[] array (the example has 4 layers), tagged "Commercial use".
- https://fal.ai/pricing: Model APIs are billed per output from prepaid credits. Serverless is billed per second while a runner is alive; image pull and queue time are not billed. Custom deployments go through sales.
- No LayerDiffuse endpoint found. The ToS clause on output ownership was not read (UNVERIFIED).

## Recommendation
(a) Development backend for SDXL+LayerDiffuse: Modal Starter. It is free up to $30/month, about 50 h of T4 or about 37 h of L4, and has a sanctioned HTTPS endpoint. Kaggle gives more hours, but its tunnel policy could not be verified. Colab free explicitly bans this kind of use.
(b) Qwen-Image-Layered: fal.ai hosted at $0.05/image, so about $5 per 100 images. Alternatively, self-host on a Modal A100-80 at about $2.50/h, which uses the free credits but has a long cold start for 20B weights.

## More platforms and Kaggle details (moved from hosting_and_matting.md §A, 2026-11-07)

All pages were checked on 2026-11-07 unless noted. "Unverified" means the primary page could not be read or did not state the fact.

#### Takeaway
On the Colab free tier, the official FAQ forbids "bypassing the notebook UI to interact primarily via a web UI" and forbids remote proxies, so a ComfyUI or API tunnel there risks termination. Kaggle could not be verified from primary sources: its docs page is rendered with JavaScript, though community projects do tunnel ComfyUI from Kaggle. Hugging Face ZeroGPU does NOT require PRO to host: free accounts can host 2 ZeroGPU Spaces, but get only 5 minutes of GPU per day. Modal's $30/month free credit is the cleanest way to run a real HTTP API for free.

#### Cited Findings
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

#### Inferences
- **Colab:** a ComfyUI or API tunnel driven from a local app is exactly the "interact primarily via a web UI" / web-service case, so it is non-compliant on the free tier. It technically works, but a session can be killed at any time.
- **Kaggle:** the most practical free notebook option, with about 30 GPU hours per week and persistent model caching via private Datasets/Models. The URL changes every session and lasts at most about 12 hours.
- **ZeroGPU:** good for short, on-demand calls such as matting or segmentation through `gradio_client`. 5 minutes per day on a free account is enough for roughly dozens of small matting or SAM calls, but not for heavy diffusion.
- **Modal:** $30 per month is roughly 50 T4-hours or 37 L4-hours at base price. It gives a stable, ToS-clean HTTPS endpoint, with weights cached in a Modal Volume.

#### Gaps
- **Kaggle:** the official docs and ToS pages render only through JavaScript and could not be read. No primary statement on tunnels or quotas was obtained; I tried a WebFetch of kaggle.com/docs/notebooks and two searches.
- **Paperspace / DigitalOcean Gradient free GPU tier:** not checked because of the call budget. Its status as of November 2026 is unverified.
- **Google Drive model caching on Colab:** not re-verified; it is standard practice via `drive.mount`.

## Licences and notebooks checked 2026-11-07 (moved from open_items_closeout.md)

### 1. Licenses
| Item | Finding | Source |
|---|---|---|
| OmniPSD code | CC BY-NC-SA 4.0 (LICENCE file), NON-COMMERCIAL | https://github.com/showlab/OmniPSD , https://raw.githubusercontent.com/showlab/OmniPSD/main/LICENCE |
| OmniPSD weights | No own weights linked in README; only dataset subset lc03lc/OmniPSD_Layered_Poster. Depends on FLUX.1-dev + FLUX.1-Kontext-dev (non-commercial) | same |
| PrismLayers / ART+ code | MIT; ART+ ckpts hosted on Google Drive (license not stated separately) | https://github.com/redredsheep/PrismLayers |
| PrismLayersPro dataset | Apache-2.0 (gating not visible) | https://huggingface.co/datasets/artplus/PrismLayersPro |
| PrismLayersPlus datasets | Apache-2.0 | https://huggingface.co/datasets/artplus/PrismLayersPlus |
| ART original (microsoft/art-msra) | MIT code, but weights + inference code REMOVED 23 Jul 2025 (training data concern) | https://github.com/microsoft/art-msra |
| ART-Release/ART_v1.0 (HF) | HTTP 401: removed or private | https://huggingface.co/ART-Release/ART_v1.0 |
| SAM 3 | "SAM License" (Meta, updated 19 Nov 2025). Royalty-free use/modify/distribute; does not explicitly forbid commercial use; bans military/weapons/ITAR uses; derivatives must carry same license. HF repo gated (share contact info, Meta privacy policy) | https://github.com/facebookresearch/sam3/blob/main/LICENSE , https://huggingface.co/facebook/sam3 |
| OmniEraser HF weights | Card says Apache-2.0, but it is a FLUX.1-dev finetune/LoRA | https://huggingface.co/theSure/Omnieraser |
| OmniEraser GitHub | No LICENSE file seen | https://github.com/PRIS-CV/Omnieraser |
| FLUX.1-dev base | Non-commercial license; commercial use needs BFL license. Outputs may be used commercially (with conditions) | https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md |
| FLUX.2 klein 4B | Apache-2.0, ~13GB VRAM | https://huggingface.co/black-forest-labs/FLUX.2-klein-4B |

Practical: OmniEraser's Apache tag does not remove the FLUX.1-dev non-commercial restriction on running the base model.

### 2. Notebooks
- FLUX.2 klein 4B: HF auto-generated Colab/Kaggle notebook links on card (https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/colab, /kaggle). Model card says ~13GB, so a T4 (15GB) is borderline. No T4 timing found.
- ComfyUI on Kaggle: kayas881/comfyui-kaggle (T4 15GB, Pinggy tunnel; 1 star, unverified). https://github.com/kayas881/comfyui-kaggle
- LayerDiffuse: no Colab/Kaggle in rootonchair/diffuser_layerdiffuse (MIT, has --cpu-offload), huchenlei/ComfyUI-layerdiffuse (Apache-2.0), sd-forge-layerdiffuse (Apache-2.0). Official repo listed Colab as "coming soon". HF Space: radames/LayerDiffuse-gradio-unofficial.
- Qwen-Image-Layered: no dedicated notebook found; Comfy-Org fp8mixed repack exists. https://huggingface.co/Comfy-Org/Qwen-Image-Layered . T4 has no native bf16.
- Qwen-Image-2.1: no notebook found; Unsloth doc says GGUF ~11GB VRAM. https://unsloth.ai/docs/jp/moderu/qwen-image-2.1.md

### Unconfirmable
- No T4 VRAM/timing measurements were published for any of these models.
- ART+ Google Drive ckpt license: not stated; base ART weights were withdrawn, so redistribution status is unclear.
- Gating of PrismLayersPro: not visible without login.
- SAM 3 commercial use: implied by license text, not explicitly stated. Not legal advice.
- Kaggle code search is not indexed well by web search, so some notebooks may have been missed.
