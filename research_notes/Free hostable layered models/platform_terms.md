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
