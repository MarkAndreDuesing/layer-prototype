# Qwen-Image-Layered: speed and hosted-API cost (researched 2026-11-09)

## A. Speed

Official settings:
1. Model card (https://huggingface.co/Qwen/Qwen-Image-Layered) uses num_inference_steps=50, true_cfg_scale=4.0, layers=4, resolution=640, bf16, and cfg_normalize. It says "For this version, 640 is recommended" (the other bucket is 1024). It gives no timings. The licence is Apache 2.0.
2. The ComfyUI tutorial (https://docs.comfy.org/tutorials/image/qwen/qwen-image-layered) says "The original sampling settings are steps: 50 and CFG: 4.0". It recommends 640px input, or 1024 for high resolution. Model files:
   - qwen_image_layered_bf16.safetensors (needs high VRAM)
   - qwen_image_layered_fp8mixed.safetensors (lower VRAM)
   - text encoder: qwen_2.5_vl_7b_fp8_scaled
   - VAE: qwen_image_layered_vae

   The tutorial says "This model is slow." It names no sampler, gives no timings and mentions no Lightning LoRA.
3. fal's API defaults to 28 steps and guidance 5, with acceleration none/regular/high (https://fal.ai/models/fal-ai/qwen-image-layered/llms.txt). This is likely how hosted runs get to "a few seconds": fewer steps, acceleration, and datacentre GPUs. That is my inference, not a stated fact.
4. Lightning or distilled Layered variant: none found. lightx2v ships 4- and 8-step Lightning for Qwen-Image, Edit-2511 and 2.1 (https://huggingface.co/lightx2v/Qwen-Image-Lightning), but I found none for Layered. People may stack the Qwen-Image Lightning LoRA on Layered, but that is UNVERIFIED. No NVFP4 Layered checkpoint was found.
5. Published per-GPU timings: none found. All timings are UNVERIFIED.

## B. Hosted cost

| Provider | Price | Notes |
|---|---|---|
| fal.ai | $0.05 / image (own page) | 1-10 layers, latency not stated |
| WaveSpeed | $0.025 base; scales with layers (own pages conflict: 4 layers $0.10 or $0.20) | https://wavespeed.ai/models/wavespeed-ai/qwen-image/layered |
| 302.ai | $0.05 / call (search snippet, not fetched) | |
| RunComfy | listed, price not checked | https://www.runcomfy.com/models/qwen/qwen-image/layered |
| Replicate | no official Layered listing found | |
| Alibaba Model Studio / DashScope | no Layered endpoint found in docs | other Qwen-Image models only |
| HF Inference Providers, Together | not checked / not found | |

Comfy Cloud (https://www.comfy.org/cloud/pricing):
- Free tier: "5 free runs on real GPUs" plus Comfy Agent credits.
- Standard: $20/mo (or $16/mo billed yearly), about 4,200 credits/mo.
- Creator: $35/mo (or $28), about 7,400 credits/mo.
- Pro: $100/mo (or $80), about 21,100 credits/mo, 1 h maximum job.
- Credits do not roll over.
- The page does not say whether a card is required.

Free credits and card requirements for fal and WaveSpeed: not verified.

Licence: the weights are Apache 2.0, so commercial use is allowed. Each host's terms of service also apply (not reviewed).
