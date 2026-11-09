# LayerDiffuse and layered-generation / layer-decomposition research models (checked 2026-11-07)

Pages were opened 2026-11-07 unless marked "(search snippet only)". VRAM numbers were not published by any of these projects. Where this file gives VRAM figures, they are inferences from the base model.

## 1. LayerDiffuse (SD1.5 / SDXL): availability, license, modes, tooling

### Takeaway
This is the most practical free option. Code is Apache-2.0, weights are CreativeML OpenRAIL-M, and it supports all four modes. Mature diffusers and ComfyUI ports exist. SD1.5 and SDXL both fit on a T4 16 GB, and SDXL has a CPU-offload flag.

### Cited Findings
- The main repo is Apache-2.0. It links to the Forge implementation (layerdiffusion/sd-forge-layerdiffuse) and to LayerDiffuse_DiffusersCLI. The official "Gradio + Diffusers + Colab" and the HF Space are still marked "Coming soon". Training code and dataset are only planned. — [lllyasviel/LayerDiffuse](https://github.com/lllyasviel/LayerDiffuse)
- The HF weights repo LayerDiffusion/layerdiffusion-v1 is licensed creativeml-openrail-m, which allows research and prototype use. — [HF LayerDiffusion/layerdiffusion-v1](https://huggingface.co/LayerDiffusion/layerdiffusion-v1)
- The diffusers port is MIT-licensed. SD1.5 supports fg-only, joint fg+bg+blend (batch 3), fg on a given bg, and bg for a given fg (batch 2), and works with ControlNet and IP-Adapter. SDXL supports fg-only via attention injection (LoRA) or conv injection, inpainting, fg/bg-to-blend, and blend-to-fg/bg. SDXL scripts have a `--cpu-offload` flag (Accelerate). The README does not mention Colab. — [rootonchair/diffuser_layerdiffuse](https://github.com/rootonchair/diffuser_layerdiffuse)
- ComfyUI-layerdiffuse is Apache-2.0 and supports "only SDXL/SD15". The joint fg+bg+blend workflows are SD15-only. It has no Flux support, and its issue tracker had 66 open issues and 6 open PRs when checked. — [huchenlei/ComfyUI-layerdiffuse](https://github.com/huchenlei/ComfyUI-layerdiffuse)

### Inferences
- **SD1.5 (~4 GB fp16):** runs comfortably on a Kaggle or Colab T4.
- **SDXL (~7 GB fp16 UNet):** fits on a T4 at fp16. CPU offload makes it safer on Colab's ~12 GB system RAM.
- **Recommended free path:** a Kaggle or Colab notebook that pip-installs rootonchair/diffuser_layerdiffuse, or runs ComfyUI with huchenlei's nodes.

### Gaps
- I did not verify a maintained, working public Colab or Kaggle notebook. The official one is still "Coming soon".

## 2. LayerDiffuse-Flux (FireRedTeam)

### Takeaway
Code and weights are released: a LoRA plus TransparentVAE on FLUX.1-dev. The FLUX.1-dev Non-Commercial License governs any use, so it is OK for research and prototypes but not commercial use. Running it on a T4 is possible only with heavy offload or quantization.

### Cited Findings
- The repo shows no LICENSE file. Weights are on HF RedAIGC/Flux-version-LayerDiffuse (`layerlora.safetensors` and `TransparentVAE.pth`). It needs a FLUX.1-dev checkpoint and provides T2I and I2I demo scripts. Requirements are torch 2.3.0 and cu121. No VRAM figure is given. — [FireRedTeam/LayerDiffuse-Flux](https://github.com/FireRedTeam/LayerDiffuse-Flux)
- The HF weights repo is labelled license afl-3.0 and has no hardware notes. — [HF RedAIGC/Flux-version-LayerDiffuse](https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse)

### Inferences
- FLUX.1-dev is a 12B DiT, about 24 GB at bf16. A T4 has no bf16 and only 16 GB, so it would need fp8/NF4/GGUF quantization or sequential CPU offload. Colab's 12 GB system RAM is a likely bottleneck. Kaggle (~29 GB RAM) is the more realistic option.
- The same FLUX.1-dev constraints apply to ART, CLD, and OmniPSD below.

### Gaps
- I found no official ComfyUI node or notebook. T4 runtime has not been verified.

## 3. Research models: code and weights status

> Merged into [joint_text_to_layers.md](../Layered%20methods%20deep%20dive/joint_text_to_layers.md#code-and-weights-table-2026-11-07) on 2026-11-10 (moved verbatim, with its citations).

The 2026-11-09 re-check in that file supersedes most rows.
