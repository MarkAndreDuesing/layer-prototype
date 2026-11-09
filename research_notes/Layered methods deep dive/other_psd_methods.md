# Other Text/Image-to-PSD & layer-decomposition methods (audit, 2026-11-09)

> **Update 2026-11-09:** measured since this audit: Stable-Layers runs on 2× T4 at reduced settings (4 steps 34 s, via our GGUF stack, not the 40 GB bf16 base); LayerD and SAM2 + LaMa were tried once (next attempts in `docs/findings-log.md`); RevealLayer runs end to end on 2× T4 (output still being fixed). The audit below is unchanged.

Legend: [U] = unverified (not confirmed directly on GitHub/HF in this pass). T4 = 16 GB, fp16 only (no bf16).
Excluded (covered elsewhere): Qwen-Image-Layered (QIL), LayerDiffuse, PSDiffusion, DreamLayer, OmniPSD, LASAGNA, LayerFusion, ART, CLD, Qwen-Image-2.1.

## Ranked summary for our loop (prompt -> layered scene -> edit layer -> recomposite -> story event)

| # | Method | Type | Code / weights | Licence | T4 fp16 | Why it matters |
|---|---|---|---|---|---|---|
| 1 | Generate -> SAM2 -> amodal inpaint (pix2gestalt / LaMa) | pipeline | yes | mixed (see below) | yes | Cheapest fully runnable image-to-PSD route today |
| 2 | LayerD (ICCV25, CyberAgent) | image -> layers (graphic design) | yes: BiRefNet matting + LaMa | Apache-2.0 | yes (~1 GB + 200 MB) | Iterative top-layer peel + inpaint; template for our decomposition step |
| 3 | RevealLayer (ICML26) | image + bboxes -> RGBA layers incl. occluded content | GitHub repo exists | not stated [U] | [U] | Box-prompted, recovers hidden parts = "move object, background stays complete" |
| 4 | Stable-Layers (Stability AI) | LoRA over QIL, RL-tuned with VLM scores | inference repo | [U] | no (base ~40 GB bf16) | Better QIL; LRZ-class GPU only |
| 5 | See-through (shitagaki-lab) | anime image -> up to 23 PSD layers, depth order | yes + ComfyUI node | [U] | [U] | Strong if art style is anime/2D |
| 6 | LayerCraft | LLM-agent layered T2I + editing | GitHub | [U] | [U] | Matches our Ollama-planner idea; orchestration reference |
| 7 | Alfie (ECCV24-W) | training-free RGBA T2I (DiT attention -> alpha) | GitHub | [U] | likely (PixArt-Sigma) [U] | Transparent sprites without training |
| 8 | Referring Layer Decomposition / RefLayer (ICLR26) | image + text -> named RGBA layer | no code found [U] | - | - | Text-targeted extraction; RefLade 1.11M dataset |
| 9 | Generative Omnimatte (CVPR25) | video -> object+effects layers | public reimpl. (VideoX-Fun, SAM2) | [U] | doubtful | Only if scenes get animated |
| 10 | TransPixar/TransPixeler (CVPR25) | text -> RGBA video | LoRA for CogVideoX-5B | HF card Apache-2.0; repo LICENSE [U] | no (~24 GB) | Animated transparent FX; LRZ only |
| 11 | LayeringDiff (2501.11197) | generate composite, then disassemble | not found [U] | - | - | Same philosophy as route #1 |
| 12 | LayerDecomp (Adobe, CVPR25) | image -> fg/bg incl. shadows/reflections | no code (5B DiT) | - | - | Idea: keep shadows with object layer |
| 13 | DiffDecompose (2505.21541) | alpha-composited image -> layers | repo placeholder ("code later") | - | - | Not usable now |
| 14 | Text2Layer (2023) | 2-layer fg/bg T2I (LDM) | code link points to zhang-zx/SINE (mismatch) [U] | - | - | Historical baseline |
| 15 | MuLAn (CVPR24) | dataset, 44K multi-layer annotations | HF dataset [U] | [U] | n/a | Train/eval data; layers from off-the-shelf models (noisy) |
| 16 | UniWorld-Design (2608.03971), LiWi (2605.14552), Inpainting->Layer Decomposition (2511.20996) | 2026 decomposition papers citing QIL | none found [U] | - | - | Watch list; LiWi targets natural photos |

Not found / not audited: PixelART, TransAnimate, SemLayer (2603.24039, icons), AmodalSVG (2604.10940).
Curated list for further sweeps: https://github.com/discopotato3376-oss/awesome-image-layer-decomposition

## Details

### 1. Generate -> segment -> amodal complete (recommended baseline)
1. Generate full scene (SDXL / SD1.5 fp16 on T4).
2. Segment objects with SAM2 (Apache-2.0), boxes from detector or LLM plan.
3. Complete hidden object parts: pix2gestalt (CVPR24, SD-based, weights on HF) https://github.com/cvlab-columbia/pix2gestalt (licence type [U]); or Open-World Amodal Appearance Completion (CVPR25) https://github.com/saraao/amodal [U].
4. Inpaint background hole with LaMa / SD inpaint.
5. Export RGBA layers + z-order to PSD (psd-tools / pytoshop).
Edit ops: move/resize free; re-prompt a layer = regenerate/inpaint that layer, then harmonise.

### 2. LayerD
https://github.com/CyberAgentAILab/LayerD , https://arxiv.org/abs/2509.25134
Raster graphic design -> layers by repeatedly matting the top layer (cyberagent/layerd-birefnet, Apache-2.0, ~1 GB) and inpainting beneath (simple-lama-inpainting, licence [U]). Trained on Crello (posters) so natural scenes are out-of-domain; fine-tune code included.

### 3. RevealLayer
https://arxiv.org/abs/2605.11818 , https://github.com/360CVGroup/RevealLayer
Diffusion; RGB + user bboxes -> visible and hidden RGBA layers. Releases RevealLayer-100K and RevealLayerBench (200 imgs). Base model, weights, licence [U].

### 4. Stable-Layers
https://github.com/Stability-AI/Stable-Layers , https://arxiv.org/abs/2605.30257
LoRA on QIL with VLM-scored RL. Inference only; ~80 GB-class GPU.

### 5. See-through
https://github.com/shitagaki-lab/see-through ; ComfyUI https://github.com/jtydhr88/ComfyUI-See-through
Anime illustration -> up to 23 part layers with hidden-part fill and depth ordering (Live2D oriented). Licence [U].

### 6. LayerCraft
https://github.com/PeterYYZhang/LayerCraft , https://arxiv.org/abs/2504.00010
LLM agents plan layout (CoT), generate background, integrate objects layer by layer; object-level edits. Weights/licence [U].

### 7. Alfie
https://github.com/aimagelab/Alfie , https://arxiv.org/abs/2408.14826
Training-free RGBA from DiT attention maps. Single objects, not scenes. Licence [U].

### 8. Referring Layer Decomposition
https://arxiv.org/abs/2602.19358 , https://yaojie-shen.github.io/project/RLD/
Prompt-conditioned layer extraction. Code/weights not found [U].

### 9. Generative Omnimatte
https://github.com/gen-omnimatte/gen-omnimatte-public , https://arxiv.org/abs/2411.16683
Video -> object layers with shadows/reflections; VideoX-Fun + SAM2; torch 2.5.1/CUDA 12.4. VRAM [U].

### 10. TransPixar
https://github.com/wileewang/TransPixar , https://huggingface.co/wileewang/TransPixar
RGBA video via LoRA on CogVideoX-5B (Mochi LoRA "coming soon"); ~24 GB reported.

### 11-16
- LayeringDiff: https://arxiv.org/abs/2501.11197 (code not found [U])
- LayerDecomp: https://rayjryang.github.io/LayerDecomp , https://arxiv.org/abs/2411.17864 (Adobe, no code)
- DiffDecompose: https://github.com/Wangzt1121/DiffDecompose , https://arxiv.org/abs/2505.21541
- Text2Layer: https://www.researchgate.net/publication/372468916
- MuLAn: https://arxiv.org/abs/2404.02790
- UniWorld-Design https://arxiv.org/abs/2608.03971 ; LiWi https://arxiv.org/abs/2605.14552 ; https://arxiv.org/abs/2511.20996

## Takeaway
On T4: build route #1 (generate -> SAM2 -> pix2gestalt/LaMa -> RGBA stack) using LayerD-style iterative peel-and-inpaint. Evaluate RevealLayer next once weights/VRAM are confirmed. Stable-Layers, TransPixar, Gen-Omnimatte are LRZ-only.
