# Multi-turn edit drift and consistency: evaluation for layered vs flat scenes (as of Nov 2026)

Scope note: about 15 tool calls. Most numbers come from arXiv HTML or alphaxiv text of the primary papers. Items taken from secondary summaries, or only partly read, are marked [unverified].

## Q1. Multi-turn, sequential and iterative editing benchmarks (2023-2026): what they measure, how much drift, which models drift most

### Takeaway
Drift over turns is now well documented for flat editors. Unedited-content consistency falls monotonically with turn count. Strong single-turn editors (Qwen-Image-Edit, FLUX Kontext, open models) collapse fastest. Benchmarks run 3 turns (EdiVal), 5 (MSE-Bench), 8 (MTEdit-Bench), or 10 or more (TailSteer Add & Preserve, AnchorEdit). None of them includes a layered editor.

### Cited Findings
- **EdiVal-Agent (ICLR 2026)** has three parts. Instruction following (EdiVal-IF) uses Grounding-DINO plus geometric rules for symbolic edits, and a VLM on detector crops for semantic edits. Content consistency (EdiVal-CC) = 1/2 x (background similarity outside every object box ever seen + mean similarity of unchanged-object crops to their base crops). The similarity is DINOv3, preferred over L1 because small shifts inflate L1. Quality uses HPSv3. The default is 3 turns, across 16 editors — [EdiVal-Agent](https://www.alphaxiv.org/abs/2509.13399.md); [arXiv](https://arxiv.org/abs/2509.13399)
- EdiVal-CC at turns 1/2/3:

  | Editor | T1 | T2 | T3 |
  |---|---|---|---|
  | FLUX.1-Kontext-dev | 95.32 | 92.24 | 90.22 |
  | GPT-Image-1.5 | 94.49 | 91.20 | 88.49 |
  | Nano Banana | 93.91 | 90.48 | 89.48 |
  | Seedream 4.0 | 92.51 | 88.03 | 85.86 |
  | Qwen-Image-Edit | 84.22 | 80.52 | 77.98 |
  | GPT-Image-1 | 81.00 | 77.78 | 75.50 |

  Every editor loses about 3-4 points per turn — [EdiVal-Agent](https://www.alphaxiv.org/abs/2509.13399.md)
- Qwen-Image-Edit's overall score falls 78.36 -> 59.56 -> 41.93 over three turns. The authors attribute this to exposure bias. Flow-matching editors show progressive brightening across turns (rising 99.9% luminance quantile); FLUX.1-Kontext-dev is the exception. In one example HPSv3 falls 6.19 -> 4.19 -> 3.34 while a VLM's quality scores stay high, so VLM judges miss quality drift — [EdiVal-Agent](https://www.alphaxiv.org/abs/2509.13399.md)
- The paper's text gives inconsistent editor counts (16 in Table 3, 13 in the contributions list, 11 on the project page) — [EdiVal-Agent](https://www.alphaxiv.org/abs/2509.13399.md); [project page](https://tianyucodings.github.io/EdiVAL-page)
- **MSE-Bench (VINCIE, 2025)** has 100 sessions of 5 turns each. GPT-4o judges both instruction success and consistency, with no ground truth. VINCIE reaches 25% success at turn 5 versus <2% for earlier academic methods [unverified, search summary] — [VINCIE](https://arxiv.org/pdf/2506.10941)
- **MTEdit-Bench (2026)** has a step-wise sub-benchmark of 100 eight-turn samples and context-aware sub-benchmarks of 50 four-turn samples each. A VLM judge (Qwen 3.5 Plus) scores IF and CC from 0 to 10. A trajectory fails once IF or CC drops below 6, and later turns then score 0. The Drift Robustness Score (DRS) is the mean of VisualQuality-R1 and RALI. — [MTEdit](https://arxiv.org/html/2610.23751)
- Turn-8 IF/CC/DRS: Nano Banana Pro 8.83/7.51/4.05; Nano Banana 2 7.99/6.80/4.20; GPT Image 2 7.11/5.81/4.17; Seedream 5.0 Lite 6.38/4.52/3.75; FLUX.1-kontext-dev 2.13/1.76/3.82; Qwen-Image-Edit-Causal 3.17/2.41/4.06; VINCIE, OmniGen2 and Lance are about 0 — [MTEdit](https://arxiv.org/html/2610.23751)
- DRS from turn 1 to turn 8: Seedream 5.0 Lite 4.30 -> 3.51; Nano Banana 2 4.32 -> 3.70; GPT Image 2 4.28 -> 3.83 — [MTEdit](https://arxiv.org/html/2610.23751)
- **TailSteer "Add & Preserve" (Nov 2026)** uses 100 images. Each gets a red box over a white region, followed by 10 sequential add/remove edits inside the box, with each output fed back as the next input. It measures PSNR, SSIM and LPIPS outside the box against the original at turn 10, plus Cumulative Object Accuracy (COA) inside the box. This design is closest to a layered "edit one object, keep the rest" test — [TailSteer](https://arxiv.org/html/2611.06741)
- TailSteer turn-10 PSNR/SSIM/LPIPS/COA: SenseNova U1.5 11.23/0.166/0.607/94.9; FLUX.2 Klein 9B 13.85/0.289/0.524/31.0; Qwen Image-2.1 16.93/0.438/0.270/67.4; FLUX.1 Kontext dev 18.64/0.585/0.388/34.8; GPT-Image-2.5 18.55/0.540/0.378/98.3. Outside-box PSNR under 19 dB after 10 turns means substantial drift even for the best — [TailSteer](https://arxiv.org/html/2611.06741)
- **AnchorEdit (2026)** proposes a high-resolution benchmark of 10 or more rounds, framed around "identity drift and error accumulation". The abstract gives no numbers or benchmark name — [AnchorEdit](https://arxiv.org/abs/2606.11751)
- **Omni IIE Bench (2026)** has a "Multi-turn Coordination" track built from human-filtered dialogue tasks. Most models degrade from low- to high-semantic-scale tasks. Per-turn numbers were not retrieved — [Omni IIE](https://arxiv.org/abs/2603.16944)
- **Multi-Edit Bench (layer-wise memory, 2025)** uses 2-5 sequential edit steps on scenes with 3-6 layers. Metrics: LLaVA captions of layer crops scored with BLEU/METEOR, CLIP on crops, and a human Likert rating for background consistency (ours 4.59 vs HD-Painter 3.71, BLD 3.43). BLD and HD-Painter "decline after three steps" on CLIP and METEOR — [Layer-wise memory](https://arxiv.org/html/2505.01079)
- **[Pass 2] AnchorEdit (2026)** uses a custom 50-sample multi-turn set of up to 10 turns, plus MSE-Bench. GPT-4o scores consistency and semantic following from 1 to 10, and a sequence stops when either score falls below 3. Custom-set consistency / semantic / success: JoyImage-Edit 5.1 / 5.4 / 29.8%; VINCIE 5.8 / 5.1 / 34.1%; AnchorEdit 6.2 / 5.3 / 52.9%. No code or licence is mentioned — [AnchorEdit HTML](https://arxiv.org/html/2606.11751)
- **[Pass 2] MSE-Bench success rate per turn, T1 to T5:**

  | Editor | T1 | T2 | T3 | T4 | T5 |
  |---|---|---|---|---|---|
  | JoyImage-Edit | .980 | .802 | .679 | .617 | .446 |
  | VINCIE | .950 | .693 | .667 | .617 | .487 |
  | AnchorEdit | .950 | .737 | .707 | .623 | .567 |

  Success roughly halves over 5 turns — [AnchorEdit HTML](https://arxiv.org/html/2606.11751)
- **[Pass 2] Omni IIE Multi-turn Coordination track:** 260 dialogue groups averaging 4.35 turns (1,131 turns in total; up to 16 turns per the introduction). Metrics are LPIPS, CLIP (foreground / background / whole image), QA, PSNR and SSIM, reported only as multi-turn aggregates, with no per-turn breakdown — [Omni IIE HTML](https://arxiv.org/html/2603.16944v1)
- **[Pass 2] Omni IIE multi-turn aggregates (whole-image LPIPS / PSNR):** Step1X 0.453 / 14.8; Qwen-image-edit 0.510 / 13.5; ICEdit 0.515 / 14.5; FLUX 0.599 / 11.9; InstructPix2Pix 0.669 / 10.9; MGIE 0.685 / 10.5 — [Omni IIE HTML](https://arxiv.org/html/2603.16944v1)
- **[Pass 2] Omni IIE background LPIPS, single-turn → multi-turn:** MGIE 0.294 → 0.580; InstructPix2Pix 0.441 → 0.552; Qwen-image-edit 0.327 → 0.390; FLUX 0.428 → 0.478; Step1X 0.259 → 0.344. The deltas are my own arithmetic from the paper's Tables 2 and 3. The paper says multi-turn causes "a substantial drop in background preservation across all models" — [Omni IIE HTML](https://arxiv.org/html/2603.16944v1)
- **[Pass 2] MagicBrush multi-turn (NeurIPS 2023)** scores only the final image of each session against the ground truth, with no per-turn breakdown. Fine-tuned InstructPix2Pix: L1 0.0625 → 0.0964 and CLIP-I 0.9332 → 0.8924 from single-turn to multi-turn [unverified, secondary summary]. HIVE multi-turn: L1 0.1521, CLIP-I 0.8004, DINO 0.6463 [UIP2P appendix, unverified] — [MagicBrush](https://arxiv.org/pdf/2306.10012); [UIP2P](https://arxiv.org/pdf/2412.15216)
- **[Pass 2] ICE-Bench (ICCV 2025)** is single-turn only: 31 tasks, 6 dimensions including source consistency, and a VLLM-QA metric. It is useful for its source-consistency metric definitions but has no drift data. I found nothing on "I2EBench" — [ICE-Bench](https://openaccess.thecvf.com/content/ICCV2025/html/Pan_ICE-Bench_A_Unified_and_Comprehensive_Benchmark_for_Image_Creating_and_ICCV_2025_paper.html)
- **[Pass 2] MIRA (CVPR 2026 Findings)** is an iterative perception–reasoning–action agent (Qwen2.5-VL-7B) wrapped around Kontext, Step1X or Qwen-Image-Edit. It is a method rather than a drift benchmark — [MIRA](https://arxiv.org/html/2511.21087v1)
- **[Pass 2] CoEdit:** I found no chat-based image-editing benchmark under this name, so it may be a misremembered name — searched [here](https://arxiv.org/html/2511.21087v1) (no match).
- Other names seen but not read: MICE-Bench (Edit-R2) [unverified] — [Edit-R2](https://arxiv.org/pdf/2606.05950); on-policy self-distillation for multi-turn editing [unverified] — [arXiv 2609.35611](https://arxiv.org/html/2609.35611)

### Inferences
- A practical drift rate for flat editors is about 3-4 EdiVal-CC points per turn. After 10 turns, outside-region PSNR is typically 11-19 dB. A layered stack that leaves untouched layers bit-identical should score infinite PSNR on those layers, so the interesting signal is recomposition and harmonisation effects plus edited-layer identity drift.
- Pair a VLM judge with HPSv3 or a no-reference quality metric, because VLMs miss gradual quality decay. Also log luminance statistics, since brightening is a known flow-model drift mode.

### Gaps
- No per-turn numbers for MagicBrush multi-turn, I2EBench, ICE-Bench, CoEdit or MIRA were retrieved in this pass.
- Omni IIE and AnchorEdit numeric drift results were not read; the full PDFs were too large to fetch.

## Q2. Metrics for preservation of unedited content, and layer-specific metrics

### Takeaway
For flat images, the current standard is masked or region-excluded similarity: DINOv3 for EdiVal-CC; PSNR, SSIM and LPIPS outside the edit region for TailSteer. Subject identity uses DreamSim with the background removed, or newer masked metrics. Layered papers measure single-shot decomposition quality: alpha-weighted RGB L1 and alpha soft IoU, following LayerD's DTW-aligned protocol. None measures recomposition error over cycles.

### Cited Findings
- EdiVal-CC combines a background mask (image minus all object boxes, base and current) with per-object crop similarity, using DINOv3 or L1. Background scoring is disabled after a background-change edit — [EdiVal-Agent](https://www.alphaxiv.org/abs/2509.13399.md)
- TailSteer uses PSNR/SSIM/LPIPS outside the edit box against turn 0, plus COA for edit-history correctness — [TailSteer](https://arxiv.org/html/2611.06741)
- MTEdit uses VLM-judged CC (0-10) with a failure-cascade rule — [MTEdit](https://arxiv.org/html/2610.23751)
- ConsiStory uses DreamSim pairwise similarity after background removal with CarveKit, replacing the background with noise. The authors note it is biased toward layouts with slight changes — [ConsiStory](https://arxiv.org/pdf/2402.03286)
- Masked patch-similarity (MaSC) and ID-Sim argue that global embeddings confuse identity with background — [MaSC](https://arxiv.org/pdf/2605.22469); [ID-Sim](https://www.alphaxiv.org/abs/2604.05039.md)
- Qwen-Image-Layered evaluates on Crello using LayerD's protocol: order-aware DTW alignment, up to 5 neighbour merges, alpha-weighted RGB L1 and alpha soft IoU. It reports L1 0.0594 / IoU 0.8705 vs LayerD 0.0709 / 0.7520 [unverified, secondary summary] — [Emergent Mind](https://www.emergentmind.com/papers/2512.15603); [CVPR paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Yin_Qwen-Image-Layered_Towards_Inherent_Editability_via_Layer_Decomposition_CVPR_2026_paper.pdf)
- RGBA-VAE reconstruction on AIM-500 is reported as PSNR/SSIM/rFID/LPIPS (reportedly PSNR 38.83 vs LayerDiffuse 32.09) [unverified] — [Emergent Mind](https://www.emergentmind.com/papers/2512.15603)
- Its compositing model is standard alpha-over: C_i = a_i*RGB_i + (1 - a_i)*C_{i-1}, requiring C_N = I. No explicit recomposition-error metric was found [unverified] — [Emergent Mind](https://www.emergentmind.com/papers/2512.15603)

### Inferences
- Proposed layer metrics for us:
  - per-layer alpha IoU and alpha-weighted L1 versus that layer at turn 0, for layers that were not edited;
  - recomposition error = PSNR or LPIPS between the composite and the flat render, if a harmonisation pass is applied;
  - a round-trip test: decompose, recompose, decompose again, repeated N times.

### Gaps
- No layered paper reports a decompose-recompose cycle error. This is confirmed as an open gap only to the extent of these searches.

## Q3. Story-visualisation consistency protocols

### Takeaway
ViStoryBench (CVPR 2026) is the reusable template. It detects and crops characters before computing identity similarity (CIDS), and adds style similarity (CSD), character-count matching (OCCM), prompt adherence, aesthetics and copy-paste detection.

### Cited Findings
- ViStoryBench has 80 stories, 344 characters and 30+ methods, with metrics validated against human ratings — [ViStoryBench](https://arxiv.org/abs/2505.24862); code at [GitHub](https://github.com/vistorybench/vistorybench)
- CIDS crops character regions because full-image CLIP is too coarse. There is a prompt-alignment vs consistency trade-off (AnimDirector CIDS 0.401) [unverified, digest] — [papernotes](https://en.papernotes.org/CVPR2026/image_generation/vistorybench_comprehensive_benchmark_suite_for_story_visualization/)
- ConsiStory protocol: 100 sets of 5 images, masked DreamSim plus CLIP-T, and a user study — [ConsiStory](https://arxiv.org/pdf/2402.03286)

- **[Pass 2] StoryDiffusion (NeurIPS 2024)** measures character-to-character CLIP image similarity and text-image CLIP, with a user study of 30 participants (72.8% preferred StoryDiffusion). Text-CLIP was 0.659 vs PhotoMaker 0.654 and IP-Adapter 0.613 [unverified, secondary summary] — [hyper.ai summary](https://hyper.ai/en/papers/2405.01434)
- **[Pass 2] StoryGen and StoryDiffusion as baselines:** Audit & Repair used a 50-person Prolific study rating visual consistency and narrative alignment on a 1–5 scale — [Audit & Repair](https://arxiv.org/pdf/2506.18900)
- **[Pass 2] ViStoryBench licences:** evaluation code is Apache-2.0 and the dataset is MIT. Metrics: CIDS (GroundingDINO detection plus face models such as InsightFace/AdaFace, plus SigLIP), CSD style, aesthetics, OCCM, and prompt alignment (needs an API key). It expects a fixed directory layout keyed to ViStory story IDs, so arbitrary sequences need a custom loader — [ViStoryBench GitHub](https://github.com/vistorybench/vistorybench)

### Gaps
- I did not find StoryGen's own metric table.

## Q4. Direct comparisons of layered vs flat editing, and error accumulation in decompose-recompose cycles

### Takeaway
I found no independent multi-turn layered-vs-flat study. Qwen-Image-Layered shows only qualitative single-edit comparisons against Qwen-Image-Edit-2509. LimeCross claims that keeping layers across iterative edits avoids re-decomposition error, but its numbers were not readable. Gap G1 stands.

### Cited Findings
- Qwen-Image-Layered argues that flat rasters entangle content. Qualitatively, it says Qwen-Image-Edit-2509 shifts pixels and struggles with resizing and moving objects — [CVPR paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Yin_Qwen-Image-Layered_Towards_Inherent_Editability_via_Layer_Decomposition_CVPR_2026_paper.pdf)
- LimeCross (2026) does context-conditioned layered editing with structural consistency and keeps the layered representation across edits [details unverified; PDF not parsed] — [LimeCross](https://arxiv.org/pdf/2605.10319)
- Layer-wise memory reports a qualitative multi-step advantage over flat inpainting baselines (Q1) — [arXiv 2505.01079](https://arxiv.org/html/2505.01079)

- **[Pass 2] LimeCross (2605.10319), read via its HTML version:** an explicit RGBA representation is kept across edits and never re-flattened or re-decomposed. Baselines that are fed their own composites "accumulate errors", but this is shown only qualitatively (Fig. 5). Table 1 covers 6,000 instances of 2- and 3-layer sequential edits:

  | Metric | LimeCross | MagicQuillV2 | KV-Edit |
  |---|---|---|---|
  | Preservation PSNR | ∞ (by construction) | 21.75 | 20.49 |
  | HPS (x10²) | 26.76 | 26.68 | 25.81 |
  | ImageReward | 0.493 | 0.421 | 0.278 |
  | mFID-α | 51.16 | 51.63 | 54.85 |

  The authors note the infinite PSNR is trivial. The paper does not measure error growth against step count or the effect of repeated decomposition. Qwen-Image-Layered-Control is used only to decompose the baselines' outputs — [LimeCross HTML](https://arxiv.org/html/2605.10319)
- **[Pass 2] Multi-Edit Bench (layer-wise memory):** the per-step drift plot exists only as a figure (App. D.6, Fig. 11), with no numbers in the text. See Q1 — [arXiv 2505.01079](https://arxiv.org/html/2505.01079)
- **[Pass 2] Decompose-recompose error:** LayerD (ICCV 2025) decomposes recursively (top-layer matting plus background completion). The authors note that failures in an earlier iteration hurt later ones. This is error accumulation within a single decomposition, not across edit cycles — [LayerD](https://openaccess.thecvf.com/content/ICCV2025/papers/Suzuki_LayerD_Decomposing_Raster_Graphic_Designs_into_Layers_ICCV_2025_paper.pdf)
- **[Pass 2] Round-trip measurements:** I found no paper that measures stability over repeated decompose → recompose cycles, or re-decomposition stability. One summary page mentions a pixel-space recomposition consistency loss used in LayerD-related work [unverified] — [Emergent Mind LayerD](https://www.emergentmind.com/topics/layerd)

### Gaps
- No paper quantifies drift across repeated decompose-recompose cycles (confirmed again in pass 2). Arm C of our protocol would be novel.

## Q5. Proposed protocol for our prototype (2x T4)

### Takeaway
Run a TailSteer-style "edit inside a region, measure outside" chain combined with EdiVal-CC. Use 10 turns (with an extension to 20 on a subset) on about 20 story scenes, in two arms: a layer stack and a flat editor.

### Inferences (design, not cited facts)
1. **Scenes:** 20 scenes, each with 3-5 RGBA layers (background, 2-3 characters, 1 prop), generated with our LayerDiffuse pipeline.
2. **Edit sequences:** 10 turns of story-style edits (add or remove a prop, change a character's pose or expression, move an object, change lighting or time of day, change the background), with a fixed seed. Instructions come from an LLM through Ollama, with the same scripts for both arms. Extend 5 scenes to 20 turns.
3. **Arms:**
   - Layered (A): edit only the target layer, then alpha-over composite.
   - Flat (B): feed the composite into a flat editor (FLUX.1-Kontext-dev quantised or Qwen-Image-Edit if it fits; otherwise SD inpainting with a mask), chaining each output into the next turn.
   - Optional (C): layered with re-decomposition every turn, which isolates cycle error.
4. **Metrics per turn, against turn 0:**
   - outside-edit PSNR/SSIM/LPIPS (TailSteer style);
   - EdiVal-CC style DINOv2/v3 similarity on unchanged object crops and the background;
   - masked DreamSim for character identity;
   - for layered arms, per-layer alpha IoU and alpha-weighted L1;
   - edit success: GroundingDINO rules, or a VLM yes/no;
   - quality: HPSv3 or MUSIQ, plus mean and 99.9% luminance.
5. **Analysis:** plot curves against turn number, fit slopes, run paired tests per scene, and run a small human check on 10% of samples.
6. **Compute:** 20 scenes x 10 turns x 3 arms = 600 edits, plus 5 x 10 x 3 for the 20-turn extension, about 750 total. At about 20-60 s per 1024 px edit on a T4 (an assumed figure, not measured), that is under 13 GPU-hours split across 2 GPUs. The metrics are cheap.

### [Pass 2] Compute grounding (cited)
- **No T4 benchmark was found** for FLUX.1-Kontext-dev or Qwen-Image-Edit in GGUF form. Nearest data points:
  - FLUX dev at 1024², 20 steps: about 80 s on an RTX 3050 6 GB (Q4), about 32 s on an RTX 3060 12 GB (Q5), about 17 s on an RTX 5060 Ti 16 GB (Q5) — [gigagpu](https://gigagpu.com/?p=34558)
  - FLUX dev on an RTX 3060 Ti 8 GB with Q4_K_S and lowvram: 90–150 s — [localaimaster](https://localaimaster.com/blog/run-flux-on-low-vram-gpu)
  - Kontext-dev: about 9 s per edit on an RTX 4090 (NF4); 12–18 s on an RTX 3060 is an estimate — [dev.to](https://dev.to/jovan_chan_9500711396d4e6/flux1-kontext-dev-for-local-ai-in-2026-image-editing-on-consumer-gpus-without-the-api-bills-3nd3)
  - GGUF weights are available — [QuantStack GGUF](https://huggingface.co/QuantStack/FLUX.1-Kontext-dev-GGUF)
  - Qwen-Image-Edit Q8 GGUF: about 48 s/step on an M1 Max — [lilting.ch](https://lilting.ch/en/articles/comfyui-qwen-mps-bf16-slowdown)
  - Qwen-Image-2512 GGUF: about 1.2 s/step on an RTX 5090 — [byteshape](https://huggingface.co/byteshape/Qwen-Image-2512-GGUF)
- **Inference for the T4 (Turing, no FP8, roughly 3060-class fp16 throughput but lower clocks):**
  - Kontext-dev Q4/Q5 at 1024², 20 steps: roughly 60–120 s per edit; 512–768 px is about 2–3x faster.
  - Qwen-Image-Edit (20B) Q4 with the 4-step Lightning LoRA: roughly 40–90 s.
  - For about 750 edits per flat arm, roughly 12–25 GPU-hours, so about 6–13 hours of wall time on 2 GPUs.
  - Run a 5-edit timing pilot before committing.

### [Pass 2] Reusable code and licences
- **EdiVal-Agent:** official ICLR 2026 code, with data on Hugging Face (`C-Tianyu/EdiVal`). The repo shows both MIT and CC BY-SA 4.0 badges; check its LICENSE file for which applies to code vs data. A custom editor plugs in through a class exposing `generate_single_edit` (`--editor-class`). `eval.py` computes IF and CC (GroundingDINO + DINOv3); HPSv3 runs in a separate environment. Instruction generation needs an OpenAI key, but we can write our own CSV of instructions instead — [EdiVal GitHub](https://github.com/tianyucodings/EdiVal)
  - Direct reuse: wrap arm A (the layer stack composite) and arm B as editor classes, and use its CC and IF evaluation unchanged, with MAX_TURNS raised to 10.
- **ViStoryBench:** Apache-2.0 code and MIT data. Reuse its CIDS and CSD evaluators on character crops; this needs a custom loader — [GitHub](https://github.com/vistorybench/vistorybench)
- **TailSteer:** has a project page, but release of code or the benchmark is unconfirmed, and the paper is CC BY-NC-SA. Replicate its Add & Preserve protocol ourselves (red box, 10 turns, PSNR/SSIM/LPIPS outside the box, COA) — [project](https://lyricccco.github.io/tailsteer/); [arXiv](https://arxiv.org/abs/2611.06741)
- **MSE-Bench (VINCIE):** 100 five-turn sessions judged by GPT-4o. It could be a secondary sanity set [licence unverified] — [VINCIE](https://arxiv.org/pdf/2506.10941)
- **Licensing judgement:** all of the above are fine for a personal, non-commercial project. Note that BY-SA or NC terms would bind any redistributed derived data.

### [Pass 2] Tightened protocol
1. Use the EdiVal harness as the backbone, with our own instruction CSV generated by local Ollama, so no OpenAI key is needed.
2. Run 10 turns, plus a 20-turn subset.
3. Add TailSteer-style outside-region PSNR/SSIM/LPIPS against turn 0.
4. Add per-layer alpha IoU and alpha-weighted L1, following the Qwen-Image-Layered / LayerD protocol.
5. For arm C, re-decompose the composite each turn with LayerDiffuse or Qwen-Image-Layered, and log the layer-matching IoU against the turn-0 layers. This is a novel measurement.
6. Score character identity with ViStoryBench CIDS on character crops.
7. Report:
   - curves per turn;
   - area under the curve (AUC) over turns;
   - the turn at which outside-region PSNR first drops below 25 dB.

### Gaps
- No T4 throughput was measured for Kontext or Qwen-Image-Edit; all T4 numbers above are extrapolated.
- The EdiVal code-vs-data licence split is unresolved.

## Coordinator addition: ImageDirector (UIST '26)

- **Citation:** Mejia Araujo & Zhang, https://doi.org/10.1145/3830398.3830709 (open access, open-source).
- **Why it belongs here:** it measures drift across iterative edits as **SSIM outside the union of the object's old and new masks**. The authors call this conservative, because it also excludes the baseline's "copied instead of moved" artifacts.
- **Results** (N=12, 3 edits each, novice users):

| Measure | ImageDirector (compositing, background never regenerated) | Nano Banana Pro (text only) |
|---|---|---|
| Background SSIM | 0.998 | 0.922 |
| Expert background-preservation rating (0–2, α 0.83) | 1.83 | 0.90 |

- **Relevance to the layered-vs-flat claim:** this is the closest published *empirical* evidence for it, although ImageDirector re-segments each turn instead of keeping layers.
- **Caveat:** experts rated lighting and shadow integration no better for the compositing approach. The flat regenerator sometimes looked more natural. The protocol should therefore keep the quality and harmonisation measures (HPSv3, expert cohesion) alongside the drift measures.
- See `hci_layered_generative_editing.md` for the full study design.
