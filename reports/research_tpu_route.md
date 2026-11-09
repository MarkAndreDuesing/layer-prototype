# Can Qwen-Image-Layered run on a free Kaggle TPU? (researched 2026-11-09)

[U] = unverified or secondary source. Our own estimates are labelled as estimates.

## 1. What Kaggle gives you
- **Weekly quota: 20 h of TPU time per week.** Sessions are capped at 9 h [U]. Sources: a developer blog (https://chloechiaw.github.io/two-weeks) and a script comment (https://huggingface.co/spaces/cuilabs/bee/blob/main/scripts/push_kaggle_tpu_kernel.py). I could not read the official page https://www.kaggle.com/docs/tpu (it renders with JavaScript only).
- **Which TPU: possibly v6e-8, not v5e-8 [U].** The same script says "Kaggle's TPU offering as of 2026-05 is v6e-8" and uses the accelerator value `TpuV6E8`. Check this in the notebook accelerator menu. A v6e chip has 32 GB of HBM and about 918 TF bf16. A v5e chip has 16 GB and 197 TF.
- **Host RAM: unknown.** Google Cloud sizes the full v5e-8 VM at 384 GB (https://docs.cloud.google.com/tpu/docs/v5e). Kaggle probably gives less [U]. Run `free -g` in a session to check.
- **Preinstalled software:** Kaggle TPU images have historically shipped with both JAX and torch_xla [U]. Run `pip list | grep -E "torch|jax"` to check the exact versions.

## 2. Software routes
**(a) PyTorch/XLA + diffusers with SPMD sharding.** Diffusers PR #13474 (https://github.com/huggingface/diffusers/pull/13474) was merged in April 2026 and adds `examples/research_projects/pytorch_xla/inference/flux/flux_inference_spmd.py`, "Tested on v5e-8".
- How it works:
  - It shards the parameters across the chips with `xs.mark_sharding`.
  - Flash attention runs with `partition_spec=("data",None,None,None)`.
  - The text encoders run on the CPU.
  - The VAE moves to the TPU only for decoding, because otherwise it "would barely fit".
- Measured in the README:
  - The first iteration includes about 195 s of VAE compilation.
  - "The second iteration shows the true steady-state speed (~1.76s)". It is not clear what this time covers (whole image or not), so it is unverified.
- Known problem: the author later reported recompilations and "Failed to deserialize executable: UNIMPLEMENTED". The thread does not say whether this was fixed.
- No memory figures or torch_xla version are given.
- Porting effort: this is the best template. QwenImageLayeredPipeline would need the same sharding, applied to the Qwen MMDiT blocks.

**(b) TorchTPU.** Google's native PyTorch backend for TPUs ("Eager First") was open-sourced on 2026-10-20 [U, secondary coverage].
- Sources:
  - https://developers.googleblog.com/torchtpu-running-pytorch-natively-on-tpus-at-google-scale/
  - https://explainx.ai/blog/hugging-face-torchtpu-transformers-diffusers-native-tpu-pytorch-2026
- In diffusers you move a pipeline with `pipe.to("tpu")`. Hugging Face reports that 96.5% of runnable tests pass.
- One diffusers example took 188 s compiled and 262 s eager [U]. That example was slow.
- Whether TorchTPU is preinstalled or installable on Kaggle: not confirmed.
- Qwen-Image support specifically: not confirmed.
- Multi-chip sharding of a 20B model under TorchTPU: not documented in what I found.

**(c) JAX/MaxDiffusion.** MaxDiffusion supports SD, SDXL, Flux, and Wan 2.1/2.2 on v5e, v5p and v6e (https://www.mintlify.com/AI-Hypercomputer/maxdiffusion, a mirror). There is **no Qwen-Image port.** A JAX port of the 20B MMDiT, the Qwen2.5-VL encoder and the layered VAE would take weeks.

**Published Qwen-Image timings on v5e or v6e:** none found.

## 3. Blockers
- **Memory fits.** The 40 GB transformer sharded over 8 chips is about 5 GB per chip on v5e (16 GB each). That leaves room for activations. The Qwen2.5-VL-7B text encoder (about 15 GB) can run on the CPU, as the Flux example does, or be sharded too.
- **Compilation.** Expect several minutes of compilation per new shape, judging by the Flux example's 195 s for the VAE alone. Fixing the resolution at 640, the layer count at 4 and the step count means it compiles once per session. That one-off compile costs 5–15 min of the 20 h quota (estimate).
- **Dynamic shapes.** The layer count changes the sequence length, and changing it triggers a recompile. Keep it fixed at 4. Also look in the pipeline for data-dependent Python code, such as `.item()` calls or variable-length text prompts. Pad the text to a fixed length.
- **Version risk.** torch_xla often lags the newest torch. Recent diffusers releases may need a newer torch than the torch_xla build on Kaggle supports. Unverified; check in a session.

## 4. Licensing
- **Model:** "Qwen-Image-Layered is licensed under Apache 2.0" (https://huggingface.co/Qwen/Qwen-Image-Layered). That is fine for this use.
- **Kaggle:** the terms allow notebooks for ML experimentation. Using the TPU for personal image generation is not banned as far as I found [U, I did not read the full terms]. Running it as a persistent API backend may conflict with interactive-use rules. Under the permissive reading this project uses, interactive personal use is OK.

## 5. Expected speed (estimate)
- **Token count.** At 640 px there are 40×40 = 1,600 latent tokens per image. The 4 output layers plus the input image give about 8k tokens. CFG doubles that to about 16k tokens.
- **Compute.** That is about 2 × 20e9 × 16k ≈ 6.4e14 FLOPs per step. 50 steps come to about 3.2e16 FLOPs.
- **v5e-8:** peak is 1.58 PF/s. At 30–40% utilisation that is **about 50–70 s per decomposition at 50 steps**, or **about 30–40 s at 30 steps**. Add a few seconds for the VAE and the text encoder.
- **v6e-8 (if that is what Kaggle now has):** about 4.6× the compute, so **about 12–20 s at 50 steps**.

## Verdict
- **Feasible, with a medium-to-high risk of engineering friction.** A ~40 s target is realistic on v5e-8 with about 30 steps at fixed shapes, and easy on v6e-8.
- **Effort:** an unverified AI estimate said "2–5 days" (and "weeks" for JAX). Treat both as guesses; the probe below measures instead.
- **Quota:** 20 h per week minus compile overhead. That allows roughly 1,000+ decompositions per week.
- **First step:** a 1-hour probe session to confirm:
  - the chip type,
  - host RAM,
  - torch_xla and torch versions,
  - whether `pip install` of a recent diffusers works.

## Probe results (2026-11-09 evening)

Notebook generator: [prototype/tools/make_tpu_probe.py](../prototype/tools/make_tpu_probe.py).
- Three runs pushed through the Kaggle CLI never got a TPU. Kaggle stores `enable_tpu: true` and `machine_shape: TpuV5E8` (`TpuV6E8` is not a valid name), but the runs used the standard CPU Docker image and JAX saw only a CPU.
- The image ships **torch 2.11 (CPU build) and JAX 0.11.1, no torch_xla**. pip's newest `torch_xla` is 2.9, so torch has to be pinned to 2.9 to match.
- **Next:** run the probe as an interactive session in the Kaggle editor with Accelerator = TPU, to see whether TPUs are only available there.
