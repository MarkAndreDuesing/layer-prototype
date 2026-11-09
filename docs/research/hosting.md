# Hosting, compute and costs

> **Role: research synthesis (current).** Summarises the sources; details and citations stay in the linked snapshots and research notes.

Merged from six reports, kept unchanged as snapshots: [../archive/2026-11-05 Image generation models hosting and costs.md](../archive/2026-11-05%20Image%20generation%20models%20hosting%20and%20costs.md), [../archive/2026-11-07 Free hostable layered models.md](../archive/2026-11-07%20Free%20hostable%20layered%20models.md), [../archive/2026-11-07 Kaggle migration plan.md](../archive/2026-11-07%20Kaggle%20migration%20plan.md), [../archive/2026-11-09 Speed and APIs.md](../archive/2026-11-09%20Speed%20and%20APIs.md), [../archive/2026-11-09 University GPU access LMU.md](../archive/2026-11-09%20University%20GPU%20access%20LMU.md), [../archive/2026-11-09 TPU route.md](../archive/2026-11-09%20TPU%20route.md).

## 1. What we use now
- **Kaggle (free):** 2× T4 16 GB (or 1× P100), about 30 GPU-h/week, sessions up to 12 h, about 29 GB RAM, internet after phone verification. **Max 2 batch GPU sessions at once.** No native bf16/fp8 on T4 → [../concepts/fp16-on-t4.md](../concepts/fp16-on-t4.md). Terms: personal, non-commercial use; no serving farms. Workflow: [../guides/kaggle-cli.md](../guides/kaggle-cli.md).
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

## 4. LMU and LRZ access (from the 11-09 research)
1. **RBG Slurm:** usable by all students and staff of the institute without registration; 15 running / 30 submitted jobs per user.
2. **abakus:** `srun -q abaki -p Abaki …`; default runtime 1 day; access by mail to the RBG with a good reason and informal advisor confirmation. GPU models are not listed publicly: ask whether a card has ≥ 40 GB and compute capability ≥ 8.0.
3. **LRZ AI Systems:** valid LRZ account + AI-enabled project with a clear research focus; a project's master user adds students to `<projectID>-ai-c`; Slurm + Enroot containers; projects renewed yearly.
4. Questions for the RBG are listed in the snapshot.

<a id="tpu"></a>
## 5. TPU
- Kaggle offers TPU v5e-8 (8 chips × 16 GB HBM, native bf16) on a separate 20 h/week quota. Template: diffusers' PyTorch/XLA FLUX SPMD example (tested on v5e-8). Estimated about 40 s per Qwen split at about 30 steps; the "2–5 days" effort figure was an unverified guess.
- **Probe ([`../../prototype/tools/make_tpu_probe.py`](../../prototype/tools/make_tpu_probe.py)), 2026-11-09:** three CLI batch runs never got a TPU (Kaggle stored `machine_shape: TpuV5E8`, the run used the standard CPU image). The image ships torch 2.11 (CPU) + JAX 0.11.1 and no torch_xla; pip's newest torch_xla is 2.9, so torch must be pinned to 2.9. **Next:** run the probe as an interactive TPU session in the editor.

## 6. Licences that matter (non-commercial project)
FLUX.1-dev (non-commercial) for RevealLayer, ART+, CLD, LayerDiffuse-Flux; Qwen Research licence for Qwen-Image-2.1; Stability Community licence for Stable-Layers; Apache-2.0 for Qwen-Image-Layered, IC-Light v1, SAM2, LaMa; OpenRAIL-M for LayerDiffuse weights. All fine for this personal, non-commercial project.
