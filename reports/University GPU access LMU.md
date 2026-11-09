# University GPU access at LMU Informatik for Qwen-Image-Layered (20B)

Researched 2026-11-09 from public pages only. **[V]** means verified on the cited page. **[U]** means unverified (from memory or inferred), so confirm it before relying on it.

Model needs: about 40 GB VRAM in bf16 (compute capability 8.0 or higher, so A100, H100, L40S or RTX 30/40 class) or about 13 GB in 4-bit (V100 or T4 class can work).

## 1. Institute level: RBG (Rechnerbetriebsgruppe, Institut für Informatik)

- **Slurm for all students [V]:** "Slurm can be used by all students and faculty members of the Institut. No further registration is required." https://www.rz.ifi.lmu.de/infos/slurm_en.html
- **Limits [V]:** 15 concurrently running jobs and 30 submitted jobs per user (same page).
- **abakus GPU servers [V]:** these servers "have more powerfull NVIDIA cards, huge RAM size, fast/local/big SSD, many CPU cores." You submit with `srun -q abaki -p Abaki rechenjob.sh`. "On the abakus servers default job runtime is 1 day and max job runtime is 5 days." (same page)
- **abakus access [V]:** "Access to the abakus servers can be requested by mail from the RBG. A good reason is required." The page lists a thesis, a publication or an official project as good reasons, each with an informal confirmation from your advisor. The contact is rbg@ifi.lmu.de.
- **GPU models and VRAM on abakus [U]:** the page does not name them. You must ask whether any card has at least 40 GB and compute capability 8.0 or higher.
- **CIP workstations [V, partly]:** compute jobs on CIP machines are described at https://www.rz.ifi.lmu.de/infos/rechenjobs_en.html. A 2024 RBG news item said a test environment had no NVIDIA or AMD drivers (https://www.rz.ifi.lmu.de/announcements_en.html). Typical CIP GPUs are consumer cards with 8 to 16 GB **[U]**, so at best they could run 4-bit.
- **Chair-level clusters (Database Systems/Data Mining (Prof. Seidl), Machine Learning chairs, Media Informatics, MCML groups) [U]:** these chairs commonly run their own GPU servers, but nothing public says students can use them. Access goes only through a supervisor at that chair.

## 2. LRZ AI Systems

Main page: https://doku.lrz.de/lrz-ai-systems-11484278.html

- **Hardware [V]:** "Available GPU/CPU partitions and hardware (P100, V100, A100, H100)." Also DGX nodes **[U]**.
- **Prerequisites [V]:** "Valid LRZ user account and AI-enabled project required for access."
- **Eligibility [V]:** "Prospective users of the AI Systems must demonstrate that their intended work or use case has a clear research focus." Also, "older projects with many members do not qualify for AI enablement." Source: https://doku.lrz.de/1-access-10746642.html
- **Students, courses and theses [U]:** none of these is mentioned explicitly. In practice a chair or project head (the PI or Master User) adds students to their project.
- **How to apply [V]:**
  - Individuals: "ask the project's master user(s)" to add you to the `<projectID>-ai-c` group (managed in the IDM portal).
  - Existing project: file a Servicedesk request, choosing "Predefined service request" then "AI Systems Project Activation/Update".
  - New project: submit the "Application for an LRZ Project" PDF with "AI Systems" written under "Sonstige Bemerkungen".
  - Alternative: apply via the BayernKI initiative.
- **Membership and renewal [V]:** "There is no limit or quota to the number of users that can be added to the -ai-c group." Projects must be renewed every year.
- **Jobs and containers [V]:** Slurm handles interactive and batch jobs. Containers run on "the Enroot container runtime for unprivileged, high-performance images" (Docker images are imported into Enroot). Jupyter, RStudio and TensorBoard run through the web interface.
- **Time limits, quotas and storage [U]:** I did not verify these. My recollection is jobs of up to about 1 to 2 days on GPU partitions and DSS project storage. See the "Usage policies" pages under https://doku.lrz.de/.
- **LRZ Compute Cloud GPU VMs [U]:** these exist (V100-class at the time of writing, according to my knowledge). Access also requires an LRZ project.

## 3. MCML (Munich Center for Machine Learning) [U]

MCML has dedicated GPU capacity (DGX/A100/H100 class, hosted at LRZ) for MCML members and their supervised students. Access goes through an MCML PI. I did not verify this on a public page.

## 4. Bavarian and national options [U]

- **BayernKI [V for its existence]:** this is an alternative route to the LRZ AI Systems, named on the LRZ access page.
- **NHR (NHR@FAU Erlangen, Alex cluster with A100/A40) [U]:** NHR has free "NHR-Starter" or test projects for researchers at German universities. Student projects need a PI (for example a professor) to apply. This route fits better than Gauss for small jobs.
- **Gauss Centre (JUWELS, SuperMUC-NG, JURECA) [U]:** proposal-based and aimed at large research projects. Training accounts are tied to courses, so this is overkill for a course prototype.

## 5. Speed estimate (FLOPs-based)

Assumptions: 6.6e14 FLOPs per step (640 px, 4 layers, with CFG) and 50 steps, giving 3.3e16 FLOPs per run.

| GPU | Peak | Assumed effective | s/step | s/run (50 steps) | 2,000 runs |
|---|---|---|---|---|---|
| T4 (measured) | 65 fp16 | 19 TFLOPS | ~35 | ~29 min | ~970 h |
| A100 bf16 | 312 | ~150 (about 50% MFU) | ~4.4 | ~3.7 min | ~122 h |
| H100 bf16 | ~990 | ~400 (about 40% MFU) | ~1.7 | ~1.4 min | ~46 h |

The T4 is also too small for bf16 (16 GB, compute capability 7.5), so on a T4 you need 4-bit plus offloading, which makes it slower still. An A100 40 GB is tight for bf16, so an 80 GB A100 or an H100 is preferable. Inference with fewer steps (for example 20 to 30) cuts these times proportionally.

## 6. Paid APIs vs university GPUs

| Option | Cost for 2,000 runs | Latency per run | Setup effort | Terms |
|---|---|---|---|---|
| fal.ai (~$0.05/run) [U price] | ~$100 | ~10 to 60 s [U] | Minutes: API key and Python client | Commercial ToS; your images go to a third party. Fine for a personal or non-commercial project. |
| WaveSpeed ($0.025/layer) [U price] | 4 layers ≈ $0.10/run, so ~$200 | similar [U] | Minutes | Same as fal.ai |
| RBG abakus | Free | Queue wait, then ~2 to 4 min if A100-class [U] | Email the RBG with an advisor confirmation; days | Institute rules; academic use only |
| LRZ AI Systems | Free | Queue (can be hours), then ~1.5 to 4 min | Needs a PI's AI-enabled project; days to weeks. Enroot container build. | Research-focus requirement |

Bottom line: for a course prototype with about 2,000 runs, a paid API costs $100 to $200 and works within the hour. The university GPUs are free but depend on a supervisor and take setup time. They become worth it for a recurring or larger workload, or for fine-tuning.

## 7. How to get access, step by step

1. **Try the RBG Slurm cluster now (no registration).** Log in to a CIP machine via SSH (see https://www.rz.ifi.lmu.de/) and run `sinfo -o "%P %G %N"` to see partitions and GPUs.
2. **Ask your course supervisor for a short confirmation** that this is an official course or research project.
3. **Email rbg@ifi.lmu.de asking for abakus access.** Include the advisor confirmation, the model's needs (40 GB in bf16 with compute capability 8.0 or higher, or 13 GB in 4-bit) and the expected volume (about 2,000 runs, about 100 GPU-hours).
4. **Test a job** with `srun -q abaki -p Abaki --gres=gpu:1 --pty bash`, then `nvidia-smi`.
5. **If abakus is insufficient, go to LRZ.** Ask your supervisor or chair whether they have an AI-enabled LRZ project. If they do, the Master User adds you to `<projectID>-ai-c` in the IDM portal.
6. **If they have no such project,** the supervisor (as PI) applies with the LRZ project PDF, writing "AI Systems" in the remarks, then files a Servicedesk request for "AI Systems Project Activation/Update". The BayernKI route is the alternative.
7. **On LRZ,** build an Enroot image from a PyTorch NGC or Docker image (`enroot import docker://nvcr.io#nvidia/pytorch:<tag>`). Store the model weights on project storage, not in your home directory.
8. **Run batch jobs** with `sbatch` on an A100 or H100 partition. Use an interactive `salloc` session only for debugging.
9. **As a fallback for deadlines,** keep a fal.ai or WaveSpeed key with a small budget cap.

## 8. Questions to ask (RBG, supervisor, LRZ)

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

## Sources

- https://www.rz.ifi.lmu.de/infos/slurm_en.html
- https://www.rz.ifi.lmu.de/infos/rechenjobs_en.html
- https://www.rz.ifi.lmu.de/announcements_en.html
- https://doku.lrz.de/lrz-ai-systems-11484278.html
- https://doku.lrz.de/1-access-10746642.html
