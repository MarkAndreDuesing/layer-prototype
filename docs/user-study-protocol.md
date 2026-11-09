# User study protocol (draft): layer controls vs prompt-only editing in a story loop

Gap G4 and RQ3 in [Research gap and questions.md](../reports/Research%20gap%20and%20questions.md); background in [Layered loop open paths.md](../reports/Layered%20loop%20open%20paths.md) and `research_notes/Layered loop open paths/hci_layered_generative_editing.md`.

**Closest prior work: ImageDirector (UIST '26, N=12, novices).** It found that object-level direct manipulation beat text-only editing:
- on spatial accuracy: 37 vs 207 px position error
- on background drift: SSIM 0.998 vs 0.922
- on perceived control: 5.9 vs 4.4

Lighting and cohesion were *not* better. It used no persistent layers, no story loop, no story events, only 3 edits, and no agency measures. This study fills those gaps.

**This is a draft for the team. An LMU ethics self-check and consent forms are still needed before running it.**

## 1. Research questions and hypotheses
- **RQ3a:** Do per-layer controls (re-prompt, move, resize) increase perceived agency and predictability compared with prompt-only editing of the whole scene, during story play?
- **RQ3b:** Do they reduce unintended changes to things the player did not edit (drift), and the number of regenerations needed to reach an intended scene?
- **H1:** layer condition shows higher PXI Autonomy and Ease of Control, and a higher per-edit expectation-match rating.
- **H2:** layer condition shows fewer unintended changes (masked SSIM, coder ratings) and fewer regenerations to target.
- **H3 (exploratory):** prompt-only is rated easier at first (learning curve, as in ImageDirector and LACE). Lighting and cohesion are rated similar or better for prompt-only, because the whole scene is regenerated.

## 2. Design
1. **Within subjects, 2 conditions, counterbalanced (AB/BA):**
   - **A, prompt-only:** players re-prompt the whole scene. Story events are applied to the whole image.
   - **B, layers:** per-layer re-prompt, move and resize, plus the global prompt. Story events are routed per layer (`layerproto/events.py`).
2. **Same backend, seed policy and latency in both conditions**, so that only the editing interface differs.
   - If B has an instant preview (shadow layer, procedural overlays), A gets an equivalent progress view.
   - AI disclosure is identical in both conditions (perceived-AI bias, Bazzaz & Cooper 2026).
3. **Two matched story seeds**, for example "forest camp at night" and "harbour town at dawn". Seeds are counterbalanced across conditions.

## 3. Participants
1. **N = 24 novices.** LMU students with little or no experience of generative AI or image editing; screen for this.
   - Power: with dz = 0.6, which is conservative compared with ImageDirector's control effect, a paired test with α = .05 two-sided and power = .80 needs about 24.
2. Exclude professional 3D or image-editing users, as ImageDirector did, to avoid ceiling effects.
3. Questionnaires are in German where a validated German version exists (German PXI); otherwise English.

## 4. Procedure (about 70 min per session)
1. Consent, demographics and an experience screen (5 min).
2. Short intro to the story loop (5 min).
3. **Condition 1** (25 min):
   1. Guided tutorial with two practice edits (5 min).
   2. **Target task:** reach a described scene state, for example "the chair is left of the fire, the fire is out, it is raining". The target is shown as text plus a reference sketch with circles and arrows, as in ImageDirector. Limit: 8 min or 10 edits.
   3. **Free story session:** 5 story turns. The system triggers 2 story events, for example "fog rolls in" and "a stranger arrives". The player edits as they like (10 min).
   4. Questionnaires (5 min).
4. **Condition 2**: same structure (25 min).
5. Preference questions and a semi-structured interview (10 min).

## 5. Measures
1. **Logged automatically:**
   - number of edits and regenerations
   - time to target
   - wait times
   - edit type (global prompt, layer re-prompt, move, resize)
2. **Drift**, using `layerproto/drift_eval.py`: SSIM and PSNR outside the regions the player edited, per turn. Two coders also rate unintended changes (0/1/2, Krippendorff's α).
3. **Target accuracy:** position error of the moved object (centroid to target, as in ImageDirector) and an expert rubric (intent achieved, background preserved, cohesion).
4. **After every edit:** a 1-item 7-point rating, "the result matched what I expected" ("Who Did It?", UIST 2024), and a 1-item "the wait was acceptable".
5. **After each condition:**
   - German PXI subscales: Autonomy, Mastery, Ease of Control
   - Creativity Support Index
   - NASA-TLX (raw)
   - SUS
   - 7-point perceived control and expressiveness items (as in ImageDirector)
   - the 4-item effectance subset (Ballou et al. 2021)
6. **End of session:** overall and per-task preference, and an interview about strategies, breakdowns and how players used layers for story intent.

## 6. Analysis
1. Paired Wilcoxon or t-tests per measure; linear mixed models with condition and order (and seed) as fixed effects and participant as a random effect.
2. Per-turn drift curves per condition; area under the curve.
3. Reflexive thematic analysis of the interviews.
4. Pre-register the hypotheses and the primary measures. Primary measures: PXI Ease of Control, expectation-match, masked SSIM.

## 7. Material and technical setup
1. Hardware:
   - **Ideal:** one GPU backend at about 4 s per layer edit (`relocate()` v5, SDXL) and about 30 s per scene split (Qwen-Image-Layered on 2× T4).
   - **For the lab:** an LRZ or RBG GPU, or pre-generated scenes so that latency is the same for all participants. See [University GPU access LMU.md](../reports/University%20GPU%20access%20LMU.md).
2. Pilot with 3 people. Check task difficulty, timing and logging, and decide whether latency becomes a secondary factor.
3. Open points:
   - ethics self-check
   - compensation
   - whether to use a commercial flat editor as the A backend (ImageDirector used Nano Banana Pro) or our own SDXL/Qwen stack, to keep the backend identical
