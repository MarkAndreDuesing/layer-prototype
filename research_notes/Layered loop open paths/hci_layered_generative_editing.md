# HCI and games work on layered/object-level generative scene editing and on generative AI in interactive storytelling (as of Nov 2026)

Scope note: this is a short search pass (about 10 queries and fetches). Items I only know from memory, or saw only in search snippets, are marked [unverified].

## 1. Object- and layer-level generative image editing interfaces (2022-2026)

### Takeaway
Layer and region interfaces for diffusion models exist, but they were evaluated mostly with artists or professionals. The studies are small (N about 12 to 21), use within-subjects comparisons against prompt-only baselines, and report SUS, creativity-support and ownership measures. SketchFlex is the main novice-focused study, and it is region-based, not layer-based.

### Cited Findings
- **Layered Diffusion Brushes** (Gholami & Xiao, UBC). It is training-free, layer-based, region-targeted diffusion editing. The user study recruited artists experienced in image editing who also had basic familiarity with AI tools. SUS was 80.35 for LDB vs about 38 for InstructPix2Pix and SD-Inpainting. The authors also claim better creativity support for iterative design. It appeared at ICCV 2025, not CHI. — [arXiv](https://arxiv.org/abs/2405.00313); [ICCV 2025 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Gholami_Streamlining_Image_Editing_with_Layered_Diffusion_Brushes_ICCV_2025_paper.pdf)
- **LACE** (Huang & Ning, 2025). It brings layer-based prompting into Photoshop for professional artists. A pilot study (N=21) found significant gains in usability, ownership and satisfaction over baseline AI workflows. The authors note it may increase the learning curve for users less familiar with advanced image editing. The venue is reported as CHI 2025 [unverified; likely an EA or workshop paper]. — [arXiv 2504.15189](https://arxiv.org/html/2504.15189v2); [GenAICHI 2025 workshop paper](https://generativeaiandhci.github.io/papers/2025/genaichi2025_17.pdf)
- **SketchFlex** (arXiv 2502.07556). Region-based sketch plus prompt for spatial-semantic coherence. It deliberately targets **novices**: participants ranged from no formal painting training to proficient. Baselines were a text-only model and a region-to-image model. — [arXiv](https://arxiv.org/pdf/2502.07556)
- **PromptCharm** (CHI '24). Mixed-initiative prompt refinement, a style database, attention visualization and inpainting-based refinement. Within-subjects study, N=12, vs a plain-text baseline and Promptist, plus an exploratory study with N=12. Users made higher-quality images that better matched their expectations. — [arXiv 2403.04014](https://arxiv.org/pdf/2403.04014)
- **DirectGPT** (Masson et al., CHI '24). Direct manipulation turned into engineered prompts for text, code and SVG editing (not diffusion). Participants were 50% faster, with 50% fewer and 72% shorter prompts than with ChatGPT. This is the key precedent for "direct manipulation vs prompt-only" as the main independent variable. — [PDF](https://cs.uwaterloo.ca/~dmasson/pdfs/directgpt.pdf)
- **Promptify** (UIST '23). LLM-assisted exploration of text-to-image prompts. It outperformed a baseline tool. — [via search; arXiv 2304.09337 [unverified]](https://arxiv.org/abs/2304.09337)
- **Noise Pilot** (Aveni et al.). A "multi-layered" creativity tool for diffusion offering several levels of interaction with the model. Here "layers" are levels of interaction, not RGBA image layers. The venue is likely 2026 [unverified]. — [PDF](https://timothyaveni.com/research/smith_noisepilot_26.pdf)
- **LayerComposer** (arXiv 2510.20820). A layered canvas where users place, resize and lock subjects, one per layer. It is a CV paper, and its evaluation centres on output quality rather than an HCI study. — [arXiv](https://arxiv.org/html/2510.20820v2)
- Model-side layered RGBA work only runs perceptual preference studies, not usability studies:
  - MRT (CVPR 2026) used pairwise blind preference tests against LayerD, Qwen-Image-Layered and others. — [supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Tang_Masked_Region_Transformer_CVPR_2026_supplemental.pdf)
  - LASAGNA (arXiv 2601.15507). — [arXiv](https://arxiv.org/abs/2601.15507v1)
- **DesignPrompt** (Peng, Koch & Mackay, DIS 2024, DOI 10.1145/3643834.3661588). A moodboard tool that combines images, colour and text into one prompt. Study with 12 **professional designers**. VERIFIED. — [HAL](https://hal.u-pec.fr/LISN-EX-SITU/hal-04652020v1)
- **PromptPaint** (Chung & Adar, UIST 2023, DOI 10.1145/3586183.3606777). Prompts mixed like paints, applied to different canvas regions and diffusion stages. This is region-level, not layers. VERIFIED. — [PDF](https://johnr0.github.io/assets/publications/UIST23-PromptPaint.pdf)
- **Collage Diffusion** (Sarukkai, Li, Ma, Ré & Fatahalian, WACV 2024). Input is RGBA layers, each with its own text, harmonized into one image with per-layer harmonization control. It is the peer-reviewed successor of the Li thesis (see section 4). The abstract mentions no user study. — [arXiv](https://arxiv.org/abs/2303.00262); [WACV](https://openaccess.thecvf.com/content/WACV2024/html/Sarukkai_Collage_Diffusion_WACV_2024_paper.html)
- Venue checks:
  - LACE (Huang, Ning et al.) is a **GenAICHI 2025 workshop** paper (CHI 2025 workshop, Yokohama), not a CHI full paper. — [arXiv 2504.15189](https://arxiv.org/abs/2504.15189v2)
  - There is a separate, similarly named "LACE" paper on turn-taking and parallel human-AI co-creation. — [arXiv 2504.14827](https://arxiv.org/html/2504.14827v1)
  - **Noise Pilot** (Smith, Almeda, Aveni, Agarwal & Hartmann, 2026) is a *node-based* environment for authoring diffusion workflows, not image layers. Its venue is still unconfirmed [unverified]. — [PDF](https://timothyaveni.com/research/smith_noisepilot_26.pdf)

### Inferences
- The standard design is a within-subjects comparison of the tool against a prompt-only or inpainting baseline (N about 12 to 24), measured with SUS, the Creativity Support Index (CSI), sometimes NASA-TLX, plus prompt or iteration counts and time.
- Predictability and control are mostly measured with ad-hoc Likert items ("the output matched my expectations"), not validated scales.

### Gaps
- I did not verify the CHI/UIST 2026 programs directly. A newer novice layer study could exist there.

## 2. Generative AI in interactive storytelling and games with images

### Takeaway
1001 Nights (AIIDE 2023) is the main precedent for player text becoming visual game reality. Player studies mostly report engagement and qualitative agency. Visual consistency is handled technically (Unbounded), or measured with small ad-hoc Likert studies (AniTales).

### Cited Findings
- **1001 Nights** (Sun et al., AIIDE 2023). Players' story keywords become in-game items, and V2 uses Stable Diffusion for visuals. The authors analysed 2000+ gameplay instances and feedback from 200+ players, and report that mixed initiative raises engagement despite generative uncertainty. — [AIIDE](https://ojs.aaai.org/index.php/AIIDE/article/view/27539); [arXiv](https://arxiv.org/abs/2308.12915v2)
- A follow-up analysis reframes agency as how it is "allocated, withheld, delegated, and made visible" rather than as a quantity. This is useful framing for layer controls, which make agency visible. — [arXiv 2502.18641](https://arxiv.org/html/2502.18641v1)
- **Unbounded** (Google, 2024). A generative infinite character life-sim. It uses a regional IP-Adapter to keep characters and environments consistent across images. Evaluation is technical. — [arXiv 2410.18975](https://arxiv.org/html/2410.18975v2)
- **AniTales**. Prompt to visual novel. In a study with 10 general users, visual/narrative consistency was rated 4.2/5, plus an expert study with 5 participants. — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/42181/46142)
- **Toyteller** (CHI 2025). Visual storytelling through toy-playing with character symbols: users move symbols directly to steer the AI narrative. This is the closest analogue to "move objects to change the story", but it does not generate scene images. — [arXiv 2501.13284](https://arxiv.org/html/2501.13284v1)
- **Static vs agentic AI game master** for solo role-play. It used the PXI to evaluate AI-assisted play. — [arXiv 2502.19519](https://arxiv.org/pdf/2502.19519)
- **Elsewise** (2026). Authoring AI-based interactive narrative with visualization of the possibility space. — [arXiv 2601.15295](https://arxiv.org/html/2601.15295v1)
- An LLM game-master thesis (ChatRPG) found that excessive compliance with player input hurts narrative consistency. This is relevant to the trade-off between agency and coherence. — [AAU](https://projekter.aau.dk/projekter/files/784680954/Narrative_Adherence_in_LLM_driven_Games.pdf)
- **StoryDrawer** (Zhang et al., CHI 2022, DOI 10.1145/3491102.3501914). Children aged 6-10. AI turns spoken story into drawings in real time and also generates abstract sketches. A **2x2 study with 64 children** (strategies varied) found more creative, elaborate stories and an engaging experience. This is the main precedent for a factorial study of AI-visualization strategies with non-experts. VERIFIED. — [summary](https://ai.updf.com/paper-detail/StoryDrawer:-A-Child%E2%80%93AI-Collaborative-Drawing-System-to-Support-Children's-Creative-zhang-yao-fb82231f4795cc003ea0c66a7da8033850237926); [SIGGRAPH blog](https://blog.siggraph.org/2022/07/exploring-collaborative-drawing-and-creative-ai.html/)
- **Hidden Door** (commercial, launched in stages from 2022; browser-based). Narrative RPG in a graphic-novel style. Its CEO said it **pre-generates art and semantically matches it to player text**, rather than generating images live, and that visual consistency is a known "hard problem". There is no academic player study. VERIFIED from the press. — [GamesBeat](https://gamesbeat.com/hidden-door-reveals-its-ai-powered-narrative-game-building-platform/); [TechCrunch 2022](https://techcrunch.com/2022/10/27/hidden-door-wants-to-turn-fiction-into-immersive-roleplaying-experiences); [PC Gamer](https://www.pcgamer.com/hidden-door-ai-game-narrative-rpg)
- **TTRPG hobbyists and GenAI** (Oct 2025, likely CHI PLAY 2025 [venue unverified]). Ethnographic interviews plus a shared image-generation activity. — [Crossref 10.1145/3748630](https://api.crossref.org/works/10.1145%2F3748630)
- **Bazzaz & Cooper (2026).** Players rated levels they *believed* were AI-generated as more frustrating, regardless of the levels' real origin. This bias matters for disclosure in our study. — [arXiv 2602.14254](https://arxiv.org/abs/2602.14254)
- Steam review analysis (508,192 reviews) of reactions to generative AI vs procedural generation. — [arXiv 2608.11539](https://arxiv.org/pdf/2608.11539)
- A visual novel with SDXL-generated pictures was published in JGGAG (2025), a minor venue. — [Garuda](https://garuda.kemdiktisaintek.go.id/documents/detail/5856333)
- A direct search of the CHI PLAY 2025 programme via web search found **no** story game with a live image-generation player study.
- AI Dungeon's image feature and Infinite Craft: no academic study found.

### Gaps
- I found no player study on scene-image consistency or editability in AI Dungeon-style games.

## 3. Measures

### Takeaway
Combine validated scales (the PXI for autonomy, mastery and ease of control, the CSI, SUS and NASA-TLX) with study-specific items for predictability and agency, which prior generative-tool studies also wrote themselves.

### Cited Findings
- **PXI**: validated over 5 studies with 529 participants, covering functional constructs (e.g., Ease of Control, Goals and Rules, Progress Feedback) and psychosocial ones (Autonomy, Mastery, Curiosity, Immersion, Meaning). A German PXI was validated with N=506, which fits an LMU sample. The miniPXI exists, but its test-retest reliability is only moderate for some items. — [PXI validation](https://www.aminer.org/pub/5de0d070df1a9c0c415b2d89/development-and-validation-of-the-player-experience-inventory-a-scale-to-measure); [German PXI (MuC'22)](https://www-live.dfki.de/fileadmin/user_upload/import/12535_MuC22__German_PXI_Version.pdf); [miniPXI retest](https://www.themoonlight.io/review/an-investigation-of-the-test-retest-reliability-of-the-minipxi)
- **Game Sense of Agency questionnaire**: 12 items in 4 factors, built with EFA and CFA. — [T&F IJHCI 2023](https://informahealthcare.com/doi/full/10.1080/10447318.2023.2241286)
- **GEQ** (Brockmyer Game Engagement Questionnaire). — [psytests](https://psytests.org/cyber/gameeqen-bl.html)
- **SUS** and **CSI** were used in LDB and LACE-style studies (see section 1, [LDB](https://arxiv.org/abs/2405.00313)). The CSI comes from Cherry & Latulipe, TOCHI 2014 [unverified URL not fetched].

- **Effectance in Games scale** (Ballou et al., CHI PLAY 2021). It has 11 items, but the authors **do not unconditionally recommend** it and propose a 4-item subset. — [PDF](https://people.csail.mit.edu/dkao/pdf/ballou2021effectance.pdf)
- A York researcher validating a player agency questionnaire (2025) says no established tool exists. This is a Reddit claim only [weak source]. — [Reddit](https://red.chatoyer.de/r/truegaming/comments/1jq2v1t/academic_survey_players_agency_experience_in)
- **"Who Did It?"** (MPI, UIST 2024). Measures agency over *generated images* with **temporal binding** (implicit) plus **magnitude estimation** (explicit). The closer an image is to what the user expected, the higher the agency judgment. It used DragGAN. This is the best methodological precedent for linking predictability and agency. — [project](https://whodidit.mpi-inf.mpg.de/); [PDF](https://whodidit.mpi-inf.mpg.de/files/who_did_it_paper.pdf)
- **Hauhio (ICCC 2024).** Predictability is a prerequisite for controllability, and controllability can only be measured per user at a given time, not objectively for a tool. — [PDF](https://computationalcreativity.net/iccc24/papers/ICCC24_paper_170.pdf)
- **GenQuery** (Son et al., CHI 2024). Low controllability frustrated some users but also prompted exploration. — [arXiv](https://arxiv.org/pdf/2310.01287)

### Gaps
- I found no validated "predictability of generative output" scale. Use custom items or behavioural proxies: regenerations per accepted edit, and an expectation-match rating per edit.

## 4. Exhaustive check of the gap (G4): a user study of layer-based generative scene editing for non-experts or players

### Takeaway
**The gap is partially refuted and mostly confirmed.** A 2023 Stanford undergraduate thesis (Linden Li) reports a user study showing that layer-based information helps *novice* users of image generation reach images they like quickly. It is unpublished, and the design details are not on the record page. Apart from that, every layer study found targets artists or professionals (LDB, LACE), and the novice study (SketchFlex) is region-based. No study puts layer editing inside a game or story loop, and none measures player agency.

### Cited Findings
- Li, "Using layer information to improve control of image generation models", Stanford CS undergraduate thesis, June 2023 (advisors Fatahalian and Ré). It has a layer abstraction (spatial layout plus per-layer appearance constraints), a web UI, and a user study with novices. N and measures are not stated on the record page. — [Stanford PURL](https://purl.stanford.edu/tm287xk2802)
- LDB: artists. LACE: professionals. SketchFlex: novices but region-based. — see section 1.
- Searches for CHI 2026 layered/RGBA interface studies found only CV preference studies. — [MRT](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Tang_Masked_Region_Transformer_CVPR_2026_supplemental.pdf), [LayerComposer](https://arxiv.org/html/2510.20820v2)

### Inferences
- Reword G4 as: "no peer-reviewed study of layer-based generative scene editing for novices/players in an interactive-story context, and none measuring agency or predictability against prompt-only editing; closest are Li 2023 (unpublished novice thesis), SketchFlex (region, novices), LDB/LACE (layers, experts)."

- Second pass: the Li thesis full text is not reachable. There is no PDF on the record page, and the SearchWorks page returned empty. Its published successor, **Collage Diffusion** (WACV 2024, RGBA layers plus text), mentions no user study in its abstract. Li's site describes the thesis as covering "layer-based editing interfaces". — [PURL](https://purl.stanford.edu/tm287xk2802); [linden-li.github.io](https://linden-li.github.io); [arXiv 2303.00262](https://arxiv.org/abs/2303.00262)
- ACM DL search pages return **HTTP 403** to the fetch tool, so the DL could not be queried directly. — (dl.acm.org/action/doSearch, tested 2026-11-09)

### Gaps
- The full Li thesis was not read (N, tasks and measures are still unknown). I did not search the ACM DL directly; it is not reachable by this tool, so a CHI '26 or CHI PLAY '26 paper may have been missed.

## 5. Proposed study design

### Takeaway
A within-subjects 2-condition study with about 24 novices, modelled on DirectGPT and PromptCharm. These are suggestions derived from the work cited above, not findings.

### Cited Findings (design precedents)
- Within-subjects tool vs prompt-only baseline, N=12: [PromptCharm](https://arxiv.org/pdf/2403.04014). Prompt counts and time as behavioural measures: [DirectGPT](https://cs.uwaterloo.ca/~dmasson/pdfs/directgpt.pdf). SUS as a usability comparator: [LDB](https://arxiv.org/abs/2405.00313).

### Inferences (proposal)
- **Participants:** 24 novices (LMU students, little or no generative-AI or image-editing experience), screened for experience. Use 24 so condition order can be counterbalanced.
- **Conditions:** (A) prompt-only re-editing: global re-prompt of the whole scene. (B) layer controls: per-layer re-prompt, move, resize, and global prompt still available. Keep the same backend, seed policy and latency so that only the editing interface differs.
- **Tasks per condition** (about 15 min each, with 2 matched story seeds counterbalanced):
  - (1) A target-replication task: reach a described scene state. This measures predictability and efficiency.
  - (2) A free story session: about 5 story turns, including system-triggered story events. This measures agency and satisfaction.
- **Measures:**
  - Behavioural: edits, regenerations, time to target, unintended changes to other objects (rated by coders).
  - Per edit: a 1-item "the result matched what I expected".
  - After each condition: PXI (German version) Autonomy, Mastery and Ease of Control subscales; CSI; NASA-TLX raw; SUS; custom agency and predictability items.
  - Final: preference and a semi-structured interview, analysed thematically.
- **Duration:** about 60 to 75 min. Analysis: paired tests or mixed models (condition and order).
- **Hypotheses:** layers give higher predictability and agency, and fewer unintended changes. Prompt-only may be rated as easier at first; this should be tested, given LACE's learning-curve note.

- **Latency tolerance (input for the design):**
  - Lee, Sundar et al. (CHI EA 2025, N=11 interviews): users were generally tolerant of image-generation waits, saw them as part of the creative process, and rarely noticed progress cues. — [PSU Pure](https://pure.psu.edu/en/publications/while-we-wait-how-users-perceive-waiting-times-and-generation-cue/)
  - I found **no study of latency tolerance in games or story play** for generated images or scenes.
  - Suggestion: log wait times and add a per-turn "the wait was acceptable" item, or manipulate latency as a secondary factor (CPU generation makes this very relevant).
- **Disclosure:** given the perceived-AI bias in Bazzaz & Cooper ([arXiv 2602.14254](https://arxiv.org/abs/2602.14254)), keep AI disclosure identical in both conditions.
- **Agency measure:** add expectation-match ratings per edit, following "Who Did It?" ([UIST 2024](https://whodidit.mpi-inf.mpg.de/)), and optionally the 4-item effectance subset ([Ballou 2021](https://people.csail.mit.edu/dkao/pdf/ballou2021effectance.pdf)).

### Gaps
- Effect sizes for a power analysis are not available from these sources.
- I found no game-specific evidence on how long players will wait for a generated scene.

## Added by the coordinator from ACM DL searches via LMU proxy (2026-11-09)

The searches ran through the LMU library proxy (dl-acm-org.emedien.ub.uni-muenchen.de). The research threads' direct ACM DL access had been blocked (HTTP 403).

**ImageDirector (UIST '26, published 1 Nov 2026, open access, open-source).**
- Citation: Mejia Araujo & Zhang, *ImageDirector: Expressing Spatial Intent via 3D Direct Manipulation for Generative Image Editing*. https://doi.org/10.1145/3830398.3830709
- **This is the closest prior work to our G4. It narrows the gap but does not close it.**
- **Pipeline:**
  - SAM3 segmentation, plus a LaMa clean plate using a 15×15 dilated mask.
  - SAM3D image-to-3D proxy, which the user can move, rotate and scale in Three.js.
  - Harmonisation with IP-Adapter and Nano Banana Pro; the user picks one of 3 variants.
  - Runs on an A100 80 GB.
- **Study design:**
  - Within-subjects, N=12, AB/BA counterbalanced, about 75 min per session.
  - Participants were mostly novices at AI image generation: 3 had never used it, 7 had tried it once or twice.
  - Baseline: text-only Nano Banana Pro.
  - Tasks: spatial art-director tasks, with targets given as red circles or arrows, 3 iterative steps per condition.
- **Measures:**
  - Position error: SAM3 centroid to target.
  - **Visual drift: SSIM outside the union of the old and new object masks.**
  - Expert ratings (Krippendorff's α): spatial intent, background preservation, visual cohesion, overall quality.
  - NASA-TLX.
  - Perceived control and expressiveness, 7-point Likert items adapted from their reference [48].
  - Preference, and a semi-structured interview.
- **Results:**

| Measure | ImageDirector | Text baseline |
|---|---|---|
| Position error | 37 px | 207 px |
| Background SSIM | 0.998 | 0.922 |
| Total time | 7.4 min | 9.3 min |
| Perceived control | 5.89 | 4.39 |
| Expressiveness | 5.56 | 4.33 |
| Preferred by | 8 of 12 | 3 of 12 |

  - NASA-TLX overall: no difference, because of the 3D learning curve.
  - **Visual cohesion and overall quality: no difference.** Experts noted the text baseline sometimes produced *more natural lighting and shadow*, because regenerating the whole scene incidentally re-lights the object. The authors frame this as a precision-vs-integration trade-off.
  - Users "preferred the immediacy of the raw 3D proxy over the latency of harmonization", which bears on our latency question.
  - Users converged on a hybrid workflow: text for semantics, direct manipulation for space.
- **What it does not cover, i.e. what remains for us:**
  - It uses no persistent RGBA layer stack: objects are re-segmented on every edit.
  - There is no story or game loop, no story events, no players, and no agency or PXI measures.
  - It has only 3 iterative steps; no long sequences.
  - It uses closed and A100-class models; nothing on free hardware.
- **Reusable for us:**
  - the masked-SSIM drift measure
  - the expert rubric with α
  - the AB/BA design
  - the control and expressiveness items
  - the finding that harmonisation latency hurts and users wanted the raw proxy first. Our UI could show the moved layer instantly and swap in the re-harmonised version when it is ready.

**Other hits from the same searches (abstracts only):**
- **DiaryPlay** (CHI '26, https://doi.org/10.1145/3772318.3790572). Generates interactive story vignettes from text stories, eliciting environment, characters and events, with an LLM "Controlled Divergence Module" for player and NPC behaviour. It is a story-game authoring system with a user study; its visuals are not generative layered scenes [check the full text].
- **FIERO** (PACMHCI CHI PLAY '26, https://doi.org/10.1145/3831305). Creative writing through collaborative gameplay.
- **"Eye Want It All!"** (CHI EA '26, https://doi.org/10.1145/3772363.3799314). Eye tracking as implicit support for generative inpainting.
- **"I Like Your Story!" / 1001 Nights** (CHI EA '25 demo, https://doi.org/10.1145/3706599.3721163).
- **Not relevant to the gap:** BlendScape (UIST '24, end-user generative customisation of video-call environments) and DiffUHaul (SIGGRAPH Asia '24, training-free object dragging; technical, no user study).
