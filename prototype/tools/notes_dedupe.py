"""One-off (2026-11-10): de-duplicate research_notes by moving sections VERBATIM into hub notes and leaving pointers.

    python prototype/tools/notes_dedupe.py
Hubs: Layered methods deep dive/joint_text_to_layers.md (methods), Free hostable layered models/platform_terms.md
(platforms, licences), Layered image generation models/modular_pipelines_and_hci.md (matting / segmentation).
Superseded drafts move unchanged to research_notes/archive/. Nothing is paraphrased, so citations are kept.
"""
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N = os.path.join(ROOT, "research_notes")
DD, CP, FH, LI = "Layered methods deep dive", "Coherent layered generation pipeline", "Free hostable layered models", "Layered image generation models"


def path(*p):
    return os.path.join(N, *p)


def read(*p):
    return open(path(*p), encoding="utf-8").read()


def write(text, *p):
    open(path(*p), "w", encoding="utf-8").write(text)


def cut(text, start, end):
    """Return (section, text_without_section). start/end are exact heading lines; end=None -> to end of file."""
    i = text.index(start)
    j = text.index(end, i + len(start)) if end else len(text)
    return text[i:j], text[:i] + text[j:]


def demote(sec, by=1):
    return re.sub(r"^(#{2,5}) ", lambda m: "#" * (len(m.group(1)) + by) + " ", sec, flags=re.M)


def strip_head(sec):
    return sec.split("\n", 1)[1] if "\n" in sec else ""


def git(*a):
    subprocess.run(["git", *a], cwd=ROOT, check=True)


hub = read(DD, "joint_text_to_layers.md")
ptr = "> Merged into [joint_text_to_layers.md](../{}) on 2026-11-10 (moved verbatim, with its citations).\n\n"

# 1. verified release table
rel = read(DD, "release_status_recheck.md")
rel_body = strip_head(rel)

# 2. paper facts: other_layered_methods Q1 (detailed)
olm = read(CP, "other_layered_methods.md")
q1, olm_rest = cut(olm, "## Q1. Which methods do", "## Q2.")
olm_new = olm_rest.replace("## Q2.", "## Q1. Native \"single prompt -> coherent layers\" methods\n\n" + ptr.format("Layered%20methods%20deep%20dive/joint_text_to_layers.md#paper-facts-per-method") + "## Q2.", 1)

# 3. prompt_to_layers §2: only the methods not covered by Q1
ptl = read(LI, "prompt_to_layers_generation.md")
s2, ptl_rest = cut(ptl, "## Multi-layer generators", "## Qwen-Image-Layered, Qwen-Image-2.1")
keep = []
for line in s2.split("\n"):
    if line.startswith("- **") and any(k in line for k in ("**ART (Anonymous", "**PrismLayers", "**LayeringDiff", "**Text2Layer", "**TransPixar", "LiWi")):
        keep.append(line)
    elif line.startswith("  - ") and keep and keep[-1].startswith(("- **ART (", "  - Inputs")):
        keep.append(line)
gaps = s2[s2.index("### Inferences"):].strip()
ptl_new = ptl_rest.replace("## Qwen-Image-Layered, Qwen-Image-2.1", "## Multi-layer generators (ART, PrismLayers, DreamLayer, LayerFusion, PSDiffusion, OmniPSD, Text2Layer, LayeringDiff)\n\n" + ptr.format("Layered%20methods%20deep%20dive/joint_text_to_layers.md#paper-facts-per-method") + "## Qwen-Image-Layered, Qwen-Image-2.1", 1)

# 4. layerdiffuse_and_research_models §3 table
lrm = read(FH, "layerdiffuse_and_research_models.md")
s3, lrm_rest = cut(lrm, "## 3. Research models", None)
lrm_new = lrm_rest.rstrip("\n") + "\n\n## 3. Research models: code and weights status\n\n" + ptr.format("Layered%20methods%20deep%20dive/joint_text_to_layers.md#code-and-weights-table-2026-11-07") + "The 2026-11-09 re-check in that file supersedes most rows.\n"

add = "\n## Release status (verified 2026-11-09)\n\n" + rel_body.strip() + "\n"
add += "\n## Paper facts per method\n\nFrom the 2026-11-07 rounds ([other_layered_methods.md](../Coherent%20layered%20generation%20pipeline/other_layered_methods.md) Q1 and [prompt_to_layers_generation.md](../Layered%20image%20generation%20models/prompt_to_layers_generation.md)); release statements there are superseded by the table above.\n\n"
add += demote(strip_head(q1)).strip() + "\n\n### Additional methods (from prompt_to_layers_generation.md)\n\n" + "\n".join(keep) + "\n\n" + demote(gaps, 1) + "\n"
add += "\n## Code and weights table (2026-11-07)\n\nFrom [layerdiffuse_and_research_models.md](../Free%20hostable%20layered%20models/layerdiffuse_and_research_models.md) §3; older than the release table above, kept for rows it does not cover (See-through, LayerD details).\n\n" + demote(strip_head(s3)).strip() + "\n"
hub = hub.replace("## Top 3 most promising for the game", add.strip() + "\n\n## Top 3 most promising for the game", 1)
hub = hub.replace("Release status of the other methods: `release_status_recheck.md`.", "Release status of all methods: section \"Release status\" below (moved in from `release_status_recheck.md`). Paper facts of all methods are collected here too, so this is the one methods note.")
write(hub, DD, "joint_text_to_layers.md")
write(olm_new, CP, "other_layered_methods.md")
write(ptl_new, LI, "prompt_to_layers_generation.md")
write(lrm_new, FH, "layerdiffuse_and_research_models.md")
git("rm", "-q", path(DD, "release_status_recheck.md"))

# 5. hosting cluster -> platform_terms
hm = read(FH, "hosting_and_matting.md")
a, hm_rest = cut(hm, "## A. Free GPU hosting", "## B.")
b = hm_rest[hm_rest.index("## B."):]
oi = read(FH, "open_items_closeout.md")
pt = read(FH, "platform_terms.md")
pt += "\n## More platforms and Kaggle details (moved from hosting_and_matting.md §A, 2026-11-07)\n\n" + demote(strip_head(a), 1).strip() + "\n"
pt += "\n## Licences and notebooks checked 2026-11-07 (moved from open_items_closeout.md)\n\n" + demote(strip_head(oi), 1).strip() + "\n"
write(pt, FH, "platform_terms.md")
mp = read(LI, "modular_pipelines_and_hci.md")
mp += "\n## Q6. Matting, segmentation, inpainting and harmonisation for hosting (moved from Free hostable layered models/hosting_and_matting.md §B)\n\n" + demote(strip_head(b), 0).strip() + "\n"
write(mp, LI, "modular_pipelines_and_hci.md")
git("rm", "-q", path(FH, "hosting_and_matting.md"), path(FH, "open_items_closeout.md"))

# 6. superseded drafts -> research_notes/archive/
os.makedirs(path("archive"), exist_ok=True)
for rel_path, new in ((os.path.join(DD, "spatial_edit_survey.md"), "spatial_edit_survey (first unverified pass).md"),
                      (os.path.join(DD, "heavy_models_inference_plans.md"), "heavy_models_inference_plans.md"),
                      (os.path.join(DD, "heavy_models_recheck_2026-11-09.md"), "heavy_models_recheck_2026-11-09.md"),
                      (os.path.join(DD, "flux_models_t4_adaptation.md"), "flux_models_t4_adaptation (untested cells).md"),
                      (os.path.join(LI, "image_to_layers_decomposition.md"), "image_to_layers_decomposition (2026-11-07).md")):
    git("mv", path(rel_path), path("archive", new))
print("done")
