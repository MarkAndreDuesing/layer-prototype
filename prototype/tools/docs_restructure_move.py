"""One-off: move docs into the new structure with `git mv`, then rewrite relative Markdown links everywhere.

    python prototype/tools/docs_restructure_move.py
Writes docs/archive/MOVED.md (old path -> new path).
"""
import os
import re
import subprocess
from urllib.parse import quote, unquote

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOVES = {
    # history
    "docs/findings-log.md": "docs/log.md",
    # research syntheses kept current
    "reports/Research gap and questions.md": "docs/research/research-gap.md",
    "reports/research_cpu_speedups.md": "docs/research/cpu-speedups.md",
    # guides
    "docs/fastsd-cpu/INSTALL.md": "docs/guides/fastsd-cpu/INSTALL.md",
    "docs/fastsd-cpu/USAGE.md": "docs/guides/fastsd-cpu/USAGE.md",
    "prototype/docs/qwen-layered-on-kaggle.md": "docs/guides/qwen-on-kaggle.md",
    # snapshots (content merged into the current docs; originals kept unchanged)
    "reports/Coherent layered generation pipeline.md": "docs/archive/2026-11-07 Coherent layered generation pipeline.md",
    "reports/Layer based architecture models and hosting.md": "docs/archive/2026-11-07 Layer based architecture models and hosting.md",
    "reports/Qwen layered multi model pipeline plan.md": "docs/archive/2026-11-07 Qwen layered multi model pipeline plan.md",
    "reports/Kaggle migration plan.md": "docs/archive/2026-11-07 Kaggle migration plan.md",
    "reports/Image generation models hosting and costs.md": "docs/archive/2026-11-05 Image generation models hosting and costs.md",
    "reports/Free hostable layered models.md": "docs/archive/2026-11-07 Free hostable layered models.md",
    "reports/research_speed_and_apis.md": "docs/archive/2026-11-09 Speed and APIs.md",
    "reports/University GPU access LMU.md": "docs/archive/2026-11-09 University GPU access LMU.md",
    "reports/research_tpu_route.md": "docs/archive/2026-11-09 TPU route.md",
    "reports/Layered image generation models.md": "docs/archive/2026-11-07 Layered image generation models.md",
    "reports/Layered methods - how to get them working.md": "docs/archive/2026-11-09 Layered methods - how to get them working.md",
    "reports/Layered loop open paths.md": "docs/archive/2026-11-09 Layered loop open paths.md",
    "docs/summary-2026-11-09-research-block.md": "docs/archive/2026-11-09 Research block summary.md",
    "docs/investigation-checklist.md": "docs/archive/2026-11-09 Investigation checklist.md",
    "docs/open-paths.md": "docs/archive/2026-11-09 Open paths.md",
    "docs/session-summary-sketch-to-image.md": "docs/archive/2026-11-07 Session summary sketch-to-image.md",
    "docs/archive/SETUP_AND_USAGE_GUIDE_3_paths.md": "docs/archive/2026-11-07 Setup and usage guide 3 paths.md",
}


def git(*a):
    subprocess.run(["git", *a], cwd=ROOT, check=True)


def main():
    for old, new in MOVES.items():
        os.makedirs(os.path.join(ROOT, os.path.dirname(new)), exist_ok=True)
        git("mv", old, new)
    rev = {os.path.normpath(n): os.path.normpath(o) for o, n in MOVES.items()}
    fwd = {os.path.normpath(o): os.path.normpath(n) for o, n in MOVES.items()}
    link = re.compile(r"(\]\()([^)\s]+)(\))")
    md = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True).stdout.split("\n")
    md = [m for m in md if os.path.exists(os.path.join(ROOT, m))]
    changed = 0
    for rel in md:
        cur = os.path.normpath(rel)
        old_loc = rev.get(cur, cur)
        p = os.path.join(ROOT, rel)
        s = open(p, encoding="utf-8").read()

        def fix(m):
            target = m.group(2)
            if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                return m.group(0)
            path, frag = (target.split("#", 1) + [""])[:2]
            if not path:
                return m.group(0)
            tgt_old = os.path.normpath(os.path.join(os.path.dirname(old_loc), unquote(path)))
            tgt_new = fwd.get(tgt_old, tgt_old)
            if not os.path.exists(os.path.join(ROOT, tgt_new)):
                return m.group(0)
            newrel = os.path.relpath(tgt_new, os.path.dirname(cur)).replace("\\", "/")
            if path.endswith("/") and not newrel.endswith("/"):
                newrel += "/"
            out = quote(newrel, safe="/._-()~") + ("#" + frag if frag else "")
            return m.group(1) + out + m.group(3)

        s2 = link.sub(fix, s)
        if s2 != s:
            open(p, "w", encoding="utf-8").write(s2); changed += 1
    lines = ["# Moved files (docs restructure, 2026-11-09)", "",
             "Every file kept its content; links inside all Markdown files were rewritten to the new paths.", "",
             "| Old path | New path |", "|---|---|"]
    lines += [f"| `{o}` | `{n}` |" for o, n in MOVES.items()]
    open(os.path.join(ROOT, "docs", "archive", "MOVED.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("moved", len(MOVES), "files; rewrote links in", changed, "files")


if __name__ == "__main__":
    main()
