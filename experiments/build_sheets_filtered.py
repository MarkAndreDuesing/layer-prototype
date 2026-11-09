"""Rebuild the comparison sheets from the CURRENT contents of all_images/.

Images you removed from all_images/ are drawn as black cells; every cell has its
all_images filename printed underneath. Reads all_images_manifest.csv for the
grid structure only. Does NOT copy anything into all_images/ (compile_all.py
would restore the removed files, so do not rerun that script).
"""
import csv
import shutil
import textwrap
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from compile_all import INPUT_FILE, PHASE_NAME, STYLE_PROMPT, STYLES

EXP = Path(r"C:\Users\Mark Other\Documents\StabeDiffusionCPU\experiments")
ALL = EXP / "all_images"
SHEETS = EXP / "sheets"
OLD = EXP / "sheets_unfiltered"

CELL, CAP_H, SIDE, TITLE_H, HEAD_H = 256, 46, 360, 40, 22
ROW_H = CELL + CAP_H


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def caption(draw, x, y, text, fnt, fill="black"):
    for i, line in enumerate(textwrap.wrap(text, width=38, break_long_words=True)[:3]):
        draw.text((x + 3, y + 3 + i * 13), line, fill=fill, font=fnt)


def main() -> None:
    rows = list(csv.DictReader(open(EXP / "all_images_manifest.csv", encoding="utf-8")))
    present = {p.name for p in ALL.glob("*.png")}

    if SHEETS.exists() and not OLD.exists():
        shutil.move(str(SHEETS), str(OLD))     # keep the earlier unfiltered sheets
    SHEETS.mkdir(exist_ok=True)

    with open(EXP / "all_images_status.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "status"])
        for r in rows:
            w.writerow([r["file"], "kept" if r["file"] in present else "removed"])

    groups = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        order = (int(r["phase"]), float(r["strength"] or 0), float(r["scale"] or 0),
                 int(r["steps"]), float(r["guidance"]), r["cfg"])
        groups[(r["path"], r["input"], int(r["seed"]))][order][r["style"]] = r

    fnt, fsm = font(12), font(11)
    for (path, inp, seed), cfgs in sorted(groups.items()):
        keys = sorted(cfgs)
        kept = sum(1 for k in keys for r in cfgs[k].values() if r["file"] in present)
        total = sum(len(cfgs[k]) for k in keys)
        sheet = Image.new("RGB", (SIDE + CELL * 5, TITLE_H + HEAD_H + ROW_H * len(keys)), "white")
        d = ImageDraw.Draw(sheet)
        d.text((6, 6), f"{path} | input: {inp} | seed {seed} | kept {kept} of {total} | "
                       f"black = removed | file names under each image are the all_images names",
               fill="black", font=fnt)
        d.text((SIDE + 6, TITLE_H + 4), "input image", fill="black", font=fnt)
        for c, st in enumerate(STYLES, start=1):
            d.text((SIDE + c * CELL + 6, TITLE_H + 4), f"{st}: {STYLE_PROMPT[st]}"[:38],
                   fill="black", font=fsm)
        inp_name = INPUT_FILE[inp]
        thumb = Image.open(EXP / inp_name).convert("RGB").resize((CELL, CELL))
        for ri, key in enumerate(keys):
            y = TITLE_H + HEAD_H + ri * ROW_H
            rec0 = next(iter(cfgs[key].values()))
            lines = [f"phase {rec0['phase']}: {PHASE_NAME[int(rec0['phase'])].split(' (')[0]}",
                     f"cfg: {rec0['cfg']}", f"src: {Path(rec0['src']).parent}"]
            if path in ("P1", "P2"):
                lines.append(f"strength {rec0['strength']}  steps {rec0['steps']}  guidance {rec0['guidance']}")
            else:
                lines.append(f"base {rec0['base']}")
                lines.append(f"scale {rec0['scale']}  steps {rec0['steps']}  guidance {rec0['guidance']}"
                             + ("  +neg prompt" if rec0["negative"] else ""))
            for li, text in enumerate(lines):
                d.text((6, y + 6 + li * 15), text, fill="black", font=fnt)
            sheet.paste(thumb, (SIDE, y))
            caption(d, SIDE, y + CELL, inp_name, fsm, fill="gray")
            for c, st in enumerate(STYLES, start=1):
                rec = cfgs[key].get(st)
                x = SIDE + c * CELL
                if rec is None:
                    continue
                if rec["file"] in present:
                    im = Image.open(ALL / rec["file"]).convert("RGB").resize((CELL, CELL))
                    sheet.paste(im, (x, y))
                    caption(d, x, y + CELL, rec["file"], fsm)
                else:
                    d.rectangle([x, y, x + CELL - 1, y + CELL - 1], fill="black")
                    caption(d, x, y + CELL, rec["file"], fsm, fill="gray")
        sheet.save(SHEETS / f"{path}_input-{inp}_seed{seed}.png")
        print(f"{path} {inp} seed{seed}: kept {kept}/{total}")


if __name__ == "__main__":
    main()
