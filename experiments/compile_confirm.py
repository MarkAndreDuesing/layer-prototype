"""Compile the confirmation-test images (experiments/confirm/) like all_images/.

  python compile_confirm.py init    copy every image to confirm_images/ with a settings-based name
                                    and write confirm_manifest.csv (run once; do NOT rerun after
                                    filtering, it would restore removed files)
  python compile_confirm.py sheets  build confirm_sheets/ from what is currently in confirm_images/;
                                    removed files become black cells, every cell shows its file name

One sheet per path and seed. Rows follow the plan: image-guided blocks first, text-only block last.
"""
import csv
import shutil
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

EXP = Path(r"C:\Users\Mark Other\Documents\StabeDiffusionCPU\experiments")
SRC = EXP / "confirm"
IMGS = EXP / "confirm_images"
SHEETS = EXP / "confirm_sheets"
STYLES = ["pixelart", "impressionist", "hyperreal", "nostyle"]
STYLE_PROMPT = {
    "pixelart": "pixel art, retro 16-bit game scene",
    "impressionist": "impressionist oil painting, visible brushstrokes, soft light",
    "hyperreal": "hyperrealistic photograph, ultra detailed, cinematic lighting",
    "nostyle": "(no style words)",
}
# block -> (path, input, description, row order)
BLOCK = {
    "A": ("P2", "filled", "SDXL-Turbo img2img, filled sketch, 4 steps, guidance 1.0", 1),
    "B": ("P2", "raw", "SDXL-Turbo img2img, plain sketch, 4 steps, guidance 1.0", 2),
    "E": ("P2", "textonly", "SDXL-Turbo text-to-image (no input), 4 steps, guidance 1.0", 3),
    "C": ("P1", "filled", "SD-Turbo img2img, filled sketch, 4 steps, guidance 1.0", 1),
    "F": ("P1", "textonly", "SD-Turbo text-to-image (no input), 4 steps, guidance 1.0", 2),
    "D": ("P3", "canny-raw", "neutral SD1.5 + LCM-LoRA + ControlNet (Canny of plain sketch), scale 0.7, 6 steps, guidance 1.0", 1),
    "G": ("P3", "textonly", "neutral SD1.5 + LCM-LoRA text-to-image (no ControlNet, no input), 6 steps, guidance 1.0", 2),
}
INPUT_FILE = {"filled": "inputs/init_filled.png", "raw": "inputs/init_raw.png",
              "canny-raw": "inputs/control_canny.png"}
CELL, CAP_H, SIDE, TITLE_H, HEAD_H = 256, 46, 360, 40, 22
ROW_H = CELL + CAP_H


def settings_of(block: str, cfg: str) -> str:
    if block in ("A", "B", "C"):
        return f"str{cfg.split('_s')[-1]}-steps4-g1.0"
    if block == "D":
        return "neutral-scale70-steps6-g1.0"
    if block == "E":
        return "sdxlturbo-steps4-g1.0"
    if block == "F":
        return "sdturbo-steps4-g1.0"
    return "lcmlora-neutral-steps6-g1.0"


def init() -> None:
    IMGS.mkdir(exist_ok=True)
    rows = []
    for png in sorted(SRC.rglob("*__*.png")):
        block, seed_dir = png.relative_to(SRC).parts[:2]
        cfg, style = png.stem.split("__")
        path, inp, _, _ = BLOCK[block]
        seed = int(seed_dir[4:])
        name = f"blk{block}__{path}__{inp}__{settings_of(block, cfg)}__seed{seed}__{style}.png"
        shutil.copy(png, IMGS / name)
        rows.append({"file": name, "block": block, "path": path, "input": inp,
                     "settings": settings_of(block, cfg), "seed": seed, "style": style,
                     "src": str(png.relative_to(EXP))})
    with open(EXP / "confirm_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("copied", len(rows))


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def caption(d, x, y, text, fnt, fill="black"):
    for i, line in enumerate(textwrap.wrap(text, width=38, break_long_words=True)[:3]):
        d.text((x + 3, y + 3 + i * 13), line, fill=fill, font=fnt)


def sheets() -> None:
    SHEETS.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(EXP / "confirm_manifest.csv", encoding="utf-8")))
    present = {p.name for p in IMGS.glob("*.png")}
    with open(EXP / "confirm_status.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "status"])
        for r in rows:
            w.writerow([r["file"], "kept" if r["file"] in present else "removed"])
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in rows:
        groups[(r["path"], int(r["seed"]))][(BLOCK[r["block"]][3], r["block"], r["settings"])][r["style"]] = r
    fnt, fsm = font(12), font(11)
    for (path, seed), cfgs in sorted(groups.items()):
        keys = sorted(cfgs)
        kept = sum(1 for k in keys for r in cfgs[k].values() if r["file"] in present)
        total = sum(len(cfgs[k]) for k in keys)
        sheet = Image.new("RGB", (SIDE + CELL * 5, TITLE_H + HEAD_H + ROW_H * len(keys)), "white")
        d = ImageDraw.Draw(sheet)
        d.text((6, 6), f"CONFIRMATION TEST | {path} | seed {seed} | kept {kept} of {total} | black = removed | "
                       f"file names under each image are the confirm_images names", fill="black", font=fnt)
        d.text((SIDE + 6, TITLE_H + 4), "input image", fill="black", font=fnt)
        for c, st in enumerate(STYLES, start=1):
            d.text((SIDE + c * CELL + 6, TITLE_H + 4), f"{st}: {STYLE_PROMPT[st]}"[:38], fill="black", font=fsm)
        for ri, key in enumerate(keys):
            _, block, settings = key
            y = TITLE_H + HEAD_H + ri * ROW_H
            _, inp, desc, _ = BLOCK[block]
            for li, text in enumerate([f"block {block}: {inp}", settings] + textwrap.wrap(desc, 44)[:3]):
                d.text((6, y + 6 + li * 15), text, fill="black", font=fnt)
            if inp == "textonly":
                d.rectangle([SIDE, y, SIDE + CELL - 1, y + CELL - 1], fill=(225, 225, 225))
                d.text((SIDE + 40, y + CELL // 2 - 6), "text only: no input image", fill="black", font=fnt)
            else:
                sheet.paste(Image.open(EXP / INPUT_FILE[inp]).convert("RGB").resize((CELL, CELL)), (SIDE, y))
                caption(d, SIDE, y + CELL, INPUT_FILE[inp], fsm, fill="gray")
            for c, st in enumerate(STYLES, start=1):
                rec = cfgs[key].get(st)
                x = SIDE + c * CELL
                if rec is None:
                    continue
                if rec["file"] in present:
                    sheet.paste(Image.open(IMGS / rec["file"]).convert("RGB").resize((CELL, CELL)), (x, y))
                    caption(d, x, y + CELL, rec["file"], fsm)
                else:
                    d.rectangle([x, y, x + CELL - 1, y + CELL - 1], fill="black")
                    caption(d, x, y + CELL, rec["file"], fsm, fill="gray")
        sheet.save(SHEETS / f"CONFIRM_{path}_seed{seed}.png")
        print(f"{path} seed{seed}: kept {kept}/{total}")


if __name__ == "__main__":
    {"init": init, "sheets": sheets}[sys.argv[1]]()
