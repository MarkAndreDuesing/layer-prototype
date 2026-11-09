"""Compile EVERY generated image under experiments/ (no selection) into:

  all_images/            one flat folder, files named by their exact settings
  all_images_manifest.csv  one row per image with all settings and the source file
  sheets/                comparison sheets per path / input image / seed
  README.md              how to read the names and sheets

Sheets: one per (path, input image, seed). Rows are in the order the runs were
made (phase, then configuration); columns are the input image plus the 4 styles.
"""
import csv
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

EXP = Path(r"C:\Users\Mark Other\Documents\StabeDiffusionCPU\experiments")
ALL = EXP / "all_images"
SHEETS = EXP / "sheets"
STYLES = ["pixelart", "impressionist", "hyperreal", "nostyle"]
STYLE_PROMPT = {
    "pixelart": "pixel art, retro 16-bit game scene",
    "impressionist": "impressionist oil painting, visible brushstrokes, soft light",
    "hyperreal": "hyperrealistic photograph, ultra detailed, cinematic lighting",
    "nostyle": "(no style words)",
}
NEG = "blurry, low quality, multiple skeletons, extra people, text, watermark"

INPUT_FILE = {
    "raw": "inputs/init_raw.png",
    "blur": "inputs/init_blur.png",
    "filled": "inputs/init_filled.png",
    "canny-raw": "inputs/control_canny.png",
    "scribble-raw": "inputs/control_scribble.png",
    "canny-filled": "inputs/control_canny_filled.png",
}
PHASE_NAME = {
    1: "batch1 (plain sketch + blur, 1 seed)",
    2: "filled1 (filled sketch, 1 seed)",
    3: "round2 (3 or 2 seeds)",
    4: "round2-extra (5/6 step variants)",
}
MODEL = {"P1": "stabilityai/sd-turbo", "P2": "stabilityai/sdxl-turbo"}


def parse(src_dir: str, cfg: str) -> dict:
    """Settings of one configuration, from the run scripts and the folder/config names."""
    d = {"steps": 4, "guidance": 1.0, "negative": "", "strength": "", "scale": "",
         "base": "", "control": "", "phase": 0}
    path = {"p1": "P1", "p2": "P2", "p3": "P3"}[src_dir.split("_")[-1] if src_dir.startswith("r2_")
                                                else src_dir.split("_")[0]]
    d["path"] = path
    if src_dir in ("p1", "p2", "p3"):
        d["phase"] = 1
    elif src_dir.endswith("_filled"):
        d["phase"] = 2
    else:
        d["phase"] = 3
    if path in ("P1", "P2"):
        d["model"] = MODEL[path]
        parts = cfg.split("_")
        d["input"] = parts[0]                      # raw | blur | filled
        d["strength"] = int(parts[1][1:]) / 100
        if cfg.endswith("5steps"):
            d["steps"] = 5
            if path == "P2":
                d["phase"] = 4
    else:
        d["model"] = "SD1.5 + LCM-LoRA + scribble ControlNet"
        if cfg.startswith("neutral"):
            d["base"] = "stable-diffusion-v1-5 (neutral)"
            d["input"] = "canny-raw"
            d["scale"] = 0.7
            if "g15neg" in cfg:
                d["guidance"], d["negative"] = 1.5, NEG
            if cfg.endswith("6steps"):
                d["steps"], d["phase"] = 6, 4
        else:
            d["base"] = "Lykon/dreamshaper-8"
            parts = cfg.split("_")
            kind = parts[0]                         # canny | scribble
            filled = parts[1] == "filled"
            d["input"] = f"{kind}-{'filled' if filled else 'raw'}"
            d["scale"] = int(parts[-1][1:]) / 100
        d["control"] = d["input"]
    return d


def short(d: dict) -> str:
    if d["path"] in ("P1", "P2"):
        core = f"str{int(round(d['strength'] * 100))}"
    else:
        b = "neutral" if "neutral" in d["base"] else "dreamshaper8"
        core = f"{b}-scale{int(round(d['scale'] * 100))}"
    g = f"g{d['guidance']}" + ("neg" if d["negative"] else "")
    return f"{core}-steps{d['steps']}-{g}"


def main() -> None:
    ALL.mkdir(exist_ok=True)
    SHEETS.mkdir(exist_ok=True)
    rows = []
    for png in sorted(EXP.rglob("*.png")):
        rel = png.relative_to(EXP)
        top = rel.parts[0]
        if top in ("inputs", "final", "all_images", "sheets") or png.name.startswith("_sheet_"):
            continue
        if "__" not in png.stem:
            continue
        cfg, style = png.stem.split("__")
        seed = 123123
        if rel.parts[1].startswith("seed"):
            seed = int(rel.parts[1][4:])
        d = parse(top, cfg)
        d.update(cfg=cfg, style=style, seed=seed, src=str(rel))
        d["file"] = f"{d['path']}__ph{d['phase']}__{d['input']}__{short(d)}__seed{seed}__{style}.png"
        rows.append(d)

    for d in rows:
        shutil.copy(EXP / d["src"], ALL / d["file"])
    fields = ["file", "path", "phase", "input", "model", "base", "control", "strength",
              "scale", "steps", "guidance", "negative", "seed", "style", "cfg", "src"]
    rows.sort(key=lambda r: (r["path"], r["input"], r["seed"], r["phase"], r["cfg"],
                             STYLES.index(r["style"])))
    with open(EXP / "all_images_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # Sheets: one per (path, input, seed)
    groups = defaultdict(lambda: defaultdict(dict))   # key -> row key -> style -> record
    for r in rows:
        order = (r["phase"], r["strength"] or 0, r["scale"] or 0, r["steps"], r["guidance"], r["cfg"])
        groups[(r["path"], r["input"], r["seed"])][order][r["style"]] = r
    cell, side, title_h, head_h = 256, 360, 36, 22
    for (path, inp, seed), cfgs in sorted(groups.items()):
        keys = sorted(cfgs)
        sheet = Image.new("RGB", (side + cell * 5, title_h + head_h + cell * len(keys)), "white")
        dr = ImageDraw.Draw(sheet)
        dr.text((6, 6), f"{path} | input: {inp} | seed {seed} | all runs, in run order "
                        f"(prompt = style words + same scene text)", fill="black")
        dr.text((side + 6, title_h + 4), "input image", fill="black")
        for c, st in enumerate(STYLES, start=1):
            dr.text((side + c * cell + 6, title_h + 4), f"{st}: {STYLE_PROMPT[st]}"[:40], fill="black")
        thumb = Image.open(EXP / INPUT_FILE[inp]).convert("RGB").resize((cell, cell))
        for ri, key in enumerate(keys):
            y = title_h + head_h + ri * cell
            any_rec = next(iter(cfgs[key].values()))
            lines = [f"phase {any_rec['phase']}: {PHASE_NAME[any_rec['phase']].split(' (')[0]}",
                     f"cfg: {any_rec['cfg']}", f"src: {Path(any_rec['src']).parent}"]
            if path in ("P1", "P2"):
                lines.insert(2, f"strength {any_rec['strength']}  steps {any_rec['steps']}  guidance {any_rec['guidance']}")
            else:
                lines.insert(2, f"base {any_rec['base']}")
                lines.insert(3, f"scale {any_rec['scale']}  steps {any_rec['steps']}  guidance {any_rec['guidance']}"
                                + ("  +neg prompt" if any_rec["negative"] else ""))
            for li, text in enumerate(lines):
                dr.text((6, y + 6 + li * 14), text, fill="black")
            sheet.paste(thumb, (side, y))
            for c, st in enumerate(STYLES, start=1):
                rec = cfgs[key].get(st)
                if rec:
                    im = Image.open(EXP / rec["src"]).convert("RGB").resize((cell, cell))
                    sheet.paste(im, (side + c * cell, y))
        sheet.save(SHEETS / f"{path}_input-{inp}_seed{seed}.png")

    print("images:", len(rows), "sheets:", len(list(SHEETS.glob('*.png'))))
    for p in ("P1", "P2", "P3"):
        print(p, sum(1 for r in rows if r["path"] == p))


if __name__ == "__main__":
    main()
