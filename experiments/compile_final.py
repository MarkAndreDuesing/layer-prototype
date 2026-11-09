"""Compile the final chosen configuration of each path into one sheet per path.

Rows = seeds, columns = [input, pixelart, impressionist, hyperreal, nostyle].
Also copies the chosen images to experiments/final/<path>/ with a mapping README.
"""
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

EXP = Path(r"C:\Users\Mark Other\Documents\StabeDiffusionCPU\experiments")
OUT = EXP / "final"
STYLES = ["pixelart", "impressionist", "hyperreal", "nostyle"]
CELL, LABEL_H, SIDE_W, TITLE_H = 320, 22, 90, 34

PATHS = {
    "path1_sdturbo_img2img": {
        "title": "Path 1: SD-Turbo img2img | filled sketch | Strength 0.8 | 4 steps | Guidance 1",
        "src": "r2_p1", "cfg": "filled_s80", "seeds": [123123, 7, 2024],
        "input": "inputs/init_filled.png",
        "note": "Style-capable setting, but the skeleton is lost or deformed in all 3 seeds. "
                "Strength 0.7 (filled_s70) keeps the skeleton but barely restyles; see r2_p1.",
    },
    "path2_sdxlturbo_img2img": {
        "title": "Path 2: SDXL-Turbo img2img | filled sketch | Strength 0.8 | 4 steps | Guidance 1",
        "src": "r2_p2", "cfg": "filled_s80", "seeds": [123123, 7, 2024],
        "input": "inputs/init_filled.png",
        "note": "Recommended overall. Seed 2024 duplicates the skeleton.",
    },
    "path3_controlnet_neutral": {
        "title": "Path 3: SD1.5 neutral base + LCM-LoRA + scribble ControlNet (Canny edges) | scale 0.7 | 4 steps | Guidance 1",
        "src": "r2_p3", "cfg": "neutral_g10_c70", "seeds": [123123, 7],
        "input": "inputs/control_canny.png",
        "note": "Best layout adherence; only 2 seeds were generated (123123, 7). "
                "6-step variant: neutral_g10_c70_6steps in the same folders.",
    },
}


def main() -> None:
    readme = ["Final chosen configuration per path (source folders are relative to experiments/)\n"]
    for name, p in PATHS.items():
        dest = OUT / name
        dest.mkdir(parents=True, exist_ok=True)
        cols = 1 + len(STYLES)
        sheet = Image.new(
            "RGB",
            (SIDE_W + CELL * cols, TITLE_H + LABEL_H + CELL * len(p["seeds"])),
            "white",
        )
        d = ImageDraw.Draw(sheet)
        d.text((6, 8), p["title"], fill="black")
        heads = ["input"] + STYLES
        for c, h in enumerate(heads):
            d.text((SIDE_W + c * CELL + 6, TITLE_H + 5), h, fill="black")
        inp = Image.open(EXP / p["input"]).convert("RGB").resize((CELL, CELL))
        for r, seed in enumerate(p["seeds"]):
            y = TITLE_H + LABEL_H + r * CELL
            d.text((6, y + 6), f"seed {seed}", fill="black")
            sheet.paste(inp, (SIDE_W, y))
            for c, style in enumerate(STYLES, start=1):
                f = EXP / p["src"] / f"seed{seed}" / f"{p['cfg']}__{style}.png"
                shutil.copy(f, dest / f"seed{seed}__{style}.png")
                sheet.paste(
                    Image.open(f).convert("RGB").resize((CELL, CELL)),
                    (SIDE_W + c * CELL, y),
                )
        sheet.save(OUT / f"FINAL_{name}.png")
        readme.append(
            f"{name}\n  config: {p['cfg']} from {p['src']}/seed*/\n"
            f"  seeds: {p['seeds']}\n  images: final/{name}/\n  sheet: final/FINAL_{name}.png\n"
            f"  note: {p['note']}\n"
        )
    (OUT / "README.txt").write_text("\n".join(readme), encoding="utf-8")
    print("done:", sorted(x.name for x in OUT.iterdir()))


if __name__ == "__main__":
    main()
