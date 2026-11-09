"""Supplementary tests using the filled-in sketch (inputs/sketch_filled.png).

Usage: env\\Scripts\\python.exe run_filled.py <p1|p2|p3>
Reuses the helpers from run_tests.py; writes to experiments/<path>_filled.
"""
import gc
import json
import sys
import time
import traceback

from run_tests import (
    EXP,
    INPUTS,
    SIZE,
    STYLES,
    contact_sheet,
    make_settings,
    prompt_for,
)
from PIL import Image


def run(path: str) -> None:
    from backend.annotators.image_control_factory import ImageControlFactory
    from backend.lcm_text_to_image import LCMTextToImage

    filled = Image.open(INPUTS / "sketch_filled.png").convert("RGB").resize((SIZE, SIZE))
    filled.save(INPUTS / "init_filled.png")
    ImageControlFactory().create_control("Canny").get_control_image(filled).save(
        INPUTS / "control_canny_filled.png"
    )

    s = make_settings(path)
    out_dir = EXP / f"{path}_filled"
    out_dir.mkdir(exist_ok=True)
    if path == "p3":
        configs = [("canny_filled_c60", 0.6), ("canny_filled_c100", 1.0)]
        s.controlnet._control_image = Image.open(INPUTS / "control_canny_filled.png")
    else:
        configs = [("filled_s60", 0.6), ("filled_s80", 0.8)]
        s.init_image = filled

    t = LCMTextToImage()
    t.init("cpu", s)
    log = []
    for name, val in configs:
        if path == "p3":
            s.controlnet.conditioning_scale = val
        else:
            s.strength = val
        for style in STYLES:
            s.prompt = prompt_for(style)
            t0 = time.time()
            try:
                t.generate(s)[0].save(out_dir / f"{name}__{style}.png")
                log.append({"cfg": name, "style": style, "secs": round(time.time() - t0, 1)})
                print(f"{path} {name} {style} {log[-1]['secs']}s", flush=True)
            except Exception:
                traceback.print_exc()
                log.append({"cfg": name, "style": style, "error": True})
            gc.collect()
        contact_sheet(out_dir, [c[0] for c in configs], f"{path}_filled")
    (out_dir / "log.json").write_text(json.dumps(log, indent=1))
    print("ALL DONE", path, flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
