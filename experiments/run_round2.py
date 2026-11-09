"""Round two: more seeds, Strength sweep for SDXL-Turbo, neutral base for ControlNet.

Usage: env\\Scripts\\python.exe run_round2.py <r2_p2|r2_p3>
Output: experiments/<name>/seed<seed>/<cfg>__<style>.png + one contact sheet per seed.
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

NEUTRAL_BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
NEGATIVE = "blurry, low quality, multiple skeletons, extra people, text, watermark"


def run(name: str) -> None:
    from backend.lcm_text_to_image import LCMTextToImage

    if name in ("r2_p1", "r2_p2"):
        s = make_settings("p2" if name == "r2_p2" else "p1")
        s.init_image = Image.open(INPUTS / "init_filled.png")
        configs = [(f"filled_s{int(v*100)}", {"strength": v}) for v in (0.7, 0.8, 0.9)]
        if name == "r2_p1":
            # 5 steps makes Strength 0.9 run 4 real steps instead of 3
            configs.append(("filled_s90_5steps", {"strength": 0.9, "steps": 5}))
        seeds = [123123, 7, 2024]
    else:
        s = make_settings("p3")
        s.lcm_lora.base_model_id = NEUTRAL_BASE
        s.controlnet._control_image = Image.open(INPUTS / "control_canny.png")
        configs = [
            ("neutral_g10_c70", {"guidance": 1.0, "neg": "", "scale": 0.7}),
            ("neutral_g15neg_c70", {"guidance": 1.5, "neg": NEGATIVE, "scale": 0.7}),
        ]
        seeds = [123123, 7]

    t = LCMTextToImage()
    t.init("cpu", s)
    log = []
    for seed in seeds:
        s.seed = seed
        out_dir = EXP / name / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for cfg_name, cfg in configs:
            if name in ("r2_p1", "r2_p2"):
                s.strength = cfg["strength"]
                s.inference_steps = cfg.get("steps", 4)
            else:
                s.guidance_scale = cfg["guidance"]
                s.negative_prompt = cfg["neg"]
                s.controlnet.conditioning_scale = cfg["scale"]
            for style in STYLES:
                s.prompt = prompt_for(style)
                t0 = time.time()
                try:
                    t.generate(s)[0].save(out_dir / f"{cfg_name}__{style}.png")
                    log.append({"seed": seed, "cfg": cfg_name, "style": style,
                                "secs": round(time.time() - t0, 1)})
                    print(f"{name} seed{seed} {cfg_name} {style} {log[-1]['secs']}s", flush=True)
                except Exception:
                    traceback.print_exc()
                    log.append({"seed": seed, "cfg": cfg_name, "style": style, "error": True})
                gc.collect()
        contact_sheet(out_dir, [c[0] for c in configs], f"{name}_seed{seed}")
    (EXP / name / "log.json").write_text(json.dumps(log, indent=1))
    print("ALL DONE", name, flush=True)


def run_extra(name: str) -> None:
    """Add one more config row to the existing r2_p2 / r2_p3 seed folders.

    r2_p2_5steps: SDXL-Turbo, Strength 0.9, 5 steps (4 real steps), seeds as before.
    r2_p3_6steps: neutral-base ControlNet, Guidance 1.0, scale 0.7, 6 steps.
    Existing images are untouched; sheets are rebuilt with all rows.
    """
    from backend.lcm_text_to_image import LCMTextToImage

    if name == "r2_p2_5steps":
        base, seeds, new_cfg = "r2_p2", [123123, 7, 2024], "filled_s90_5steps"
        s = make_settings("p2")
        s.init_image = Image.open(INPUTS / "init_filled.png")
        s.strength, s.inference_steps = 0.9, 5
        rows = ["filled_s70", "filled_s80", "filled_s90", new_cfg]
    else:
        base, seeds, new_cfg = "r2_p3", [123123, 7], "neutral_g10_c70_6steps"
        s = make_settings("p3")
        s.lcm_lora.base_model_id = NEUTRAL_BASE
        s.controlnet._control_image = Image.open(INPUTS / "control_canny.png")
        s.controlnet.conditioning_scale = 0.7
        s.inference_steps = 6
        rows = ["neutral_g10_c70", "neutral_g15neg_c70", new_cfg]

    t = LCMTextToImage()
    t.init("cpu", s)
    log = []
    for seed in seeds:
        s.seed = seed
        out_dir = EXP / base / f"seed{seed}"
        for style in STYLES:
            s.prompt = prompt_for(style)
            t0 = time.time()
            try:
                t.generate(s)[0].save(out_dir / f"{new_cfg}__{style}.png")
                log.append({"seed": seed, "cfg": new_cfg, "style": style,
                            "secs": round(time.time() - t0, 1)})
                print(f"{name} seed{seed} {new_cfg} {style} {log[-1]['secs']}s", flush=True)
            except Exception:
                traceback.print_exc()
            gc.collect()
        contact_sheet(out_dir, rows, f"{base}_seed{seed}")
    (EXP / base / f"log_{new_cfg}.json").write_text(json.dumps(log, indent=1))
    print("ALL DONE", name, flush=True)


if __name__ == "__main__":
    if sys.argv[1].endswith(("_5steps", "_6steps")):
        run_extra(sys.argv[1])
    else:
        run(sys.argv[1])
