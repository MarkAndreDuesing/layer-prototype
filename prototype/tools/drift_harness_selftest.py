r"""Self-test of layerproto/drift_eval.py on CPU with stand-in editors (no diffusion models).

    fastsdcpu\env\Scripts\python.exe prototype\tools\drift_harness_selftest.py

The editors here are deliberately simple so the harness logic can be checked:
  layered: moves the target layer (pure pixel move) -> untouched regions must stay identical (PSNR inf)
  flat:    the same move on the flat image, followed by a mild "regeneration" perturbation of the whole image
           (re-encode through a slight blur + brightness creep + noise), mimicking what EdiVal-Agent reports for
           flat flow editors (drift + brightening per turn). The numbers are NOT a result about real editors.
  redecompose: layered + a stand-in decompose that re-extracts layers from the composite with small alpha errors.
Real runs replace these with Qwen-Image-Layered / Qwen-Image-2.1 / Kontext on Kaggle.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from layerproto.drift_eval import composite, run_arm, summarise  # noqa: E402

PROTO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = os.path.join(PROTO, "kaggle", "results_qwen_v7_layers", "layers_forest_512px_4L_3st")
OUT = os.path.join(PROTO, "kaggle", "results_local_drift_selftest")
os.makedirs(OUT, exist_ok=True)
layers0 = [Image.open(os.path.join(D, f"layer_{i}.png")).convert("RGBA") for i in range(4)]
rng = np.random.default_rng(0)
edits = [{"layer": int(1 + t % 3), "op": "move", "dx": int(rng.integers(-20, 21)), "dy": int(rng.integers(-8, 9))}
         for t in range(10)]


def move(layer, dx, dy):
    m = Image.new("RGBA", layer.size, (0, 0, 0, 0)); m.paste(layer, (dx, dy), layer); return m


def layered_edit(layers, e):
    out = list(layers); out[e["layer"]] = move(layers[e["layer"]], e["dx"], e["dy"]); return out


_flat_layers = {"cur": [l.copy() for l in layers0]}


def flat_edit(img, e):
    # same semantic edit, but the editor regenerates the whole frame from the previous output
    _flat_layers["cur"] = layered_edit(_flat_layers["cur"], e)
    target = np.asarray(composite(_flat_layers["cur"]), np.float32)
    prev = np.asarray(img.filter(ImageFilter.GaussianBlur(0.6)), np.float32)
    keep = np.asarray(composite(_flat_layers["cur"]), np.float32) == target
    regen = 0.85 * prev + 0.15 * target                       # carries previous turn's degradation forward
    regen = regen * 1.01 + rng.normal(0, 1.5, regen.shape)    # brightness creep + resampling noise
    return Image.fromarray(np.clip(regen, 0, 255).astype(np.uint8))


def decompose(img, n):
    out = []
    for l in _layers_ref["cur"]:
        a = np.asarray(l.getchannel("A"), np.float32)
        a = np.clip(a + rng.normal(0, 6, a.shape), 0, 255).astype(np.uint8)   # small alpha errors per cycle
        x = l.copy(); x.putalpha(Image.fromarray(a)); out.append(x)
    return out


_layers_ref = {"cur": layers0}


def layered_edit_tracked(layers, e):
    out = layered_edit(layers, e); _layers_ref["cur"] = out; return out


res = {}
for arm in ("layered", "flat", "redecompose"):
    _flat_layers["cur"] = [l.copy() for l in layers0]; _layers_ref["cur"] = layers0
    rows = run_arm(arm, layers0, edits, layered_edit=layered_edit_tracked, flat_edit=flat_edit, decompose=decompose)
    res[arm] = {"summary": summarise(rows), "turns": rows}
    print(arm, json.dumps(res[arm]["summary"], default=float))
json.dump(res, open(os.path.join(OUT, "selftest.json"), "w"), indent=1, default=float)

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for arm, r in res.items():
        t = [x["turn"] for x in r["turns"][1:]]
        ax[0].plot(t, [min(x["psnr_out"], 60) for x in r["turns"][1:]], marker="o", label=arm)
        ax[1].plot(t, [x["lum_shift"] for x in r["turns"][1:]], marker="o", label=arm)
    ax[0].set_title("PSNR outside edited regions (dB, capped 60)"); ax[1].set_title("mean luminance shift")
    for a in ax: a.set_xlabel("turn"); a.legend(fontsize=8)
    fig.suptitle("Harness self-test with STAND-IN editors (not a result)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "selftest_curves.png"), dpi=110)
except Exception as e:
    print("plot skipped:", e)
