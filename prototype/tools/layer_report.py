r"""Numbers behind 'did the model really split the scene?' for a benchmark result folder (CPU).

    fastsdcpu/env/Scripts/python.exe prototype/tools/layer_report.py <results dir with <tag>_layerN.png> <tag> [<tag> ...]

Per layer: share of the canvas that is opaque, bounding box, and the share of the fox / cottage / tree boxes that is opaque (the fox fills
only about a third of its box, so about 32 % means the whole fox). For layer 0 (the background): the mean colour difference to the input
inside the SAM2 masks of the cottage, the tree and the fox. Near 0 = the object is still in the background layer (nothing was taken
out); large = the object was removed and the hole filled. The recomposite difference alone cannot tell these cases apart.
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
SAM = os.path.join(PROTO, "kaggle", "results_segment_inpaint_v2", "seginp_v2", "layers_A_hand_boxes")
REGIONS = {"fox": (205, 325, 375, 452), "cottage": (95, 100, 475, 330), "tree": (300, 0, 512, 352)}
INPUT = np.asarray(Image.open(os.path.join(PROTO, "test_scene", "fox_cottage.png")).convert("RGB").resize((512, 512)), float)
MASKS = {n: np.asarray(Image.open(os.path.join(SAM, f)).convert("RGBA").resize((512, 512)))[..., 3] > 128
         for n, f in (("cottage", "2_cottage.png"), ("tree", "4_big_tree.png"), ("fox", "5_red_fox.png"))}


def report(folder, tag):
    print("==", tag)
    files = sorted(glob.glob(os.path.join(folder, f"{tag}_layer*.png")), key=lambda f: int(f.rsplit("layer", 1)[1][:-4]))
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGBA").resize((512, 512))
        a = np.asarray(im)[..., 3] > 40
        s = f"  layer{i}: opaque {a.mean() * 100:5.1f}%"
        ys, xs = np.nonzero(a)
        if len(xs):
            s += f"  bbox x{xs.min()}-{xs.max()} y{ys.min()}-{ys.max()}"
        for n, (x0, y0, x1, y1) in REGIONS.items():
            s += f" | {n} box {a[y0:y1, x0:x1].mean() * 100:3.0f}%"
        if i == 0:
            rgb = np.asarray(im, float)[..., :3]
            s += "  || background vs input in mask: " + " ".join(f"{n} {np.abs(rgb - INPUT)[m].mean():5.1f}" for n, m in MASKS.items())
        print(s)


if __name__ == "__main__":
    for t in sys.argv[2:]:
        report(sys.argv[1], t)
