r"""CPU analysis of the drift experiment: replays the saved per-turn outputs of arm A (layers after each relocate() move)
and arm B (Qwen-Image-2.1 editing the flat composite turn after turn) through layerproto/drift_eval.py.

    fastsdcpu/env/Scripts/python.exe prototype/tools/drift_analyse.py <dir_A> <dir_B> <edits.json> <out_dir>
"""
import glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from PIL import Image
from layerproto import drift_eval as D

dirA, dirB, edits_path, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
E = json.load(open(edits_path)); T = E["target_layer"]; n_turns = len(E["turns"])
nl = len(glob.glob(f"{dirA}/t0_layer*.png"))
layersAt = lambda t: [Image.open(f"{dirA}/t{t}_layer{i}.png").convert("RGBA") for i in range(nl)]
layers0 = layersAt(0)
edits = [dict(layer=T, op="move", region=None) for _ in range(n_turns)]
state = {"t": 0}
def layered_replay(layers, e):
    state["t"] += 1; return layersAt(state["t"])
rowsA = D.run_arm("layered", layers0, edits, layered_edit=layered_replay)
turnsB = sorted(int(os.path.basename(f).split("_")[0][1:]) for f in glob.glob(f"{dirB}/t*_composite.png"))
nB = max(turnsB)
stateB = {"t": 0}
def flat_replay(img, e):
    stateB["t"] += 1; return Image.open(f"{dirB}/t{stateB['t']}_composite.png").convert("RGB").resize(img.size)
# flat editor has no layer masks: the allowed-change region is the old+new fox area taken from arm A's fox layer (same ground-truth edit)
class _Fake:  # reuse the layered ground-truth regions for arm B
    pass
def run_flat():
    img0 = D.composite(layers0); size = img0.size
    import numpy as np
    edited = np.zeros((size[1], size[0]), bool); rows = [dict(turn=0, psnr_out=float("inf"), ssim_out=1.0, lum_shift=0.0)]
    for t in range(1, nB + 1):
        edited |= D.alpha_mask(layersAt(t - 1)[T]) | D.alpha_mask(layersAt(t)[T])
        edited = edited | D.box_mask(size, None)
        img = Image.open(f"{dirB}/t{t}_composite.png").convert("RGB").resize(size)
        rows.append(dict(turn=t, psnr_out=D.psnr_outside(img0, img, edited), ssim_out=D.ssim_outside(img0, img, edited),
                         lum_shift=D.lum_shift(img0, img), edited_frac=float(edited.mean())))
    return rows
rowsB = run_flat()
res = {"A_layered_relocate": dict(rows=rowsA, summary=D.summarise(rowsA)), "B_flat_qwen21": dict(rows=rowsB, summary=D.summarise(rowsB))}
D.save(res, f"{out}/drift_results.json")
print("turn | A psnr_out  ssim_out  | B psnr_out  ssim_out  lum_shift")
for t in range(1, max(len(rowsA), len(rowsB))):
    a = rowsA[t] if t < len(rowsA) else None; b = rowsB[t] if t < len(rowsB) else None
    f = lambda r, k: ("%.1f" % min(r[k], 99) if k == "psnr_out" else "%.3f" % r[k]) if r else "-"
    print(f"{t:4d} | {f(a,'psnr_out'):>9} {f(a,'ssim_out'):>9} | {f(b,'psnr_out'):>9} {f(b,'ssim_out'):>9} {('%.1f' % b['lum_shift']) if b else '-'}")
for k, v in res.items(): print(k, v["summary"])
# strips: composites per turn for both arms
S = 160; rows = []
for name, get in (("A layered", lambda t: D.composite(layersAt(t))), ("B flat", lambda t: Image.open(f"{dirB}/t{t}_composite.png").convert("RGB"))):
    tiles = []
    for t in range(0, (n_turns if name.startswith("A") else nB) + 1):
        tiles.append(get(t).convert("RGB").resize((S, S)))
    rows.append(tiles)
W = max(len(r) for r in rows) * S
canvas = Image.new("RGB", (W, S * len(rows)))
for j, r in enumerate(rows):
    for i, tile in enumerate(r): canvas.paste(tile, (i * S, j * S))
canvas.save(f"{out}/drift_strip.jpg", quality=90)
