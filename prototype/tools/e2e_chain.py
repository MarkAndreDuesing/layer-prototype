"""Headless end-to-end test of the chain engine: comma split -> background -> elements into the scene -> move."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image  # noqa: E402
from layerproto.config import load_settings  # noqa: E402
from layerproto.engine import Engine  # noqa: E402

S = load_settings()
S.step_refine = S.step_split = S.step_words = S.step_captions = S.step_word_check = False
eng = Engine(S)
text = sys.argv[1] if len(sys.argv) > 1 else (
    "pixel art, night forest of tall pine trees, full moon and stars, "
    "a large campfire with a stone ring in the center, a skeleton dancing on the right side of the fire")
sc = eng.plan_scene(text)
print("scene", sc.id, "| style", repr(sc.style))
for l in sc.layers:
    print(f"  [{l.depth}] {l.role:10s} {l.prompt}")
t = time.time()
eng.generate_all(sc, 1, progress=lambda m: print("  ..", m, flush=True))
print(f"GENERATE {time.time() - t:.0f}s")
for l in eng._ordered(sc):
    v = l.current()
    print(f"  {l.name}: {v.seconds}s, {v.transparent_pct}% transparent, file={v.file}")
eng.compose(sc)
el = eng.object_order(sc)[0]
eng.transform_layer(sc, el.id, -0.15, 0.0, 1.0)
moved = eng.compose(sc, save=False)
folder = sc.folder(S.scenes_dir)
moved.save(f"{folder}/final_moved.png")
tiles = [Image.open(f"{folder}/final.png"), moved] + [Image.open(f"{folder}/{l.current().file}") for l in eng._ordered(sc)]
grid = Image.new("RGBA", (256 * len(tiles), 256), (40, 160, 60, 255))
for i, im in enumerate(tiles):
    tile = Image.new("RGBA", (256, 256), (40, 160, 60, 255)); tile.alpha_composite(im.convert("RGBA").resize((256, 256)))
    grid.paste(tile, (i * 256, 0))
grid.save(os.path.join(S.scenes_dir, "_e2e_chain_grid.png"))
print("FOLDER", folder)
