"""End-to-end engine test without the UI: plan -> generate all layers -> update one layer -> captions."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.stdout.reconfigure(encoding="utf-8")

from layerproto.config import load_settings  # noqa: E402
from layerproto.engine import Engine  # noqa: E402

text = sys.argv[1] if len(sys.argv) > 1 else \
    "pixel art, night forest, campfire in the middle, a skeleton dances happily on the right side of the fire, starry sky"
eng = Engine(load_settings())
t = time.time()
if len(sys.argv) > 2:  # reuse an existing scene id: skip planning/generation, test update only
    from layerproto.scene import Scene
    sc = Scene.load(os.path.join(eng.s.scenes_dir, sys.argv[2], "scene.json"))
    obj = next(l for l in sc.layers if l.role == "object")
    eng.update_layer(sc, obj.id, prompt=obj.prompt + ", glowing red eyes", mode="refine", strength=0.6)
    print(f"UPDATE {obj.name} {time.time() - t:.0f}s")
    for l in sc.layers:
        for i, v in enumerate(l.variants):
            print(f"  {l.name} #{i} {v.mode} {v.seconds}s file={v.file}\n      caption: {v.caption}\n      found={v.matched_words} missing={v.missing_words}")
    sys.exit()
sc = eng.plan_scene(text)
print(f"PLAN {time.time() - t:.0f}s  scene={sc.id}  style={sc.style!r}")
for l in sc.layers:
    print(f"  [{l.depth}] {l.name} ({l.role}) adj={l.adjectives} box={l.box}\n      prompt: {l.prompt}")
print("  relations:", sc.relations)

t = time.time()
eng.generate_all(sc, 1, progress=lambda m: print("  ..", m, flush=True))
print(f"GENERATE {time.time() - t:.0f}s")

obj = next(l for l in sc.layers if l.role == "object")
t = time.time()
eng.update_layer(sc, obj.id, prompt=obj.prompt + ", glowing red eyes", mode="refine", strength=0.6)
print(f"UPDATE {obj.name} {time.time() - t:.0f}s")

for l in sc.layers:
    for i, v in enumerate(l.variants):
        print(f"  {l.name} #{i} {v.mode} {v.seconds}s file={v.file}\n      caption: {v.caption}\n      found={v.matched_words} missing={v.missing_words}")
print("FOLDER", os.path.join(eng.s.scenes_dir, sc.id))
