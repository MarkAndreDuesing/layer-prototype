"""CPU test of layerproto.event_apply with a stub editor on the fox & cottage layers."""
import glob, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from layerproto import events as E
from layerproto.event_apply import apply_event

RL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kaggle", "results_reveallayer_fox_v15", "reveal_out", "gs1.0_st12", "fox_cottage")
layers = [Image.open(f"{RL}/layer_{i}_rgba.png").convert("RGBA").resize((256, 256)) for i in range(5)]
names = ["background", "cottage", "arch", "tree", "fox"]
plan = E.normalise_plan({"strategy": "per_layer", "background_prompt": "dark night", "effects": ["rain"],
                         "layers": [{"name": "background", "action": "relight", "prompt": ""}, {"name": "cottage", "action": "relight", "prompt": ""},
                                    {"name": "arch", "action": "remove", "prompt": ""}, {"name": "tree", "action": "relight", "prompt": ""},
                                    {"name": "fox", "action": "reprompt", "prompt": "a golden statue of a fox"}], "reason": "test"}, names)
calls = []
def editor(img, prompt, seed):
    calls.append((prompt[:30], seed))
    out = E.grade(img, "night")
    if seed == 42 and "fox" in prompt:                      # first try of the fox returns an emptied layer -> must be retried
        out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    return out
out, rep = apply_event(layers, plan, editor, event="a rainy night")
assert all(r["name"] != "arch" or r["action"] == "remove" for r in rep)
fox = [r for r in rep if r["name"] == "fox"][0]
assert fox["accepted"] and len(fox["tries"]) == 2, fox            # retry kicked in
assert len(out) >= 4 + 3, len(out)                                 # 4 kept layers + shadows + rain overlay
assert out[0].size == (256, 256)
print("event_apply test passed:", len(out), "layers out;", [(r["name"], r.get("accepted"), r.get("shadow")) for r in rep])
