"""CPU test of the /chain/relocate route + RemoteChain.relocate + Engine.relocate_layer with a fake chain (no model)."""
import os
import sys
import tempfile

import requests
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fastapi.testclient import TestClient  # noqa: E402

from layerproto.backends.remote_chain import RemoteChain  # noqa: E402
from layerproto.config import Settings  # noqa: E402
from layerproto.engine import Engine  # noqa: E402
from layerproto.remote_server import build_app  # noqa: E402
from layerproto.scene import Layer, Scene, Variant  # noqa: E402

seen = {}


class FakeChain:
    def info(self):
        return {"backend": "fake"}

    def relocate(self, scene, layer, box, prompt, seed, strength=0.5, pin_frac=0.4, w=512, h=512, steps=20, guidance=7.0,
                 negative="", **k):
        seen.update(box=list(box), prompt=prompt, seed=seed, strength=strength)
        out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        x, y, bw, bh = box
        out.paste((0, 255, 0, 255), (x, y, x + bw, y + bh))
        return scene, scene, out


client = TestClient(build_app(FakeChain(), "tok"))
# route the client's HTTP calls through the in-process app
requests.post = lambda url, json=None, headers=None, timeout=None: client.post(url.replace("http://t", ""), json=json, headers=headers)
tmp = tempfile.mkdtemp()
s = Settings()
s.scenes_dir = tmp
s.width = s.height = 200
s.step_captions = False
s.step_word_check = False
eng = Engine.__new__(Engine)
eng.s = s
eng.chain = RemoteChain("http://t", "tok")
sc = Scene(id="t", user_input="", refined_prompt="", n_layers=2)
folder = sc.folder(tmp)
os.makedirs(folder, exist_ok=True)
Image.new("RGBA", (200, 200), (0, 0, 255, 255)).save(f"{folder}/bg.png")
o = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
o.paste((255, 0, 0, 255), (20, 40, 60, 100))
o.save(f"{folder}/o.png")
bg = Layer(id="background", name="background", role="background", depth=0, prompt="bg")
bg.variants = [Variant(file="bg.png", seed=1, prompt="bg")]
bg.selected = 0
ob = Layer(id="o1", name="obj", role="object", depth=1, prompt="a red box")
ob.box = [0.5, 0.5, 0.4, 0.4]
ob.variants = [Variant(file="o.png", seed=7, prompt="a red box")]
ob.selected = 0
ob.dx = 0.1
sc.layers = [bg, ob]
eng.relocate_layer(sc, "o1")
assert seen["seed"] == 7 and seen["prompt"] == "a red box", seen
x, y, w, h = seen["box"]
assert w <= 81 and h <= 81 and abs(x + w / 2 - 140) <= 1 and abs(y + h / 2 - 140) <= 1, seen
assert len(ob.variants) == 2 and ob.variants[-1].mode == "relocate" and ob.selected == 1 and ob.dx == 0 and ob.scale == 1
print("relocate route test passed:", seen["box"])
