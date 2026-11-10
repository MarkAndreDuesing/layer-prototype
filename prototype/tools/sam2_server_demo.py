"""Real /decompose route end to end on a GPU box: Sam2Decomposer behind layerproto.remote_server.build_app, called through the
RemoteDecomposer client over an in-process FastAPI TestClient, then Engine.decompose_image -> scene.json -> compose.

    python prototype/tools/sam2_server_demo.py --image prototype/test_scene/fox_cottage.png --out out/
"""
import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from layerproto.backends import decomposer as D  # noqa: E402
from layerproto.backends.sam2_decomposer import Sam2Decomposer  # noqa: E402
from layerproto.config import Settings  # noqa: E402
from layerproto.engine import Engine  # noqa: E402
from layerproto.remote_server import build_app  # noqa: E402
from layerproto.scene import Scene, new_scene_id  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
J = json.load(open(os.path.join(os.path.dirname(a.image), "fox_cottage.json"), encoding="utf-8"))
names = [layer["name"] for layer in J["layers_back_to_front"]]
t0 = time.time()
dec = Sam2Decomposer(device="cuda")
print("load", round(time.time() - t0, 1), "s", flush=True)
client = TestClient(build_app(chain=type("C", (), {"info": lambda self: {}})(), token="t", decomposer=dec))


class _R:
    def __init__(self, r):
        self.r = r

    def raise_for_status(self):
        self.r.raise_for_status()

    def json(self):
        return self.r.json()


D.requests = type("rq", (), {"post": staticmethod(lambda url, json=None, headers=None, timeout=None:
                                                  _R(client.post("/" + url.rsplit("/", 1)[1], json=json, headers=headers)))})
s = Settings()
s.scenes_dir = tempfile.mkdtemp()
s.remote_url = "http://x"
s.remote_token = "t"
s.width = s.height = 512
eng = Engine(s)
img = Image.open(a.image).convert("RGB")
boxes = [layer["box"] for layer in J["layers_back_to_front"]]
for tag, p in (("by_noun", D.DecomposeParams(names=names, method="sam2-lama")),
               ("by_boxes", D.DecomposeParams(names=names, boxes=boxes, method="sam2-lama"))):
    sc = Scene(id=new_scene_id(tag), user_input=J["caption"])
    t = time.time()
    eng.decompose_image(sc, img, p)
    out = eng.compose(sc)
    print(tag, "layers", [layer.name for layer in sc.layers], "seconds", round(time.time() - t, 1), flush=True)
    folder = sc.folder(s.scenes_dir)
    tiles = [img.resize((256, 256)), out.convert("RGB").resize((256, 256))]
    for layer in sc.layers:
        g = Image.new("RGBA", (512, 512), (40, 160, 60, 255))
        g.alpha_composite(Image.open(f"{folder}/{layer.current().file}").convert("RGBA"))
        tiles.append(g.convert("RGB").resize((256, 256)))
    strip = Image.new("RGB", (256 * len(tiles), 256))
    for i, im in enumerate(tiles):
        strip.paste(im, (256 * i, 0))
    strip.save(f"{a.out}/{tag}_strip.jpg", quality=90)
