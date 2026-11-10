"""Registry of server profiles (one profile per Kaggle session) and their fox & cottage benchmark cases (paths A and B).

A profile = pip packages + Kaggle kernel sources (pre-downloaded weights) + a factory that returns a decomposer + the benchmark
cases. `tools/make_profile_notebook.py --profile NAME --mode bench|serve` turns one into a notebook.

Paths (see docs/models/TEMPLATE.md): A = best case, up to 8 minutes of compute per image; B = live-demo budget, about 30 s of
compute per image. Compute = the generation call only, never the model load.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..backends.decomposer import DecomposeParams

CAPTION = ("Storybook illustration of a thatched stone cottage with a large leafy tree on the right, a stone gate arch on the left, "
           "small village houses behind, a cobblestone path leading into the foreground and a red fox standing on the path.")
NAMES = ["village houses", "cottage", "stone gate arch", "big tree", "red fox"]
BOXES = [[0, 195, 115, 280], [95, 100, 475, 330], [62, 245, 150, 322], [180, 0, 512, 352], [205, 325, 375, 452]]


@dataclass
class Case:
    tag: str                       # file name stem, e.g. "A"
    path: str                      # "A" best case or "B" live-demo budget
    params: DecomposeParams
    note: str = ""
    repeats: int = 2               # the first call warms up (kernels, caches); report the last


@dataclass
class Profile:
    name: str
    pip: list = field(default_factory=list)
    kernel_sources: list = field(default_factory=list)
    gpu: str = "NvidiaTeslaT4"
    factory: str = ""              # "module:function" returning a decomposer callable (image, DecomposeParams) -> layers
    cases: list = field(default_factory=list)
    doc: str = ""


def P(**kw):
    return DecomposeParams(caption=CAPTION, **kw)


PROFILES = {
    "qwen-layered": Profile(
        name="qwen-layered",
        pip=["diffusers==0.37.0", "transformers>=4.51", "accelerate", "gguf"],
        kernel_sources=["markandreduesing/qwen-layered-weights-transformer", "markandreduesing/qwen-layered-weights-base"],
        factory="layerproto.serve.qwen_layered:make",
        doc="docs/models/qwen-image-layered.md (also stable-layers.md, qwen-lightning.md)",
        cases=[
            Case("qwenlayered_A", "A", P(method="qwen-layered", resolution=512, n_layers=5, steps=16), "512 px, 5 layers, 16 Euler steps"),
            Case("qwenlayered_B", "B", P(method="qwen-layered", resolution=384, n_layers=4, steps=4), "384 px, 4 layers, 4 Euler steps"),
            Case("stablelayers_A", "A", P(method="stable-layers", resolution=512, n_layers=5, steps=8, sampler="heun"), "512 px, 5 layers, 8 Heun steps"),
            Case("stablelayers_B", "B", P(method="stable-layers", resolution=384, n_layers=5, steps=3), "384 px, 5 layers, 3 Euler steps"),
            Case("lightning_A", "A", P(method="qwen-lightning", resolution=512, n_layers=5, steps=4), "512 px, 5 layers, 4 steps"),
            Case("lightning_B", "B", P(method="qwen-lightning", resolution=384, n_layers=5, steps=2), "384 px, 5 layers, 2 steps"),
        ]),
    "qwen21": Profile(
        name="qwen21",
        pip=["transformers>=5.17", "gguf>=0.10", "accelerate", "git+https://github.com/huggingface/diffusers"],
        factory="layerproto.serve.qwen21:make",
        doc="docs/models/qwen-image-2.1.md",
        cases=[
            Case("qwen21_A", "A", P(method="qwen21-extract", names=NAMES[1:], boxes=BOXES[1:]), "background + 4 objects, 8 steps, 512 px"),
            Case("qwen21_B", "B", P(method="qwen21-extract", names=["red fox"], boxes=[BOXES[4]]), "one object per request, 8 steps, 512 px", repeats=2),
        ]),
}
