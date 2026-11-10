"""Image -> layers backend (Qwen-Image-Layered, Stable-Layers, RevealLayer, CLD, SAM2 + LaMa ... on a GPU server).

Contract: decompose(image, DecomposeParams) -> list of full-canvas RGBA images, background first, then objects back to front.
The remote server (layerproto/remote_server.py, route /decompose) hosts whichever model the GPU notebook loaded; the
`method` string is passed through so one server can offer several (see docs/models/README.md for what each needs).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

import requests
from PIL import Image

from .remote import b64_to_img, img_to_b64


@dataclass
class DecomposeParams:
    caption: str = ""
    method: str = "qwen-layered"        # qwen-layered | stable-layers | reveallayer | cld | sam2-lama
    n_layers: int = 4
    boxes: list = field(default_factory=list)   # [[x0, y0, x1, y1], ...] in pixels, back to front (RevealLayer / CLD / SAM2)
    names: list = field(default_factory=list)   # one name per box (Grounding DINO / SAM2 by noun)
    resolution: int = 384
    steps: int = 4
    seed: int = 0
    amodal: bool = False                        # SAM2: fill the parts of each object hidden by objects in front (SDXL-inpaint)


class Decomposer(ABC):
    name = "base"

    @abstractmethod
    def decompose(self, image: Image.Image, p: DecomposeParams) -> list[Image.Image]:
        """RGBA layers, background first."""

    def info(self) -> dict:
        return {"decomposer": self.name}


class RemoteDecomposer(Decomposer):
    name = "remote-decompose"

    def __init__(self, url: str, token: str = "", timeout: float = 900):
        self.url = url.rstrip("/")
        self.headers = {"X-Token": token} if token else {}
        self.timeout = timeout

    def decompose(self, image: Image.Image, p: DecomposeParams) -> list[Image.Image]:
        r = requests.post(f"{self.url}/decompose", json={"params": asdict(p), "image": img_to_b64(image.convert("RGB"))},
                          headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return [b64_to_img(s) for s in r.json()["layers"]]


def recomposite(layers: list[Image.Image]) -> Image.Image:
    canvas = layers[0].convert("RGBA").copy()
    for L in layers[1:]:
        canvas.alpha_composite(L.convert("RGBA").resize(canvas.size))
    return canvas
