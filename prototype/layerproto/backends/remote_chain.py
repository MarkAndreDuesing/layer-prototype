"""Client for the scene-to-layer chain running on a remote GPU (Kaggle / Modal) via layerproto/remote_server.py."""
from __future__ import annotations

from dataclasses import asdict

import requests
from PIL import Image

from .chain_base import ChainBackend, ChainParams
from .remote import b64_to_img, img_to_b64


class RemoteChain(ChainBackend):
    name = "remote-chain"

    def __init__(self, url: str, token: str = "", timeout: float = 900):
        self.url = url.rstrip("/")
        self.headers = {"X-Token": token} if token else {}
        self.timeout = timeout

    def _post(self, route: str, payload: dict) -> Image.Image:
        r = requests.post(f"{self.url}/{route}", json=payload, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return b64_to_img(r.json()["image"])

    def background(self, p: ChainParams) -> Image.Image:
        return self._post("chain/background", {"params": asdict(p)})

    def blend(self, scene: Image.Image, p: ChainParams) -> Image.Image:
        return self._post("chain/blend", {"params": asdict(p), "scene": img_to_b64(scene)})

    def extract(self, scene: Image.Image, blended: Image.Image, p: ChainParams) -> Image.Image:
        return self._post("chain/extract", {"params": asdict(p), "scene": img_to_b64(scene),
                                            "blended": img_to_b64(blended)})

    def info(self) -> dict:
        try:
            return requests.get(f"{self.url}/health", headers=self.headers, timeout=15).json()
        except Exception as e:  # noqa: BLE001
            return {"backend": self.name, "url": self.url, "error": str(e)}
