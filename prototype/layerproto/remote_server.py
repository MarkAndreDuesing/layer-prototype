"""HTTP server exposing the LayerDiffuse scene-to-layer chain, for running it on a GPU (Kaggle, Modal, ...).

    python -m layerproto.remote_server --port 8765 --token <secret>

Endpoints (JSON, images as base64 PNG):
    GET  /health
    POST /chain/background {params}
    POST /chain/blend      {params, scene}
    POST /chain/extract    {params, scene, blended}
Locally, point prototype/config.json at it:
    {"image_backend": "remote", "remote_url": "https://<tunnel-url>", "remote_token": "<secret>"}
"""
from __future__ import annotations

import argparse

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from .backends.chain_base import ChainParams
from .backends.layerdiffuse_chain import LayerDiffuseChain
from .backends.remote import b64_to_img, img_to_b64


class Req(BaseModel):
    params: dict
    scene: str | None = None
    blended: str | None = None


def build_app(chain, token: str = "") -> FastAPI:
    app = FastAPI(title="layerproto chain backend")

    def check(x_token):
        if token and x_token != token:
            raise HTTPException(401, "bad token")

    @app.get("/health")
    def health(x_token: str | None = Header(default=None)):
        check(x_token)
        return chain.info()

    @app.post("/chain/background")
    def background(req: Req, x_token: str | None = Header(default=None)):
        check(x_token)
        return {"image": img_to_b64(chain.background(ChainParams(**req.params)))}

    @app.post("/chain/blend")
    def blend(req: Req, x_token: str | None = Header(default=None)):
        check(x_token)
        return {"image": img_to_b64(chain.blend(b64_to_img(req.scene), ChainParams(**req.params)))}

    @app.post("/chain/extract")
    def extract(req: Req, x_token: str | None = Header(default=None)):
        check(x_token)
        img = chain.extract(b64_to_img(req.scene), b64_to_img(req.blended), ChainParams(**req.params))
        return {"image": img_to_b64(img)}

    return app


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--token", default="")
    ap.add_argument("--alpha-quality", default="best")
    a = ap.parse_args()
    uvicorn.run(build_app(LayerDiffuseChain(model=a.model, alpha_quality=a.alpha_quality), a.token),
                host=a.host, port=a.port)
