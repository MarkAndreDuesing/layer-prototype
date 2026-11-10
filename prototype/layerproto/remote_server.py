"""HTTP server exposing the LayerDiffuse scene-to-layer chain, for running it on a GPU (Kaggle, Modal, ...).

    python -m layerproto.remote_server --port 8765 --token <secret>

Endpoints (JSON, images as base64 PNG):
    GET  /health
    POST /chain/background {params}
    POST /chain/blend      {params, scene}
    POST /chain/extract    {params, scene, blended}
    POST /chain/relocate   {params, scene, layer, box, strength, pin_frac} -> {layer, composite}   (move / resize with re-harmonisation)
    GET  /methods          -> {profile, methods: [{method, presets: [{label, path, params}]}]}   (only with a decomposer)
    POST /decompose        {params, image} -> {layers: [b64 PNG, background first]}   (only if a decomposer callable is given)
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


class RelocateReq(BaseModel):
    params: dict
    scene: str              # canvas without the layer (RGB)
    layer: str              # the layer's RGBA (canvas-sized)
    box: list               # new [x, y, w, h] in pixels for the layer's visible content
    strength: float = 0.5
    pin_frac: float = 0.4


class DecomposeReq(BaseModel):
    params: dict
    image: str


class Req(BaseModel):
    params: dict
    scene: str | None = None
    blended: str | None = None


def describe_methods(decomposer, profile: str | None) -> dict:
    """Methods the loaded decomposer offers, with the presets (benchmark paths A / B) of the running profile for the frontend."""
    from dataclasses import asdict
    names = sorted(getattr(decomposer, "by_method", {}) or [])
    presets: dict[str, list] = {}
    if profile:
        try:
            from .serve.profiles import PROFILES
            for c in PROFILES[profile].cases:
                params = {k: v for k, v in asdict(c.params).items() if k in ("method", "resolution", "n_layers", "steps", "sampler", "amodal")}
                label = f"{'A best case' if c.path == 'A' else 'B live demo (~30 s)'}: {c.note}"
                presets.setdefault(c.params.method, []).append({"label": label, "path": c.path, "params": params})
        except Exception:  # noqa: BLE001 - presets are optional
            pass
    names = names or sorted(presets)
    return {"profile": profile, "methods": [{"method": m, "presets": presets.get(m, [])} for m in names]}


def build_app(chain, token: str = "", decomposer=None, profile: str | None = None) -> FastAPI:
    """decomposer: optional callable (PIL image, DecomposeParams) -> list of RGBA PIL images, background first."""
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

    @app.post("/chain/relocate")
    def relocate(req: RelocateReq, x_token: str | None = Header(default=None)):
        check(x_token)
        if not hasattr(chain, "relocate"):
            raise HTTPException(501, "this chain has no relocate()")
        p = ChainParams(**req.params)
        _naive, harm, new_layer = chain.relocate(b64_to_img(req.scene), b64_to_img(req.layer), req.box, p.prompt, p.seed,
                                                 strength=req.strength, pin_frac=req.pin_frac, w=p.width, h=p.height,
                                                 steps=p.steps, guidance=p.guidance, negative=p.negative_prompt)
        return {"layer": img_to_b64(new_layer), "composite": img_to_b64(harm)}

    if decomposer is not None:
        from .backends.decomposer import DecomposeParams

        @app.get("/methods")
        def methods(x_token: str | None = Header(default=None)):
            check(x_token)
            return describe_methods(decomposer, profile)

        @app.post("/decompose")
        def decompose(req: DecomposeReq, x_token: str | None = Header(default=None)):
            check(x_token)
            try:
                layers = decomposer(b64_to_img(req.image).convert("RGB"), DecomposeParams(**req.params))
            except ValueError as e:                      # unknown method, no names, nothing found: the caller's input, not a crash
                raise HTTPException(400, str(e))
            return {"layers": [img_to_b64(L) for L in layers]}

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
