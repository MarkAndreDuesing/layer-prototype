"""Modal deployment of the LayerDiffuse SDXL scene-to-layer chain (live backend for the local UI).

Why Modal: web endpoints are a supported feature (no tunnel), and the Starter plan includes $30/month of credit
(T4 about $0.59/h, L4 about $0.80/h; checked 2026-11-07). Kaggle's Acceptable Use Policy forbids "server farming"
and "tools for circumvention", so a tunnelled server on Kaggle is not clearly allowed; use Kaggle for in-notebook
runs instead (prototype/kaggle/layerproto_chain.ipynb).

One-time setup (from the StabeDiffusionCPU folder):
    pip install modal
    modal setup                                   # opens the browser to log in
    modal secret create layerproto LAYERPROTO_TOKEN=<pick-a-secret>
Deploy:
    modal deploy prototype/remote/modal_app.py
Then in prototype/config.json:
    {"image_backend": "remote", "remote_url": "https://<your-workspace>--layerproto-chain-serve.modal.run",
     "remote_token": "<the same secret>", "width": 1024, "height": 1024}
"""
import os
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0", "diffusers==0.37.0", "transformers>=4.51", "accelerate", "safetensors",
        "huggingface_hub", "peft", "einops", "opencv-python-headless", "pillow", "requests",
        "fastapi[standard]", "pydantic",
    )
    # the prototype code incl. the vendored (patched) LayerDiffuse port; scenes are not uploaded
    .add_local_dir(PROTO, "/root/prototype", ignore=["scenes", "**/__pycache__", "*.log", "kaggle", "remote"])
)

app = modal.App("layerproto-chain", image=image)
cache = modal.Volume.from_name("layerproto-hf-cache", create_if_missing=True)  # model weights survive restarts


@app.function(
    gpu="L4",                      # 24 GB; "T4" also works (16 GB) and is cheaper but slower
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_name("layerproto")],
    timeout=1800,
    scaledown_window=300,          # shut down 5 min after the last request, so idle time costs nothing
)
@modal.concurrent(max_inputs=1)    # one generation at a time per container
@modal.asgi_app()
def serve():
    os.environ.setdefault("HF_HOME", "/cache/hf")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    sys.path.insert(0, "/root/prototype")
    from layerproto.backends.layerdiffuse_chain import LayerDiffuseChain
    from layerproto.remote_server import build_app

    chain = LayerDiffuseChain(model="stabilityai/stable-diffusion-xl-base-1.0", device="cuda", alpha_quality="best")
    return build_app(chain, token=os.environ.get("LAYERPROTO_TOKEN", ""))
