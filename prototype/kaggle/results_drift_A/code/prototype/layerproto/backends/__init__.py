from __future__ import annotations

from ..config import Settings
from .chain_base import ChainBackend, ChainParams


def make_chain_backend(s: Settings) -> ChainBackend:
    """Scene-to-layer chain: local LayerDiffuse SDXL, or the same chain on a remote GPU."""
    if s.image_backend == "remote":
        from .remote_chain import RemoteChain

        return RemoteChain(s.remote_url, s.remote_token)
    from .layerdiffuse_chain import LayerDiffuseChain

    return LayerDiffuseChain(model=s.sdxl_model, device=s.device, alpha_quality=s.alpha_quality)


__all__ = ["ChainBackend", "ChainParams", "make_chain_backend"]
