"""Edit one RGBA layer with a generative editor (Qwen-Image-2.1 RGBA edit, ...) and keep its transparency.

Measured on fox & cottage (docs/concepts/editing.md §5): the editor returned a fully opaque scene or an emptied layer on
6 of 10 first tries; accepting an edit only when the transparent share stays close to the input and retrying with the next
seed gave 10 of 10 accepted edits (4 on the second try).

    edited, report = edit_layer_keep_alpha(lambda img, seed: qwen21_edit(img, prompt, seed), layer_rgba, seeds=(42, 7, 123, 2024))
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from PIL import Image


def transparent_share(img: Image.Image, thr: int = 16) -> float:
    return float((np.asarray(img.convert("RGBA"))[..., 3] < thr).mean())


def edit_layer_keep_alpha(editor: Callable[[Image.Image, int], Image.Image], layer: Image.Image,
                          seeds: Sequence[int] = (42, 7, 123, 2024), tol: float = 0.15,
                          lock_alpha: bool = False) -> tuple[Image.Image, dict]:
    """Run `editor(layer, seed)` until the transparent share is within `tol` of the input's.

    Returns (edited layer, report). If no seed passes, the closest attempt is returned with report["accepted"] = False
    (the caller can then fall back to the unedited layer). lock_alpha=True also multiplies the edit's alpha by the
    input alpha, trimming spill outside the original shape (useful when the share is right but the outline drifted).
    """
    layer = layer.convert("RGBA")
    want = transparent_share(layer)
    best, best_err, tries = None, 1e9, []
    for seed in seeds:
        out = editor(layer, seed).convert("RGBA")
        if out.size != layer.size:
            out = out.resize(layer.size)
        err = abs(transparent_share(out) - want)
        tries.append({"seed": seed, "transparent": round(transparent_share(out), 3), "ok": err < tol})
        if err < best_err:
            best, best_err = out, err
        if err < tol:
            break
    if lock_alpha and best is not None:
        a = np.minimum(np.asarray(best.getchannel("A")), np.asarray(layer.getchannel("A")))
        best = best.copy()
        best.putalpha(Image.fromarray(a.astype(np.uint8)))
    return best, {"accepted": best_err < tol, "input_transparent": round(want, 3), "tries": tries}
