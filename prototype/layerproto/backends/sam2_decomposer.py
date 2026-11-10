"""Image -> layers with Grounding DINO + SAM2 + LaMa: segment BY NOUN, fill the background (7 s + fill on a T4).

    dec = Sam2Decomposer(device="cuda")   # needs: pip install git+https://github.com/facebookresearch/sam2 simple-lama-inpainting
    layers = dec(image, DecomposeParams(names=["cottage", "red fox"]))   # background first, then objects in the given order
    # or boxes: DecomposeParams(boxes=[[x0, y0, x1, y1], ...], names=[...]) skips Grounding DINO

Pass `dec` as `decomposer=` to layerproto.remote_server.build_app to serve it on /decompose.
Measured on fox & cottage: docs/models/sam2-lama.md.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from .decomposer import DecomposeParams


class Sam2Decomposer:
    def __init__(self, device: str = "cuda", sam_id: str = "facebook/sam2-hiera-small",
                 dino_id: str = "IDEA-Research/grounding-dino-tiny"):
        import torch
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from simple_lama_inpainting import SimpleLama
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch, self.device = torch, device
        self.pred = SAM2ImagePredictor(build_sam2_hf(sam_id, device=device))
        self.proc = AutoProcessor.from_pretrained(dino_id)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device)
        self.lama = SimpleLama()

    def _box_for(self, im: Image.Image, name: str):
        H, W = im.size[1], im.size[0]
        inp = self.proc(images=im, text=name.lower() + ".", return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.dino(**inp)
        r = self.proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.3, text_threshold=0.25,
                                                             target_sizes=[(H, W)])[0]
        if len(r["boxes"]) == 0:
            return None
        return [float(x) for x in r["boxes"][int(r["scores"].argmax())].tolist()]

    def _mask(self, arr, box):
        self.pred.set_image(arr)
        m, _s, _ = self.pred.predict(box=np.array(box, dtype=np.float32), multimask_output=False)
        return m[0].astype(bool)

    def __call__(self, image: Image.Image, p: DecomposeParams) -> list[Image.Image]:
        im = image.convert("RGB")
        arr = np.asarray(im)
        H, W = arr.shape[:2]
        boxes = list(p.boxes) or [self._box_for(im, n) for n in p.names]
        names = list(p.names) + [f"object {i + 1}" for i in range(len(p.names), len(boxes))]
        masks = [(n, self._mask(arr, b)) for n, b in zip(names, boxes) if b is not None]
        if not masks:
            raise ValueError("no object found: pass names (nouns) or boxes")
        taken = np.zeros((H, W), bool)
        owned = []
        for name, m in reversed(masks):                     # front layers own overlapping pixels
            mm = m & ~taken
            taken |= mm
            owned.append((name, mm))
        owned.reverse()
        hole = Image.fromarray((taken * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9))
        layers = [self.lama(im, hole).resize(im.size).convert("RGBA")]
        for _name, mm in owned:
            a = Image.fromarray((mm * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
            L = im.convert("RGBA")
            L.putalpha(a)
            layers.append(L)
        return layers
