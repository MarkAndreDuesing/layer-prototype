"""Scene engine built on the LayerDiffuse SDXL scene-to-layer chain.

Core loop (no LLM needed):
    background layer   = chain.background(background prompt)
    each element layer = generated INTO the composite of the layers below it:
                             blended = chain.blend(scene_below, element prompt)      # bg2ble
                             layer   = chain.extract(scene_below, blended, prompt)    # bgble2fg -> aligned RGBA
    re-prompt a layer  = same two steps against the layers below it; nothing else changes
    story event        = a new element generated into the current full composite (new top layer)
    move / resize      = canvas transform on the stored RGBA layer (dx, dy, scale); no model call
    final image        = background + every visible layer, bottom to top, with transforms

Optional add-ons (switchable): LLM prompt refinement / layer split / word breakdown, captions + word check.
"""
from __future__ import annotations

import random
import time
from typing import Callable, Optional

from PIL import Image

from .backends import ChainParams, make_chain_backend
from .config import Settings
from .llm import breakdown, make_captioner, make_llm, word_check
from .scene import Layer, Scene, Variant, new_scene_id, slug

Progress = Optional[Callable[[str], None]]


class Engine:
    def __init__(self, settings: Settings):
        self.s = settings
        self.chain = make_chain_backend(settings)
        self.llm = make_llm(settings)
        self.captioner = make_captioner(settings)

    def reconfigure(self, settings: Settings) -> None:
        """Apply new settings live; the image model is only reloaded if a setting it depends on changed."""
        old = self.s
        self.s = settings
        keys = ("image_backend", "sdxl_model", "device", "alpha_quality", "remote_url", "remote_token")
        if any(getattr(old, k) != getattr(settings, k) for k in keys):
            self.chain = make_chain_backend(settings)
        self.llm = make_llm(settings)
        self.captioner = make_captioner(settings)

    # ---- describe -> layers (LLM split is an optional add-on; comma split otherwise) -------------
    def plan_scene(self, user_text: str, mode: str = "objects", n_layers: int | None = None,
                   scene_id: str | None = None) -> Scene:
        n = int(n_layers or self.s.max_layers)
        s = self.s
        d = breakdown(self.llm, user_text, n, mode, refine=s.step_refine, split=s.step_split, words=s.step_words)
        scene = Scene(id=scene_id or new_scene_id(user_text), user_input=user_text,
                      refined_prompt=d.get("refined_prompt", ""), style=d.get("style", ""),
                      relations=d.get("relations", []), words=d.get("words", []), split_mode=mode, n_layers=n)
        seen = set()
        for l in d["layers"]:
            lid = slug(l["name"])
            while lid in seen:
                lid += "x"
            seen.add(lid)
            scene.layers.append(Layer(id=lid, name=l["name"], role=l["role"], depth=int(l["depth"]),
                                      prompt=l["prompt"], adjectives=list(l.get("adjectives", [])),
                                      box=list(l.get("box", [0, 0, 1, 1]))))
        scene.layers.sort(key=lambda l: l.depth)
        scene.save(self.s.scenes_dir)
        return scene

    # ---- generation -----------------------------------------------------------------------------
    def generate_all(self, scene: Scene, variants: int | None = None, progress: Progress = None,
                     missing_only: bool = False) -> Scene:
        """Background first, then every element bottom-to-top, each generated into the scene below it."""
        n = self._n(variants)
        todo = [l for l in self._ordered(scene) if l.enabled and not (missing_only and l.variants)]
        for i, layer in enumerate(todo):
            self._say(progress, f"layer {i + 1}/{len(todo)}: {layer.name}")
            self._generate_layer(scene, layer, n, progress)
            layer.selected = len(layer.variants) - n  # first of the new variants
            scene.save(self.s.scenes_dir)
        return scene

    def update_layer(self, scene: Scene, layer_id: str, prompt: str | None = None, variants: int = 1,
                     progress: Progress = None) -> Scene:
        """Re-prompt one layer: regenerated against the layers below it; all other layers stay as they are."""
        layer = scene.layer(layer_id)
        if prompt:
            layer.prompt = prompt
        n = self._n(variants)
        self._generate_layer(scene, layer, n, progress)
        layer.selected = len(layer.variants) - 1
        scene.save(self.s.scenes_dir)
        return scene

    def add_event_layer(self, scene: Scene, name: str, prompt: str, variants: int = 1,
                        progress: Progress = None) -> Scene:
        """Story event: a new element generated into the current full composite, as the new top layer."""
        lid = slug(name)
        while any(l.id == lid for l in scene.layers):
            lid += "x"
        top = max([l.depth for l in scene.layers] or [0]) + 1
        layer = Layer(id=lid, name=name, role="object", depth=top, prompt=prompt)
        scene.layers.append(layer)
        n = self._n(variants)
        self._generate_layer(scene, layer, n, progress)
        layer.selected = len(layer.variants) - 1
        scene.save(self.s.scenes_dir)
        return scene

    def _generate_layer(self, scene: Scene, layer: Layer, n: int, progress: Progress):
        folder = scene.folder(self.s.scenes_dir)
        if layer.role == "background":
            for _ in range(n):
                p = self._params(scene, layer.prompt)
                t0 = time.time()
                self._say(progress, f"{layer.name}: generating background")
                img = self.chain.background(p)
                self._store(scene, layer, img, p, "new", time.time() - t0, progress)
            return
        below = self.compose(scene, below_depth=layer.depth, save=False)
        tag = f"{layer.id}__scene_in_{int(time.time())}.png"
        below.save(f"{folder}/{tag}")
        for _ in range(n):
            p = self._params(scene, layer.prompt)
            t0 = time.time()
            self._say(progress, f"{layer.name}: drawing into the scene (bg2ble)")
            blended = self.chain.blend(below, p)
            self._say(progress, f"{layer.name}: extracting the layer (bgble2fg)")
            img = self.chain.extract(below, blended, p)
            v = self._store(scene, layer, img, p, "new", time.time() - t0, progress)
            v.blend = scene.save_image(self.s.scenes_dir, layer, blended, f"v{len(layer.variants) - 1:02d}-blend-{p.seed}")
            v.scene_in = tag

    def _store(self, scene, layer, img, p, mode, secs, progress) -> Variant:
        rel = scene.save_image(self.s.scenes_dir, layer, img, f"v{len(layer.variants):02d}-{mode}-{p.seed}")
        hist = img.split()[-1].histogram()
        v = Variant(file=rel, seed=p.seed, prompt=p.prompt, mode=mode, seconds=round(secs, 1),
                    transparent_pct=round(100 * sum(hist[:10]) / max(1, sum(hist)), 1))
        if self.s.step_captions:
            self._say(progress, f"{layer.name}: captioning")
            try:
                v.caption = self.captioner.caption(img)
            except Exception as e:  # noqa: BLE001 - captions are optional
                v.caption = f"(caption failed: {e})"
        if self.s.step_word_check and v.caption and not v.caption.startswith("("):
            v.matched_words, v.missing_words = word_check(v.caption, [layer.name] + list(layer.adjectives))
        layer.variants.append(v)
        return v

    # ---- composition (Photoshop-style stack) --------------------------------------------------------
    def _ordered(self, scene: Scene) -> list[Layer]:
        return sorted(scene.layers, key=lambda l: (l.role != "background", l.depth))

    def object_order(self, scene: Scene) -> list[Layer]:
        """Element layers from bottom to top."""
        return [l for l in self._ordered(scene) if l.role != "background"]

    def move_layer(self, scene: Scene, layer_id: str, direction: str) -> Scene:
        order = self.object_order(scene)
        i = next(k for k, l in enumerate(order) if l.id == layer_id)
        j = i + 1 if direction == "up" else i - 1
        if 0 <= j < len(order):
            order[i], order[j] = order[j], order[i]
        for depth, l in enumerate(order, start=1):
            l.depth = depth
        scene.layers = [l for l in scene.layers if l.role == "background"] + order
        scene.save(self.s.scenes_dir)
        return scene

    def transform_layer(self, scene: Scene, layer_id: str, dx: float, dy: float, scale: float) -> Scene:
        layer = scene.layer(layer_id)
        layer.dx, layer.dy, layer.scale = float(dx), float(dy), max(0.05, float(scale))
        scene.save(self.s.scenes_dir)
        return scene

    def _placed(self, img: Image.Image, layer: Layer, W: int, H: int) -> Image.Image:
        """Apply a layer's move/resize to its full-canvas RGBA image."""
        img = img.convert("RGBA").resize((W, H))
        if layer.dx == 0 and layer.dy == 0 and layer.scale == 1:
            return img
        bbox = img.split()[-1].getbbox() or (0, 0, W, H)
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        if layer.scale != 1:
            sw, sh = max(1, int(W * layer.scale)), max(1, int(H * layer.scale))
            scaled = img.resize((sw, sh), Image.LANCZOS)
            ox, oy = int(cx - cx * layer.scale), int(cy - cy * layer.scale)  # keep the visible centre in place
        else:
            scaled, ox, oy = img, 0, 0
        out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        out.paste(scaled, (ox + int(layer.dx * W), oy + int(layer.dy * H)))
        return out

    def compose(self, scene: Scene, below_depth: int | None = None, save: bool = True) -> Image.Image:
        """Stack the selected variants. below_depth: only the background and elements under that depth."""
        folder = scene.folder(self.s.scenes_dir)
        W, H = self.s.width, self.s.height
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        for l in self._ordered(scene):
            if l.role != "background" and below_depth is not None and l.depth >= below_depth:
                continue
            v = l.current()
            if not (l.enabled and v):
                continue
            canvas.alpha_composite(self._placed(Image.open(f"{folder}/{v.file}"), l, W, H))
        if save and below_depth is None:
            canvas.save(f"{folder}/final.png")
        return canvas

    def select(self, scene: Scene, layer_id: str, index: int) -> Scene:
        layer = scene.layer(layer_id)
        layer.selected = max(0, min(index, len(layer.variants) - 1))
        scene.save(self.s.scenes_dir)
        return scene

    # ---- helpers -----------------------------------------------------------------------------------
    def _n(self, variants) -> int:
        return max(1, int(variants or self.s.variants_per_layer)) if self.s.step_variants else 1

    def _params(self, scene: Scene, prompt: str) -> ChainParams:
        style = scene.style.strip() if self.s.step_style else ""
        full = prompt if not style or style.lower() in prompt.lower() else f"{prompt}, {style}"
        return ChainParams(prompt=full, seed=random.randint(0, 2**31 - 1), width=self.s.width, height=self.s.height,
                           steps=self.s.steps, guidance=self.s.guidance, negative_prompt=self.s.negative_prompt)

    def estimate_seconds(self, n_backgrounds: int, n_elements: int) -> int:
        """Measured on this laptop: ~16 s per step at 512 px, CFG 7 (SDXL base, CPU); scales with pixels."""
        if self.s.image_backend == "remote":
            per_pass = 15  # [est] T4 / L4 at 1024 px
        else:
            per_pass = 16 * self.s.steps * (self.s.width * self.s.height) / (512 * 512)
        cap = (14 if "moondream" in self.s.caption_model else 55) if self.s.step_captions else 0
        return int(n_backgrounds * (per_pass + cap) + n_elements * (2 * per_pass + cap))

    @staticmethod
    def _say(progress: Progress, msg: str):
        if progress:
            progress(msg)
