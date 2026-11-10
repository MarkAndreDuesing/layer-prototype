"""Settings for the layer prototype.

Defaults live here. Override any of them in prototype/config.json, e.g.
    {"image_backend": "remote", "remote_url": "https://<tunnel>.example"}
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

PROTO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROTO_DIR, "config.json")


@dataclass
class Settings:
    # --- image backend -----------------------------------------------------
    # "local"  = LayerDiffuse SDXL scene-to-layer chain in this process (CPU or GPU)
    # "remote" = the same chain served by layerproto/remote_server.py (Kaggle, Modal, ...)
    image_backend: str = "local"
    remote_url: str = "http://127.0.0.1:8765"
    remote_token: str = ""  # optional shared secret sent as X-Token header

    # The released LayerDiffuse SDXL conditional models (bg2ble / bgble2fg) need SDXL base at their
    # released settings: on SDXL-Turbo the extraction step failed (0-5% transparent, tested 2026-11-07).
    sdxl_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    steps: int = 20
    guidance: float = 7.0
    negative_prompt: str = "bad, ugly"
    width: int = 512   # 512 keeps CPU runs at ~11 min per element; use 1024 on a GPU
    height: int = 512
    # "fast" = one pass of the alpha decoder, "best" = 8-pass augmented median (slower)
    alpha_quality: str = "fast"
    device: str = "auto"  # auto | cpu | cuda

    # --- LLM (prompt refiner + word breakdown) -----------------------------
    # "ollama" = local Ollama; "openai" = any OpenAI-compatible endpoint
    # (OpenRouter, a vLLM/llama.cpp server on Kaggle, LM Studio, ...)
    llm_backend: str = "ollama"
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen3:4b"
    openai_base_url: str = "https://openrouter.ai/api/v1"
    openai_api_key_env: str = "OPENAI_API_KEY"  # name of the env var, never the key itself
    llm_temperature: float = 0.4

    # --- image-to-text per layer --------------------------------------------
    # "ollama" uses a local vision model; "none" disables captions
    captioner: str = "ollama"
    # qwen2.5vl:3b: accurate, ~55 s/layer on this CPU; "moondream": ~14 s but often garbled
    caption_model: str = "qwen2.5vl:3b"

    # --- misc ---------------------------------------------------------------
    variants_per_layer: int = 1
    max_layers: int = 6
    scenes_dir: str = field(default_factory=lambda: os.path.join(PROTO_DIR, "scenes"))

    # --- pipeline steps: each can be switched off to save time -----------------
    step_refine: bool = True            # LLM rewrites the players' text into one clean prompt
    step_split: bool = True             # LLM splits the scene into layers (off = split on commas, no LLM)
    step_words: bool = True             # LLM word types + relations (off = shorter, faster LLM answer)
    step_variants: bool = True          # several variants per layer (off = always 1)
    step_style: bool = True             # append the shared style to every layer prompt
    step_captions: bool = True          # image-to-text per generated variant
    step_word_check: bool = True        # compare caption with the layer's words
    keep_models_loaded: bool = False    # keep LLM / caption model in RAM between calls (faster, ~3-5 GB more)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self) -> str:
        """Persist everything except the scenes folder to prototype/config.json."""
        d = {k: v for k, v in self.to_dict().items() if k != "scenes_dir"}
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        return CONFIG_PATH


# label shown in the UI -> Settings attribute (order = pipeline order)
STEPS = {
    "Prompt refinement (LLM)": "step_refine",
    "Layer split (LLM)": "step_split",
    "Word & relation breakdown (LLM)": "step_words",
    "Multiple variants per layer": "step_variants",
    "Shared style on every layer": "step_style",
    "Image-to-text captions": "step_captions",
    "Word check against captions": "step_word_check",
    "Keep models loaded between steps (faster, more RAM)": "keep_models_loaded",
}


def load_settings() -> Settings:
    s = Settings()
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            for k, v in json.load(f).items():
                if hasattr(s, k):
                    setattr(s, k, v)
    return s
