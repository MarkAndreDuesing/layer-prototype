"""LLM prompt refiner + word breakdown, and per-layer image-to-text.

Backends:
  OllamaLLM        local Ollama (default), JSON forced through Ollama's schema `format`
  OpenAICompatLLM  any OpenAI-compatible server: OpenRouter, a vLLM / llama.cpp server on Kaggle, LM Studio ...
Both expose chat_json(system, user, schema) -> dict, so the engine never cares which one runs.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from abc import ABC, abstractmethod

import requests
from PIL import Image

from .config import Settings

# ---------------------------------------------------------------------------
# Breakdown schema: nouns -> layers, adjectives -> that layer's look, verbs/prepositions -> relations
# ---------------------------------------------------------------------------
BREAKDOWN_SCHEMA = {
    "type": "object",
    "properties": {
        "refined_prompt": {"type": "string"},
        "style": {"type": "string"},
        "layers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string", "enum": ["background", "object"]},
                    "depth": {"type": "integer"},
                    "prompt": {"type": "string"},
                    "adjectives": {"type": "array", "items": {"type": "string"}},
                    "box": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["name", "role", "depth", "prompt", "adjectives", "box"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "relation": {"type": "string"},
                    "object": {"type": "string"},
                },
                "required": ["subject", "relation", "object"],
            },
        },
        "words": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                    "type": {"type": "string", "enum": ["noun", "adjective", "verb", "preposition", "other"]},
                    "layer": {"type": "string"},
                },
                "required": ["word", "type", "layer"],
            },
        },
    },
    "required": ["refined_prompt", "style", "layers", "relations", "words"],
}

_HEADER = ("You turn a short, possibly messy scene description (keywords from several players) into a plan "
           "for a text-to-image model.\n\nDo all of this:\n")
_PART_REFINE = """- refined_prompt: one clean, faithful scene description. Keep every subject, action, colour and spatial relation the players gave. Do not invent new objects. No rendered text, letters or logos.
- style: the visual style the players asked for (e.g. "pixel art", "watercolor"). Empty string if none was given; do not invent one.
"""
_PART_LAYERS = """- layers: exactly ONE layer with role "background" (depth 0), then the object layers (role "object"). {split_rule}
   - The background prompt must contain ALL scenery words from the input (place, landscape, time of day, weather, sky, lighting), e.g. "night forest clearing under a starry sky", and no characters or main objects.
   - name: short (e.g. "tower", "Bob", "campfire"; for a group "campfire + logs").
   - depth: 0 for the background, higher numbers are closer to the viewer.
   - prompt: the element as it should appear IN this scene: the noun plus its own adjectives, pose/action and where it is (it is drawn into the scene below it, so its placement words matter).
   - adjectives: only words from the players' input that describe that layer's LOOK (colour, mood, material, size, time of day). Positions like "on the right" are NOT adjectives; they go into box.
   - box: rough placement in the final scene as [x, y, width, height], each 0..1 (background is [0,0,1,1]).
"""
_PART_WORDS = """- relations: verbs and prepositions that connect the subjects, as subject / relation / object (e.g. Bob / stands in front of / tower). Use layer names if layers are given.
- words: every meaningful word of the ORIGINAL input with its type (noun, adjective, verb, preposition, other) and the layer name it belongs to (empty string if none).
"""


class LLM(ABC):
    @abstractmethod
    def chat_json(self, system: str, user: str, schema: dict) -> dict: ...

    def describe(self) -> str:
        return self.__class__.__name__


def _parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    return json.loads(m.group(0) if m else text)


class OllamaLLM(LLM):
    def __init__(self, url: str, model: str, temperature: float = 0.4, keep_alive: str | int = 0):
        self.url, self.model, self.temperature, self.keep_alive = url.rstrip("/"), model, temperature, keep_alive

    def chat_json(self, system: str, user: str, schema: dict) -> dict:
        r = requests.post(f"{self.url}/api/chat", timeout=600, json={
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "stream": False,
            "think": False,
            # unload right after the call so SDXL gets the RAM back
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature},
        })
        r.raise_for_status()
        return _parse_json(r.json()["message"]["content"])

    def describe(self) -> str:
        return f"ollama:{self.model}"


class OpenAICompatLLM(LLM):
    def __init__(self, base_url: str, model: str, api_key: str, temperature: float = 0.4):
        self.base_url, self.model, self.api_key, self.temperature = base_url.rstrip("/"), model, api_key, temperature

    def chat_json(self, system: str, user: str, schema: dict) -> dict:
        r = requests.post(f"{self.base_url}/chat/completions", timeout=300,
                          headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                          json={
                              "model": self.model,
                              "temperature": self.temperature,
                              "messages": [{"role": "system", "content": system},
                                           {"role": "user", "content": user}],
                              "response_format": {"type": "json_schema",
                                                  "json_schema": {"name": "breakdown", "schema": schema}},
                          })
        r.raise_for_status()
        return _parse_json(r.json()["choices"][0]["message"]["content"])

    def describe(self) -> str:
        return f"openai-compatible:{self.model}@{self.base_url}"


def make_llm(s: Settings) -> LLM:
    if s.llm_backend == "openai":
        return OpenAICompatLLM(s.openai_base_url, s.llm_model, os.environ.get(s.openai_api_key_env, ""), s.llm_temperature)
    return OllamaLLM(s.ollama_url, s.llm_model, s.llm_temperature, keep_alive="15m" if s.keep_models_loaded else 0)


SPLIT_MODES = {
    "objects": "One layer per object",
    "bands": "Depth bands (back / mid / front)",
    "exact": "Exact number of layers",
}


def split_rule(mode: str, n: int) -> str:
    if mode == "bands":
        bands = max(1, min(3, n - 1))
        names = ["midground", "foreground", "front foreground"][:bands]
        return (f"Group ALL objects into {bands} depth band layer(s): {', '.join(names)} (back to front), "
                f"so {bands + 1} layers in total. Each band layer contains every object at that distance; "
                f"name it like 'midground: campfire, logs' and describe all its objects in its prompt.")
    if mode == "exact":
        return (f"Return EXACTLY {n} layers in total (1 background + {n - 1} object layers). If there are more "
                f"subjects than object slots, group closely related subjects into one layer; if there are fewer, "
                f"split a subject into clearly separable parts (e.g. 'campfire' -> 'flames', 'logs'). Never invent "
                f"new subjects the players did not mention or clearly imply.")
    return f"One layer per concrete noun/subject the players mentioned, at most {n} layers in total."


def build_request(refine: bool, split: bool, words: bool, mode: str, n: int) -> tuple[str, dict] | None:
    """System prompt + JSON schema containing only the enabled parts (fewer fields = faster answer)."""
    props, text = {}, _HEADER
    if refine:
        text += _PART_REFINE
        props["refined_prompt"] = BREAKDOWN_SCHEMA["properties"]["refined_prompt"]
        props["style"] = BREAKDOWN_SCHEMA["properties"]["style"]
    if split:
        text += _PART_LAYERS.replace("{split_rule}", split_rule(mode, n))
        props["layers"] = BREAKDOWN_SCHEMA["properties"]["layers"]
    if words:
        text += _PART_WORDS
        props["relations"] = BREAKDOWN_SCHEMA["properties"]["relations"]
        props["words"] = BREAKDOWN_SCHEMA["properties"]["words"]
    if not props:
        return None
    return text + "Answer with JSON only.", {"type": "object", "properties": props, "required": list(props)}


def breakdown(llm: LLM | None, user_text: str, max_layers: int, mode: str = "objects", refine: bool = True,
              split: bool = True, words: bool = True) -> dict:
    """Run only the enabled LLM parts; anything switched off falls back to a fast non-LLM version."""
    req = build_request(refine, split, words, mode, max_layers) if llm else None
    data = llm.chat_json(req[0], user_text, req[1]) if req else {}
    if not refine:
        data["refined_prompt"] = user_text.strip()
        data["style"] = detect_style(user_text)
    if not split:
        data["layers"] = heuristic_layers(user_text, mode, max_layers)
    data["_llm_parts"] = [k for k, on in (("refine", refine), ("split", split), ("words", words)) if on and req]
    return normalise_breakdown(data, max_layers)


_STYLE_WORDS = re.compile(r"\b(pixel art|8-bit|16-bit|watercolou?r|oil painting|impressionis\w*|anime|manga|cartoon|"
                          r"comic|sketch|pencil drawing|line art|3d render|low poly|photo\w*|hyperreal\w*|"
                          r"realistic|cinematic|isometric|claymation|ink)\b", re.I)
_SCENERY_WORDS = re.compile(r"\b(sky|night|day|dawn|dusk|sunset|sunrise|evening|morning|forest|woods|field|meadow|"
                            r"desert|mountains?|hills?|sea|ocean|lake|river|beach|city|street|town|village|room|hall|"
                            r"cave|dungeon|castle courtyard|clearing|landscape|starry|stars|moon\w*|rain\w*|snow\w*|"
                            r"fog\w*|mist\w*|cloud\w*|weather|background)\b", re.I)


def detect_style(text: str) -> str:
    return ", ".join(dict.fromkeys(m.group(0).lower() for m in _STYLE_WORDS.finditer(text)))


_SKIP = set("a an the one two three some large small big tall tiny huge little old young dark bright happy "
             "glowing burning dancing standing sitting running flying red orange yellow green blue purple white "
             "black brown grey gray golden".split())


def _chunk_name(chunk: str) -> str:
    """Name a comma chunk by its head noun: the first word that is not an article/adjective, cut at a preposition
    ("a large campfire with a stone ring" -> "campfire")."""
    words = re.findall(r"[A-Za-z']+", chunk)
    for prep in ("with", "in", "on", "at", "next", "near", "beside", "behind", "under", "over", "of"):
        if prep in [w.lower() for w in words[1:]]:
            words = words[:[w.lower() for w in words].index(prep, 1)]
            break
    for w in words:
        if w.lower() not in _SKIP:
            return w.lower()[:24]
    return (words[-1].lower()[:24] if words else "")


def heuristic_layers(text: str, mode: str, n: int) -> list[dict]:
    """No-LLM split: comma-separated chunks; scenery chunks -> background, the rest -> one object layer each."""
    chunks = [c.strip() for c in re.split(r"[,;\n]+", text) if c.strip()]
    chunks = [c for c in chunks if not (_STYLE_WORDS.fullmatch(c.strip()) or _STYLE_WORDS.sub("", c).strip() == "")]
    scenery = [c for c in chunks if _SCENERY_WORDS.search(c)]
    objects = [c for c in chunks if c not in scenery]
    bg = {"name": "background", "role": "background", "depth": 0, "adjectives": [], "box": [0, 0, 1, 1],
          "prompt": ", ".join(scenery) or "empty scenery"}
    slots = max(1, n - 1)
    if mode == "bands":
        slots = max(1, min(3, slots))
    if len(objects) > slots:  # group the overflow into the last slot
        objects = objects[:slots - 1] + [" and ".join(objects[slots - 1:])]
    layers = [bg]
    for i, c in enumerate(objects, start=1):
        name = _chunk_name(c) or f"object {i}"
        layers.append({"name": name, "role": "object", "depth": i, "adjectives": [], "prompt": c,
                       "box": [0.25, 0.25, 0.5, 0.5]})
    return layers


_POSITIONAL = re.compile(r"\b(left|right|middle|center|centre|front|behind|above|below|next to|beside|side|top|bottom)\b", re.I)


def normalise_breakdown(d: dict, max_layers: int) -> dict:
    """Make small-model output safe: exactly one background first, clamp boxes, cap layer count."""
    layers = [l for l in d.get("layers", []) if l.get("prompt")]
    bgs = [l for l in layers if l.get("role") == "background"]
    objs = [l for l in layers if l.get("role") != "background"]
    if not bgs:
        bgs = [{"name": "background", "role": "background", "depth": 0, "adjectives": [], "box": [0, 0, 1, 1],
                "prompt": d.get("refined_prompt", "") + ", empty scenery, no characters"}]
    bg = dict(bgs[0], role="background", depth=0, box=[0, 0, 1, 1])
    out = [bg]
    for i, l in enumerate(sorted(objs, key=lambda x: x.get("depth", 1)), start=1):
        box = (list(l.get("box") or []) + [0.25, 0.25, 0.5, 0.5])[:4]
        box = [min(1.0, max(0.0, float(v))) for v in box]
        adj = [a for a in l.get("adjectives", []) if not _POSITIONAL.search(a)]
        out.append(dict(l, role="object", depth=max(1, int(l.get("depth", i) or i)), box=box, adjectives=adj))
    d["layers"] = out[:max_layers]
    d.setdefault("relations", [])
    d.setdefault("words", [])
    d.setdefault("style", "")
    return d


# ---------------------------------------------------------------------------
# Image-to-text per layer
# ---------------------------------------------------------------------------
class Captioner(ABC):
    @abstractmethod
    def caption(self, img: Image.Image) -> str: ...


class NoCaptioner(Captioner):
    def caption(self, img: Image.Image) -> str:
        return ""


class OllamaCaptioner(Captioner):
    def __init__(self, url: str, model: str, keep_alive: str | int = 0):
        self.url, self.model, self.keep_alive = url.rstrip("/"), model, keep_alive

    def caption(self, img: Image.Image) -> str:
        # transparent layers are shown on mid-grey so the vision model sees the object, not a black box
        flat = Image.new("RGB", img.size, (128, 128, 128))
        flat.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
        buf = io.BytesIO()
        flat.save(buf, format="PNG")
        r = requests.post(f"{self.url}/api/generate", timeout=600, json={
            "model": self.model,
            "prompt": "Describe this image in one sentence: the main subject, its colours, pose and style.",
            "images": [base64.b64encode(buf.getvalue()).decode()],
            "stream": False,
            "keep_alive": self.keep_alive,
        })
        r.raise_for_status()
        return r.json().get("response", "").strip()


def make_captioner(s: Settings) -> Captioner:
    if s.captioner == "ollama" and s.step_captions:
        return OllamaCaptioner(s.ollama_url, s.caption_model, keep_alive="15m" if s.keep_models_loaded else 0)
    return NoCaptioner()


def word_check(caption: str, words: list[str]) -> tuple[list[str], list[str]]:
    """Crude 'ignored words' check: which of the layer's words show up in the caption."""
    cap_tokens = re.findall(r"[a-z]+", caption.lower())
    matched, missing = [], []

    def hit(tok: str) -> bool:
        stem = tok[:max(4, len(tok) - 2)]
        # match stems both ways and compound parts ("campfire" ~ "fire", "bonfire" ~ "fire")
        return any(c.startswith(stem) or tok.startswith(c[:max(4, len(c) - 2)]) or (len(c) >= 4 and c in tok)
                   for c in cap_tokens)

    for w in words:
        toks = [t for t in re.findall(r"[a-z]+", w.lower()) if len(t) > 2]
        if not toks:
            continue
        (matched if any(hit(t) for t in toks) else missing).append(w.strip())
    return matched, missing
