"""Layer prototype UI (localhost, single user for now).

Run from the StabeDiffusionCPU folder:
    fastsdcpu\\env\\Scripts\\python.exe prototype\\app.py
Then open http://127.0.0.1:7870
"""
from __future__ import annotations

import faulthandler
import glob
import warnings
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")  # printed by Gradio on every click

# Native crashes (access violations) kill Python without a traceback; log them to a file instead.
_CRASH_LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash.log"), "a", buffering=1)
faulthandler.enable(file=_CRASH_LOG, all_threads=True)

# Pin Windows' current C++ runtime before any package can load an outdated bundled copy
# (a 2020 msvcp140.dll shipped inside PyQt5 crashed generation once on 2026-11-07).
if os.name == "nt":
    import ctypes

    for _dll in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"):
        try:
            ctypes.WinDLL(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", _dll))
        except OSError:
            pass

import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402

import dataclasses  # noqa: E402

from layerproto.config import STEPS, load_settings  # noqa: E402
from layerproto.engine import Engine  # noqa: E402
from layerproto.llm import SPLIT_MODES  # noqa: E402
from layerproto.scene import Layer, Scene, slug  # noqa: E402

S = load_settings()
ENGINE = Engine(S)

LAYER_HEADERS = ["generate?", "name", "role", "depth", "prompt", "adjectives"]
LAYER_TYPES = ["bool", "str", "str", "number", "str", "str"]
MODE_BY_LABEL = {v: k for k, v in SPLIT_MODES.items()}


# ---------------------------------------------------------------------------- helpers
def load(scene_id: str) -> Scene:
    return Scene.load(os.path.join(S.scenes_dir, scene_id, "scene.json"))


def layer_rows(sc: Scene):
    return [[l.enabled, l.name, l.role, l.depth, l.prompt, ", ".join(l.adjectives)] for l in sc.layers]


def _rows(rows):
    return rows.values.tolist() if hasattr(rows, "values") else list(rows or [])


def _truthy(v) -> bool:
    return v is True or str(v).strip().lower() in ("true", "1", "yes", "x")


def apply_rows(sc: Scene, rows) -> Scene:
    """Write table edits back into the scene; layers keep their images when their name is unchanged."""
    old = {l.name: l for l in sc.layers}
    new_layers, seen = [], set()
    for r in _rows(rows):
        r = (list(r) + [""] * 6)[:6]
        name, prompt = str(r[1]).strip(), str(r[4]).strip()
        if not name or not prompt:
            continue
        prev = old.get(name)
        lid = prev.id if prev else slug(name)
        while lid in seen:
            lid += "x"
        seen.add(lid)
        l = prev or Layer(id=lid, name=name, role="object", depth=1, prompt="")
        l.enabled = _truthy(r[0]) if str(r[0]).strip() != "" else True
        l.role = "background" if str(r[2]).strip().lower().startswith("b") else "object"
        try:
            l.depth = int(float(r[3] or 0))
        except ValueError:
            l.depth = 1
        l.prompt = prompt
        l.adjectives = [a.strip() for a in str(r[5]).split(",") if a.strip()]
        new_layers.append(l)
    sc.layers = sorted(new_layers, key=lambda l: l.depth)
    return sc


def img(sc: Scene, file: str) -> str:
    return os.path.join(S.scenes_dir, sc.id, file)


def overview(sc: Scene):
    return [(img(sc, l.current().file), f"[{l.depth}] {l.name}") for l in sc.layers if l.current()]


def layer_info(l: Layer) -> str:
    v = l.current()
    if not v:
        return "_Not generated yet._"
    warn = " ⚠️ extraction failed (no transparency) - try another seed" if l.role == "object" and v.transparent_pct < 5 else ""
    out = f"Selected **#{l.selected}** · seed {v.seed} · {v.seconds}s"
    if l.role != "background":
        out += f" · {v.transparent_pct}% transparent{warn}"
    if S.step_captions:
        out += f"\n\n**Image-to-text:** {v.caption or '(no caption for this variant)'}"
    if S.step_captions and S.step_word_check:
        out += (f"\n\n**Found in caption:** {', '.join(v.matched_words) or '-'}  \n"
                f"**Not found (possibly ignored):** {', '.join(v.missing_words) or '-'}")
    return out


def names(rows):
    return [str(r[1]).strip() for r in _rows(rows) if len(r) > 1 and str(r[1]).strip()]


def eta_text(rows, n_var) -> str:
    rs = [r for r in _rows(rows) if len(r) > 4 and str(r[1]).strip() and str(r[4]).strip()
          and (_truthy(r[0]) or str(r[0]).strip() == "")]
    n_bg = sum(1 for r in rs if str(r[2]).lower().startswith("b"))
    nv = int(n_var or 1) if S.step_variants else 1
    secs = ENGINE.estimate_seconds(n_bg * nv, (len(rs) - n_bg) * nv)
    where = "remote GPU [est]" if S.image_backend == "remote" else "this laptop"
    return (f"**Estimate:** {len(rs)} layer(s) × {nv} variant(s) ≈ **{secs // 60} min {secs % 60:02d} s** on {where} "
            f"(each element = 2 passes: draw into scene, then extract).")


def scene_list():
    paths = sorted(glob.glob(os.path.join(S.scenes_dir, "*", "scene.json")), reverse=True)
    return [os.path.basename(os.path.dirname(p)) for p in paths]


def err(e: Exception) -> str:
    traceback.print_exc()
    return f"**Error:** {type(e).__name__}: {e}"


def md_table(headers, rows) -> str:
    """Plain Markdown table: gr.Dataframe inside a collapsed Accordion froze the browser (2026-11-07)."""
    if not rows:
        return "_none_"
    esc = lambda v: str(v).replace("|", "/")  # noqa: E731
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(esc(v) for v in r) + " |" for r in rows])


def plan_outputs(sc: Scene, msg: str):
    rows = layer_rows(sc)
    return (sc.id, sc.refined_prompt, sc.style, rows,
            md_table(["subject", "relation", "object"],
                     [[r.get("subject", ""), r.get("relation", ""), r.get("object", "")] for r in sc.relations]),
            md_table(["word", "type", "layer"],
                     [[w.get("word", ""), w.get("type", ""), w.get("layer", "")] for w in sc.words]),
            gr.update(choices=names(rows), value=[]), msg)


# ---------------------------------------------------------------------------- settings
def active_labels():
    return [label for label, key in STEPS.items() if getattr(S, key)]


def steps_hint() -> str:
    off = [label for label, key in STEPS.items() if not getattr(S, key)]
    llm = [k for k in ("step_refine", "step_split", "step_words") if getattr(S, k)]
    msg = "LLM call: " + ("skipped entirely" if not llm else f"{len(llm)} of 3 parts")
    return f"{msg} · switched off: {', '.join(off) or 'nothing'}"


def _apply(**changes) -> None:
    global S
    S = dataclasses.replace(S, **changes)
    ENGINE.reconfigure(S)


def plan_label() -> str:
    parts = [n for n, k in (("refine", "step_refine"), ("split", "step_split"), ("word breakdown", "step_words"))
             if getattr(S, k)]
    return ("Run LLM: " + " + ".join(parts)) if parts else "Split on commas (no LLM)"


PIPELINE = [("Refine", "step_refine"), ("Split", "step_split"), ("Words", "step_words"),
            ("Background", None), ("Elements drawn into scene + extracted", None), ("Variants ×N", "step_variants"),
            ("Captions", "step_captions"), ("Word check", "step_word_check")]


def pipeline_html() -> str:
    chips = []
    for name, key in PIPELINE:
        on = key is None or getattr(S, key)
        style = ("background:#f97316;color:white" if on else
                 "background:#e5e7eb;color:#9ca3af;text-decoration:line-through")
        if key == "step_split" and not on:
            name, style = "Split on commas", "background:#fde68a;color:#78350f"
        chips.append(f'<span style="{style};padding:4px 10px;border-radius:12px;font-size:13px;'
                     f'white-space:nowrap">{name}</span>')
    return ('<div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:4px 0">'
            '<b style="margin-right:4px">Pipeline:</b>' + ' <span>→</span> '.join(chips) + '</div>')


def ui_for_steps():
    """Visible effects of the step switches on the Scene tab."""
    off = [label for label, key in STEPS.items() if not getattr(S, key)]
    print(f"[steps] switched off: {', '.join(off) or 'nothing'}", flush=True)
    return (steps_hint(),
            gr.update(visible=S.step_variants),
            gr.update(visible=S.step_words), gr.update(value=plan_label()), pipeline_html(),
            gr.update(visible=S.step_split), gr.update(visible=S.step_split),      # split settings, Re-split
            gr.update(visible=S.step_refine), gr.update(visible=S.step_style))    # refined prompt, shared style


def step_checkboxes(per_row: int = 3) -> dict:
    """One gr.Checkbox per pipeline step, laid out a few per row."""
    items, boxes = list(STEPS.items()), {}
    for i in range(0, len(items), per_row):
        with gr.Row():
            for label, key in items[i:i + per_row]:
                boxes[key] = gr.Checkbox(value=getattr(S, key), label=label)
    return boxes


def set_step(key: str, value: bool):
    """One checkbox = one step. Each click only sets its own step, so clicks arriving out of order
    can't overwrite each other (a whole-checklist update did, 2026-11-07)."""
    _apply(**{key: bool(value)})
    return (gr.update(value=bool(value)),) + ui_for_steps()


def apply_settings(*vals):
    step_vals, rest = vals[:len(STEPS)], vals[len(STEPS):]
    llm, cap, llm_backend, openai_url, backend, remote, alpha, steps, w, h, var, model, negative, persist = rest
    changes = {key: bool(v) for key, v in zip(STEPS.values(), step_vals)}
    changes.update(llm_model=llm.strip(), caption_model=cap.strip(), llm_backend=llm_backend,
                   openai_base_url=openai_url.strip(), image_backend=backend, remote_url=remote.strip(),
                   alpha_quality=alpha, steps=int(steps), width=int(w), height=int(h), variants_per_layer=int(var),
                   sdxl_model=model.strip(), negative_prompt=negative)
    try:
        _apply(**changes)
        msg = "Applied." + (f" Saved to `{S.save()}`." if persist else " (not saved: click 'Save as default')")
    except Exception as e:  # noqa: BLE001
        msg = err(e)
    return (msg,) + tuple(gr.update(value=getattr(S, k)) for k in STEPS.values()) + ui_for_steps()


# ---------------------------------------------------------------------------- step 1-2 callbacks
def do_plan(text, mode_label, n_layers, scene_id=None):
    try:
        mode = MODE_BY_LABEL.get(mode_label, "objects")
        sc = ENGINE.plan_scene(text, mode, int(n_layers), scene_id=scene_id)
        return plan_outputs(sc, f"{'Re-split' if scene_id else 'Planned'} scene `{sc.id}`: {len(sc.layers)} layers "
                                f"({SPLIT_MODES[mode]}, {int(n_layers)}) via {ENGINE.llm.describe()}. "
                                f"Edit the table, then generate.")
    except Exception as e:  # noqa: BLE001
        return (scene_id, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), err(e))


def do_resplit(scene_id, text, mode_label, n_layers, refined, style):
    if not scene_id:
        return do_plan(text, mode_label, n_layers)
    out = list(do_plan(text, mode_label, n_layers, scene_id=scene_id))
    if refined.strip():  # keep the user's hand edits of prompt and style
        out[1], out[2] = refined, style
        sc = load(scene_id)
        sc.refined_prompt, sc.style = refined, style
        sc.save(S.scenes_dir)
    return tuple(out)


def do_add_row(rows):
    rs = _rows(rows)
    depth = max([int(float(r[3] or 0)) for r in rs if len(r) > 3 and str(r[3]).strip()] or [0]) + 1
    rs.append([True, f"new layer {len(rs)}", "object", depth, "", ""])
    return rs, gr.update(choices=names(rs))


def do_delete_rows(rows, picked):
    rs = [r for r in _rows(rows) if str(r[1]).strip() not in set(picked or [])]
    return rs, gr.update(choices=names(rs), value=[])


def do_merge_rows(rows, picked):
    picked = list(picked or [])
    rs = _rows(rows)
    if len(picked) < 2:
        return rs, gr.update(), "Pick at least two layers to merge."
    sel = [r for r in rs if str(r[1]).strip() in picked]
    rest = [r for r in rs if str(r[1]).strip() not in picked]
    if any(str(r[2]).lower().startswith("b") for r in sel):
        return rs, gr.update(), "The background can't be merged into an object layer."
    merged = [True, " + ".join(str(r[1]).strip() for r in sel), "object", min(int(float(r[3] or 1)) for r in sel),
              " and ".join(str(r[4]).strip() for r in sel),
              ", ".join(a for r in sel for a in str(r[5]).split(",") if a.strip())]
    rs = sorted(rest + [merged], key=lambda r: int(float(r[3] or 0)))
    return rs, gr.update(choices=names(rs), value=[]), f"Merged {len(sel)} layers into '{merged[1]}'."


# ---------------------------------------------------------------------------- step 3 callbacks
def do_generate(scene_id, refined, style, rows, n, version, missing_only=False, progress=gr.Progress()):
    if not scene_id:
        return [], version, "Plan a scene first."
    try:
        sc = apply_rows(load(scene_id), rows)
        sc.refined_prompt, sc.style = refined, style
        sc.save(S.scenes_dir)
        ENGINE.generate_all(sc, int(n), progress=lambda m: progress(0, desc=m), missing_only=missing_only)
        return overview(sc), version + 1, f"Done. Backend: {ENGINE.chain.info()}"
    except Exception as e:  # noqa: BLE001
        return gr.update(), version + 1, err(e)


def do_generate_missing(scene_id, refined, style, rows, n, version, progress=gr.Progress()):
    return do_generate(scene_id, refined, style, rows, n, version, missing_only=True, progress=progress)


# ---------------------------------------------------------------------------- step 4: final image
def order_md(sc: Scene) -> str:
    objs = ENGINE.object_order(sc)[::-1]  # top of the stack first, like a Photoshop layer panel
    lines = [f"{i}. **{l.name}**" + ("" if l.enabled and l.current() else " _(not shown)_")
             for i, l in enumerate(objs, start=1)]
    bg = next((l for l in sc.layers if l.role == "background"), None)
    if bg:
        lines.append(f"{len(objs) + 1}. **{bg.name}** _(base layer, always at the bottom)_")
    return "**Layer order (top → bottom)**\n\n" + "\n".join(lines)


def _final_outputs(sc: Scene, keep=None):
    img_ = ENGINE.compose(sc)
    objs = [(l.name, l.id) for l in ENGINE.object_order(sc)[::-1]]
    val = keep if keep in [v for _, v in objs] else (objs[0][1] if objs else None)
    sel = next((l for l in sc.layers if l.id == val), None)
    return (img_, order_md(sc), gr.update(choices=objs, value=val), layer_rows(sc), overview(sc),
            sel.dx if sel else 0.0, sel.dy if sel else 0.0, sel.scale if sel else 1.0)


NO_FINAL = (gr.update(),) * 8


def do_compose(scene_id, rows, picked_layer):
    if not scene_id:
        return (None, "_Plan or load a scene first._") + (gr.update(),) * 6
    sc = load(scene_id)
    if rows is not None and len(_rows(rows)):
        sc = apply_rows(sc, rows)  # picks up 'generate?' ticks (hidden layers) and prompt edits
        sc.save(S.scenes_dir)
    return _final_outputs(sc, picked_layer)


def do_move(scene_id, layer_id, direction):
    if not scene_id or not layer_id:
        return NO_FINAL
    return _final_outputs(ENGINE.move_layer(load(scene_id), layer_id, direction), layer_id)


def do_transform(scene_id, layer_id, dx, dy, scale):
    """Photoshop-style move / resize of one layer: a canvas transform, no model call."""
    if not scene_id or not layer_id:
        return NO_FINAL
    return _final_outputs(ENGINE.transform_layer(load(scene_id), layer_id, dx, dy, scale), layer_id)


def do_fit_box(scene_id, layer_id):
    """Move / resize the layer so its visible content fills the box the planner gave it (CPU, no model call)."""
    if not scene_id or not layer_id:
        return NO_FINAL
    return _final_outputs(ENGINE.fit_layer_to_box(load(scene_id), layer_id), layer_id)


def do_pick_move(scene_id, layer_id):
    if not scene_id or not layer_id:
        return 0.0, 0.0, 1.0
    l = load(scene_id).layer(layer_id)
    return l.dx, l.dy, l.scale


def do_event(scene_id, name, prompt, n, version, progress=gr.Progress()):
    """Story event (plan steps 6-8): a new element generated into the current scene as a new top layer."""
    if not scene_id or not str(prompt).strip():
        return gr.update(), version, "Load a generated scene and describe the event first."
    try:
        name = str(name).strip() or str(prompt).strip().split(",")[0][:24]
        sc = ENGINE.add_event_layer(load(scene_id), name, prompt, variants=int(n),
                                    progress=lambda m: progress(0, desc=m))
        return overview(sc), version + 1, f"Added '{name}' as a new layer, generated into the current scene."
    except Exception as e:  # noqa: BLE001
        return gr.update(), version, err(e)


def do_load(scene_id, version):
    if not scene_id:
        return (None, "", "", [], "_none_", "_none_", gr.update(), [], version, "")
    if scene_id not in scene_list():  # e.g. a half-typed name in the dropdown
        return (gr.update(),) * 9 + (f"**No saved scene called `{scene_id}`.** Pick one from the list.",)
    sc = load(scene_id)
    print(f"[load] {sc.id}: {len(sc.layers)} layers", flush=True)
    out = plan_outputs(sc, f"Loaded `{sc.id}`")
    return out[:7] + (overview(sc), version + 1, out[7])


# ---------------------------------------------------------------------------- layout
# ---------------------------------------------------------------------------- image -> layers (GPU decomposer)
def do_decompose(img, caption, method, n_layers, boxes_json, names_csv, amodal=False, progress=gr.Progress()):
    """Split an uploaded / generated image into RGBA layers on the remote GPU server and store it as a scene."""
    import json as _json
    from layerproto.backends.decomposer import DecomposeParams
    from layerproto.scene import new_scene_id
    if img is None:
        return "Upload an image first.", [], None
    try:
        boxes = _json.loads(boxes_json) if boxes_json.strip() else []
        names = [n.strip() for n in names_csv.split(",") if n.strip()]
        sc = Scene(id=new_scene_id(caption or "decomposed"), user_input=caption, refined_prompt=caption, n_layers=int(n_layers))
        ENGINE.decompose_image(sc, img.convert("RGB"), DecomposeParams(caption=caption, method=method, n_layers=int(n_layers),
                               boxes=boxes, names=names, amodal=bool(amodal)), progress=lambda m: progress(0, desc=m))
        final = ENGINE.compose(sc)
        return f"Scene `{sc.id}` saved with {len(sc.layers)} layers. Open it in the Scene tab (Load old scene).", overview(sc), final
    except Exception as e:  # noqa: BLE001
        return err(e), [], None



with gr.Blocks(title="Layer scene prototype") as demo:
    scene_id = gr.State(None)
    version = gr.State(0)  # bumped whenever images change, re-renders the layer cards
    with gr.Tab("Scene"):
        gr.Markdown("# Layer scene prototype\nDescribe → background layer → each element is drawn **into the scene** "
                    "and extracted as its own layer (LayerDiffuse SDXL) → re-prompt, move or resize single layers → "
                    "final image.")
        status = gr.Markdown()
        with gr.Accordion("Pipeline steps (switch off to save time; full options in the Settings tab)", open=False):
            quick = step_checkboxes()
            eta_hint = gr.Markdown(steps_hint())
        pipe_strip = gr.HTML(pipeline_html())

        # 1. describe + split settings
        with gr.Row():
            user_text = gr.Textbox(label="1. Describe the scene (keywords from several players are fine)", lines=3,
                                   scale=3,
                                   placeholder="pixel art, night forest, campfire in the middle, skeleton dancing on the right")
            with gr.Column(scale=2, visible=S.step_split) as split_col:
                split_mode = gr.Radio(list(SPLIT_MODES.values()), value=SPLIT_MODES["objects"], label="How to split",
                                      info="per object · grouped by depth · exactly N layers")
                n_layers = gr.Slider(2, 8, value=S.max_layers, step=1, label="Layers (incl. background)",
                                     info="Maximum for 'per object', exact count for 'exact', 1 + number of bands for 'bands'")
            with gr.Column(scale=1, min_width=200):
                plan_btn = gr.Button(plan_label(), variant="primary")
                resplit_btn = gr.Button("Re-split with these settings", visible=S.step_split)
                old = gr.Dropdown(scene_list(), label="…or load a saved scene")
                load_btn = gr.Button("Load")

        # 2. breakdown
        gr.Markdown("### 2. Refined prompt and layers (editable)")
        with gr.Row():
            refined = gr.Textbox(label="Refined prompt", lines=3, scale=3, visible=S.step_refine)
            style = gr.Textbox(label="Shared style (appended to every layer)", scale=1, visible=S.step_style)
        layers_df = gr.Dataframe(headers=LAYER_HEADERS, datatype=LAYER_TYPES, interactive=True,
                                 column_count=(6, "fixed"), wrap=True,
                                 label="Layers: one row = one layer. Untick 'generate?' to skip a layer.")
        with gr.Row():
            picked = gr.CheckboxGroup([], label="Select layers for Delete / Merge", scale=3)
            add_btn = gr.Button("➕ Add layer", scale=1)
            del_btn = gr.Button("🗑 Delete selected", scale=1)
            merge_btn = gr.Button("🔗 Merge selected", scale=1)
        with gr.Accordion("Word breakdown and relations", open=False, visible=S.step_words) as words_acc:
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Relations (verbs / prepositions)**")
                    rel_df = gr.Markdown("_none_")
                with gr.Column():
                    gr.Markdown("**Words**")
                    words_df = gr.Markdown("_none_")

        # 3. generate
        gr.Markdown("### 3. Generate the layers")
        with gr.Row():
            n_var = gr.Slider(1, 4, value=S.variants_per_layer, step=1, scale=2, visible=S.step_variants,
                              label="Variants per layer")
            gen_btn = gr.Button("Generate all layers", variant="primary", scale=1)
            miss_btn = gr.Button("Generate missing only", scale=1)
        eta = gr.Markdown()
        over = gr.Gallery(label="Selected variant of each layer (back to front)", columns=6, height=240)

        gr.Markdown("### 4. Final image (all layers stacked, background at the bottom)")
        with gr.Row():
            final_img = gr.Image(label="Final image (saved as final.png in the scene folder)", type="pil",
                                 image_mode="RGBA", height=512, scale=3, interactive=False)
            with gr.Column(scale=2):
                order_view = gr.Markdown("_Generate or load a scene first._")
                move_pick = gr.Dropdown([], label="Layer")
                with gr.Row():
                    up_btn = gr.Button("⬆ Layer up")
                    down_btn = gr.Button("⬇ Layer down")
                gr.Markdown("**Move / resize the selected layer** (no regeneration)")
                t_dx = gr.Slider(-1, 1, value=0, step=0.01, label="Move left / right (fraction of width)")
                t_dy = gr.Slider(-1, 1, value=0, step=0.01, label="Move up / down (fraction of height)")
                t_scale = gr.Slider(0.2, 3, value=1, step=0.05, label="Size (1 = as generated)")
                with gr.Row():
                    t_apply = gr.Button("Apply move / resize", variant="primary")
                    t_reset = gr.Button("Reset")
                    t_fit = gr.Button("Fit to planned box")
                compose_btn = gr.Button("Recompose (after selecting variants or editing the table)")

        gr.Markdown("### Story event: add a new element into the current scene")
        with gr.Row():
            ev_name = gr.Textbox(label="Name", placeholder="forest fire", scale=1)
            ev_prompt = gr.Textbox(label="What happens", placeholder="the pine trees on the left catch fire, "
                                   "flames and smoke", scale=3)
            ev_btn = gr.Button("Add event layer", variant="primary", scale=1)

        # 5. per-layer cards
        gr.Markdown("### 5. Fix one layer (only that layer is regenerated)")
        gr.Markdown("_Re-prompting a layer regenerates it **into the layers below it** (bg2ble then bgble2fg), so "
                    "it keeps the scene's lighting and perspective; every other layer stays exactly as it is._")

        final_outs = [final_img, order_view, move_pick, layers_df, over, t_dx, t_dy, t_scale]
        recompose = (do_compose, [scene_id, layers_df, move_pick], final_outs)

        @gr.render(inputs=[scene_id, version])
        def layer_cards(sid, _ver):
            if not sid:
                gr.Markdown("_Plan or load a scene first._")
                return
            sc = load(sid)
            if not any(l.variants for l in sc.layers):
                gr.Markdown("_Generate the layers first._")
                return
            for layer in sc.layers:
                lid = layer.id
                with gr.Group():
                    gr.Markdown(f"#### [{layer.depth}] {layer.name} · {layer.role}"
                                + ("" if layer.enabled else " · _skipped_"))
                    with gr.Row():
                        with gr.Column(scale=3):
                            # no selected_index: a preselected gallery + .select handler froze the browser
                            gal = gr.Gallery([(img(sc, v.file), f"#{i} {v.mode}" + (" ✔" if i == layer.selected else ""))
                                              for i, v in enumerate(layer.variants)],
                                             columns=4, height=220, label="Variants (click to select; ✔ = selected)")
                            info = gr.Markdown(layer_info(layer))
                        with gr.Column(scale=2):
                            lp = gr.Textbox(layer.prompt, label="Layer prompt", lines=2)
                            nv = gr.Slider(1, 4, value=1, step=1, label="Variants", visible=S.step_variants)
                            ub = gr.Button(f"Regenerate '{layer.name}' in the scene", variant="primary")

                    def pick(evt: gr.SelectData, _lid=lid, _sid=sid):
                        s2 = ENGINE.select(load(_sid), _lid, evt.index)
                        return overview(s2), layer_info(s2.layer(_lid))

                    gal.select(pick, None, [over, info]).then(*recompose)

                    def upd(prompt, n, ver, _lid=lid, _sid=sid, progress=gr.Progress()):
                        try:
                            s2 = ENGINE.update_layer(load(_sid), _lid, prompt=prompt, variants=int(n),
                                                     progress=lambda m: progress(0, desc=m))
                            return overview(s2), ver + 1, f"Updated '{s2.layer(_lid).name}' only; other layers unchanged."
                        except Exception as e:  # noqa: BLE001
                            return gr.update(), ver, err(e)

                    ub.click(upd, [lp, nv, version], [over, version, status]).then(*recompose)

        # wiring
        plan_outs = [scene_id, refined, style, layers_df, rel_df, words_df, picked, status]
        plan_btn.click(do_plan, [user_text, split_mode, n_layers], plan_outs)
        resplit_btn.click(do_resplit, [scene_id, user_text, split_mode, n_layers, refined, style], plan_outs)
        add_btn.click(do_add_row, [layers_df], [layers_df, picked])
        del_btn.click(do_delete_rows, [layers_df, picked], [layers_df, picked])
        merge_btn.click(do_merge_rows, [layers_df, picked], [layers_df, picked, status])
        layers_df.change(lambda r: gr.update(choices=names(r)), [layers_df], [picked])
        for comp in (layers_df, n_var):
            comp.change(eta_text, [layers_df, n_var], [eta])
        gen_btn.click(do_generate, [scene_id, refined, style, layers_df, n_var, version],
                      [over, version, status]).then(*recompose)
        miss_btn.click(do_generate_missing, [scene_id, refined, style, layers_df, n_var, version],
                       [over, version, status]).then(*recompose)
        load_btn.click(do_load, [old, version],
                       [scene_id, refined, style, layers_df, rel_df, words_df, picked, over, version, status]
                       ).then(*recompose)
        compose_btn.click(*recompose)
        up_btn.click(lambda s, l: do_move(s, l, "up"), [scene_id, move_pick], final_outs)
        down_btn.click(lambda s, l: do_move(s, l, "down"), [scene_id, move_pick], final_outs)
        move_pick.input(do_pick_move, [scene_id, move_pick], [t_dx, t_dy, t_scale])
        t_apply.click(do_transform, [scene_id, move_pick, t_dx, t_dy, t_scale], final_outs)
        t_reset.click(lambda s, l: do_transform(s, l, 0, 0, 1), [scene_id, move_pick], final_outs)
        t_fit.click(do_fit_box, [scene_id, move_pick], final_outs).then(do_pick_move, [scene_id, move_pick], [t_dx, t_dy, t_scale])
        ev_btn.click(do_event, [scene_id, ev_name, ev_prompt, n_var, version], [over, version, status]).then(*recompose)

    # ------------------------------------------------------------------ image -> layers tab
    with gr.Tab("Decompose an image (GPU)"):
        gr.Markdown("Split an existing image into RGBA layers with a model on the remote GPU server "
                    "(`remote_url` in Settings; server started with a decomposer, see `layerproto/remote_server.py`). "
                    "Methods and what they need: `docs/models/README.md`.")
        with gr.Row():
            dc_img = gr.Image(label="Image", type="pil", height=320)
            with gr.Column():
                dc_caption = gr.Textbox(label="Caption (what the image shows)", lines=3)
                dc_method = gr.Dropdown(["qwen-layered", "stable-layers", "reveallayer", "cld", "sam2-lama"], value="qwen-layered", label="Method")
                dc_n = gr.Slider(2, 8, value=4, step=1, label="Layers (Qwen-Image-Layered / Stable-Layers)")
                dc_boxes = gr.Textbox(label="Boxes, back to front, pixels [[x0,y0,x1,y1],...] (RevealLayer, CLD, SAM2)", lines=2)
                dc_names = gr.Textbox(label="Names for the boxes, comma separated (also used for SAM2 by noun)")
                dc_amodal = gr.Checkbox(False, label="Complete hidden parts (SAM2 only, SDXL-inpaint, about 30 s per hidden object)")
                dc_btn = gr.Button("Decompose", variant="primary")
        dc_status = gr.Markdown()
        with gr.Row():
            dc_gallery = gr.Gallery(label="Layers (background first)", columns=6, height=240)
            dc_final = gr.Image(label="Recomposited", type="pil", image_mode="RGBA", height=320, interactive=False)
        dc_btn.click(do_decompose, [dc_img, dc_caption, dc_method, dc_n, dc_boxes, dc_names, dc_amodal], [dc_status, dc_gallery, dc_final])

    # ------------------------------------------------------------------ settings tab
    with gr.Tab("Settings"):
        gr.Markdown("### Pipeline steps\nSwitch steps off to save time. Changes apply immediately; "
                    "**Save as default** writes them to `prototype/config.json`.")
        full = step_checkboxes()
        gr.Markdown("### Models and generation")
        with gr.Row():
            set_llm = gr.Textbox(S.llm_model, label="LLM model (Ollama tag or API model id)")
            set_cap = gr.Dropdown(["qwen2.5vl:3b", "moondream"], value=S.caption_model, allow_custom_value=True,
                                  label="Caption model", info="qwen2.5vl:3b ≈ 55 s, accurate · moondream ≈ 14 s, garbled")
            set_llm_backend = gr.Radio(["ollama", "openai"], value=S.llm_backend, label="LLM backend",
                                       info="openai = any OpenAI-compatible endpoint (OpenRouter, Kaggle server)")
            set_openai_url = gr.Textbox(S.openai_base_url, label="OpenAI-compatible base URL")
        with gr.Row():
            set_backend = gr.Radio(["local", "remote"], value=S.image_backend, label="Image backend")
            set_remote = gr.Textbox(S.remote_url, label="Remote backend URL (Modal endpoint)")
            set_alpha = gr.Radio(["fast", "best"], value=S.alpha_quality, label="Alpha edge quality",
                                 info="best = 8-pass decoder, ~10 s slower per object")
        with gr.Row():
            set_steps = gr.Slider(1, 50, value=S.steps, step=1, label="Diffusion steps (chain released at 20)")
            set_w = gr.Slider(256, 1024, value=S.width, step=64, label="Width")
            set_h = gr.Slider(256, 1024, value=S.height, step=64, label="Height")
            set_var = gr.Slider(1, 4, value=S.variants_per_layer, step=1, label="Default variants per layer")
        with gr.Row():
            set_model = gr.Textbox(S.sdxl_model, label="SDXL model (the LayerDiffuse chain needs SDXL base)")
            set_neg = gr.Textbox(S.negative_prompt, label="Negative prompt")
        with gr.Row():
            apply_btn = gr.Button("Apply", variant="primary")
            save_btn = gr.Button("Save as default")
        set_status = gr.Markdown()

    setting_inputs = list(full.values()) + [set_llm, set_cap, set_llm_backend, set_openai_url, set_backend, set_remote,
                      set_alpha, set_steps, set_w, set_h, set_var, set_model, set_neg]
    step_ui = [eta_hint, n_var, words_acc, plan_btn, pipe_strip, split_col, resplit_btn, refined, style]
    bump = (lambda v: v + 1, [version], [version])
    # .input ignores clicks while an update is pending (trigger_mode "once"); "always_last" re-runs with the
    # latest ticks instead, so fast clicking never loses a switch
    for key in STEPS.values():
        fn = (lambda v, _k=key: set_step(_k, v))
        # "multiple": every click is processed; order no longer matters because each click sets only its own step
        # one shared queue (concurrency_id) runs step updates one at a time in click order, so the last
        # click's page state is always the one that ends up on screen
        quick[key].input(fn, [quick[key]], [full[key]] + step_ui, trigger_mode="multiple",
                         concurrency_id="steps", concurrency_limit=1).then(*bump)
        full[key].input(fn, [full[key]], [quick[key]] + step_ui, trigger_mode="multiple",
                        concurrency_id="steps", concurrency_limit=1).then(*bump)
    quick_list = list(quick.values())
    apply_btn.click(apply_settings, setting_inputs + [gr.State(False)], [set_status] + quick_list + step_ui).then(*bump)
    save_btn.click(apply_settings, setting_inputs + [gr.State(True)], [set_status] + quick_list + step_ui).then(*bump)


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7870)
