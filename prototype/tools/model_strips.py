r"""Builds the standard fox & cottage strip images for docs/models from saved results (CPU, no model).

    fastsdcpu/env/Scripts/python.exe prototype/tools/model_strips.py

Strip layout (tile 256 px, transparent layers shown on magenta): input | recomposite of the layers | background (if any) | layers ...
Qwen-Image-Layered / Stable-Layers / Lightning already have strips of this form (copied as they are).
"""
import glob
import os
import shutil

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.dirname(HERE)
ROOT = os.path.dirname(PROTO)
K = os.path.join(PROTO, "kaggle")
OUT = os.path.join(ROOT, "docs", "img")
T = 256
INPUT = Image.open(os.path.join(PROTO, "test_scene", "fox_cottage.png")).convert("RGB")


def tile(im, magenta=True):
    im = im.convert("RGBA").resize((T, T), Image.LANCZOS)
    bg = Image.new("RGBA", im.size, (200, 60, 200, 255) if magenta else (255, 255, 255, 255))
    bg.alpha_composite(im)
    return bg.convert("RGB")


def recomposite(layers):
    c = layers[0].convert("RGBA").resize((512, 512))
    for L in layers[1:]:
        c.alpha_composite(L.convert("RGBA").resize((512, 512)))
    return c


def strip(name, layers, label=None, recomp=None, first_is_background=True):
    """layers: RGBA images, background first when first_is_background."""
    tiles = [tile(INPUT, False), tile(recomp if recomp is not None else recomposite(layers), False)]
    tiles += [tile(L, magenta=not (first_is_background and i == 0)) for i, L in enumerate(layers)]
    S = Image.new("RGB", (T * len(tiles), T), (255, 255, 255))
    for i, t in enumerate(tiles):
        S.paste(t, (T * i, 0))
    if label:
        d = ImageDraw.Draw(S)
        d.rectangle((0, 0, 420, 14), fill=(0, 0, 0))
        d.text((3, 1), label, fill=(255, 255, 255))
    path = os.path.join(OUT, f"fox_{name}.jpg")
    S.save(path, quality=88)
    print("wrote", os.path.relpath(path, ROOT), len(layers), "layers")


def load(paths):
    return [Image.open(p).convert("RGBA") for p in paths]


def main():
    # RevealLayer (5 boxes): filled background + 5 layers
    for tag, d in (("reveallayer_A", "gs3.5_st16"), ("reveallayer_B", "gs1.0_st12")):
        p = os.path.join(K, "results_reveallayer_fox_v15", "reveal_out", d, "fox_cottage")
        strip(tag, load([f"{p}/bg_rgba.png"] + [f"{p}/layer_{i}_rgba.png" for i in range(5)]), label=f"RevealLayer {d}")
    # CLD 512 px 7 boxes, 16 steps (best case so far) and 768 px 16 steps
    for tag, p, lab in (("cld_512", os.path.join(K, "results_cld_v5", "cld_out", "case0_16st"), "CLD 512 px, 16 steps"),
                        ("cld_768", os.path.join(K, "results_cld_768_full", "cld_out", "case0_16st"), "CLD 768 px, 16 steps")):
        files = sorted(glob.glob(f"{p}/layer_*_rgba.png"), key=lambda f: int(os.path.basename(f).split("_")[1]))
        strip(tag, load([f"{p}/background.png"] + files), label=lab)
    # ART+ 512 px 16 steps (v11) and 1024 px 8 steps
    for tag, p, lab in (("artplus_512", os.path.join(K, "results_artplus_v11", "art_out", "case0_16st"), "ART+ 512 px, 16 steps"),
                        ("artplus_1024", os.path.join(K, "results_artplus_1024_8st", "art_out", "case0_8st"), "ART+ 1024 px, 8 steps")):
        files = sorted(glob.glob(f"{p}/layer_*_rgba.png"), key=lambda f: int(os.path.basename(f).split("_")[1]))
        strip(tag, load([f"{p}/background.png"] + files), label=lab)
    # LayerD with the generic BiRefNet matting
    p = os.path.join(K, "results_layerd_v11", "layerd_v2", "generic_birefnet", "fox_cottage")
    files = sorted(glob.glob(f"{p}/layer_*.png"), key=lambda f: int(os.path.basename(f).split("_")[1].split(".")[0]))
    if files:
        strip("layerd_generic", load(files), label="LayerD + generic BiRefNet", first_is_background=False)
    # SAM2 + LaMa with hand boxes (T4, 8 s)
    p = os.path.join(K, "results_segment_inpaint_v2", "seginp_v2", "layers_A_hand_boxes")
    files = sorted(glob.glob(f"{p}/*.png"), key=lambda f: int(os.path.basename(f).split("_")[0]))
    strip("sam2_lama_boxes", load(files), label="SAM2 + LaMa, hand boxes")
    # Qwen-Image-2.1 extraction through the /decompose route
    p = os.path.join(K, "results_qwen21_route", "route")
    strip("qwen21_route", load([f"{p}/layer_{i}.png" for i in range(5)]), label="Qwen-Image-2.1 extract, 114 s")
    # strips that already exist (tile 256: input | recomposite | layers)
    q = os.path.join(K, "results_qwen_fox_v11", "qwen_layered_test")
    for name, f in (("qwenlayered_384_4L", "v9_base_euler_fox_cottage_384px_4L_4st.png"), ("qwenlayered_512_5L", "v9_base_euler_fox_cottage_512px_5L_4st.png"),
                    ("stablelayers_euler", "v9_stable_layers_euler_fox_cottage_384px_5L_4st.png"),
                    ("stablelayers_heun", "v9_stable_layers_heun_fox_cottage_384px_5L_4st.png")):
        shutil.copy(os.path.join(q, f), os.path.join(OUT, f"fox_{name}.png"))
        print("copied", name)
    q = os.path.join(K, "results_qwen_v10", "qwen_layered_test")
    for name, f in (("lightning_2st", "v9_lightning1.0_euler_fox_cottage_384px_5L_2st.png"), ("lightning_4st", "v9_lightning1.0_euler_fox_cottage_384px_5L_4st.png")):
        shutil.copy(os.path.join(q, f), os.path.join(OUT, f"fox_{name}.png"))
        print("copied", name)


if __name__ == "__main__":
    main()
