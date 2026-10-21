#!/usr/bin/env python3
"""
Build spotless text-free XKCD stage scenes from the real XKCD dataset using
connected-component glyph & leader-line removal, and record exact speaker head
and speech-bubble anchors for crisp hand-lettered `xkcd-script.ttf` compositing.
"""
from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = Path("/Users/lika/labs/xkcd-bench/comics/images")
CLEAN_DIR = ROOT / "data" / "clean_panels"
SCENE_DIR = ROOT / "docs" / "scenes"


def clean_glyphs_in_region(
    arr: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    max_glyph_h: int = 23,
    remove_leaders: bool = True,
) -> np.ndarray:
    out = arr.copy()
    sub = out[y0:y1, x0:x1].copy()
    gray = sub.mean(axis=2)
    ink = gray < 235
    sh, sw = ink.shape
    lbl, _ = ndimage.label(ink, structure=np.ones((3, 3), dtype=int))
    erase = np.zeros((sh, sw), dtype=bool)
    for idx, slc in enumerate(ndimage.find_objects(lbl), start=1):
        if slc is None:
            continue
        ys, xs = slc
        h = ys.stop - ys.start
        w = xs.stop - xs.start
        if ys.start <= 2 and ys.stop >= sh - 3:
            continue
        comp = lbl[ys, xs] == idx
        n_pix = int(comp.sum())
        if h <= max_glyph_h and w < sw * 0.85:
            erase[ys, xs] |= comp
        elif remove_leaders and w < 65 and n_pix < max(h, w) * 4.2 and h < sh * 0.85:
            erase[ys, xs] |= comp
    erase = ndimage.binary_dilation(erase, iterations=2)
    sub[erase] = [255, 255, 255]
    out[y0:y1, x0:x1] = sub
    return out


def build_canonical_scenes() -> dict[str, dict]:
    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Scene: Armchair + Standing Cueball (#149, 350x295)
    im149 = np.array(Image.open(RAW_DIR / "149.png").convert("RGB"))
    im149 = clean_glyphs_in_region(im149, 4, 192, 4, im149.shape[1] - 4, max_glyph_h=26, remove_leaders=True)
    Image.fromarray(im149).save(SCENE_DIR / "scene_armchair.png")

    # 2. Scene: Machine Learning Linear Algebra Pile (#1838, 391x466)
    im1838 = np.array(Image.open(RAW_DIR / "1838.png").convert("RGB"))
    im1838 = clean_glyphs_in_region(im1838, 4, 224, 4, im1838.shape[1] - 4, max_glyph_h=27, remove_leaders=True)
    # Also remove the long left leader line above the left character's head (head top is at y=318)
    im1838 = clean_glyphs_in_region(im1838, 140, 314, 35, 95, max_glyph_h=25, remove_leaders=True)
    # Also remove the short right leader tail above the right character's head (head top is at y=226)
    im1838[200:225, 220:280] = 255
    Image.fromarray(im1838).save(SCENE_DIR / "scene_ml_pile.png")

    # 3. Scene: Compiling Sword Fight on Office Chairs (#303, 402x369)
    im303 = np.array(Image.open(RAW_DIR / "303.png").convert("RGB"))
    im303[4:120, 4:-4] = 255  # top banner
    im303 = clean_glyphs_in_region(im303, 120, 360, 15, 250, max_glyph_h=26, remove_leaders=False)
    Image.fromarray(im303).save(SCENE_DIR / "scene_compiling.png")

    # 4. Scene: Dependency Jenga Tower (#2347, 381x495)
    im2347 = np.array(Image.open(RAW_DIR / "2347.png").convert("RGB"))
    im2347[4:92, 20:360] = 255
    sub = im2347[230:446, 205:375].copy()
    ink = sub.mean(axis=2) < 235
    lbl, _ = ndimage.label(ink, structure=np.ones((3, 3), dtype=int))
    er = np.zeros_like(ink, dtype=bool)
    for idx, slc in enumerate(ndimage.find_objects(lbl), start=1):
        if slc is None:
            continue
        ys, xs = slc
        if (ys.stop - ys.start) <= 28 and xs.start > 2:
            er[ys, xs] |= lbl[ys, xs] == idx
    er = ndimage.binary_dilation(er, iterations=2)
    sub[er] = [255, 255, 255]
    im2347[230:446, 205:375] = sub
    Image.fromarray(im2347).save(SCENE_DIR / "scene_dependency.png")

    # 5. Scene: Duty Calls Computer Desk (#386, 300x330)
    im386 = np.array(Image.open(RAW_DIR / "386.png").convert("RGB"))
    im386 = clean_glyphs_in_region(im386, 4, 178, 4, im386.shape[1] - 4, max_glyph_h=26, remove_leaders=True)
    im386[4:240, 4:135] = 255  # off-screen voice line on left
    im386[220:246, 178:198] = 255  # 'CLICK' above keyboard without touching Cueball's face
    Image.fromarray(im386).save(SCENE_DIR / "scene_desk.png")

    # 6. Scene: Cueball + Ponytail Dialogue Panel (#927 middle panel, 202x247)
    im927 = np.array(Image.open(RAW_DIR / "927.png").convert("RGB"))
    p927 = im927[36:, 148:350].copy()
    p927 = clean_glyphs_in_region(p927, 4, 162, 4, p927.shape[1] - 4, max_glyph_h=22, remove_leaders=False)
    Image.fromarray(p927).save(SCENE_DIR / "scene_dialogue.png")

    # Also clean all 226 panels in data/clean_panels for training/dataset consistency
    for p in sorted(RAW_DIR.glob("*.png")):
        try:
            im = np.array(Image.open(p).convert("RGB"))
            H, W, _ = im.shape
            im_c = clean_glyphs_in_region(im, 4, int(H * 0.52), 4, W - 4, max_glyph_h=23, remove_leaders=True)
            Image.fromarray(im_c).resize((256, 256), Image.Resampling.LANCZOS).save(CLEAN_DIR / f"panel_{p.stem}.png")
        except Exception:
            pass

    scenes_meta = {
        "scene_armchair": {
            "file": "scene_armchair.png",
            "width": int(im149.shape[1]),
            "height": int(im149.shape[0]),
            "left_head": [93, 198],
            "right_head": [206, 184],
        },
        "scene_ml_pile": {
            "file": "scene_ml_pile.png",
            "width": int(im1838.shape[1]),
            "height": int(im1838.shape[0]),
            "left_head": [54, 332],
            "right_head": [212, 246],
        },
        "scene_compiling": {
            "file": "scene_compiling.png",
            "width": int(im303.shape[1]),
            "height": int(im303.shape[0]),
            "header_box": [200, 56],
            "bubbles": [
                {"cx": 96, "cy": 168, "max_chars": 15},
                {"cx": 175, "cy": 240, "max_chars": 14},
                {"cx": 92, "cy": 332, "max_chars": 16},
            ],
        },
        "scene_dependency": {
            "file": "scene_dependency.png",
            "width": int(im2347.shape[1]),
            "height": int(im2347.shape[0]),
            "top_caption": [138, 36],
            "side_caption": [292, 310],
        },
        "scene_desk": {
            "file": "scene_desk.png",
            "width": int(im386.shape[1]),
            "height": int(im386.shape[0]),
            "left_voice": [75, 195],
            "right_head": [162, 182],
        },
        "scene_dialogue": {
            "file": "scene_dialogue.png",
            "width": int(p927.shape[1]),
            "height": int(p927.shape[0]),
            "left_head": [66, 134],
            "right_head": [126, 142],
        },
    }

    # Embed base64 data URIs so JS and Python SVG/PNG engines work standalone
    for k, v in scenes_meta.items():
        raw_bytes = (SCENE_DIR / v["file"]).read_bytes()
        v["data_uri"] = "data:image/png;base64," + base64.b64encode(raw_bytes).decode("ascii")

    (SCENE_DIR / "scenes_meta.json").write_text(json.dumps(scenes_meta, indent=2))
    print(f"Built {len(scenes_meta)} spotless canonical XKCD scenes in {SCENE_DIR}")
    return scenes_meta


if __name__ == "__main__":
    build_canonical_scenes()
