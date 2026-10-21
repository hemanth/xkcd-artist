#!/usr/bin/env python3
"""
Build spotless text-free single-panel AND multi-panel XKCD stage canvases from
the real XKCD dataset using connected-component glyph & leader-line removal.
"""
from __future__ import annotations

import base64
import json
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
    im1838 = clean_glyphs_in_region(im1838, 140, 314, 35, 95, max_glyph_h=25, remove_leaders=True)
    im1838[200:225, 220:280] = 255
    Image.fromarray(im1838).save(SCENE_DIR / "scene_ml_pile.png")

    # 3. Scene: Compiling Sword Fight on Office Chairs (#303, 402x369)
    im303 = np.array(Image.open(RAW_DIR / "303.png").convert("RGB"))
    im303[4:120, 4:-4] = 255
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
    im386[4:240, 4:135] = 255
    im386[220:246, 178:198] = 255
    Image.fromarray(im386).save(SCENE_DIR / "scene_desk.png")

    # 6. Scene: Cueball + Ponytail Dialogue Panel (#927 middle panel, 202x247)
    im927 = np.array(Image.open(RAW_DIR / "927.png").convert("RGB"))
    p927 = im927[36:, 148:350].copy()
    p927 = clean_glyphs_in_region(p927, 4, 162, 4, p927.shape[1] - 4, max_glyph_h=22, remove_leaders=False)
    Image.fromarray(p927).save(SCENE_DIR / "scene_dialogue.png")

    # 7. Multi-Panel Scene: 3-Panel Megan & Cueball Debate Strip (#552, expanded height to 240px for roomy dialogue)
    im552_raw = np.array(Image.open(RAW_DIR / "552.png").convert("RGB"))
    col_dark = (im552_raw.mean(axis=2) < 100).mean(axis=0)
    for x in range(im552_raw.shape[1]):
        if col_dark[x] < 0.7:
            im552_raw[4:75, x] = [255, 255, 255]
    im552_raw[4:88, 345:454] = [255, 255, 255]
    im552_raw[70:85, 225:255] = [255, 255, 255]
    im552_raw[70:85, 330:355] = [255, 255, 255]
    # Add 55px of clean upper headroom inside each of the 3 panels so 3-4 lines of dialogue fit effortlessly
    extra_h = 56
    H5, W5, _ = im552_raw.shape
    im552 = np.full((H5 + extra_h, W5, 3), 255, dtype=np.uint8)
    im552[:4, :] = im552_raw[:4, :]
    for y in range(4, extra_h + 4):
        im552[y, :] = im552_raw[10, :]
    im552[extra_h + 4 :, :] = im552_raw[4:, :]
    Image.fromarray(im552).save(SCENE_DIR / "scene_552_3panel.png")

    # 8. Multi-Panel Scene: 4-Panel Laptop Code/System Review Strip (#1513, 740x258)
    im1513 = np.array(Image.open(RAW_DIR / "1513.png").convert("RGB"))
    col_dark1513 = (im1513.mean(axis=2) < 100).mean(axis=0)
    for x in range(im1513.shape[1]):
        if col_dark1513[x] < 0.7:
            im1513[4:134, x] = [255, 255, 255]
    im1513[96:152, 20:54] = [255, 255, 255]
    im1513[126:139, 50:115] = [255, 255, 255]
    im1513[126:148, 95:130] = [255, 255, 255]
    im1513[128:158, 195:230] = [255, 255, 255]
    im1513[128:158, 382:418] = [255, 255, 255]
    im1513[128:158, 565:605] = [255, 255, 255]
    im1513[128:154, 635:732] = [255, 255, 255]
    Image.fromarray(im1513).save(SCENE_DIR / "scene_1513_4panel.png")

    # 9. Multi-Panel Scene: 3-Panel Debugging Escalation Strip (Calm Desk -> Team Consult -> Shaking Monitor)
    # Panel A: Calm Cueball at desk (cropped from scene_desk, 220x260)
    pA = np.full((260, 220, 3), 255, dtype=np.uint8)
    desk_crop = im386[80:326, 95:295]  # 246x200
    pA[10:256, 10:210] = desk_crop
    # Panel B: Ponytail & Cueball consulting at laptop (from panel 2 of #1513, 154x250)
    pB = np.full((260, 220, 3), 255, dtype=np.uint8)
    consult_crop = im1513[4:254, 188:342]  # 250x154 without right border line
    pB[6:256, 33:187] = consult_crop
    # Panel C: Cueball shaking monitor in despair (from #979 right panel, 252x270)
    im979 = np.array(Image.open(RAW_DIR / "979.png").convert("RGB"))
    p979 = im979[4:266, 236:481].copy()
    p979[:88, :] = 255
    p979[76:100, :96] = 255
    p979[90:108, 60:86] = 255
    pC_img = Image.fromarray(p979).resize((220, 252), Image.Resampling.LANCZOS)
    pC = np.full((260, 220, 3), 255, dtype=np.uint8)
    pC[4:256, :] = np.array(pC_img)

    # Stitch Panel A, B, C into a 3-panel strip (684x264)
    strip_debug = Image.new("RGB", (220 * 3 + 24, 264), (255, 255, 255))
    d_draw = ImageDraw.Draw(strip_debug)
    for idx, panel_arr in enumerate([pA, pB, pC]):
        px = 4 + idx * 228
        strip_debug.paste(Image.fromarray(panel_arr), (px, 2))
        d_draw.rectangle([px, 2, px + 220, 261], outline=(15, 15, 15), width=2)
    strip_debug.save(SCENE_DIR / "scene_debug_3panel.png")

    # 10. Scene: Git / Distributed Systems Team (#1597, 330x420)
    im1597 = np.array(Image.open(RAW_DIR / "1597.png").convert("RGB"))
    im1597[4:288, 4:-4] = 255
    im1597[280:304, 210:250] = 255
    im1597_crop = im1597[60:, :].copy()
    im1597_crop[:3, :] = 0
    Image.fromarray(im1597_crop).save(SCENE_DIR / "scene_1597_team.png")

    scenes_meta = {}
    for fname in [
        "scene_armchair.png",
        "scene_ml_pile.png",
        "scene_compiling.png",
        "scene_dependency.png",
        "scene_desk.png",
        "scene_dialogue.png",
        "scene_552_3panel.png",
        "scene_1513_4panel.png",
        "scene_debug_3panel.png",
        "scene_1597_team.png",
    ]:
        fpath = SCENE_DIR / fname
        im = Image.open(fpath)
        key = fpath.stem
        scenes_meta[key] = {
            "file": fname,
            "width": im.width,
            "height": im.height,
            "data_uri": "data:image/png;base64," + base64.b64encode(fpath.read_bytes()).decode("ascii"),
        }

    (SCENE_DIR / "scenes_meta.json").write_text(json.dumps(scenes_meta, indent=2))
    js_code = "export const SCENES_META = " + json.dumps(scenes_meta, indent=2) + ";\n"
    (ROOT / "js" / "src" / "scenes_data.js").write_text(js_code)
    (ROOT / "docs" / "src").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "src" / "scenes_data.js").write_text(js_code)
    print(f"Built {len(scenes_meta)} spotless single & multi-panel XKCD scenes!")
    return scenes_meta


if __name__ == "__main__":
    build_canonical_scenes()
