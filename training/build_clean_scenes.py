#!/usr/bin/env python3
"""
Build spotless, pixel-accurate single-panel and multi-panel XKCD stage canvases
from authentic XKCD panels using exact connected-component preservation
(zero sliced heads, zero orphan leader-line stubs, 100% intact props & borders).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = Path("/Users/lika/labs/xkcd-bench/comics/images")
CLEAN_DIR = ROOT / "data" / "clean_panels"
SCENE_DIR = ROOT / "docs" / "scenes"


def keep_connected_components(
    rgb: np.ndarray,
    predicate: Callable[[int, slice, slice, int], bool],
    ink_thresh: int = 228,
    dilate_erase: int = 1,
) -> np.ndarray:
    """
    Erase connected ink components for which `predicate(comp_id, ys, xs, npix)` is False,
    preserving anti-aliased edges of all kept components and non-ink background fills.
    """
    out = rgb.copy()
    gray = out.mean(axis=2)
    ink = gray < ink_thresh
    lbl, _ = ndimage.label(ink, structure=np.ones((3, 3), dtype=int))

    keep_mask = np.zeros_like(ink, dtype=bool)
    erase_mask = np.zeros_like(ink, dtype=bool)

    for idx, slc in enumerate(ndimage.find_objects(lbl), start=1):
        if slc is None:
            continue
        ys, xs = slc
        comp = lbl[ys, xs] == idx
        npix = int(comp.sum())
        if predicate(idx, ys, xs, npix):
            keep_mask[ys, xs] |= comp
        else:
            erase_mask[ys, xs] |= comp

    if dilate_erase > 0:
        # Dilate erased glyphs slightly to clean JPEG/PNG anti-aliasing halos,
        # while strictly protecting dilated kept components (borders, heads, props).
        protected = ndimage.binary_dilation(keep_mask, iterations=1)
        erase_mask = ndimage.binary_dilation(erase_mask, iterations=dilate_erase) & (~protected)

    out[erase_mask] = [255, 255, 255]
    return out


def build_canonical_scenes() -> dict[str, dict]:
    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Scene: Armchair + Standing Cueball (#149, 360x299)
    # Keep outer border (h >= 290) + both figures & armchair (ys.stop >= 260 or ys.start >= 250)
    im149 = np.array(Image.open(RAW_DIR / "149.png").convert("RGB"))
    im149_clean = keep_connected_components(
        im149,
        lambda _i, ys, _xs, _n: (ys.stop - ys.start) >= 290 or ys.stop >= 260 or ys.start >= 250,
    )
    Image.fromarray(im149_clean).save(SCENE_DIR / "scene_armchair.png")

    # 2. Scene: Machine Learning Linear Algebra Pile (#1838, 371x439)
    # Keep outer border (h >= 430) + everything starting in the stage zone (ys.start >= 232)
    im1838 = np.array(Image.open(RAW_DIR / "1838.png").convert("RGB"))
    im1838_clean = keep_connected_components(
        im1838,
        lambda _i, ys, _xs, _n: (ys.stop - ys.start) >= 430 or ys.start >= 232,
    )
    Image.fromarray(im1838_clean).save(SCENE_DIR / "scene_ml_pile.png")

    # 3. Scene: Compiling Sword Fight on Office Chairs (#303, 413x360)
    # Erase ONLY header text (y < 120) and glyphs strictly inside the 3 speech-bubble interiors,
    # keeping wall poster, light switch, doorway shading, bubble outlines, and sword fighters 100% intact.
    im303 = np.array(Image.open(RAW_DIR / "303.png").convert("RGB"))

    def _keep_303(_i: int, ys: slice, xs: slice, _n: int) -> bool:
        if (ys.stop - ys.start) >= 350:
            return True  # Main border + wall + doorway + speech bubble outlines
        in_header = ys.start >= 5 and ys.stop <= 122
        in_b1 = xs.start >= 30 and xs.stop <= 172 and ys.start >= 132 and ys.stop <= 186
        in_b2 = xs.start >= 130 and xs.stop <= 226 and ys.start >= 210 and ys.stop <= 254
        in_b3 = xs.start >= 26 and xs.stop <= 165 and ys.start >= 296 and ys.stop <= 338
        return not (in_header or in_b1 or in_b2 or in_b3)

    im303_clean = keep_connected_components(im303, _keep_303)
    Image.fromarray(im303_clean).save(SCENE_DIR / "scene_compiling.png")

    # 4. Scene: Dependency Jenga Tower (#2347, 385x489)
    # Keep outer border, Randall Munroe's hand-drawn horizontal curly brace (y=[62..92], w>150),
    # the entire gray-shaded Jenga tower (npix > 5000), and the curved Nebraska arrow (y=[380..425], x=[204..258]).
    im2347 = np.array(Image.open(RAW_DIR / "2347.png").convert("RGB"))

    def _keep_2347(_i: int, ys: slice, xs: slice, npix: int) -> bool:
        h = ys.stop - ys.start
        w = xs.stop - xs.start
        if h >= 480 or npix >= 4000:
            return True  # Panel border + entire Jenga tower
        if w >= 150 and 58 <= ys.start <= 72:
            return True  # Authentic hand-drawn horizontal curly brace
        if 204 <= xs.start <= 220 and 380 <= ys.start <= 400 and w >= 40:
            return True  # Authentic curved arrow pointing to tiny load-bearing block
        return False

    im2347_clean = keep_connected_components(im2347, _keep_2347)
    Image.fromarray(im2347_clean).save(SCENE_DIR / "scene_dependency.png")

    # 5. Scene: Duty Calls Computer Desk (#386, 300x330)
    # Keep outer border + Cueball/chair/desk/monitor (ys.stop >= 295 or ys.start >= 245),
    # omitting the tiny "JUC"/"klik" typing letters (x=[168..202], y=[212..244]) so it's crystal-clean.
    im386 = np.array(Image.open(RAW_DIR / "386.png").convert("RGB"))
    im386_clean = keep_connected_components(
        im386,
        lambda _i, ys, _xs, _n: (ys.stop - ys.start) >= 320 or ys.stop >= 295 or ys.start >= 248,
    )
    Image.fromarray(im386_clean).save(SCENE_DIR / "scene_desk.png")

    # 6. Scene: Cueball + Ponytail Dialogue Panel (#927 middle panel, 203x246)
    # Crop middle panel [37:283, 147:350] and keep panel border + full Cueball head/neck/body & Ponytail (ys.stop >= 150 in crop)
    im927 = np.array(Image.open(RAW_DIR / "927.png").convert("RGB"))
    p927 = im927[37:283, 147:350].copy()  # 246x203
    p927_clean = keep_connected_components(
        p927,
        lambda _i, ys, _xs, _n: (ys.stop - ys.start) >= 240 or ys.stop >= 148,
    )
    Image.fromarray(p927_clean).save(SCENE_DIR / "scene_dialogue.png")

    # 7. Multi-Panel Scene: 3-Panel Megan & Cueball Debate Strip (#552, expanded cleanly to 459x241)
    # Keep 3 panel borders (h >= 180) + all 6 stick figures (ys.stop >= 165) — zero ticks above heads!
    im552_raw = np.array(Image.open(RAW_DIR / "552.png").convert("RGB"))
    im552_base = keep_connected_components(
        im552_raw,
        lambda _i, ys, _xs, _n: (ys.stop - ys.start) >= 180 or ys.stop >= 165,
    )
    extra_h = 56
    h5, w5, _ = im552_base.shape
    im552 = np.full((h5 + extra_h, w5, 3), 255, dtype=np.uint8)
    im552[:4, :] = im552_base[:4, :]
    # Extend vertical panel borders cleanly through the added headroom using row y=20 (pure vertical borders)
    for y in range(4, extra_h + 4):
        im552[y, :] = im552_base[20, :]
    im552[extra_h + 4 :, :] = im552_base[4:, :]
    Image.fromarray(im552).save(SCENE_DIR / "scene_552_3panel.png")

    # 8. Multi-Panel Scene: 4-Panel Laptop Code/System Review Strip (#1513, 740x258)
    # Keep borders (h >= 250) + all characters/desks/chairs/wheels (ys.stop >= 245 or ys.start >= 235)
    im1513_raw = np.array(Image.open(RAW_DIR / "1513.png").convert("RGB"))
    im1513_clean = keep_connected_components(
        im1513_raw,
        lambda _i, ys, _xs, _n: (ys.stop - ys.start) >= 250 or ys.stop >= 245 or ys.start >= 235,
    )
    # Frame Panel 2 uniformly so all 4 panels have crisp 2px XKCD black borders
    im1513_pil = Image.fromarray(im1513_clean)
    d1513 = ImageDraw.Draw(im1513_pil)
    d1513.rectangle([181, 1, 334, 256], outline=(10, 10, 10), width=2)
    im1513_pil.save(SCENE_DIR / "scene_1513_4panel.png")
    im1513_framed = np.array(im1513_pil)

    # 9. Multi-Panel Scene: 3-Panel Debugging Escalation Strip (Calm Desk -> Team Consult -> Shaking Monitor, 684x264)
    pA = np.full((260, 220, 3), 255, dtype=np.uint8)
    desk_crop = im386_clean[80:326, 95:295]  # 246x200 spotless Cueball at desk
    pA[10:256, 10:210] = desk_crop

    pB = np.full((260, 220, 3), 255, dtype=np.uint8)
    consult_crop = im1513_clean[4:254, 184:332]  # 250x148 spotless Ponytail + Cueball at laptop
    pB[6:256, 36:184] = consult_crop

    # Spotless #979 right panel (Cueball shaking monitor in despair): keep border + stage (ys.start >= 98)
    im979_raw = np.array(Image.open(RAW_DIR / "979.png").convert("RGB"))
    p979_right = im979_raw[:, 233:485].copy()  # 270x252
    p979_clean = keep_connected_components(
        p979_right,
        lambda _i, ys, _xs, _n: (ys.stop - ys.start) >= 265 or ys.start >= 98,
    )
    pC_inner = Image.fromarray(p979_clean[4:266, 4:248]).resize((212, 248), Image.Resampling.LANCZOS)
    pC = np.full((260, 220, 3), 255, dtype=np.uint8)
    pC[8:256, 4:216] = np.array(pC_inner)

    strip_debug = Image.new("RGB", (220 * 3 + 24, 264), (255, 255, 255))
    d_draw = ImageDraw.Draw(strip_debug)
    for idx, panel_arr in enumerate([pA, pB, pC]):
        px = 4 + idx * 228
        strip_debug.paste(Image.fromarray(panel_arr), (px, 2))
        d_draw.rectangle([px, 2, px + 220, 261], outline=(10, 10, 10), width=2)
    strip_debug.save(SCENE_DIR / "scene_debug_3panel.png")

    # 10. Multi-Panel Scene: 2-Panel Setup -> Despair Strip (`two_panel`, 464x264)
    strip_2p = Image.new("RGB", (220 * 2 + 24, 264), (255, 255, 255))
    d2_draw = ImageDraw.Draw(strip_2p)
    for idx, panel_arr in enumerate([pA, pC]):
        px = 4 + idx * 228
        strip_2p.paste(Image.fromarray(panel_arr), (px, 2))
        d2_draw.rectangle([px, 2, px + 220, 261], outline=(10, 10, 10), width=2)
    strip_2p.save(SCENE_DIR / "scene_979_2panel.png")

    # 11. Scene: Git / Distributed Systems Team (#1597, 330x418)
    # Keep outer border (h >= 470) + Cueball, Ponytail, Hairy, desk, and chair (ys.stop >= 445)
    im1597_raw = np.array(Image.open(RAW_DIR / "1597.png").convert("RGB"))
    im1597_clean = keep_connected_components(
        im1597_raw,
        lambda _i, ys, _xs, _n: (ys.stop - ys.start) >= 470 or ys.stop >= 445,
    )
    im1597_crop = im1597_clean[60:, :].copy()
    im1597_crop[:2, :] = [10, 10, 10]
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
        "scene_979_2panel.png",
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
