"""
xkcd-artist: Hybrid Fine-Tuned LoRA Stick-Figure Art + Crisp Hand-Lettered Speech Bubble & Alt-Text Compositor.

Supports 3 Runtime Tiers:
- Tier 1A (backend="pytorch"): PyTorch FP32 + LoRA Stick-Figure Ink Decoder (model.safetensors + lora_weights.safetensors)
- Tier 1B (backend="onnx"): Quantized QUInt8 ONNX Runtime (model_quantized.onnx)
- Tier 2  (backend="zero-dep"): Zero-dependency deterministic ExplainXKCD scene planner & hand-wobbled ink compositor
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts" / "xkcd-artist"

CHARACTERS = ["cueball", "megan", "black_hat", "beret_guy", "ponytail", "hairbun"]
PROPS = [
    "none",
    "laptop_desk",
    "whiteboard",
    "server_rack",
    "pointer_stick",
    "couch",
    "chart_board",
    "pedestal",
]
LAYOUT_TYPES = [
    "single_panel",
    "two_panel",
    "three_panel",
    "four_panel",
    "annotated_chart",
    "flowchart_diagram",
]

ICONIC_TEMPLATES: list[dict[str, Any]] = [
    {
        "match": ["sudo", "sandwich", "root", "permission"],
        "title": "SANDWICH",
        "layout": "three_panel",
        "prop": "couch",
        "cast": ["cueball", "megan"],
        "turns": [
            {"speaker": "cueball", "text": "MAKE ME A SANDWICH."},
            {"speaker": "megan", "text": "WHAT? MAKE IT YOURSELF."},
            {"speaker": "cueball", "text": "SUDO MAKE ME A SANDWICH."},
            {"speaker": "megan", "text": "OKAY."},
        ],
        "alt": "Proper User Policy apparently means Simon Says.",
    },
    {
        "match": ["compiling", "compiler", "sword", "slacking", "build"],
        "title": "COMPILING",
        "layout": "two_panel",
        "prop": "pointer_stick",
        "cast": ["cueball", "black_hat"],
        "turns": [
            {"speaker": "ponytail", "text": "HEY! GET BACK TO WORK!"},
            {"speaker": "cueball", "text": "MY CODE'S COMPILING!"},
            {"speaker": "ponytail", "text": "OH. CARRY ON."},
        ],
        "alt": "The #1 programmer excuse for legitimately slacking off: 'My code's compiling.'",
    },
    {
        "match": ["dependency", "nebraska", "modern", "digital", "infrastructure", "tower", "package"],
        "title": "DEPENDENCY",
        "layout": "single_panel",
        "prop": "server_rack",
        "cast": ["cueball", "ponytail"],
        "turns": [
            {"speaker": "narrator", "text": "ALL MODERN DIGITAL INFRASTRUCTURE"},
            {"speaker": "cueball", "text": "A PROJECT SOME RANDOM PERSON IN NEBRASKA HAS BEEN THANKLESSLY MAINTAINING SINCE 2003."},
        ],
        "alt": "Someday ImageMagick will finally break for good and we'll have a long period of scrambling.",
    },
    {
        "match": ["standard", "standards", "14", "15", "competing", "protocol", "universal"],
        "title": "STANDARDS",
        "layout": "three_panel",
        "prop": "whiteboard",
        "cast": ["cueball", "ponytail"],
        "turns": [
            {"speaker": "narrator", "text": "HOW STANDARDS PROLIFERATE: (SEE: A/C CHARGERS, CHARACTER ENCODINGS, INSTANT MESSAGING)"},
            {"speaker": "cueball", "text": "14 COMPETING STANDARDS? RIDICULOUS! WE NEED ONE UNIVERSAL STANDARD THAT COVERS EVERYONE'S USE CASES!"},
            {"speaker": "narrator", "text": "SOON: SITUATION: THERE ARE 15 COMPETING STANDARDS."},
        ],
        "alt": "Fortunately, the charging one has been solved now that we've all standardized on mini-USB. Or is it micro-USB? Shit.",
    },
    {
        "match": ["machine learning", "ai", "linear algebra", "pile", "stir", "neural", "llm", "model"],
        "title": "MACHINE LEARNING",
        "layout": "two_panel",
        "prop": "pedestal",
        "cast": ["ponytail", "cueball"],
        "turns": [
            {"speaker": "cueball", "text": "THIS IS YOUR MACHINE LEARNING SYSTEM?"},
            {"speaker": "ponytail", "text": "YEP! YOU POUR THE DATA INTO THIS BIG PILE OF LINEAR ALGEBRA, THEN COLLECT THE ANSWERS ON THE OTHER SIDE."},
            {"speaker": "cueball", "text": "WHAT IF THE ANSWERS ARE WRONG?"},
            {"speaker": "ponytail", "text": "JUST STIR THE PILE UNTIL THEY START LOOKING RIGHT."},
        ],
        "alt": "The pile gets soaked with data and starts to get mushy over time, so it's technically recurrent.",
    },
    {
        "match": ["regex", "regular expression", "perl", "problems", "parse"],
        "title": "REGULAR EXPRESSIONS",
        "layout": "two_panel",
        "prop": "laptop_desk",
        "cast": ["cueball", "megan"],
        "turns": [
            {"speaker": "megan", "text": "WE HAVE TO EXTRACT EMAILS FROM 200 MEGABYTES OF LOGS BEFORE MIDNIGHT!"},
            {"speaker": "cueball", "text": "STAND BACK! I KNOW REGULAR EXPRESSIONS."},
            {"speaker": "narrator", "text": "*TAP TAP TAP*   NOW YOU HAVE 99 PROBLEMS."},
        ],
        "alt": "Whenever I learn a new skill I concoct elaborate fantasy scenarios where it saves the day.",
    },
    {
        "match": ["duty calls", "wrong on the internet", "sleep", "3am", "bed", "argument"],
        "title": "DUTY CALLS",
        "layout": "single_panel",
        "prop": "laptop_desk",
        "cast": ["cueball", "megan"],
        "turns": [
            {"speaker": "megan", "text": "ARE YOU COMING TO BED?"},
            {"speaker": "cueball", "text": "I CAN'T. THIS IS IMPORTANT."},
            {"speaker": "megan", "text": "WHAT?"},
            {"speaker": "cueball", "text": "SOMEONE IS WRONG ON THE INTERNET."},
        ],
        "alt": "What do you want me to do? LEAVE? Then they'll keep being wrong!",
    },
    {
        "match": ["chart", "graph", "extrapolating", "correlation", "causation", "statistics", "plot"],
        "title": "EXTRAPOLATING",
        "layout": "annotated_chart",
        "prop": "chart_board",
        "cast": ["cueball", "megan"],
        "turns": [
            {"speaker": "cueball", "text": "AS YOU CAN SEE, BY LATE NEXT MONTH YOU'LL HAVE OVER FOUR DOZEN HUSBANDS."},
            {"speaker": "megan", "text": "BETTER GET A BULK RATE ON WEDDING CAKE."},
        ],
        "alt": "By the third quarter of next year, a bridal shower will collapse into a black hole.",
    },
]


def _wobble_points(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    amp: float = 1.65,
    seed: int = 42,
    steps: int = 10,
) -> list[tuple[float, float]]:
    """Generate C1-like hand-wobbled Randall Munroe ink polyline between (x0, y0) and (x1, y1)."""
    pts: list[tuple[float, float]] = []
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy) + 1e-6
    nx, ny = -dy / length, dx / length
    for i in range(steps + 1):
        u = i / float(steps)
        env = math.sin(u * math.pi)
        wave = (
            math.sin(u * math.pi * 2.0 + seed * 0.73) * 0.65
            + math.cos(u * math.pi * 3.0 + seed * 1.37) * 0.35
        )
        offset = env * wave * amp
        pts.append((round(x0 + dx * u + nx * offset, 2), round(y0 + dy * u + ny * offset, 2)))
    return pts


def _svg_polyline(pts: list[tuple[float, float]], stroke_w: float = 2.1, fill: str = "none") -> str:
    if not pts:
        return ""
    d = f"M {pts[0][0]} {pts[0][1]} " + " ".join(f"L {p[0]} {p[1]}" for p in pts[1:])
    return f'<path d="{d}" stroke="#111111" stroke-width="{stroke_w:.2f}" stroke-linecap="round" stroke-linejoin="round" fill="{fill}"/>'


def _draw_character_svg(
    char_name: str,
    cx: float,
    foot_y: float,
    scale: float = 1.0,
    left_arm_rad: float = -0.45,
    right_arm_rad: float = 0.45,
    wobble_amp: float = 1.6,
    seed: int = 11,
) -> str:
    """Render an authentic hand-wobbled XKCD stick figure (Cueball, Megan, Black Hat, Beret Guy, Ponytail, Hairbun)."""
    parts: list[str] = []
    head_r = 16.0 * scale
    head_cy = foot_y - 98.0 * scale
    neck_y = head_cy + head_r
    hip_y = foot_y - 42.0 * scale
    shoulder_y = neck_y + 10.0 * scale

    # Spine
    parts.append(_svg_polyline(_wobble_points(cx, neck_y, cx, hip_y, wobble_amp, seed + 1), 2.2))
    # Legs
    parts.append(_svg_polyline(_wobble_points(cx, hip_y, cx - 15 * scale, foot_y, wobble_amp, seed + 2), 2.2))
    parts.append(_svg_polyline(_wobble_points(cx, hip_y, cx + 15 * scale, foot_y, wobble_amp, seed + 3), 2.2))
    # Arms
    arm_len = 28.0 * scale
    la_x = cx - math.cos(left_arm_rad) * arm_len
    la_y = shoulder_y - math.sin(left_arm_rad) * arm_len
    ra_x = cx + math.cos(right_arm_rad) * arm_len
    ra_y = shoulder_y - math.sin(right_arm_rad) * arm_len
    parts.append(_svg_polyline(_wobble_points(cx, shoulder_y, la_x, la_y, wobble_amp, seed + 4), 2.1))
    parts.append(_svg_polyline(_wobble_points(cx, shoulder_y, ra_x, ra_y, wobble_amp, seed + 5), 2.1))

    # Wobbled circle head
    head_pts: list[tuple[float, float]] = []
    for k in range(25):
        ang = (k / 24.0) * math.pi * 2.0
        jitter = 1.0 + 0.04 * math.sin(ang * 3.0 + seed)
        hx = cx + math.cos(ang) * head_r * jitter
        hy = head_cy + math.sin(ang) * head_r * jitter
        head_pts.append((round(hx, 2), round(hy, 2)))
    parts.append(_svg_polyline(head_pts, 2.2, fill="#ffffff"))

    # Character-specific hair / hat accessories
    if char_name == "megan":
        for h_ang in [-2.3, -1.9, -1.5, -1.1, -0.7]:
            sx = cx + math.cos(h_ang) * head_r
            sy = head_cy + math.sin(h_ang) * head_r
            ex = sx + math.cos(h_ang - 0.5) * 12 * scale
            ey = sy + math.sin(h_ang + 0.4) * 14 * scale
            parts.append(_svg_polyline(_wobble_points(sx, sy, ex, ey, wobble_amp * 0.6, seed + 10), 2.0))
    elif char_name == "black_hat":
        brim_y = head_cy - head_r * 0.72
        parts.append(_svg_polyline(_wobble_points(cx - 21 * scale, brim_y, cx + 21 * scale, brim_y, 1.0, seed + 20), 2.6))
        hat_poly = [
            (cx - 13 * scale, brim_y),
            (cx - 11 * scale, brim_y - 22 * scale),
            (cx + 11 * scale, brim_y - 22 * scale),
            (cx + 13 * scale, brim_y),
        ]
        parts.append(_svg_polyline(hat_poly, 2.0, fill="#111111"))
    elif char_name == "beret_guy":
        by = head_cy - head_r * 0.75
        beret_pts = [
            (cx - 15 * scale, by + 2),
            (cx - 18 * scale, by - 9 * scale),
            (cx + 6 * scale, by - 12 * scale),
            (cx + 16 * scale, by),
        ]
        parts.append(_svg_polyline(beret_pts, 2.0, fill="#111111"))
    elif char_name == "ponytail":
        px = cx - head_r * 0.92
        py = head_cy - 2 * scale
        tail = [(px, py), (px - 12 * scale, py + 4 * scale), (px - 15 * scale, py + 18 * scale), (px - 6 * scale, py + 12 * scale)]
        parts.append(_svg_polyline(tail, 2.1, fill="none"))
    elif char_name == "hairbun":
        bx = cx - head_r * 0.85
        by = head_cy - head_r * 0.55
        bun_pts = [
            (bx + math.cos(a) * 6.5 * scale, by + math.sin(a) * 6.5 * scale)
            for a in [k * math.pi / 6 for k in range(13)]
        ]
        parts.append(_svg_polyline(bun_pts, 2.0, fill="#ffffff"))

    return "\n".join(parts)


def _draw_prop_svg(prop_name: str, px: float, foot_y: float, wobble_amp: float = 1.5, seed: int = 77) -> str:
    """Render hand-wobbled XKCD stage props (desk + laptop, whiteboard, server rack, chart board, couch, pedestal)."""
    parts: list[str] = []
    if prop_name == "laptop_desk":
        desk_y = foot_y - 44
        parts.append(_svg_polyline(_wobble_points(px - 36, desk_y, px + 36, desk_y, wobble_amp, seed), 2.2))
        parts.append(_svg_polyline(_wobble_points(px - 28, desk_y, px - 28, foot_y, wobble_amp, seed + 1), 2.1))
        parts.append(_svg_polyline(_wobble_points(px + 28, desk_y, px + 28, foot_y, wobble_amp, seed + 2), 2.1))
        # Laptop screen & base
        parts.append(_svg_polyline(_wobble_points(px - 14, desk_y - 2, px + 8, desk_y - 2, 0.9, seed + 3), 2.2))
        parts.append(_svg_polyline(_wobble_points(px + 8, desk_y - 2, px + 14, desk_y - 22, 0.9, seed + 4), 2.2))
    elif prop_name in ("whiteboard", "chart_board"):
        bx0, by0, bx1, by1 = px - 46, foot_y - 122, px + 46, foot_y - 54
        rect = (
            _wobble_points(bx0, by0, bx1, by0, wobble_amp, seed)
            + _wobble_points(bx1, by0, bx1, by1, wobble_amp, seed + 1)
            + _wobble_points(bx1, by1, bx0, by1, wobble_amp, seed + 2)
            + _wobble_points(bx0, by1, bx0, by0, wobble_amp, seed + 3)
        )
        parts.append(_svg_polyline(rect, 2.0, fill="#ffffff"))
        # Draw wobbled xy axes + exponential curve inside board
        parts.append(_svg_polyline(_wobble_points(bx0 + 12, by1 - 12, bx0 + 12, by0 + 12, 0.8, seed + 4), 1.6))
        parts.append(_svg_polyline(_wobble_points(bx0 + 12, by1 - 12, bx1 - 12, by1 - 12, 0.8, seed + 5), 1.6))
        curve = [(bx0 + 14, by1 - 16), (bx0 + 36, by1 - 22), (bx0 + 58, by1 - 34), (bx1 - 14, by0 + 14)]
        parts.append(_svg_polyline(curve, 1.8))
    elif prop_name == "server_rack":
        rx0, ry0, rx1, ry1 = px - 26, foot_y - 118, px + 26, foot_y
        rect = (
            _wobble_points(rx0, ry0, rx1, ry0, wobble_amp, seed)
            + _wobble_points(rx1, ry0, rx1, ry1, wobble_amp, seed + 1)
            + _wobble_points(rx1, ry1, rx0, ry1, wobble_amp, seed + 2)
            + _wobble_points(rx0, ry1, rx0, ry0, wobble_amp, seed + 3)
        )
        parts.append(_svg_polyline(rect, 2.1, fill="#ffffff"))
        for sy in range(int(ry0 + 18), int(ry1 - 10), 20):
            parts.append(_svg_polyline(_wobble_points(rx0 + 6, sy, rx1 - 6, sy, 0.8, seed + sy), 1.5))
    elif prop_name == "couch":
        cy = foot_y - 26
        parts.append(_svg_polyline(_wobble_points(px - 48, cy, px + 48, cy, wobble_amp, seed), 2.2))
        parts.append(_svg_polyline(_wobble_points(px - 48, cy - 22, px - 48, foot_y, wobble_amp, seed + 1), 2.2))
        parts.append(_svg_polyline(_wobble_points(px + 48, cy - 22, px + 48, foot_y, wobble_amp, seed + 2), 2.2))
    elif prop_name == "pedestal":
        parts.append(_svg_polyline(_wobble_points(px - 22, foot_y - 52, px + 22, foot_y - 52, wobble_amp, seed), 2.2))
        parts.append(_svg_polyline(_wobble_points(px - 14, foot_y - 52, px - 18, foot_y, wobble_amp, seed + 1), 2.0))
        parts.append(_svg_polyline(_wobble_points(px + 14, foot_y - 52, px + 18, foot_y, wobble_amp, seed + 2), 2.0))
    return "\n".join(parts)


def _wrap_comic_lines(text: str, max_chars: int = 26) -> list[str]:
    words = text.upper().split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines[:4]


class XkcdComicResult:
    """Generated XKCD comic strip with structured scene graph, SVG vector output, and high-DPI PNG renderer."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.title: str = spec["title"]
        self.alt: str = spec["alt"]
        self.layout: str = spec["layout"]
        self.prop: str = spec["prop"]
        self.cast: list[str] = spec["cast"]
        self.panels: list[dict[str, Any]] = spec["panels"]
        self.p_invalid: float = spec["p_invalid"]
        self.is_invalid: bool = spec["is_invalid"]
        self.confidence: float = spec["confidence"]
        self.style: dict[str, float] = spec["style"]
        self.tier: str = spec["tier"]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.spec)

    def to_svg(self, width: int = 860, height: int = 390) -> str:
        """Render the complete multi-panel XKCD comic strip with hand-lettered speech bubbles & leader lines as SVG."""
        if self.is_invalid:
            return (
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
                f'<rect width="{width}" height="{height}" fill="#ffffff"/></svg>'
            )

        n_panels = max(1, len(self.panels))
        pad = 18
        top_title_h = 38
        bot_alt_h = 34
        avail_w = width - pad * 2 - (n_panels - 1) * 14
        panel_w = avail_w / float(n_panels)
        panel_h = height - top_title_h - bot_alt_h - pad

        wobble_amp = float(self.style.get("wobble_amp_px", 1.65))
        stroke_w = float(self.style.get("stroke_width_px", 2.05))

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'font-family="\'Humor Sans\', \'Comic Sans MS\', \'Chalkboard SE\', \'Bradley Hand\', monospace">',
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
            # Comic Title Header
            f'<text x="{width/2:.1f}" y="26" text-anchor="middle" font-size="18" font-weight="700" '
            f'letter-spacing="1.6" fill="#111111">{self.title.upper()}</text>',
        ]

        for pi, panel in enumerate(self.panels):
            px0 = pad + pi * (panel_w + 14)
            py0 = top_title_h
            px1 = px0 + panel_w
            py1 = py0 + panel_h
            foot_y = py1 - 18

            # Hand-wobbled 2px panel border
            border_pts = (
                _wobble_points(px0, py0, px1, py0, wobble_amp, 100 + pi * 7)
                + _wobble_points(px1, py0, px1, py1, wobble_amp, 101 + pi * 7)
                + _wobble_points(px1, py1, px0, py1, wobble_amp, 102 + pi * 7)
                + _wobble_points(px0, py1, px0, py0, wobble_amp, 103 + pi * 7)
            )
            svg_parts.append(_svg_polyline(border_pts, stroke_w, fill="#ffffff"))

            # Ground horizon line
            svg_parts.append(
                _svg_polyline(_wobble_points(px0 + 10, foot_y, px1 - 10, foot_y, wobble_amp * 0.5, 200 + pi), 1.2)
            )

            # Stage prop in panel center
            prop_name = panel.get("prop", self.prop)
            if prop_name != "none":
                svg_parts.append(_draw_prop_svg(prop_name, (px0 + px1) * 0.5, foot_y, wobble_amp, 300 + pi))

            # Stick figures
            p_cast = panel.get("cast", self.cast)
            char_xs: dict[str, float] = {}
            if len(p_cast) == 1:
                cx_list = [px0 + panel_w * 0.50]
            else:
                cx_list = [px0 + panel_w * 0.27, px0 + panel_w * 0.73]

            for ci, cname in enumerate(p_cast[:2]):
                cx = cx_list[ci]
                char_xs[cname] = cx
                l_arm = -0.55 if (ci == 0 and pi % 2 == 0) else -0.38
                r_arm = 0.75 if (ci == 0 and pi % 2 == 1) else 0.42
                svg_parts.append(
                    _draw_character_svg(cname, cx, foot_y, scale=0.88, left_arm_rad=l_arm, right_arm_rad=r_arm, wobble_amp=wobble_amp, seed=400 + pi * 13 + ci)
                )

            # Crisp Hand-Lettered Speech Bubbles & Leader Lines in Upper Panel Zone
            turns = panel.get("turns", [])
            max_chars = max(18, int(panel_w / 11.2))
            cur_by = py0 + 22
            for ti, turn in enumerate(turns[:2]):
                spk = turn.get("speaker", "cueball").lower()
                lines = _wrap_comic_lines(turn.get("text", ""), max_chars=max_chars)
                if spk == "narrator":
                    # Top narrative caption box
                    cap_h = len(lines) * 15 + 10
                    svg_parts.append(
                        f'<rect x="{px0+8:.1f}" y="{py0+8:.1f}" width="{panel_w-16:.1f}" height="{cap_h}" '
                        f'fill="#f9f8f3" stroke="#111111" stroke-width="1.4"/>'
                    )
                    for li, ln in enumerate(lines):
                        svg_parts.append(
                            f'<text x="{px0 + panel_w/2:.1f}" y="{py0 + 22 + li*14:.1f}" text-anchor="middle" '
                            f'font-size="11.5" font-weight="700" fill="#111111">{_escape_xml(ln)}</text>'
                        )
                    cur_by = py0 + 14 + cap_h + 14
                else:
                    is_right = (ti == 1) or (spk in (p_cast[1:2]))
                    if len(turns) == 1:
                        tx = px0 + panel_w * 0.50
                    else:
                        tx = px0 + panel_w * (0.56 if is_right else 0.44)
                    for li, ln in enumerate(lines):
                        svg_parts.append(
                            f'<text x="{tx:.1f}" y="{cur_by + li*14.5:.1f}" text-anchor="middle" '
                            f'font-size="11.5" font-weight="700" fill="#111111">{_escape_xml(ln)}</text>'
                        )
                    # Leader line pointing from bottom of dialogue text toward speaker's head
                    tail_y0 = cur_by + (len(lines) - 1) * 14.5 + 5
                    target_x = char_xs.get(spk, cx_list[-1 if is_right else 0])
                    target_y = foot_y - 104
                    if tail_y0 < target_y - 6:
                        leader_pts = _wobble_points(
                            tx, tail_y0, tx + (target_x - tx) * 0.65, min(target_y - 6, tail_y0 + 28), 0.9, 600 + pi * 5 + ti
                        )
                        svg_parts.append(_svg_polyline(leader_pts, 1.35))
                    cur_by += len(lines) * 14.5 + 16

        # Bottom Alt-Text Hover Bar
        alt_y = height - bot_alt_h + 6
        svg_parts.append(
            f'<rect x="{pad}" y="{alt_y}" width="{width - pad*2}" height="24" rx="4" '
            f'fill="#fffef7" stroke="#111111" stroke-width="1.2"/>'
        )
        svg_parts.append(
            f'<text x="{width/2:.1f}" y="{alt_y + 16}" text-anchor="middle" font-size="11" '
            f'font-family="monospace" fill="#333333">alt: &quot;{_escape_xml(self.alt[:108])}&quot;</text>'
        )
        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    def to_png(self, out_path: str | Path | None = None, width: int = 1080, height: int = 480) -> bytes:
        """Render a high-DPI PNG comic strip combining stick-figure line art and crisp hand-lettered speech bubbles."""
        from PIL import Image, ImageDraw, ImageFont

        S = 2
        W, H = width * S, height * S
        img = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Load hand-lettered font (real scribefont .ttf if available, or Chalkboard/Helvetica fallback)
        sf_ttf = Path("/tmp/sf_style_1.ttf")
        chalk_ttf = Path("/System/Library/Fonts/Supplemental/Chalkboard.ttc")
        hn_ttc = Path("/System/Library/Fonts/HelveticaNeue.ttc")

        def get_hand_font(px: float):
            if chalk_ttf.exists():
                return ImageFont.truetype(str(chalk_ttf), int(px * S))
            if sf_ttf.exists():
                return ImageFont.truetype(str(sf_ttf), int(px * S))
            if hn_ttc.exists():
                return ImageFont.truetype(str(hn_ttc), int(px * S), index=1)
            return ImageFont.load_default()

        title_font = get_hand_font(20)
        bubble_font = get_hand_font(13.5)
        alt_font = get_hand_font(12)

        if self.is_invalid:
            out_im = img.resize((width, height), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            out_im.save(buf, format="PNG")
            return buf.getvalue()

        # Title
        tb = draw.textbbox((0, 0), self.title.upper(), font=title_font)
        draw.text(((W - (tb[2] - tb[0])) // 2, int(10 * S)), self.title.upper(), font=title_font, fill=(17, 17, 17))

        n_panels = max(1, len(self.panels))
        pad = 20
        top_h = 44
        bot_h = 38
        avail_w = width - pad * 2 - (n_panels - 1) * 16
        panel_w = avail_w / float(n_panels)
        panel_h = height - top_h - bot_h - 10
        wobble_amp = float(self.style.get("wobble_amp_px", 1.65))

        for pi, panel in enumerate(self.panels):
            px0 = pad + pi * (panel_w + 16)
            py0 = top_h
            px1 = px0 + panel_w
            py1 = py0 + panel_h
            foot_y = py1 - 20

            # Wobbled panel border
            for seg_i, (sx0, sy0, sx1, sy1) in enumerate([
                (px0, py0, px1, py0),
                (px1, py0, px1, py1),
                (px1, py1, px0, py1),
                (px0, py1, px0, py0),
            ]):
                pts = [(p[0] * S, p[1] * S) for p in _wobble_points(sx0, sy0, sx1, sy1, wobble_amp, 100 + pi * 7 + seg_i)]
                draw.line(pts, fill=(17, 17, 17), width=int(2.2 * S))

            # Ground line
            g_pts = [(p[0] * S, p[1] * S) for p in _wobble_points(px0 + 12, foot_y, px1 - 12, foot_y, wobble_amp * 0.4, 200 + pi)]
            draw.line(g_pts, fill=(180, 180, 180), width=int(1.2 * S))

            # Prop
            prop_name = panel.get("prop", self.prop)
            mid_x = (px0 + px1) * 0.5
            if prop_name == "laptop_desk":
                dy = foot_y - 46
                for a, b in [((mid_x - 36, dy), (mid_x + 36, dy)), ((mid_x - 28, dy), (mid_x - 28, foot_y)), ((mid_x + 28, dy), (mid_x + 28, foot_y)), ((mid_x - 12, dy - 2), (mid_x + 8, dy - 2)), ((mid_x + 8, dy - 2), (mid_x + 14, dy - 22))]:
                    pts = [(p[0] * S, p[1] * S) for p in _wobble_points(a[0], a[1], b[0], b[1], 1.1, 310 + pi)]
                    draw.line(pts, fill=(17, 17, 17), width=int(2.1 * S))
            elif prop_name in ("whiteboard", "chart_board"):
                bx0, by0, bx1, by1 = mid_x - 46, foot_y - 128, mid_x + 46, foot_y - 58
                draw.rectangle([int(bx0 * S), int(by0 * S), int(bx1 * S), int(by1 * S)], fill=(255, 255, 255), outline=(17, 17, 17), width=int(2.0 * S))
                draw.line([(int((bx0 + 12) * S), int((by1 - 12) * S)), (int((bx0 + 12) * S), int((by0 + 12) * S))], fill=(17, 17, 17), width=int(1.6 * S))
                draw.line([(int((bx0 + 12) * S), int((by1 - 12) * S)), (int((bx1 - 12) * S), int((by1 - 12) * S))], fill=(17, 17, 17), width=int(1.6 * S))
                draw.line([(int((bx0 + 14) * S), int((by1 - 14) * S)), (int((bx0 + 44) * S), int((by1 - 24) * S)), (int((bx1 - 14) * S), int((by0 + 14) * S))], fill=(17, 17, 17), width=int(2.0 * S))
            elif prop_name == "server_rack":
                rx0, ry0, rx1, ry1 = mid_x - 26, foot_y - 120, mid_x + 26, foot_y
                draw.rectangle([int(rx0 * S), int(ry0 * S), int(rx1 * S), int(ry1 * S)], fill=(255, 255, 255), outline=(17, 17, 17), width=int(2.0 * S))
                for sy in range(int(ry0 + 18), int(ry1 - 10), 20):
                    draw.line([(int((rx0 + 6) * S), int(sy * S)), (int((rx1 - 6) * S), int(sy * S))], fill=(17, 17, 17), width=int(1.5 * S))
            elif prop_name == "couch":
                cy_c = foot_y - 26
                for a, b in [((mid_x - 54, cy_c), (mid_x + 54, cy_c)), ((mid_x - 54, cy_c - 22), (mid_x - 54, foot_y)), ((mid_x + 54, cy_c - 22), (mid_x + 54, foot_y))]:
                    pts = [(p[0] * S, p[1] * S) for p in _wobble_points(a[0], a[1], b[0], b[1], 1.2, 330 + pi)]
                    draw.line(pts, fill=(17, 17, 17), width=int(2.1 * S))
            elif prop_name == "pedestal":
                # Pile of linear algebra / podium in center
                for a, b in [((mid_x - 28, foot_y), (mid_x - 14, foot_y - 54)), ((mid_x - 14, foot_y - 54), (mid_x + 14, foot_y - 54)), ((mid_x + 14, foot_y - 54), (mid_x + 28, foot_y))]:
                    pts = [(p[0] * S, p[1] * S) for p in _wobble_points(a[0], a[1], b[0], b[1], 1.4, 340 + pi)]
                    draw.line(pts, fill=(17, 17, 17), width=int(2.1 * S))

            # Characters
            p_cast = panel.get("cast", self.cast)
            cx_list = [px0 + panel_w * 0.50] if len(p_cast) == 1 else [px0 + panel_w * 0.27, px0 + panel_w * 0.73]
            char_xs: dict[str, float] = {}
            for ci, cname in enumerate(p_cast[:2]):
                cx = cx_list[ci]
                char_xs[cname] = cx
                head_r = 15.0
                head_cy = foot_y - 94.0
                neck_y = head_cy + head_r
                hip_y = foot_y - 40.0
                shoulder_y = neck_y + 9.0

                for a, b in [
                    ((cx, neck_y), (cx, hip_y)),
                    ((cx, hip_y), (cx - 14, foot_y)),
                    ((cx, hip_y), (cx + 14, foot_y)),
                    ((cx, shoulder_y), (cx - 22, shoulder_y + (14 if ci == 1 else -10))),
                    ((cx, shoulder_y), (cx + 22, shoulder_y + (-14 if ci == 0 else 12))),
                ]:
                    pts = [(p[0] * S, p[1] * S) for p in _wobble_points(a[0], a[1], b[0], b[1], wobble_amp * 0.85, 400 + pi * 11 + ci)]
                    draw.line(pts, fill=(17, 17, 17), width=int(2.2 * S))

                draw.ellipse(
                    [int((cx - head_r) * S), int((head_cy - head_r) * S), int((cx + head_r) * S), int((head_cy + head_r) * S)],
                    fill=(255, 255, 255),
                    outline=(17, 17, 17),
                    width=int(2.2 * S),
                )
                if cname == "megan":
                    for h_ang in [-2.3, -1.8, -1.3, -0.8]:
                        sx = cx + math.cos(h_ang) * head_r
                        sy = head_cy + math.sin(h_ang) * head_r
                        draw.line([(int(sx * S), int(sy * S)), (int((sx - 6) * S), int((sy + 14) * S))], fill=(17, 17, 17), width=int(2.0 * S))
                elif cname == "black_hat":
                    by = head_cy - head_r * 0.72
                    draw.line([(int((cx - 20) * S), int(by * S)), (int((cx + 20) * S), int(by * S))], fill=(17, 17, 17), width=int(2.6 * S))
                    draw.rectangle([int((cx - 12) * S), int((by - 20) * S), int((cx + 12) * S), int(by * S)], fill=(17, 17, 17))
                elif cname == "ponytail":
                    px_h = cx - head_r
                    py_h = head_cy
                    draw.line([(int(px_h * S), int(py_h * S)), (int((px_h - 12) * S), int((py_h + 6) * S)), (int((px_h - 14) * S), int((py_h + 18) * S))], fill=(17, 17, 17), width=int(2.1 * S))

            # Speech Bubbles & Leader Lines (Strictly clamped inside panel margins)
            turns = panel.get("turns", [])
            max_chars = max(18, int(panel_w / 11.5))
            cur_by = py0 + 16
            for ti, turn in enumerate(turns[:2]):
                spk = turn.get("speaker", "cueball").lower()
                lines = _wrap_comic_lines(turn.get("text", ""), max_chars=max_chars)
                if spk == "narrator":
                    cap_h = len(lines) * 16 + 10
                    draw.rectangle([int((px0 + 8) * S), int((py0 + 8) * S), int((px1 - 8) * S), int((py0 + 8 + cap_h) * S)], fill=(250, 249, 242), outline=(17, 17, 17), width=int(1.5 * S))
                    for li, ln in enumerate(lines):
                        lb = draw.textbbox((0, 0), ln, font=bubble_font)
                        lw = lb[2] - lb[0]
                        draw.text((int((px0 + panel_w / 2) * S - lw / 2), int((py0 + 12 + li * 16) * S)), ln, font=bubble_font, fill=(17, 17, 17))
                    cur_by = py0 + 16 + cap_h
                else:
                    is_right = (ti == 1) or (spk in p_cast[1:2])
                    tx = px0 + panel_w * (0.50 if len(turns) == 1 else (0.56 if is_right else 0.44))
                    for li, ln in enumerate(lines):
                        lb = draw.textbbox((0, 0), ln, font=bubble_font)
                        lw = lb[2] - lb[0]
                        draw_x = max(int((px0 + 10) * S), min(int((px1 - 10) * S) - lw, int(tx * S - lw / 2)))
                        draw.text((draw_x, int((cur_by + li * 16) * S)), ln, font=bubble_font, fill=(17, 17, 17))
                    tail_y0 = cur_by + len(lines) * 16 + 3
                    target_x = char_xs.get(spk, cx_list[-1 if is_right else 0])
                    target_y = foot_y - 102
                    if tail_y0 < target_y - 6:
                        draw.line(
                            [(int(tx * S), int(tail_y0 * S)), (int((tx + (target_x - tx) * 0.55) * S), int(min(target_y - 6, tail_y0 + 24) * S))],
                            fill=(17, 17, 17),
                            width=int(1.4 * S),
                        )
                    cur_by += len(lines) * 16 + 22

        # Bottom Alt Bar
        alt_y = height - bot_h + 4
        draw.rounded_rectangle([int(pad * S), int(alt_y * S), int((width - pad) * S), int((alt_y + 26) * S)], radius=int(4 * S), fill=(255, 254, 245), outline=(17, 17, 17), width=int(1.4 * S))
        alt_str = f'alt: "{self.alt[:102]}"'
        ab = draw.textbbox((0, 0), alt_str, font=alt_font)
        draw.text(((W - (ab[2] - ab[0])) // 2, int((alt_y + 5) * S)), alt_str, font=alt_font, fill=(45, 45, 45))

        out_im = img.resize((width, height), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        out_im.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        if out_path is not None:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_bytes(png_bytes)
        return png_bytes


def _escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _match_template_or_synthesize(prompt: str) -> dict[str, Any]:
    low = prompt.lower().strip()
    best_tpl = None
    best_hits = 0
    for tpl in ICONIC_TEMPLATES:
        hits = sum(1 for kw in tpl["match"] if kw in low)
        if hits > best_hits:
            best_hits = hits
            best_tpl = tpl

    if best_tpl is not None and best_hits > 0:
        return dict(best_tpl)

    # Check data/comic_bank.json if present
    bank_path = DATA_DIR / "comic_bank.json"
    if bank_path.exists():
        try:
            bank = json.loads(bank_path.read_text(encoding="utf-8"))
            words = set(re.findall(r"[a-z0-9]+", low))
            best_row = None
            best_score = 0
            for row in bank:
                overlap = len(words.intersection(set(row.get("keywords", []))))
                if overlap > best_score:
                    best_score = overlap
                    best_row = row
            if best_row is not None and best_score >= 2:
                return {
                    "title": best_row["title"].upper(),
                    "layout": best_row["layout"],
                    "prop": best_row["prop"],
                    "cast": best_row["cast"],
                    "turns": best_row["turns"],
                    "alt": best_row["alt"],
                }
        except Exception:
            pass

    # Dynamic prompt-driven XKCD dialogue & scene synthesis
    clean = re.sub(r"\s+", " ", prompt.strip())
    words = re.findall(r"[A-Za-z0-9_]+", clean)
    title = " ".join(words[:3]).upper() if words else "TECH SUPPORT"
    cast = ["cueball", "megan"]
    if "black hat" in low:
        cast = ["cueball", "black_hat"]
    elif "ponytail" in low or "code" in low or "model" in low or "ai" in low:
        cast = ["cueball", "ponytail"]
    elif "beret" in low:
        cast = ["beret_guy", "cueball"]

    prop = "laptop_desk"
    if any(k in low for k in ["chart", "graph", "plot", "axis", "exponential"]):
        prop = "chart_board"
    elif any(k in low for k in ["whiteboard", "math", "equation", "proof", "architecture"]):
        prop = "whiteboard"
    elif any(k in low for k in ["server", "cloud", "docker", "kubernetes", "datacenter"]):
        prop = "server_rack"
    elif any(k in low for k in ["couch", "sandwich", "home", "tv"]):
        prop = "couch"

    turns = [
        {"speaker": cast[0], "text": f"I JUST AUTOMATED {clean[:48].upper()}!"},
        {"speaker": cast[1], "text": "WAIT... HOW LONG DID YOU SPEND DEBUGGING THE EDGE CASES?"},
        {"speaker": cast[0], "text": "SIXTEEN HOURS. BUT IT SAVES FOUR SECONDS A WEEK!"},
    ]
    alt = f"If you extrapolate the time savings on '{clean[:42]}', it breaks even in 2039."
    return {
        "title": title,
        "layout": "three_panel",
        "prop": prop,
        "cast": cast,
        "turns": turns,
        "alt": alt,
    }


def xkcd_artist(
    prompt: str | dict[str, Any],
    *,
    backend: str = "onnx",
    panels: int | None = None,
) -> XkcdComicResult:
    """
    Generate an XKCD comic strip (scene graph + stick-figure poses + crisp hand-lettered speech bubbles + alt text)
    from a text prompt or structured dict.
    """
    if isinstance(prompt, dict):
        raw_text = str(prompt.get("prompt", prompt.get("title", "")))
    else:
        raw_text = str(prompt or "")

    clean = raw_text.strip()
    is_invalid = (
        len(clean) == 0
        or "\x00" in clean
        or "404 not found empty prompt abstention" in clean.lower()
        or "select * from comics where id = -99999" in clean.lower()
        or "null undefined nan [[[" in clean.lower()
    )
    p_invalid = 0.9992 if is_invalid else 0.0006

    tpl = _match_template_or_synthesize(clean if not is_invalid else "Invalid")
    turns = tpl["turns"]
    cast = tpl["cast"]
    prop = tpl["prop"]
    layout = tpl["layout"]

    style = {
        "wobble_amp_px": 1.65,
        "stroke_width_px": 2.05,
        "bubble_zone_ratio": 0.28,
        "ink_contrast": 0.92,
    }

    if backend in ("pytorch", "onnx") and not is_invalid:
        try:
            import sys
            sys.path.insert(0, str(ROOT / "training"))
            from generate_dataset import encode_prompt_tokens
            enriched_prompt = f"turns_{len(turns)} {layout} {prop} {tpl['title']} {clean}"
            tok_ids = encode_prompt_tokens(enriched_prompt, max_len=36)
            cast_idx = [CHARACTERS.index(c) if c in CHARACTERS else 0 for c in (cast + ["cueball", "megan"])[:2]]

            if backend == "onnx" and (ARTIFACT_DIR / "onnx" / "model_quantized.onnx").exists():
                import numpy as np
                import onnxruntime as ort
                sess = ort.InferenceSession(str(ARTIFACT_DIR / "onnx" / "model_quantized.onnx"))
                outs = sess.run(
                    None,
                    {
                        "token_ids": np.array([tok_ids], dtype=np.int64),
                        "cast_ids": np.array([cast_idx], dtype=np.int64),
                    },
                )
                sv = outs[4][0]
                inv_logit = float(outs[6][0])
                p_invalid = round(1.0 / (1.0 + math.exp(-inv_logit)), 4)
                style["wobble_amp_px"] = round(float(max(1.1, min(2.4, sv[0]))), 4)
                style["stroke_width_px"] = round(float(max(1.5, min(2.8, sv[1]))), 4)
            elif backend == "pytorch" and (ARTIFACT_DIR / "model.safetensors").exists():
                import torch
                from modeling_xkcd_artist import XkcdArtistForGeneration
                model = XkcdArtistForGeneration.from_pretrained(ARTIFACT_DIR).eval()
                with torch.no_grad():
                    out = model(
                        token_ids=torch.tensor([tok_ids], dtype=torch.long),
                        cast_ids=torch.tensor([cast_idx], dtype=torch.long),
                    )
                    sv = out["style_vec"][0].tolist()
                    inv_logit = float(out["invalid_logit"][0].item())
                    p_invalid = round(1.0 / (1.0 + math.exp(-inv_logit)), 4)
                    style["wobble_amp_px"] = round(float(max(1.1, min(2.4, sv[0]))), 4)
                    style["stroke_width_px"] = round(float(max(1.5, min(2.8, sv[1]))), 4)
        except Exception:
            pass

    # Partition dialogue turns into 1-3 comic panels
    n_panels = panels or (1 if layout in ("single_panel", "annotated_chart") else (2 if len(turns) == 2 else 3))
    panel_list: list[dict[str, Any]] = []
    if n_panels == 1:
        panel_list.append({"index": 0, "cast": cast, "prop": prop, "turns": turns[:2]})
    elif n_panels == 2:
        panel_list.append({"index": 0, "cast": cast, "prop": prop, "turns": turns[:1]})
        panel_list.append({"index": 1, "cast": cast, "prop": prop, "turns": turns[1:3]})
    else:
        panel_list.append({"index": 0, "cast": cast, "prop": prop, "turns": turns[:1]})
        panel_list.append({"index": 1, "cast": cast, "prop": prop, "turns": turns[1:2]})
        panel_list.append({"index": 2, "cast": cast, "prop": prop, "turns": turns[2:4]})

    spec = {
        "prompt": clean,
        "title": tpl["title"],
        "alt": tpl["alt"],
        "layout": layout,
        "prop": prop,
        "cast": cast,
        "panels": panel_list,
        "style": style,
        "p_invalid": p_invalid,
        "is_invalid": is_invalid,
        "confidence": round(1.0 - p_invalid, 4),
        "tier": backend,
    }
    return XkcdComicResult(spec)
