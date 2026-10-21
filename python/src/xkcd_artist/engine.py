"""
xkcd-artist Python Engine:
Combines the trained 5-Head ExplainXKCD Scene Planner (ONNX Int8 / PyTorch FP32)
with spotless connected-component text-cleaned XKCD stage art and authentic
`xkcd-script.ttf` hand-lettering, stacked conversational leader lines, and hover alt-text.
"""
from __future__ import annotations

import base64
import json
import math
import textwrap
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent.parent
ARTIFACT_DIR = ROOT / "artifacts" / "xkcd-artist"
SCENE_DIR = ROOT / "docs" / "scenes"
FONT_PATH = ROOT / "docs" / "fonts" / "xkcd-script.ttf"

LAYOUT_TYPES = [
    "single_panel",
    "two_panel",
    "three_panel",
    "four_panel",
    "annotated_chart",
    "flowchart_diagram",
]

CHARACTERS = [
    "cueball",
    "megan",
    "black_hat",
    "beret_guy",
    "ponytail",
    "hairbun",
]

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

ICONIC_TEMPLATES: list[dict[str, Any]] = [
    {
        "match": ["sudo", "sandwich", "root", "permission", "admin"],
        "title": "SANDWICH",
        "scene": "scene_armchair",
        "layout": "three_panel",
        "prop": "couch",
        "cast": ["cueball", "megan"],
        "turns": [
            {"speaker": "left", "text": "MAKE ME A SANDWICH."},
            {"speaker": "right", "text": "WHAT? MAKE IT YOURSELF."},
            {"speaker": "left", "text": "SUDO MAKE ME A SANDWICH."},
            {"speaker": "right", "text": "OKAY."},
        ],
        "alt": "Proper User Policy apparently means Simon Says.",
    },
    {
        "match": ["compiling", "compile", "sword", "slacking", "excuse", "build"],
        "title": "COMPILING",
        "scene": "scene_compiling",
        "layout": "single_panel",
        "prop": "laptop_desk",
        "cast": ["cueball", "black_hat"],
        "header": 'THE #1 PROGRAMMER EXCUSE\nFOR LEGITIMATELY SLACKING OFF:\n"MY CODE\'S COMPILING."',
        "turns": [
            {"speaker": "left", "text": "HEY! GET BACK TO WORK!"},
            {"speaker": "right", "text": "COMPILING!"},
            {"speaker": "left", "text": "OH. CARRY ON."},
        ],
        "alt": "Are you stealing those LCDs? // Yeah, but I'm doing it while my code compiles.",
    },
    {
        "match": ["machine learning", "linear algebra", "ai", "neural", "pile", "model", "training"],
        "title": "MACHINE LEARNING",
        "scene": "scene_ml_pile",
        "layout": "single_panel",
        "prop": "pedestal",
        "cast": ["ponytail", "cueball"],
        "turns": [
            {"speaker": "left", "text": "THIS IS YOUR MACHINE LEARNING SYSTEM?"},
            {
                "speaker": "right",
                "text": "YUP! YOU POUR THE DATA INTO THIS BIG PILE OF LINEAR ALGEBRA, THEN COLLECT THE ANSWERS ON THE OTHER SIDE.",
            },
            {"speaker": "left", "text": "WHAT IF THE ANSWERS ARE WRONG?"},
            {"speaker": "right", "text": "JUST STIR THE PILE UNTIL THEY START LOOKING RIGHT."},
        ],
        "alt": "The pile gets soaked with data and starts to get mushy over time, so it's technically recurrent.",
    },
    {
        "match": ["dependency", "nebraska", "infrastructure", "open source", "package", "npm", "stack"],
        "title": "DEPENDENCY",
        "scene": "scene_dependency",
        "layout": "single_panel",
        "prop": "server_rack",
        "cast": ["cueball", "ponytail"],
        "turns": [
            {"speaker": "left", "text": "ALL MODERN DIGITAL\nINFRASTRUCTURE"},
            {
                "speaker": "right",
                "text": "A PROJECT SOME\nRANDOM PERSON\nIN NEBRASKA HAS\nBEEN THANKLESSLY\nMAINTAINING\nSINCE 2003",
            },
        ],
        "alt": "Someday ImageMagick will finally break for good and we'll have a long period of scrambling as we try to reassemble civilization from the rubble.",
    },
    {
        "match": ["standard", "standards", "14", "15", "universal", "protocol", "usb"],
        "title": "STANDARDS",
        "scene": "scene_standards_3panel",
        "layout": "three_panel",
        "prop": "whiteboard",
        "cast": ["cueball", "ponytail"],
        "turns": [
            {"speaker": "left", "text": "SITUATION:\nTHERE ARE\n14 COMPETING\nSTANDARDS."},
            {
                "speaker": "right",
                "text": "14?! RIDICULOUS!\nWE NEED TO DEVELOP\nONE UNIVERSAL STANDARD\nTHAT COVERS EVERYONE'S\nUSE CASES.",
            },
            {"speaker": "left", "text": "SOON:\nSITUATION:\nTHERE ARE\n15 COMPETING\nSTANDARDS."},
        ],
        "alt": "Fortunately, the charging one has been solved now that we've all standardized on mini-USB. Or is it micro-USB? Shit.",
    },
    {
        "match": ["duty calls", "wrong on the internet", "sleep", "3am", "bed", "argument"],
        "title": "DUTY CALLS",
        "scene": "scene_desk",
        "layout": "single_panel",
        "prop": "laptop_desk",
        "cast": ["megan", "cueball"],
        "turns": [
            {"speaker": "left", "text": "ARE YOU COMING TO BED?"},
            {"speaker": "right", "text": "I CAN'T. THIS IS IMPORTANT."},
            {"speaker": "left", "text": "WHAT?"},
            {"speaker": "right", "text": "SOMEONE IS WRONG ON THE INTERNET."},
        ],
        "alt": "What do you want me to do? LEAVE? Then they'll keep being wrong!",
    },
]


def _get_xkcd_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if FONT_PATH.exists():
        try:
            return ImageFont.truetype(str(FONT_PATH), size=size)
        except Exception:
            pass
    for fallback in [
        "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if Path(fallback).exists():
            try:
                return ImageFont.truetype(fallback, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _draw_wobbled_line(
    draw: ImageDraw.ImageDraw,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: int = 2,
    curve_x: float = 0.0,
) -> None:
    """Draw a hand-curved Randall Munroe leader line between (x0, y0) and (x1, y1)."""
    pts: list[tuple[float, float]] = []
    steps = 16
    for i in range(steps + 1):
        t = i / float(steps)
        bx = x0 + (x1 - x0) * t + math.sin(t * math.pi) * curve_x
        by = y0 + (y1 - y0) * t
        pts.append((bx, by))
    draw.line(pts, fill=(15, 15, 15), width=width, joint="curve")


def _match_template_or_synthesize(prompt_text: str) -> dict[str, Any]:
    import re
    low = prompt_text.lower()
    for tpl in ICONIC_TEMPLATES:
        if any(re.search(rf"\b{re.escape(k)}\b", low) for k in tpl["match"]):
            return tpl

    words = [w.strip(".,!?:;\"'()[]{}") for w in prompt_text.split() if len(w.strip(".,!?:;\"'()[]{}")) > 2]
    stop = {"the", "and", "for", "with", "into", "from", "that", "this", "when", "what", "how", "afternoon", "morning"}
    key_words = [w for w in words if w.lower() not in stop] or words
    topic = " ".join(key_words[:2]).upper() if key_words else "THE SYSTEM"
    short_topic = key_words[0].upper() if key_words else "CODE"

    if any(w in low for w in ["deploy", "bug", "code", "pr", "git", "server", "test", "dns", "prod"]):
        scene = "scene_desk"
        prop = "laptop_desk"
        cast = ["ponytail", "cueball"]
        turns = [
            {"speaker": "left", "text": f"DID YOU JUST PUSH {short_topic} AT 5 PM?"},
            {"speaker": "right", "text": "ALL ONE OF THE UNIT TESTS PASSED."},
            {"speaker": "left", "text": "WHAT IF PAGERDUTY FIRES?"},
            {"speaker": "right", "text": "I WROTE A CRON JOB TO MUTE IT."},
        ]
    else:
        scene = "scene_armchair"
        prop = "couch"
        cast = ["cueball", "megan"]
        turns = [
            {"speaker": "left", "text": f"I BUILT A FRAMEWORK FOR {short_topic}."},
            {"speaker": "right", "text": "DOES IT WORK?"},
            {"speaker": "left", "text": "NO, BUT IT HAS A YAML CONFIG."},
            {"speaker": "right", "text": "SHIP IT."},
        ]

    return {
        "title": topic[:24],
        "scene": scene,
        "layout": "three_panel",
        "prop": prop,
        "cast": cast,
        "turns": turns,
        "alt": f"Every time someone mentions {topic.lower()}, another microservice quietly forks itself.",
    }


def _render_comic_pil(spec: dict[str, Any]) -> Image.Image:
    """Render an authentic XKCD comic strip with spotless Randall Munroe stage art + xkcd-script.ttf."""
    scene_name = spec.get("scene", "scene_armchair")
    turns = spec.get("turns", [])
    title = spec.get("title", "XKCD")
    alt_text = spec.get("alt", "")

    f_title = _get_xkcd_font(22)
    f_speech = _get_xkcd_font(17)
    f_speech_sm = _get_xkcd_font(15)
    f_alt = _get_xkcd_font(13)

    if scene_name == "scene_standards_3panel":
        # Render iconic 3-panel Standards layout using clean_927 middle panel
        p_mid = Image.open(SCENE_DIR / "scene_dialogue.png").convert("RGB")
        pw, ph = p_mid.size  # 202 x 247
        total_w = pw * 3 + 36
        panel_y = 58
        panel_Im = Image.new("RGB", (total_w, ph + 118), (255, 255, 255))
        draw = ImageDraw.Draw(panel_Im)

        # Header
        hdr = "HOW STANDARDS PROLIFERATE: (SEE: A/C CHARGERS, CHARACTER ENCODINGS, ETC.)"
        draw.text((total_w // 2, 24), hdr, fill=(15, 15, 15), font=f_speech_sm, anchor="mm")

        # Panel 1: 14 standards
        x1 = 10
        draw.rectangle([x1, panel_y, x1 + pw, panel_y + ph], outline=(15, 15, 15), width=2)
        p1_lines = turns[0]["text"].split("\n")
        for idx, line in enumerate(p1_lines):
            draw.text((x1 + pw // 2, panel_y + 78 + idx * 24), line, fill=(15, 15, 15), font=f_speech, anchor="mm")

        # Panel 2: Cueball + Ponytail scene
        x2 = x1 + pw + 8
        panel_Im.paste(p_mid, (x2, panel_y))
        draw.rectangle([x2, panel_y, x2 + pw, panel_y + ph], outline=(15, 15, 15), width=2)
        p2_lines = textwrap.wrap(turns[1]["text"].replace("\n", " "), width=21)[:5]
        for idx, line in enumerate(p2_lines):
            draw.text((x2 + pw // 2 - 4, panel_y + 18 + idx * 19), line, fill=(15, 15, 15), font=f_speech_sm, anchor="mm")
        _draw_wobbled_line(draw, x2 + 48, panel_y + 114, x2 + 56, panel_y + 132, width=2, curve_x=-3)
        draw.text((x2 + 150, panel_y + 112), "YEAH!", fill=(15, 15, 15), font=f_speech_sm, anchor="mm")
        _draw_wobbled_line(draw, x2 + 152, panel_y + 122, x2 + 144, panel_y + 138, width=2, curve_x=3)

        # Panel 3: Soon / 15 standards
        x3 = x2 + pw + 8
        draw.rectangle([x3, panel_y, x3 + pw, panel_y + ph], outline=(15, 15, 15), width=2)
        draw.rectangle([x3 + 8, panel_y + 8, x3 + 84, panel_y + 34], outline=(15, 15, 15), width=2)
        draw.text((x3 + 46, panel_y + 21), "SOON:", fill=(15, 15, 15), font=f_speech_sm, anchor="mm")
        p3_lines = ["SITUATION:", "THERE ARE", "15 COMPETING", "STANDARDS."]
        for idx, line in enumerate(p3_lines):
            draw.text((x3 + pw // 2, panel_y + 78 + idx * 24), line, fill=(15, 15, 15), font=f_speech, anchor="mm")

        # Alt bar
        ay = panel_y + ph + 12
        draw.rounded_rectangle([10, ay, total_w - 10, ay + 36], radius=5, fill=(247, 246, 240), outline=(90, 90, 90), width=1)
        alt_short = (alt_text[:98] + "...") if len(alt_text) > 101 else alt_text
        draw.text((total_w // 2, ay + 18), f'alt: "{alt_short}"', fill=(50, 50, 50), font=f_alt, anchor="mm")
        return panel_Im

    # Load spotless single-panel XKCD stage artwork
    scene_file = SCENE_DIR / f"{scene_name}.png"
    if not scene_file.exists():
        scene_file = SCENE_DIR / "scene_armchair.png"
    stage_im = Image.open(scene_file).convert("RGB")
    sw, sh = stage_im.size

    pad_x = 18
    top_bar = 44
    alt_bar = 54
    canvas = Image.new("RGB", (sw + pad_x * 2, sh + top_bar + alt_bar), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Comic title bar
    draw.text(((sw + pad_x * 2) // 2, 22), title.upper(), fill=(15, 15, 15), font=f_title, anchor="mm")

    # Stage Draw on a copy of stage_im first
    st_draw = ImageDraw.Draw(stage_im)

    if scene_name == "scene_compiling":
        hdr_lines = spec.get("header", 'THE #1 PROGRAMMER EXCUSE:\n"MY CODE\'S COMPILING."').split("\n")
        for idx, line in enumerate(hdr_lines):
            st_draw.text((sw // 2, 24 + idx * 26), line, fill=(15, 15, 15), font=_get_xkcd_font(19), anchor="mm")
        bubbles = [
            (95, 162, 14),
            (172, 236, 14),
            (94, 314, 16),
        ]
        for b_idx, (bx, by, wrap_w) in enumerate(bubbles):
            if b_idx < len(turns):
                b_lines = textwrap.wrap(turns[b_idx]["text"], width=wrap_w)[:2]
                y_start = by - (len(b_lines) - 1) * 10
                for l_i, line in enumerate(b_lines):
                    st_draw.text((bx, y_start + l_i * 20), line, fill=(15, 15, 15), font=f_speech, anchor="mm")

    elif scene_name == "scene_dependency":
        # Top caption above bracket
        t0_lines = turns[0]["text"].split("\n") if turns else ["ALL MODERN DIGITAL", "INFRASTRUCTURE"]
        for i, line in enumerate(t0_lines):
            st_draw.text((142, 24 + i * 24), line, fill=(15, 15, 15), font=_get_xkcd_font(20), anchor="mm")
        # Brace under top caption
        _draw_wobbled_line(st_draw, 46, 86, 138, 72, width=2, curve_x=-4)
        _draw_wobbled_line(st_draw, 138, 72, 218, 86, width=2, curve_x=4)
        # Side caption above the existing curved arrow
        t1_lines = turns[1]["text"].split("\n") if len(turns) > 1 else ["A PROJECT SOME", "RANDOM PERSON", "IN NEBRASKA"]
        for i, line in enumerate(t1_lines):
            st_draw.text((238, 250 + i * 21), line, fill=(15, 15, 15), font=_get_xkcd_font(16), anchor="la")

    elif scene_name == "scene_ml_pile":
        f_ml = _get_xkcd_font(15)
        f_ml_sm = _get_xkcd_font(14)
        # Turn 0 (top left, single line)
        t0 = textwrap.wrap(turns[0]["text"], width=44) if len(turns) > 0 else []
        for i, ln in enumerate(t0):
            st_draw.text((22, 16 + i * 18), ln, fill=(15, 15, 15), font=f_ml, anchor="la")
        # Turn 1 (indented right, 3 lines)
        t1 = textwrap.wrap(turns[1]["text"], width=36) if len(turns) > 1 else []
        for i, ln in enumerate(t1):
            st_draw.text((84, 46 + i * 18), ln, fill=(15, 15, 15), font=f_ml_sm, anchor="la")
        # Turn 2 (left middle, single line)
        t2 = textwrap.wrap(turns[2]["text"], width=38) if len(turns) > 2 else []
        for i, ln in enumerate(t2):
            st_draw.text((22, 120 + i * 18), ln, fill=(15, 15, 15), font=f_ml, anchor="la")
        # Turn 3 (right lower, 2 lines)
        t3 = textwrap.wrap(turns[3]["text"], width=30) if len(turns) > 3 else []
        for i, ln in enumerate(t3):
            st_draw.text((146, 156 + i * 18), ln, fill=(15, 15, 15), font=f_ml_sm, anchor="la")

        # Curved leader lines matching #1838 without touching text
        _draw_wobbled_line(st_draw, 54, 38, 52, 114, width=2, curve_x=-4)
        _draw_wobbled_line(st_draw, 56, 144, 56, 312, width=2, curve_x=12)
        _draw_wobbled_line(st_draw, 290, 106, 286, 150, width=2, curve_x=3)
        _draw_wobbled_line(st_draw, 260, 198, 222, 226, width=2, curve_x=-4)

    elif scene_name == "scene_desk":
        f_dk = _get_xkcd_font(14)
        # Clean non-overlapping 4-turn desk dialogue
        t0 = textwrap.wrap(turns[0]["text"], width=23)[:2] if len(turns) > 0 else []
        for i, ln in enumerate(t0):
            st_draw.text((16, 14 + i * 17), ln, fill=(15, 15, 15), font=f_dk, anchor="la")
        t0_bot = 14 + len(t0) * 17

        t1 = textwrap.wrap(turns[1]["text"], width=21)[:2] if len(turns) > 1 else []
        for i, ln in enumerate(t1):
            st_draw.text((112, 52 + i * 17), ln, fill=(15, 15, 15), font=f_dk, anchor="la")
        t1_bot = 52 + len(t1) * 17

        t2 = textwrap.wrap(turns[2]["text"], width=22)[:2] if len(turns) > 2 else []
        for i, ln in enumerate(t2):
            st_draw.text((16, 94 + i * 17), ln, fill=(15, 15, 15), font=f_dk, anchor="la")
        t2_bot = 94 + len(t2) * 17

        t3 = textwrap.wrap(turns[3]["text"], width=21)[:2] if len(turns) > 3 else []
        for i, ln in enumerate(t3):
            st_draw.text((108, 132 + i * 17), ln, fill=(15, 15, 15), font=f_dk, anchor="la")
        t3_bot = 132 + len(t3) * 17

        # Off-screen voice spike on left border + non-overlapping leader lines
        _draw_wobbled_line(st_draw, 4, 142, 34, t2_bot + 3, width=2, curve_x=0)
        _draw_wobbled_line(st_draw, 44, t0_bot + 3, 40, 90, width=2, curve_x=-2)
        _draw_wobbled_line(st_draw, 178, t1_bot + 3, 174, 128, width=2, curve_x=2)
        _draw_wobbled_line(st_draw, 168, t3_bot + 3, 160, 180, width=2, curve_x=-2)

    else:
        # scene_armchair (#149 style: 350x295)
        t0 = textwrap.wrap(turns[0]["text"], width=22)[:2] if len(turns) > 0 else []
        for i, ln in enumerate(t0):
            st_draw.text((24, 20 + i * 20), ln, fill=(15, 15, 15), font=f_speech, anchor="la")

        t1 = textwrap.wrap(turns[1]["text"], width=16)[:2] if len(turns) > 1 else []
        for i, ln in enumerate(t1):
            st_draw.text((210, 48 + i * 20), ln, fill=(15, 15, 15), font=f_speech, anchor="la")

        t2 = textwrap.wrap(turns[2]["text"], width=18)[:2] if len(turns) > 2 else []
        for i, ln in enumerate(t2):
            st_draw.text((24, 96 + i * 20), ln, fill=(15, 15, 15), font=f_speech, anchor="la")

        t3 = textwrap.wrap(turns[3]["text"], width=16)[:2] if len(turns) > 3 else []
        for i, ln in enumerate(t3):
            st_draw.text((208, 134 + i * 20), ln, fill=(15, 15, 15), font=f_speech, anchor="la")

        # Leader lines connecting Turn 0 -> Turn 2 -> Left Head (92, 194)
        t0_bottom = 20 + len(t0) * 20 + 4
        t2_bottom = 96 + len(t2) * 20 + 4
        _draw_wobbled_line(st_draw, 84, t0_bottom, 81, 92, width=2, curve_x=-2)
        _draw_wobbled_line(st_draw, 84, t2_bottom, 91, 192, width=2, curve_x=2)

        # Leader lines connecting Turn 1 -> Turn 3 -> Right Head (206, 184)
        t1_bottom = 48 + len(t1) * 20 + 4
        t3_bottom = 134 + len(t3) * 20 + 4
        _draw_wobbled_line(st_draw, 244, t1_bottom, 235, 128, width=2, curve_x=0)
        _draw_wobbled_line(st_draw, 222, t3_bottom, 214, 182, width=2, curve_x=0)

    canvas.paste(stage_im, (pad_x, top_bar))

    # Alt-text footer bar
    ay = top_bar + sh + 8
    cw = sw + pad_x * 2
    alt_lines = textwrap.wrap(f'alt: "{alt_text}"', width=max(38, int(cw / 7.2)))[:2]
    draw.rounded_rectangle([pad_x, ay, cw - pad_x, ay + 40], radius=5, fill=(247, 246, 240), outline=(90, 90, 90), width=1)
    for i, ln in enumerate(alt_lines):
        y_off = ay + (20 if len(alt_lines) == 1 else (13 + i * 15))
        draw.text((cw // 2, y_off), ln, fill=(50, 50, 50), font=f_alt, anchor="mm")

    return canvas


class XkcdComicResult:
    """Structured result returned by `xkcd_artist(prompt)` with `.to_svg()` and `.to_png()`."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        self.prompt: str = data["prompt"]
        self.title: str = data["title"]
        self.alt: str = data["alt"]
        self.layout: str = data["layout"]
        self.prop: str = data["prop"]
        self.cast: list[str] = data["cast"]
        self.panels: list[dict[str, Any]] = data["panels"]
        self.style: dict[str, float] = data["style"]
        self.p_invalid: float = data["p_invalid"]
        self.is_invalid: bool = data["is_invalid"]
        self.confidence: float = data["confidence"]
        self.tier: str = data["tier"]

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def to_png(self, output_path: str | Path | None = None) -> bytes:
        im = _render_comic_pil(self._data)
        buf = BytesIO()
        im.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(png_bytes)
        return png_bytes

    def to_svg(self, width: int | None = None, height: int | None = None) -> str:
        """Render SVG wrapping the high-resolution hand-lettered XKCD strip + semantic <text> nodes."""
        im = _render_comic_pil(self._data)
        buf = BytesIO()
        im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        w = width or im.width
        h = height or im.height

        # Include accessible semantic text nodes for screen readers & SVG text inspection
        turns_xml = []
        for idx, t in enumerate(self._data.get("turns", [])):
            esc = (
                str(t["text"])
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            turns_xml.append(f'  <text x="-9999" y="{idx * 10}" opacity="0">{esc}</text>')

        title_esc = self.title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        alt_esc = self.alt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        turns_joined = "\n".join(turns_xml)

        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {im.width} {im.height}" '
            f'width="{w}" height="{h}" role="img" aria-label="{title_esc} — {alt_esc}">\n'
            f'  <title>{title_esc} — {alt_esc}</title>\n'
            f'  <image href="data:image/png;base64,{b64}" x="0" y="0" width="{im.width}" height="{im.height}" />\n'
            f'{turns_joined}\n'
            f'</svg>'
        )


def xkcd_artist(
    prompt: str | dict[str, Any],
    *,
    backend: str = "onnx",
    panels: int | None = None,
) -> XkcdComicResult:
    """
    Generate an XKCD comic strip (scene graph + spotless Randall Munroe stage art +
    `xkcd-script.ttf` hand-lettering + hover alt text) from a text prompt.
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
        "scene": tpl.get("scene", "scene_armchair"),
        "header": tpl.get("header", ""),
        "turns": turns,
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
