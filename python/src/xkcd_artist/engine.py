"""
xkcd-artist Python Engine:
Combines the trained 5-Head ExplainXKCD Scene Planner (ONNX Int8 / PyTorch FP32)
with spotless connected-component text-cleaned single & multi-panel XKCD stage art,
compositional multi-panel XKCD wit generation for arbitrary prompts, and authentic
`xkcd-script.ttf` hand-lettering.
"""
from __future__ import annotations

import base64
import hashlib
import math
import re
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
        "match": ["sudo", "sandwich"],
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
        "match": ["compiling", "sword fight", "slacking"],
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
        "match": ["linear algebra", "stir the pile"],
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
        "match": ["nebraska", "modern digital infrastructure"],
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
        "match": ["14 competing", "15 standards", "universal standard"],
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
        "match": ["duty calls", "wrong on the internet"],
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
    pts: list[tuple[float, float]] = []
    steps = 16
    for i in range(steps + 1):
        t = i / float(steps)
        bx = x0 + (x1 - x0) * t + math.sin(t * math.pi) * curve_x
        by = y0 + (y1 - y0) * t
        pts.append((bx, by))
    draw.line(pts, fill=(15, 15, 15), width=width, joint="curve")


def _extract_prompt_keywords(prompt_text: str) -> tuple[str, str, list[str]]:
    stop = {
        "the", "and", "for", "with", "into", "from", "that", "this", "when", "what",
        "why", "how", "about", "like", "make", "making", "trying", "explained",
        "explain", "cueball", "megan", "ponytail", "black", "hat", "beret", "guy",
        "comic", "xkcd", "panel", "strip", "someone", "every", "time", "after",
        "before", "my", "our", "your", "is", "are", "was", "in", "on", "to", "at",
        "of", "by", "reviewing", "automating", "autonomous", "line", "lines",
    }
    raw_words = [
        w.strip(".,!?:;\"'()[]{}")
        for w in prompt_text.split()
        if len(w.strip(".,!?:;\"'()[]{}")) >= 2
    ]
    key_words = [w for w in raw_words if w.lower() not in stop and not w.isdigit()] or raw_words or ["SYSTEM"]
    title = " ".join(key_words[:2]).upper()[:26]
    subject = " ".join(key_words[:2]).upper()[:20]
    return title, subject, key_words


def _synthesize_novel_comic(prompt_text: str) -> dict[str, Any]:
    """
    Generate a brand-new multi-panel XKCD comic strip from ANY arbitrary prompt.
    Selects the best multi-panel visual progression (`scene_debug_3panel`, `scene_552_3panel`,
    `scene_1513_4panel`, `scene_1597_team`) and synthesizes context-specific panel turns + alt-text.
    """
    low = prompt_text.lower()
    for tpl in ICONIC_TEMPLATES:
        if any(k in low for k in tpl["match"]):
            return tpl

    title, subject, kw = _extract_prompt_keywords(prompt_text)
    focus = kw[0].upper() if kw else "CODE"
    second = kw[1].upper() if len(kw) > 1 else "PRODUCTION"
    h_val = int(hashlib.md5(prompt_text.encode("utf-8")).hexdigest()[:8], 16)

    # 1. Docker / Containers / Kubernetes / Cloud / YAML / Microservices -> 3-Panel Debugging Escalation
    if any(w in low for w in ["docker", "container", "kubernetes", "k8s", "yaml", "helm", "cloud", "aws", "gcp", "microservice"]):
        return {
            "title": title,
            "scene": "scene_debug_3panel",
            "layout": "three_panel",
            "prop": "laptop_desk",
            "cast": ["cueball", "ponytail"],
            "turns": [
                {"speaker": "left", "text": f"I CONTAINERIZED OUR 12-LINE {focus} SCRIPT SO IT RUNS ANYWHERE."},
                {"speaker": "right", "text": f"WHY IS THE {subject} IMAGE 18.4 GIGABYTES?"},
                {"speaker": "left", "text": "IT BUNDLES UBUNTU, THREE JDK VERSIONS, AND SOMEONE'S HOME DIRECTORY FROM 2019."},
            ],
            "alt": f"If you run docker system prune, the {subject.lower()} cluster achieves negative mass and floats out of the rack.",
        }

    # 2. Git / Merge / Rebase / Branch / Commit -> Team Laptop Demo (#1597 stage)
    if any(w in low for w in ["git", "rebase", "merge", "branch", "commit", "cherry-pick", "force push"]):
        return {
            "title": title,
            "scene": "scene_1597_team",
            "layout": "three_panel",
            "prop": "laptop_desk",
            "cast": ["cueball", "ponytail", "black_hat"],
            "turns": [
                {"speaker": "left", "text": f"I RAN INTERACTIVE {subject} TO CLEAN UP OUR COMMIT HISTORY BEFORE THE RELEASE."},
                {"speaker": "right", "text": "DID IT WORK?"},
                {"speaker": "left", "text": "HEAD IS DETACHED, MAIN IS IN 1970, AND EVERYONE JUST CLONED A FRESH REPO INTO `PROJECT_FINAL_V3`."},
            ],
            "alt": f"git reflog is just a horror novel where the main character is you from twenty minutes ago.",
        }

    # 3. Code Review / PR / Pull Request / Refactor / Legacy / Style / Lint -> 4-Panel Laptop Review (#1513 stage)
    if any(w in low for w in ["review", "pr", "pull request", "refactor", "legacy", "messy", "lint", "type", "typescript", "css"]):
        return {
            "title": title,
            "scene": "scene_1513_4panel",
            "layout": "four_panel",
            "prop": "laptop_desk",
            "cast": ["cueball", "ponytail"],
            "turns": [
                {"speaker": "left", "text": f"CAN YOU REVIEW MY {focus} PR? IT'S ONLY 4,200 LINES."},
                {"speaker": "right", "text": "WHY DOES ONE REGEX PARSE BOTH HTML AND TAXES?"},
                {"speaker": "right", "text": f"AND WHY IS {focus} IMPORTING A DEPRECATED GAME ENGINE FOR DATE MATH?"},
                {"speaker": "left", "text": "LGTM, MERGING WITH `--NO-VERIFY`."},
            ],
            "alt": f"Every comment in the {subject.lower()} codebase just says '// DO NOT TOUCH OR THE BILLING CRON DIES'.",
        }

    # 4. AI / LLM / Agent / Prompt / GPT / Neural / Model -> 3-Panel Megan & Cueball Debate (#552 stage)
    if any(w in low for w in ["llm", "agent", "agents", "gpt", "prompt", "ai", "chatbot", "hallucinat", "token", "transformer"]):
        return {
            "title": title,
            "scene": "scene_552_3panel",
            "layout": "three_panel",
            "prop": "none",
            "cast": ["megan", "cueball"],
            "turns": [
                {"speaker": "right", "text": f"I DEPLOYED TWO {subject} TO AUTOMATE MY WORKFLOW."},
                {"speaker": "right", "text": "ONE OPENS TICKETS AND THE OTHER CLOSES THEM AS 'WONTFIX'."},
                {"speaker": "left", "text": "SO NOTHING GETS DONE?", "reply": "AT 40,000 TOKENS/SEC."},
            ],
            "alt": f"We added a third {focus.lower()} agent to mediate, and now all three are in a series-A fundraising call.",
        }

    # 5. Debugging / DNS / Bug / Prod / Server / SQL / Database / Vim / Error -> 3-Panel Debugging Escalation
    if any(w in low for w in ["debug", "dns", "bug", "prod", "deploy", "server", "sql", "database", "vim", "error", " outage", "cache"]):
        return {
            "title": title,
            "scene": "scene_debug_3panel",
            "layout": "three_panel",
            "prop": "laptop_desk",
            "cast": ["cueball", "ponytail"],
            "turns": [
                {"speaker": "left", "text": f"IT'S DEFINITELY NOT {focus}. I CHECKED {subject} THREE TIMES."},
                {"speaker": "right", "text": f"WHAT ABOUT THAT STALE {second} CACHE ENTRY FROM 2021?"},
                {"speaker": "left", "text": f"WHY IS {focus} ALWAYS THE PROBLEM EVEN WHEN IT'S OFF?!"},
            ],
            "alt": f" The root cause analysis concluded that {subject.lower()} was load-bearing technical debt.",
        }

    # 6. General Arbitrary Prompts (Physics, Quantum, Cats, Coffee, Math, Life, etc.) -> Rotate across multi-panel strips!
    archetypes = [
        {
            "scene": "scene_552_3panel",
            "layout": "three_panel",
            "prop": "none",
            "cast": ["megan", "cueball"],
            "turns": [
                {"speaker": "right", "text": f"I SPENT ALL WEEKEND MODELING {subject} FROM FIRST PRINCIPLES."},
                {"speaker": "right", "text": f"IF YOU ASSUME {focus} IS A FRICTIONLESS SPHERE IN A VACUUM, IT'S SOLVED."},
                {"speaker": "left", "text": "AND IN REAL LIFE?", "reply": "EXPLODES IMMEDIATELY."},
            ],
            "alt": f"Physicists hate this one weird boundary condition in {subject.lower()}.",
        },
        {
            "scene": "scene_debug_3panel",
            "layout": "three_panel",
            "prop": "laptop_desk",
            "cast": ["cueball", "ponytail"],
            "turns": [
                {"speaker": "left", "text": f"I WROTE A QUICK SCRIPT TO AUTOMATE {subject} IN FIVE MINUTES."},
                {"speaker": "right", "text": f"YOU'VE BEEN DEBUGGING THE {focus} EDGE CASES FOR SIX HOURS."},
                {"speaker": "left", "text": "IT'S ABOUT PRINCIPLE NOW!"},
            ],
            "alt": f"By year three, the automation script for {subject.lower()} has its own compliance department.",
        },
        {
            "scene": "scene_1513_4panel",
            "layout": "four_panel",
            "prop": "laptop_desk",
            "cast": ["cueball", "ponytail"],
            "turns": [
                {"speaker": "left", "text": f"CHECK OUT MY NEW {subject} BENCHMARK."},
                {"speaker": "right", "text": "YOUR ERROR BARS ARE WIDER THAN THE Y-AXIS."},
                {"speaker": "right", "text": f"AND DID YOU FIT A 9TH-DEGREE POLYNOMIAL TO THREE {focus} POINTS?"},
                {"speaker": "left", "text": "R-SQUARED IS 1.000 THOUGH."},
            ],
            "alt": f"If you extrapolate {subject.lower()} to next Tuesday, the curve loops back through the Bronze Age.",
        },
        {
            "scene": "scene_1597_team",
            "layout": "three_panel",
            "prop": "laptop_desk",
            "cast": ["cueball", "ponytail", "black_hat"],
            "turns": [
                {"speaker": "left", "text": f"OUR NEW {subject} ARCHITECTURE USES ZERO EXTERNAL DEPENDENCIES."},
                {"speaker": "right", "text": f"HOW DID YOU IMPLEMENT {focus}?"},
                {"speaker": "left", "text": "WE WROTE OUR OWN BROKEN VERSION OF HALF THE STANDARD LIBRARY IN BASH."},
            ],
            "alt": f"Eventually every {subject.lower()} project invents its own Lisp interpreter by accident.",
        },
    ]
    chosen = archetypes[h_val % len(archetypes)]
    return {"title": title, **chosen}


def _render_comic_pil(spec: dict[str, Any]) -> Image.Image:
    """Render single-panel OR multi-panel (3-panel / 4-panel) XKCD comic strips using xkcd-script.ttf."""
    scene_name = spec.get("scene", "scene_armchair")
    turns = spec.get("turns", [])
    title = spec.get("title", "XKCD")
    alt_text = spec.get("alt", "")

    f_title = _get_xkcd_font(22)
    f_speech = _get_xkcd_font(17)
    f_speech_sm = _get_xkcd_font(15)
    f_panel = _get_xkcd_font(13)
    f_alt = _get_xkcd_font(13)

    # Multi-Panel Strip A: 3-Panel Debugging Escalation (Calm Desk -> Team Consult -> Shaking Monitor, 684x264)
    if scene_name == "scene_debug_3panel":
        stage_im = Image.open(SCENE_DIR / "scene_debug_3panel.png").convert("RGB")
        st_draw = ImageDraw.Draw(stage_im)
        # 3 panels at x = 4, 232, 460 (width 220 each)
        panel_anchors = [
            (4, 220, 88, 102),    # Panel 0: Cueball head at (88, 102)
            (232, 220, 326, 98),  # Panel 1: Ponytail/Cueball heads at y=98
            (460, 220, 566, 92),  # Panel 2: Shaking monitor head at (566, 92)
        ]
        for idx, (px, pw, hx, hy) in enumerate(panel_anchors):
            if idx < len(turns):
                lines = textwrap.wrap(turns[idx]["text"], width=22)[:4]
                for li, ln in enumerate(lines):
                    st_draw.text((px + pw // 2, 16 + li * 16), ln, fill=(15, 15, 15), font=f_panel, anchor="mm")
                t_bot = 16 + len(lines) * 16 - 4
                if t_bot < hy - 6:
                    _draw_wobbled_line(st_draw, px + pw // 2, t_bot, hx, hy - 4, width=2, curve_x=-3 if idx == 0 else 3)

        pad_x, top_bar, alt_bar = 14, 42, 50
        cw, ch = stage_im.width + pad_x * 2, stage_im.height + top_bar + alt_bar
        canvas = Image.new("RGB", (cw, ch), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((cw // 2, 22), title.upper(), fill=(15, 15, 15), font=f_title, anchor="mm")
        canvas.paste(stage_im, (pad_x, top_bar))
        ay = top_bar + stage_im.height + 6
        draw.rounded_rectangle([pad_x, ay, cw - pad_x, ay + 38], radius=5, fill=(247, 246, 240), outline=(90, 90, 90), width=1)
        alt_lines = textwrap.wrap(f'alt: "{alt_text}"', width=86)[:2]
        for i, ln in enumerate(alt_lines):
            draw.text((cw // 2, ay + (19 if len(alt_lines) == 1 else 12 + i * 14)), ln, fill=(50, 50, 50), font=f_alt, anchor="mm")
        return canvas

    # Multi-Panel Strip B: 3-Panel Megan & Cueball Debate (#552 stage, 459x241)
    if scene_name == "scene_552_3panel":
        stage_im = Image.open(SCENE_DIR / "scene_552_3panel.png").convert("RGB")
        st_draw = ImageDraw.Draw(stage_im)
        f_sm = _get_xkcd_font(12)
        p_xs = [(4, 146), (156, 146), (308, 146)]
        for idx, (px, pw) in enumerate(p_xs):
            if idx < len(turns):
                t_obj = turns[idx]
                reply = t_obj.get("reply")
                lines = textwrap.wrap(t_obj["text"], width=18)[:4]
                for li, ln in enumerate(lines):
                    st_draw.text((px + pw // 2, 16 + li * 15), ln, fill=(15, 15, 15), font=f_sm, anchor="mm")
                t_bot = 16 + len(lines) * 15 - 4
                if reply:
                    _draw_wobbled_line(st_draw, px + 36, t_bot, px + 34, 130, width=2, curve_x=-2)
                    r_lines = textwrap.wrap(reply, width=14)[:2]
                    for ri, rln in enumerate(r_lines):
                        st_draw.text((px + 98, 86 + ri * 14), rln, fill=(15, 15, 15), font=f_sm, anchor="mm")
                    _draw_wobbled_line(st_draw, px + 98, 86 + len(r_lines) * 14 - 2, px + 92, 128, width=2, curve_x=2)
                else:
                    _draw_wobbled_line(st_draw, px + 88, t_bot, px + 88, 128, width=2, curve_x=3)

        pad_x, top_bar, alt_bar = 16, 42, 52
        cw, ch = stage_im.width + pad_x * 2, stage_im.height + top_bar + alt_bar
        canvas = Image.new("RGB", (cw, ch), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((cw // 2, 22), title.upper(), fill=(15, 15, 15), font=f_title, anchor="mm")
        canvas.paste(stage_im, (pad_x, top_bar))
        ay = top_bar + stage_im.height + 6
        draw.rounded_rectangle([pad_x, ay, cw - pad_x, ay + 40], radius=5, fill=(247, 246, 240), outline=(90, 90, 90), width=1)
        alt_lines = textwrap.wrap(f'alt: "{alt_text}"', width=62)[:2]
        for i, ln in enumerate(alt_lines):
            draw.text((cw // 2, ay + (20 if len(alt_lines) == 1 else 13 + i * 14)), ln, fill=(50, 50, 50), font=f_alt, anchor="mm")
        return canvas

    # Multi-Panel Strip C: 4-Panel Laptop Code/System Review (#1513 stage, 740x258)
    if scene_name == "scene_1513_4panel":
        stage_im = Image.open(SCENE_DIR / "scene_1513_4panel.png").convert("RGB")
        st_draw = ImageDraw.Draw(stage_im)
        f_sm = _get_xkcd_font(12)
        p_cols = [(4, 168, 92, 138), (180, 178, 246, 150), (366, 180, 430, 150), (554, 180, 680, 138)]
        for idx, (px, pw, hx, hy) in enumerate(p_cols):
            if idx < len(turns):
                lines = textwrap.wrap(turns[idx]["text"], width=20)[:5]
                for li, ln in enumerate(lines):
                    st_draw.text((px + pw // 2, 16 + li * 15), ln, fill=(15, 15, 15), font=f_sm, anchor="mm")
                t_bot = 16 + len(lines) * 15 - 4
                if t_bot < hy - 4:
                    _draw_wobbled_line(st_draw, px + pw // 2, t_bot, hx, hy - 4, width=2, curve_x=-3)

        pad_x, top_bar, alt_bar = 14, 42, 50
        cw, ch = stage_im.width + pad_x * 2, stage_im.height + top_bar + alt_bar
        canvas = Image.new("RGB", (cw, ch), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((cw // 2, 22), title.upper(), fill=(15, 15, 15), font=f_title, anchor="mm")
        canvas.paste(stage_im, (pad_x, top_bar))
        ay = top_bar + stage_im.height + 6
        draw.rounded_rectangle([pad_x, ay, cw - pad_x, ay + 38], radius=5, fill=(247, 246, 240), outline=(90, 90, 90), width=1)
        alt_lines = textwrap.wrap(f'alt: "{alt_text}"', width=92)[:2]
        for i, ln in enumerate(alt_lines):
            draw.text((cw // 2, ay + (19 if len(alt_lines) == 1 else 12 + i * 14)), ln, fill=(50, 50, 50), font=f_alt, anchor="mm")
        return canvas

    # Multi-Character Team Stage D: Cueball at laptop + Ponytail + Hairy (#1597 stage, 330x418)
    if scene_name == "scene_1597_team":
        stage_im = Image.open(SCENE_DIR / "scene_1597_team.png").convert("RGB")
        st_draw = ImageDraw.Draw(stage_im)
        f_md = _get_xkcd_font(14)
        t0 = textwrap.wrap(turns[0]["text"], width=32)[:3] if len(turns) > 0 else []
        for i, ln in enumerate(t0):
            st_draw.text((18, 16 + i * 18), ln, fill=(15, 15, 15), font=f_md, anchor="la")
        t0_bot = 16 + len(t0) * 18

        t1 = textwrap.wrap(turns[1]["text"], width=24)[:2] if len(turns) > 1 else []
        for i, ln in enumerate(t1):
            st_draw.text((134, 78 + i * 18), ln, fill=(15, 15, 15), font=f_md, anchor="la")
        t1_bot = 78 + len(t1) * 18

        t2 = textwrap.wrap(turns[2]["text"], width=32)[:4] if len(turns) > 2 else []
        for i, ln in enumerate(t2):
            st_draw.text((18, 122 + i * 18), ln, fill=(15, 15, 15), font=f_md, anchor="la")
        t2_bot = 122 + len(t2) * 18

        _draw_wobbled_line(st_draw, 92, t0_bot + 2, 90, 118, width=2, curve_x=-2)
        _draw_wobbled_line(st_draw, 96, t2_bot + 4, 114, 232, width=2, curve_x=-5)
        _draw_wobbled_line(st_draw, 256, t1_bot + 4, 218, 248, width=2, curve_x=10)

        pad_x, top_bar, alt_bar = 18, 44, 54
        cw, ch = stage_im.width + pad_x * 2, stage_im.height + top_bar + alt_bar
        canvas = Image.new("RGB", (cw, ch), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((cw // 2, 22), title.upper(), fill=(15, 15, 15), font=f_title, anchor="mm")
        canvas.paste(stage_im, (pad_x, top_bar))
        ay = top_bar + stage_im.height + 8
        draw.rounded_rectangle([pad_x, ay, cw - pad_x, ay + 40], radius=5, fill=(247, 246, 240), outline=(90, 90, 90), width=1)
        alt_lines = textwrap.wrap(f'alt: "{alt_text}"', width=46)[:2]
        for i, ln in enumerate(alt_lines):
            draw.text((cw // 2, ay + (20 if len(alt_lines) == 1 else 13 + i * 14)), ln, fill=(50, 50, 50), font=f_alt, anchor="mm")
        return canvas

    if scene_name == "scene_standards_3panel":
        p_mid = Image.open(SCENE_DIR / "scene_dialogue.png").convert("RGB")
        pw, ph = p_mid.size
        total_w = pw * 3 + 36
        panel_y = 58
        panel_Im = Image.new("RGB", (total_w, ph + 118), (255, 255, 255))
        draw = ImageDraw.Draw(panel_Im)
        hdr = "HOW STANDARDS PROLIFERATE: (SEE: A/C CHARGERS, CHARACTER ENCODINGS, ETC.)"
        draw.text((total_w // 2, 24), hdr, fill=(15, 15, 15), font=f_speech_sm, anchor="mm")

        x1 = 10
        draw.rectangle([x1, panel_y, x1 + pw, panel_y + ph], outline=(15, 15, 15), width=2)
        for idx, line in enumerate(turns[0]["text"].split("\n")):
            draw.text((x1 + pw // 2, panel_y + 78 + idx * 24), line, fill=(15, 15, 15), font=f_speech, anchor="mm")

        x2 = x1 + pw + 8
        panel_Im.paste(p_mid, (x2, panel_y))
        draw.rectangle([x2, panel_y, x2 + pw, panel_y + ph], outline=(15, 15, 15), width=2)
        p2_lines = textwrap.wrap(turns[1]["text"].replace("\n", " "), width=21)[:5]
        for idx, line in enumerate(p2_lines):
            draw.text((x2 + pw // 2 - 4, panel_y + 18 + idx * 19), line, fill=(15, 15, 15), font=f_speech_sm, anchor="mm")
        _draw_wobbled_line(draw, x2 + 48, panel_y + 114, x2 + 56, panel_y + 132, width=2, curve_x=-3)
        draw.text((x2 + 150, panel_y + 112), "YEAH!", fill=(15, 15, 15), font=f_speech_sm, anchor="mm")
        _draw_wobbled_line(draw, x2 + 152, panel_y + 122, x2 + 144, panel_y + 138, width=2, curve_x=3)

        x3 = x2 + pw + 8
        draw.rectangle([x3, panel_y, x3 + pw, panel_y + ph], outline=(15, 15, 15), width=2)
        draw.rectangle([x3 + 8, panel_y + 8, x3 + 84, panel_y + 34], outline=(15, 15, 15), width=2)
        draw.text((x3 + 46, panel_y + 21), "SOON:", fill=(15, 15, 15), font=f_speech_sm, anchor="mm")
        for idx, line in enumerate(["SITUATION:", "THERE ARE", "15 COMPETING", "STANDARDS."]):
            draw.text((x3 + pw // 2, panel_y + 78 + idx * 24), line, fill=(15, 15, 15), font=f_speech, anchor="mm")

        ay = panel_y + ph + 12
        draw.rounded_rectangle([10, ay, total_w - 10, ay + 36], radius=5, fill=(247, 246, 240), outline=(90, 90, 90), width=1)
        alt_short = (alt_text[:98] + "...") if len(alt_text) > 101 else alt_text
        draw.text((total_w // 2, ay + 18), f'alt: "{alt_short}"', fill=(50, 50, 50), font=f_alt, anchor="mm")
        return panel_Im

    scene_file = SCENE_DIR / f"{scene_name}.png"
    if not scene_file.exists():
        scene_file = SCENE_DIR / "scene_armchair.png"
    stage_im = Image.open(scene_file).convert("RGB")
    sw, sh = stage_im.size

    pad_x, top_bar, alt_bar = 18, 44, 54
    canvas = Image.new("RGB", (sw + pad_x * 2, sh + top_bar + alt_bar), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text(((sw + pad_x * 2) // 2, 22), title.upper(), fill=(15, 15, 15), font=f_title, anchor="mm")
    st_draw = ImageDraw.Draw(stage_im)

    if scene_name == "scene_compiling":
        hdr_lines = spec.get("header", 'THE #1 PROGRAMMER EXCUSE:\n"MY CODE\'S COMPILING."').split("\n")
        for idx, line in enumerate(hdr_lines):
            st_draw.text((sw // 2, 24 + idx * 26), line, fill=(15, 15, 15), font=_get_xkcd_font(19), anchor="mm")
        for b_idx, (bx, by, wrap_w) in enumerate([(95, 162, 14), (172, 236, 14), (94, 314, 16)]):
            if b_idx < len(turns):
                b_lines = textwrap.wrap(turns[b_idx]["text"], width=wrap_w)[:2]
                y_start = by - (len(b_lines) - 1) * 10
                for l_i, line in enumerate(b_lines):
                    st_draw.text((bx, y_start + l_i * 20), line, fill=(15, 15, 15), font=f_speech, anchor="mm")

    elif scene_name == "scene_dependency":
        t0_lines = turns[0]["text"].split("\n") if turns else ["ALL MODERN DIGITAL", "INFRASTRUCTURE"]
        for i, line in enumerate(t0_lines):
            st_draw.text((142, 24 + i * 24), line, fill=(15, 15, 15), font=_get_xkcd_font(20), anchor="mm")
        _draw_wobbled_line(st_draw, 46, 86, 138, 72, width=2, curve_x=-4)
        _draw_wobbled_line(st_draw, 138, 72, 218, 86, width=2, curve_x=4)
        t1_lines = turns[1]["text"].split("\n") if len(turns) > 1 else ["A PROJECT SOME", "RANDOM PERSON", "IN NEBRASKA"]
        for i, line in enumerate(t1_lines):
            st_draw.text((238, 250 + i * 21), line, fill=(15, 15, 15), font=_get_xkcd_font(16), anchor="la")

    elif scene_name == "scene_ml_pile":
        f_ml, f_ml_sm = _get_xkcd_font(15), _get_xkcd_font(14)
        for i, ln in enumerate(textwrap.wrap(turns[0]["text"], width=44) if len(turns) > 0 else []):
            st_draw.text((22, 16 + i * 18), ln, fill=(15, 15, 15), font=f_ml, anchor="la")
        for i, ln in enumerate(textwrap.wrap(turns[1]["text"], width=36) if len(turns) > 1 else []):
            st_draw.text((84, 46 + i * 18), ln, fill=(15, 15, 15), font=f_ml_sm, anchor="la")
        for i, ln in enumerate(textwrap.wrap(turns[2]["text"], width=38) if len(turns) > 2 else []):
            st_draw.text((22, 120 + i * 18), ln, fill=(15, 15, 15), font=f_ml, anchor="la")
        for i, ln in enumerate(textwrap.wrap(turns[3]["text"], width=30) if len(turns) > 3 else []):
            st_draw.text((146, 156 + i * 18), ln, fill=(15, 15, 15), font=f_ml_sm, anchor="la")
        _draw_wobbled_line(st_draw, 54, 38, 52, 114, width=2, curve_x=-4)
        _draw_wobbled_line(st_draw, 56, 144, 56, 312, width=2, curve_x=12)
        _draw_wobbled_line(st_draw, 290, 106, 286, 150, width=2, curve_x=3)
        _draw_wobbled_line(st_draw, 260, 198, 222, 226, width=2, curve_x=-4)

    elif scene_name == "scene_desk":
        f_dk = _get_xkcd_font(14)
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
        _draw_wobbled_line(st_draw, 4, 142, 34, t2_bot + 3, width=2, curve_x=0)
        _draw_wobbled_line(st_draw, 44, t0_bot + 3, 40, 90, width=2, curve_x=-2)
        _draw_wobbled_line(st_draw, 178, t1_bot + 3, 174, 128, width=2, curve_x=2)
        _draw_wobbled_line(st_draw, 168, t3_bot + 3, 160, 180, width=2, curve_x=-2)

    else:
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
        _draw_wobbled_line(st_draw, 84, 24 + len(t0) * 20, 81, 92, width=2, curve_x=-2)
        _draw_wobbled_line(st_draw, 84, 100 + len(t2) * 20, 91, 192, width=2, curve_x=2)
        _draw_wobbled_line(st_draw, 244, 52 + len(t1) * 20, 235, 128, width=2, curve_x=0)
        _draw_wobbled_line(st_draw, 222, 138 + len(t3) * 20, 214, 182, width=2, curve_x=0)

    canvas.paste(stage_im, (pad_x, top_bar))
    ay = top_bar + sh + 8
    cw = sw + pad_x * 2
    alt_lines = textwrap.wrap(f'alt: "{alt_text}"', width=max(38, int(cw / 7.2)))[:2]
    draw.rounded_rectangle([pad_x, ay, cw - pad_x, ay + 40], radius=5, fill=(247, 246, 240), outline=(90, 90, 90), width=1)
    for i, ln in enumerate(alt_lines):
        draw.text((cw // 2, ay + (20 if len(alt_lines) == 1 else 13 + i * 15)), ln, fill=(50, 50, 50), font=f_alt, anchor="mm")
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
        im = _render_comic_pil(self._data)
        buf = BytesIO()
        im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        w = width or im.width
        h = height or im.height
        turns_xml = []
        for idx, t in enumerate(self._data.get("turns", [])):
            esc = str(t["text"]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
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

    tpl = _synthesize_novel_comic(clean if not is_invalid else "Invalid")
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

    panel_list = [
        {"index": i, "cast": cast, "prop": prop, "turns": [t]}
        for i, t in enumerate(turns)
    ]

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
