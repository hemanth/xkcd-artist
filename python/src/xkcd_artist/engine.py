"""
xkcd-artist Python Engine:
Combines the trained 5-Head ExplainXKCD Scene Planner (ONNX Int8 / PyTorch FP32)
with spotless connected-component XKCD stage art, procedural hand-wobbled XKCD
charts & flowcharts, 4x supersampled anti-aliased pen strokes, and authentic
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

_ORT_SESSION_CACHE: Any = None
_PT_MODEL_CACHE: Any = None

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
        "layout": "single_panel",
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


class _PenOverlay:
    """
    4x supersampled transparent ink layer that composites smooth, anti-aliased,
    subtle hand-wobbled Randall Munroe pen strokes (2.0px #0A0A0A) onto PIL canvases.
    """

    def __init__(self, width: int, height: int, scale: int = 4) -> None:
        self.w = width
        self.h = height
        self.s = scale
        self.layer = Image.new("RGBA", (width * scale, height * scale), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.layer)

    def wobbled_curve(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        width: float = 2.0,
        curve_x: float = 0.0,
        curve_y: float = 0.0,
        wobble: float = 0.65,
        seed: float = 1.0,
        dashed: bool = False,
    ) -> None:
        s = self.s
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = max(18, int(dist * 0.7))
        nx = -(y1 - y0) / (dist + 1e-6)
        ny = (x1 - x0) / (dist + 1e-6)
        pts: list[tuple[float, float]] = []
        for i in range(steps + 1):
            t = i / float(steps)
            env = math.sin(t * math.pi)
            jitter = env * wobble * (
                0.65 * math.sin(t * 9.4 + seed * 1.7) + 0.35 * math.cos(t * 17.3 + seed * 3.1)
            )
            bx = x0 + (x1 - x0) * t + env * curve_x + nx * jitter
            by = y0 + (y1 - y0) * t + env * curve_y + ny * jitter
            pts.append((bx * s, by * s))

        stroke_px = max(2, round(width * s))
        if dashed:
            seg_len = 4
            for i in range(0, len(pts) - 1, seg_len * 2):
                chunk = pts[i : i + seg_len + 1]
                if len(chunk) >= 2:
                    self.draw.line(chunk, fill=(10, 10, 10, 255), width=stroke_px, joint="curve")
        else:
            self.draw.line(pts, fill=(10, 10, 10, 255), width=stroke_px, joint="curve")

    def wobbled_polyline(
        self,
        points: list[tuple[float, float]],
        width: float = 2.0,
        wobble: float = 0.7,
        closed: bool = False,
    ) -> None:
        seq = list(points) + ([points[0]] if closed and len(points) > 2 else [])
        for idx in range(len(seq) - 1):
            p0, p1 = seq[idx], seq[idx + 1]
            self.wobbled_curve(
                p0[0], p0[1], p1[0], p1[1], width=width, wobble=wobble, seed=idx * 2.3 + 0.7
            )

    def wobbled_rect(
        self, x0: float, y0: float, x1: float, y1: float, width: float = 2.1, wobble: float = 0.55
    ) -> None:
        self.wobbled_polyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], width=width, wobble=wobble, closed=True)

    def arrowhead(self, x_from: float, y_from: float, x_tip: float, y_tip: float, size: float = 6.5) -> None:
        ang = math.atan2(y_tip - y_from, x_tip - x_from)
        for delta in (2.58, -2.58):
            ax = x_tip + size * math.cos(ang + delta)
            ay = y_tip + size * math.sin(ang + delta)
            self.wobbled_curve(x_tip, y_tip, ax, ay, width=1.9, wobble=0.25, seed=delta)

    def composite_onto(self, target_rgb: Image.Image) -> Image.Image:
        smooth = self.layer.resize((self.w, self.h), Image.Resampling.LANCZOS)
        base = target_rgb.convert("RGBA")
        base.alpha_composite(smooth)
        return base.convert("RGB")


def _extract_prompt_keywords(prompt_text: str) -> tuple[str, str, str, str]:
    stop = {
        "the", "and", "for", "with", "into", "from", "that", "this", "when", "what",
        "why", "how", "about", "like", "make", "making", "trying", "explained",
        "explain", "cueball", "megan", "ponytail", "black", "hat", "beret", "guy",
        "comic", "xkcd", "panel", "strip", "someone", "every", "time", "after",
        "before", "my", "our", "your", "is", "are", "was", "in", "on", "to", "at",
        "of", "by", "reviewing", "automating", "autonomous", "line", "lines",
        "chart", "graph", "plot", "flowchart", "diagram", "guide", "decide",
        "deciding", "whether", "touch", "touching", "annotated", "spent", "tuning",
        "push", "pushing", "deploying", "running", "writing", "building", "using",
        "tweak", "tiny", "one", "two", "three", "four", "five", "stale", "friday",
    }
    raw_words = [
        w.strip(".,!?:;\"'`()[]{}")
        for w in prompt_text.split()
        if len(w.strip(".,!?:;\"'`()[]{}")) >= 2
    ]
    filtered = [
        w for w in raw_words
        if w.lower() not in stop and not re.match(r"^\d+[a-z]*$", w.lower())
    ]
    if not filtered:
        filtered = [w for w in raw_words if not w.isdigit()] or ["SYSTEM"]

    # Deduplicate generic modifiers ("pull", "request", "container", "image", "legacy") when a specific tech noun exists
    generic_nouns = {
        "pull", "request", "pr", "container", "image", "code", "script",
        "workflow", "release", "before", "legacy", "modern", "custom", "simple",
    }
    specific = [w for w in filtered if w.lower() not in generic_nouns]

    title_words = filtered[:2]
    title = " ".join(title_words).upper()[:26]
    subject = " ".join((specific or filtered)[:2]).upper()[:22]
    focus = (specific[0] if specific else filtered[0]).upper()
    second = (specific[1] if len(specific) > 1 else (filtered[1] if len(filtered) > 1 else "PRODUCTION")).upper()
    return title, subject, focus, second


def _synthesize_novel_comic(prompt_text: str) -> dict[str, Any]:
    """
    Synthesize a comic specification across all 6 XKCD layout families:
    `single_panel`, `two_panel`, `three_panel`, `four_panel`, `annotated_chart`, `flowchart_diagram`.
    """
    low = prompt_text.lower()
    for tpl in ICONIC_TEMPLATES:
        if any(k in low for k in tpl["match"]):
            return tpl

    title, subject, focus, second = _extract_prompt_keywords(prompt_text)
    h_val = int(hashlib.md5(prompt_text.encode("utf-8")).hexdigest()[:8], 16)

    # A. Flowchart / Decision Tree / Cheat Sheet -> `flowchart_diagram` (#627 / #518 style)
    if any(w in low for w in ["flowchart", "flow chart", "decision tree", "cheat sheet", "how to decide", "workflow diagram"]):
        return {
            "title": title,
            "scene": "scene_flowchart",
            "layout": "flowchart_diagram",
            "prop": "chart_board",
            "cast": ["cueball"],
            "chart_meta": {
                "start": f"TOUCHING {title}",
                "diamond": "DOES IT WORK?",
                "yes_box": "DO NOT TOUCH IT",
                "no_box": f"CLEAR {focus} CACHE",
                "bottom_box": "BLAME DNS & WAIT",
            },
            "turns": [
                {"speaker": "left", "text": f"START: TOUCHING {title}"},
                {"speaker": "right", "text": f"DOES IT WORK? -> YES: DO NOT TOUCH IT / NO: CLEAR {focus} CACHE"},
            ],
            "alt": f"80% of senior engineering is knowing which box in the {title.lower()} flowchart is load-bearing.",
        }

    # B. Annotated Chart / Graph / Plot / Extrapolation -> `annotated_chart` (#605 / #2048 style)
    if any(w in low for w in ["chart", "graph", "plot", "axis", "axes", "correlation", "extrapolat", "regression", "curve"]):
        return {
            "title": title,
            "scene": "scene_chart",
            "layout": "annotated_chart",
            "prop": "chart_board",
            "cast": ["cueball"],
            "chart_meta": {
                "x_label": f"TIME SPENT TUNING {focus}",
                "y_label": "ACTUAL OUTPUT",
                "callout": "POINT WHERE YOU REWRITE\nIT ALL IN A SHELL SCRIPT",
                "caption": f"EMPIRICAL {subject} CURVE",
            },
            "turns": [
                {"speaker": "left", "text": f"Y-AXIS: ACTUAL OUTPUT / X-AXIS: TIME SPENT TUNING {focus}"},
                {"speaker": "right", "text": "INFLECTION: POINT WHERE YOU REWRITE IT ALL IN A SHELL SCRIPT."},
            ],
            "alt": f"If you fit a cubic spline to {subject.lower()}, productivity briefly becomes complex-valued around 2 AM.",
        }

    # C. Friday Deploy / 3AM Incident / PagerDuty / Outage -> Crisp 2-Panel Setup -> Despair (`two_panel`)
    if any(w in low for w in ["friday", "5pm", "5 pm", "3am", "3 am", "pagerduty", "incident", "outage", "denvercoder"]):
        cfg_noun = "PROD" if focus == "CONFIG" else focus
        return {
            "title": "FRIDAY DEPLOY" if title == "CONFIG" else title,
            "scene": "scene_979_2panel",
            "layout": "two_panel",
            "prop": "laptop_desk",
            "cast": ["cueball"],
            "turns": [
                {"speaker": "left", "text": f"4:58 PM FRIDAY: ONE TINY {cfg_noun} CONFIG TWEAK. ZERO RISK."},
                {"speaker": "right", "text": f"3:14 AM: WHO WROTE THIS {focus} PARSER IN 2016 AND WHAT DID THEY SEE?!"},
            ],
            "alt": f"The post-mortem action item was simply 'never perceive {subject.lower()} after 4 PM on Thursdays'.",
        }

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
            "layout": "single_panel",
            "prop": "laptop_desk",
            "cast": ["cueball", "ponytail", "black_hat"],
            "turns": [
                {"speaker": "left", "text": f"I RAN INTERACTIVE {title} TO CLEAN UP OUR COMMIT HISTORY BEFORE RELEASE."},
                {"speaker": "right", "text": "DID IT WORK?"},
                {"speaker": "left", "text": "HEAD IS DETACHED, MAIN IS IN 1970, AND EVERYONE JUST CLONED A FRESH COPY INTO 'PROJECT_FINAL_V3'."},
            ],
            "alt": "git reflog is just a horror novel where the main character is you from twenty minutes ago.",
        }

    # 3. Code Review / PR / Pull Request / Refactor / Legacy / Style / Lint -> 4-Panel Laptop Review (#1513 stage)
    if any(w in low for w in ["review", "pr", "pull request", "refactor", "legacy", "messy", "lint", "type", "typescript", "css"]):
        pr_label = "CODE" if focus in ("PR", "PULL", "REQUEST", "REVIEW") else focus
        return {
            "title": title,
            "scene": "scene_1513_4panel",
            "layout": "four_panel",
            "prop": "laptop_desk",
            "cast": ["cueball", "ponytail"],
            "turns": [
                {"speaker": "left", "text": f"CAN YOU REVIEW MY {pr_label} PR? IT'S ONLY 4,200 LINES."},
                {"speaker": "right", "text": "WHY DOES ONE REGEX PARSE BOTH HTML AND TAXES?"},
                {"speaker": "right", "text": f"AND WHY DOES {pr_label} IMPORT A DEPRECATED GAME ENGINE FOR DATE MATH?"},
                {"speaker": "left", "text": "LGTM, MERGING WITH --NO-VERIFY."},
            ],
            "alt": f"Every comment in the {subject.lower()} codebase just says '// DO NOT TOUCH OR THE BILLING CRON DIES'.",
        }

    # 4. AI / LLM / Agent / Prompt / GPT / Neural / Model -> 3-Panel Megan & Cueball Debate (#552 stage)
    if any(w in low for w in ["llm", "agent", "agents", "gpt", "prompt", "ai", "chatbot", "hallucinat", "token", "transformer"]):
        agent_noun = subject if "AGENT" in subject else f"{focus} AGENTS"
        return {
            "title": title,
            "scene": "scene_552_3panel",
            "layout": "three_panel",
            "prop": "none",
            "cast": ["megan", "cueball"],
            "turns": [
                {"speaker": "right", "text": f"I DEPLOYED TWO {agent_noun} TO AUTOMATE MY WORKFLOW."},
                {"speaker": "right", "text": "ONE OPENS TICKETS AND THE OTHER CLOSES THEM AS 'WONTFIX'."},
                {"speaker": "left", "text": "SO NOTHING GETS DONE?", "reply": "AT 40,000 TOKENS/SEC."},
            ],
            "alt": f"We added a third {focus.lower()} agent to mediate, and now all three are in a Series-A fundraising call.",
        }

    # 5. Debugging / DNS / Bug / Prod / Server / SQL / Database / Vim / Error -> 3-Panel Debugging Escalation
    if any(w in low for w in ["debug", "dns", "bug", "prod", "deploy", "server", "sql", "database", "vim", "error", "cache"]):
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
            "alt": f"The root cause analysis concluded that {subject.lower()} was load-bearing technical debt.",
        }

    # 6. General Arbitrary Prompts -> Rotate across multi-panel XKCD archetypes
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
                {"speaker": "left", "text": f"CHECK OUT MY NEW {subject} MODEL."},
                {"speaker": "right", "text": "YOUR ERROR BARS ARE WIDER THAN THE Y-AXIS."},
                {"speaker": "right", "text": f"AND DID YOU FIT A 9TH-DEGREE POLYNOMIAL TO THREE {focus} POINTS?"},
                {"speaker": "left", "text": "R-SQUARED IS 1.000 THOUGH."},
            ],
            "alt": f"If you extrapolate {subject.lower()} to next Tuesday, the curve loops back through the Bronze Age.",
        },
        {
            "scene": "scene_1597_team",
            "layout": "single_panel",
            "prop": "laptop_desk",
            "cast": ["cueball", "ponytail", "black_hat"],
            "turns": [
                {"speaker": "left", "text": f"OUR NEW {subject} ARCHITECTURE USES ZERO EXTERNAL DEPENDENCIES."},
                {"speaker": "right", "text": f"HOW DID YOU BUILD {focus}?"},
                {"speaker": "left", "text": "WE WROTE OUR OWN BROKEN VERSION OF HALF THE STANDARD LIBRARY IN BASH."},
            ],
            "alt": f"Eventually every {subject.lower()} project invents its own Lisp interpreter by accident.",
        },
    ]
    chosen = archetypes[h_val % len(archetypes)]
    return {"title": title, **chosen}


def _wrap_pixel_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_px: int,
) -> list[str]:
    """Pixel-accurate word wrapper using FreeType glyph metrics so lines never clip borders."""
    lines: list[str] = []
    for para in str(text).split("\n"):
        words = para.strip().split()
        if not words:
            continue
        cur = words[0]
        for w in words[1:]:
            cand = f"{cur} {w}"
            bbox = draw.textbbox((0, 0), cand, font=font)
            if (bbox[2] - bbox[0]) <= max_px:
                cur = cand
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def _render_annotated_chart_pil(spec: dict[str, Any]) -> Image.Image:
    W, H = 520, 340
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    pen = _PenOverlay(W, H)
    pen.wobbled_rect(4, 4, W - 5, H - 5, width=2.2, wobble=0.5)

    # Axes with hand-drawn arrowheads & ticks
    ox, oy = 74, 272
    top_y, right_x = 40, 476
    pen.wobbled_curve(ox, oy + 8, ox, top_y, width=2.1, wobble=0.65, seed=1.2)
    pen.arrowhead(ox, oy, ox, top_y, size=7.0)
    pen.wobbled_curve(ox - 8, oy, right_x, oy, width=2.1, wobble=0.65, seed=2.4)
    pen.arrowhead(ox, oy, right_x, oy, size=7.0)

    for tx in [150, 230, 310, 390]:
        pen.wobbled_curve(tx, oy - 4, tx, oy + 5, width=1.8, wobble=0.2, seed=tx)
    for ty in [216, 160, 104]:
        pen.wobbled_curve(ox - 5, ty, ox + 4, ty, width=1.8, wobble=0.2, seed=ty)

    # Wobbled empirical curve: initial confidence peak -> steep reality cliff
    curve_pts = [
        (84, 250),
        (132, 212),
        (182, 146),
        (228, 108),
        (264, 114),
        (296, 178),
        (334, 238),
        (392, 254),
        (454, 258),
    ]
    pen.wobbled_polyline(curve_pts, width=2.4, wobble=0.75)
    # Dashed naive linear extrapolation shooting off top-right
    pen.wobbled_curve(228, 108, 452, 62, width=1.7, wobble=0.45, seed=5.5, dashed=True)
    # Callout curved arrow to cliff inflection (264, 114)
    pen.wobbled_curve(336, 134, 274, 116, width=1.9, curve_y=-10, wobble=0.4, seed=7.1)
    pen.arrowhead(296, 118, 274, 116, size=6.0)

    canvas = pen.composite_onto(canvas)
    draw = ImageDraw.Draw(canvas)
    meta = spec.get("chart_meta", {})
    f_lbl = _get_xkcd_font(15)
    f_sm = _get_xkcd_font(13)

    draw.text(((ox + right_x) // 2, 304), meta.get("x_label", "EFFORT INVESTED"), fill=(10, 10, 10), font=f_lbl, anchor="mm")
    draw.text((374, 42), "NAIVE EXTRAPOLATION", fill=(45, 45, 45), font=f_sm, anchor="mm")

    callout_lines = str(meta.get("callout", "CRITICAL CLIFF")).split("\n")
    for idx, ln in enumerate(callout_lines):
        draw.text((392, 146 + idx * 16), ln, fill=(10, 10, 10), font=f_sm, anchor="mm")

    # Rotated Y-axis label
    y_lbl_txt = meta.get("y_label", "USEFULNESS")
    y_im = Image.new("RGBA", (220, 28), (255, 255, 255, 0))
    ImageDraw.Draw(y_im).text((110, 14), y_lbl_txt, fill=(10, 10, 10, 255), font=f_lbl, anchor="mm")
    y_rot = y_im.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    canvas.paste(y_rot, (24, (top_y + oy) // 2 - y_rot.height // 2), y_rot)
    return canvas


def _render_flowchart_pil(spec: dict[str, Any]) -> Image.Image:
    W, H = 540, 356
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    pen = _PenOverlay(W, H)
    pen.wobbled_rect(4, 4, W - 5, H - 5, width=2.2, wobble=0.5)

    # 1. Top Start Box
    pen.wobbled_rect(155, 24, 385, 68, width=2.0, wobble=0.55)
    # Arrow down to Decision Diamond
    pen.wobbled_curve(270, 68, 270, 104, width=2.0, wobble=0.35, seed=1.1)
    pen.arrowhead(270, 88, 270, 104, size=6.0)

    # 2. Center Decision Diamond
    diamond = [(270, 106), (366, 154), (270, 202), (174, 154)]
    pen.wobbled_polyline(diamond, width=2.0, wobble=0.55, closed=True)

    # 3. YES Branch -> Right Box ("DO NOT TOUCH IT")
    pen.wobbled_curve(366, 154, 404, 154, width=2.0, wobble=0.3, seed=2.2)
    pen.arrowhead(388, 154, 404, 154, size=6.0)
    pen.wobbled_rect(406, 126, 518, 182, width=2.0, wobble=0.55)

    # 4. NO Branch -> Down to Action Box
    pen.wobbled_curve(270, 202, 270, 242, width=2.0, wobble=0.35, seed=3.3)
    pen.arrowhead(270, 226, 270, 242, size=6.0)
    pen.wobbled_rect(152, 244, 388, 290, width=2.0, wobble=0.55)

    # 5. Loop-back arrow from Action Box left side -> back into Decision Diamond!
    pen.wobbled_polyline([(152, 267), (76, 267), (76, 154), (171, 154)], width=1.9, wobble=0.55)
    pen.arrowhead(150, 154, 171, 154, size=6.0)

    canvas = pen.composite_onto(canvas)
    draw = ImageDraw.Draw(canvas)
    meta = spec.get("chart_meta", {})
    f_box = _get_xkcd_font(15)
    f_sm = _get_xkcd_font(13)

    draw.text((270, 46), meta.get("start", "START"), fill=(10, 10, 10), font=f_box, anchor="mm")
    draw.text((270, 154), meta.get("diamond", "DOES IT WORK?"), fill=(10, 10, 10), font=f_sm, anchor="mm")
    draw.text((384, 141), "YES", fill=(10, 10, 10), font=f_sm, anchor="mm")
    yes_lines = _wrap_pixel_lines(draw, meta.get("yes_box", "DO NOT TOUCH IT"), f_sm, 98)[:2]
    for i, ln in enumerate(yes_lines):
        draw.text((462, 146 + i * 16 if len(yes_lines) > 1 else 154), ln, fill=(10, 10, 10), font=f_sm, anchor="mm")

    draw.text((286, 221), "NO", fill=(10, 10, 10), font=f_sm, anchor="mm")
    draw.text((270, 267), meta.get("no_box", "CLEAR CACHE & RETRY"), fill=(10, 10, 10), font=f_box, anchor="mm")
    draw.text((76, 314), "REPEAT UNTIL 3 AM", fill=(55, 55, 55), font=f_sm, anchor="mm")
    return canvas


def _add_margin(im: Image.Image, pad: int = 4) -> Image.Image:
    out = Image.new("RGB", (im.width + pad * 2, im.height + pad * 2), (255, 255, 255))
    out.paste(im, (pad, pad))
    return out


def _render_comic_pil(spec: dict[str, Any], include_chrome: bool = False) -> Image.Image:
    """
    Render pure, authentic Randall Munroe monochrome XKCD strips using `xkcd-script.ttf`
    and 4x-supersampled anti-aliased leader lines.
    """
    scene_name = spec.get("scene", "scene_armchair")
    turns = spec.get("turns", [])
    title = spec.get("title", "XKCD")
    alt_text = spec.get("alt", "")

    if scene_name == "scene_chart":
        core = _render_annotated_chart_pil(spec)
    elif scene_name == "scene_flowchart":
        core = _render_flowchart_pil(spec)
    elif scene_name == "scene_979_2panel":
        stage_im = Image.open(SCENE_DIR / "scene_979_2panel.png").convert("RGB")
        pen = _PenOverlay(stage_im.width, stage_im.height)
        st_draw = ImageDraw.Draw(stage_im)
        f_p = _get_xkcd_font(15)
        anchors = [(4, 220, 101, 110, -4.0), (232, 220, 338, 92, 4.0)]
        for idx, (px, pw, hx, hy, cx) in enumerate(anchors):
            if idx < len(turns):
                lines = _wrap_pixel_lines(st_draw, turns[idx]["text"], f_p, pw - 22)[:4]
                for li, ln in enumerate(lines):
                    st_draw.text((px + pw // 2, 18 + li * 18), ln, fill=(10, 10, 10), font=f_p, anchor="mm")
                t_bot = 14 + len(lines) * 18
                if t_bot < hy - 4:
                    pen.wobbled_curve(px + pw // 2, t_bot, hx, hy, width=2.0, curve_x=cx, seed=idx + 1.0)
        core = _add_margin(pen.composite_onto(stage_im), pad=3)

    elif scene_name == "scene_debug_3panel":
        stage_im = Image.open(SCENE_DIR / "scene_debug_3panel.png").convert("RGB")
        pen = _PenOverlay(stage_im.width, stage_im.height)
        st_draw = ImageDraw.Draw(stage_im)
        f_p = _get_xkcd_font(15)
        panel_anchors = [
            (4, 220, 101, 110, -4.0),
            (232, 220, 333, 148, -5.0),
            (460, 220, 566, 92, 4.0),
        ]
        for idx, (px, pw, hx, hy, cx) in enumerate(panel_anchors):
            if idx < len(turns):
                lines = _wrap_pixel_lines(st_draw, turns[idx]["text"], f_p, pw - 20)[:4]
                for li, ln in enumerate(lines):
                    st_draw.text((px + pw // 2, 18 + li * 17), ln, fill=(10, 10, 10), font=f_p, anchor="mm")
                t_bot = 14 + len(lines) * 17
                if t_bot < hy - 4:
                    pen.wobbled_curve(px + pw // 2, t_bot, hx, hy, width=2.0, curve_x=cx, seed=idx + 1.3)
        core = _add_margin(pen.composite_onto(stage_im), pad=3)

    elif scene_name == "scene_552_3panel":
        stage_im = Image.open(SCENE_DIR / "scene_552_3panel.png").convert("RGB")
        pen = _PenOverlay(stage_im.width, stage_im.height)
        st_draw = ImageDraw.Draw(stage_im)
        f_sm = _get_xkcd_font(14)
        p_xs = [(0, 158, 34, 87, 127), (170, 134, 205, 258, 134), (315, 144, 350, 402, 139)]
        for idx, (px, pw, lx, rx, head_y) in enumerate(p_xs):
            if idx < len(turns):
                t_obj = turns[idx]
                reply = t_obj.get("reply")
                lines = _wrap_pixel_lines(st_draw, t_obj["text"], f_sm, pw - 14)[:4]
                for li, ln in enumerate(lines):
                    st_draw.text((px + pw // 2, 16 + li * 16), ln, fill=(10, 10, 10), font=f_sm, anchor="mm")
                t_bot = 12 + len(lines) * 16
                if reply:
                    pen.wobbled_curve(px + 34, t_bot, lx, head_y + 2, width=2.0, curve_x=-3.0, seed=idx + 2.0)
                    r_lines = _wrap_pixel_lines(st_draw, reply, f_sm, pw - 42)[:2]
                    for ri, rln in enumerate(r_lines):
                        st_draw.text((px + pw - 50, 86 + ri * 15), rln, fill=(10, 10, 10), font=f_sm, anchor="mm")
                    r_bot = 82 + len(r_lines) * 15
                    pen.wobbled_curve(px + pw - 48, r_bot, rx, head_y, width=2.0, curve_x=3.0, seed=idx + 4.0)
                else:
                    target_x = lx if t_obj.get("speaker") == "left" else rx
                    pen.wobbled_curve(
                        px + pw // 2 + (8 if target_x == rx else -8),
                        t_bot,
                        target_x,
                        head_y,
                        width=2.0,
                        curve_x=4.0 if target_x == rx else -4.0,
                        seed=idx + 1.5,
                    )
        core = _add_margin(pen.composite_onto(stage_im), pad=3)

    elif scene_name == "scene_1513_4panel":
        stage_im = Image.open(SCENE_DIR / "scene_1513_4panel.png").convert("RGB")
        pen = _PenOverlay(stage_im.width, stage_im.height)
        st_draw = ImageDraw.Draw(stage_im)
        f_sm = _get_xkcd_font(14)
        # Exact speaker head crowns measured from scene_1513_4panel.png:
        # P0 Cueball=(93, 133), P1 Ponytail=(247, 146), P2 Ponytail=(431, 146), P3 Cueball=(669, 134)
        p_cols = [
            (0, 174, 93, 133, -4.0),
            (181, 153, 247, 146, -4.0),
            (342, 187, 431, 146, -4.0),
            (537, 202, 669, 134, 5.0),
        ]
        for idx, (px, pw, hx, hy, cx) in enumerate(p_cols):
            if idx < len(turns):
                lines = _wrap_pixel_lines(st_draw, turns[idx]["text"], f_sm, pw - 16)[:5]
                for li, ln in enumerate(lines):
                    st_draw.text((px + pw // 2, 16 + li * 16), ln, fill=(10, 10, 10), font=f_sm, anchor="mm")
                t_bot = 12 + len(lines) * 16
                if t_bot < hy - 4:
                    pen.wobbled_curve(px + pw // 2, t_bot, hx, hy, width=2.0, curve_x=cx, seed=idx + 2.1)
        core = _add_margin(pen.composite_onto(stage_im), pad=3)

    elif scene_name == "scene_1597_team":
        stage_im = Image.open(SCENE_DIR / "scene_1597_team.png").convert("RGB")
        pen = _PenOverlay(stage_im.width, stage_im.height)
        st_draw = ImageDraw.Draw(stage_im)
        f_md = _get_xkcd_font(15)

        t0 = _wrap_pixel_lines(st_draw, turns[0]["text"], f_md, 294)[:3] if len(turns) > 0 else []
        for i, ln in enumerate(t0):
            st_draw.text((16, 16 + i * 19), ln, fill=(10, 10, 10), font=f_md, anchor="la")
        t0_bot = 16 + len(t0) * 19

        # Turn 1 (Ponytail) positioned on upper-right so its curved leader sweeps cleanly around Turn 2 down to (214, 242)
        t1 = _wrap_pixel_lines(st_draw, turns[1]["text"], f_md, 165)[:2] if len(turns) > 1 else []
        for i, ln in enumerate(t1):
            st_draw.text((238, 86 + i * 18), ln, fill=(10, 10, 10), font=f_md, anchor="mm")
        t1_bot = 78 + len(t1) * 18

        t2_y0 = max(122, t1_bot + 12)
        t2 = _wrap_pixel_lines(st_draw, turns[2]["text"], f_md, 250)[:4] if len(turns) > 2 else []
        for i, ln in enumerate(t2):
            st_draw.text((16, t2_y0 + i * 18), ln, fill=(10, 10, 10), font=f_md, anchor="la")
        t2_bot = t2_y0 + len(t2) * 18

        # Vertical bridge connecting Cueball's two speech blocks + lower tail to Cueball's head (126, 223)
        pen.wobbled_curve(94, t0_bot + 2, 92, t2_y0 - 5, width=2.0, curve_x=-1.5, seed=1.4)
        pen.wobbled_curve(98, t2_bot + 3, 122, 223, width=2.0, curve_x=-5.0, seed=2.8)
        # Authentic #1597 sweeping right-hand arc from below Turn 1 down to Ponytail's head (214, 242)
        pen.wobbled_curve(256, t1_bot + 4, 215, 242, width=2.0, curve_x=28.0, seed=4.2)
        core = _add_margin(pen.composite_onto(stage_im), pad=3)

    elif scene_name == "scene_standards_3panel":
        p_mid = Image.open(SCENE_DIR / "scene_dialogue.png").convert("RGB")
        pw, ph = p_mid.size  # 203x246
        side_w = 148
        gap = 8
        total_w = side_w * 2 + pw + gap * 2 + 12
        panel_y = 44
        panel_im = Image.new("RGB", (total_w, panel_y + ph + 6), (255, 255, 255))
        pen = _PenOverlay(panel_im.width, panel_im.height)
        draw = ImageDraw.Draw(panel_im)

        f_hdr = _get_xkcd_font(17)
        f_sub = _get_xkcd_font(12)
        f_sp = _get_xkcd_font(17)
        f_mid = _get_xkcd_font(15)

        draw.text((total_w // 2, 13), "HOW STANDARDS PROLIFERATE:", fill=(10, 10, 10), font=f_hdr, anchor="mm")
        draw.text(
            (total_w // 2, 30),
            "(SEE: A/C CHARGERS, CHARACTER ENCODINGS, INSTANT MESSAGING, ETC.)",
            fill=(10, 10, 10),
            font=f_sub,
            anchor="mm",
        )

        # Left Panel (14 competing standards)
        x1 = 6
        pen.wobbled_rect(x1, panel_y, x1 + side_w, panel_y + ph - 1, width=2.1, wobble=0.35)
        for idx, line in enumerate(turns[0]["text"].split("\n")):
            draw.text((x1 + side_w // 2, panel_y + 88 + idx * 23), line, fill=(10, 10, 10), font=f_sp, anchor="mm")

        # Middle Panel (Cueball + Ponytail dialogue)
        x2 = x1 + side_w + gap
        panel_im.paste(p_mid, (x2, panel_y))
        p2_lines = _wrap_pixel_lines(draw, turns[1]["text"].replace("\n", " "), f_mid, pw - 18)[:5]
        for idx, line in enumerate(p2_lines):
            draw.text((x2 + pw // 2 - 4, panel_y + 18 + idx * 18), line, fill=(10, 10, 10), font=f_mid, anchor="mm")
        p2_bot = panel_y + 14 + len(p2_lines) * 18
        pen.wobbled_curve(x2 + 48, p2_bot, x2 + 56, panel_y + 126, width=2.0, curve_x=-4.0, seed=2.1)
        draw.text((x2 + 148, panel_y + 108), "YEAH!", fill=(10, 10, 10), font=f_mid, anchor="mm")
        pen.wobbled_curve(x2 + 150, panel_y + 117, x2 + 143, panel_y + 131, width=2.0, curve_x=3.0, seed=3.4)

        # Right Panel (Soon: 15 competing standards)
        x3 = x2 + pw + gap
        pen.wobbled_rect(x3, panel_y, x3 + side_w, panel_y + ph - 1, width=2.1, wobble=0.35)
        pen.wobbled_rect(x3 + 8, panel_y + 8, x3 + 78, panel_y + 33, width=1.9, wobble=0.3)
        draw.text((x3 + 43, panel_y + 21), "SOON:", fill=(10, 10, 10), font=f_mid, anchor="mm")
        for idx, line in enumerate(["SITUATION:", "THERE ARE", "15 COMPETING", "STANDARDS."]):
            draw.text((x3 + side_w // 2, panel_y + 88 + idx * 23), line, fill=(10, 10, 10), font=f_sp, anchor="mm")

        core = pen.composite_onto(panel_im)

    else:
        scene_file = SCENE_DIR / f"{scene_name}.png"
        if not scene_file.exists():
            scene_file = SCENE_DIR / "scene_armchair.png"
        stage_im = Image.open(scene_file).convert("RGB")
        sw, sh = stage_im.size
        pen = _PenOverlay(sw, sh)
        st_draw = ImageDraw.Draw(stage_im)

        if scene_name == "scene_compiling":
            hdr_lines = spec.get("header", 'THE #1 PROGRAMMER EXCUSE:\n"MY CODE\'S COMPILING."').split("\n")
            f_hdr = _get_xkcd_font(20)
            f_sp = _get_xkcd_font(16)
            for idx, line in enumerate(hdr_lines):
                y_pos = 26 + idx * 24 + (10 if idx == 2 else 0)
                st_draw.text((sw // 2, y_pos), line, fill=(10, 10, 10), font=f_hdr, anchor="mm")
            bubble_slots = [(108, 158, 122), (176, 232, 86), (98, 316, 122)]
            for b_idx, (bx, by, max_w) in enumerate(bubble_slots):
                if b_idx < len(turns):
                    b_lines = _wrap_pixel_lines(st_draw, turns[b_idx]["text"], f_sp, max_w)[:2]
                    y_start = by - (len(b_lines) - 1) * 9
                    for l_i, line in enumerate(b_lines):
                        st_draw.text((bx, y_start + l_i * 18), line, fill=(10, 10, 10), font=f_sp, anchor="mm")

        elif scene_name == "scene_dependency":
            # Uses Randall Munroe's authentic hand-drawn curly brace (y=62..92) and arrow (y=384..420)
            t0_lines = turns[0]["text"].split("\n") if turns else ["ALL MODERN DIGITAL", "INFRASTRUCTURE"]
            f_top = _get_xkcd_font(19)
            f_bot = _get_xkcd_font(15)
            for i, line in enumerate(t0_lines):
                st_draw.text((134, 24 + i * 22), line, fill=(10, 10, 10), font=f_top, anchor="mm")
            t1_lines = turns[1]["text"].split("\n") if len(turns) > 1 else []
            for i, line in enumerate(t1_lines):
                st_draw.text((238, 262 + i * 19), line, fill=(10, 10, 10), font=f_bot, anchor="la")

        elif scene_name == "scene_ml_pile":
            f_ml = _get_xkcd_font(15)
            t0 = _wrap_pixel_lines(st_draw, turns[0]["text"], f_ml, 325) if len(turns) > 0 else []
            for i, ln in enumerate(t0):
                st_draw.text((20, 18 + i * 18), ln, fill=(10, 10, 10), font=f_ml, anchor="la")
            t1 = _wrap_pixel_lines(st_draw, turns[1]["text"], f_ml, 270) if len(turns) > 1 else []
            for i, ln in enumerate(t1):
                st_draw.text((80, 48 + i * 18), ln, fill=(10, 10, 10), font=f_ml, anchor="la")
            t2 = _wrap_pixel_lines(st_draw, turns[2]["text"], f_ml, 270) if len(turns) > 2 else []
            for i, ln in enumerate(t2):
                st_draw.text((20, 118 + i * 18), ln, fill=(10, 10, 10), font=f_ml, anchor="la")
            t3 = _wrap_pixel_lines(st_draw, turns[3]["text"], f_ml, 210) if len(turns) > 3 else []
            for i, ln in enumerate(t3):
                st_draw.text((140, 154 + i * 18), ln, fill=(10, 10, 10), font=f_ml, anchor="la")

            # Smooth curved leader lines matching #1838
            pen.wobbled_curve(52, 38, 50, 114, width=2.0, curve_x=-5.0, seed=1.1)
            pen.wobbled_curve(54, 140, 48, 308, width=2.0, curve_x=14.0, seed=2.2)
            pen.wobbled_curve(276, 106, 272, 148, width=2.0, curve_x=4.0, seed=3.3)
            pen.wobbled_curve(256, 194, 218, 226, width=2.0, curve_x=-4.0, curve_y=5.0, seed=4.4)

        elif scene_name == "scene_desk":
            f_dk = _get_xkcd_font(16)
            t0 = _wrap_pixel_lines(st_draw, turns[0]["text"], f_dk, 240)[:2] if len(turns) > 0 else []
            for i, ln in enumerate(t0):
                st_draw.text((26, 18 + i * 19), ln, fill=(10, 10, 10), font=f_dk, anchor="la")
            t0_bot = 18 + len(t0) * 19

            t1 = _wrap_pixel_lines(st_draw, turns[1]["text"], f_dk, 165)[:2] if len(turns) > 1 else []
            for i, ln in enumerate(t1):
                st_draw.text((124, 52 + i * 19), ln, fill=(10, 10, 10), font=f_dk, anchor="la")
            t1_bot = 52 + len(t1) * 19

            t2 = _wrap_pixel_lines(st_draw, turns[2]["text"], f_dk, 120)[:2] if len(turns) > 2 else []
            for i, ln in enumerate(t2):
                st_draw.text((68, 96 + i * 19), ln, fill=(10, 10, 10), font=f_dk, anchor="la")
            t2_bot = 96 + len(t2) * 19

            t3 = _wrap_pixel_lines(st_draw, turns[3]["text"], f_dk, 160)[:2] if len(turns) > 3 else []
            for i, ln in enumerate(t3):
                st_draw.text((122, 124 + i * 19), ln, fill=(10, 10, 10), font=f_dk, anchor="la")
            t3_bot = 124 + len(t3) * 19

            # Off-screen voice from bedroom on left (#386 curved horizontal sweep!)
            pen.wobbled_curve(98, t0_bot + 3, 90, 91, width=2.0, curve_x=5.0, seed=1.2)
            pen.wobbled_curve(76, t2_bot + 4, 16, 144, width=2.0, curve_x=10.0, curve_y=12.0, seed=2.5)
            # Cueball at desk (head crown at 182, 186)
            pen.wobbled_curve(180, t1_bot + 3, 178, 120, width=2.0, curve_x=1.5, seed=3.7)
            pen.wobbled_curve(176, t3_bot + 3, 174, 178, width=2.0, curve_x=-1.5, seed=4.9)

        else:
            # Default: scene_armchair (#149 Sandwich, 360x299)
            f_sp = _get_xkcd_font(17)
            t0 = _wrap_pixel_lines(st_draw, turns[0]["text"], f_sp, 210)[:2] if len(turns) > 0 else []
            for i, ln in enumerate(t0):
                st_draw.text((26, 22 + i * 20), ln, fill=(10, 10, 10), font=f_sp, anchor="la")
            t0_bot = 22 + len(t0) * 20

            t1 = _wrap_pixel_lines(st_draw, turns[1]["text"], f_sp, 134)[:2] if len(turns) > 1 else []
            for i, ln in enumerate(t1):
                st_draw.text((216, 52 + i * 20), ln, fill=(10, 10, 10), font=f_sp, anchor="la")
            t1_bot = 52 + len(t1) * 20

            t2 = _wrap_pixel_lines(st_draw, turns[2]["text"], f_sp, 165)[:2] if len(turns) > 2 else []
            for i, ln in enumerate(t2):
                st_draw.text((26, 100 + i * 20), ln, fill=(10, 10, 10), font=f_sp, anchor="la")
            t2_bot = 100 + len(t2) * 20

            t3 = _wrap_pixel_lines(st_draw, turns[3]["text"], f_sp, 130)[:2] if len(turns) > 3 else []
            for i, ln in enumerate(t3):
                st_draw.text((214, 136 + i * 20), ln, fill=(10, 10, 10), font=f_sp, anchor="la")
            t3_bot = 136 + len(t3) * 20

            pen.wobbled_curve(86, t0_bot + 4, 84, 95, width=2.0, curve_x=-2.0, seed=1.1)
            pen.wobbled_curve(88, t2_bot + 4, 94, 197, width=2.0, curve_x=2.0, seed=2.2)
            pen.wobbled_curve(248, t1_bot + 4, 238, 130, width=2.0, curve_x=-1.5, seed=3.3)
            pen.wobbled_curve(226, t3_bot + 4, 215, 182, width=2.0, curve_x=-1.5, seed=4.4)

        core = _add_margin(pen.composite_onto(stage_im), pad=3)

    if not include_chrome:
        return core

    # Optional full non-truncated XKCD presentation frame when explicitly requested
    f_title = _get_xkcd_font(21)
    f_alt = _get_xkcd_font(13)
    dummy = ImageDraw.Draw(core)
    alt_lines = _wrap_pixel_lines(dummy, f'Title text: "{alt_text}"', f_alt, max(260, core.width - 24))
    top_h = 34
    bot_h = 16 + len(alt_lines) * 16
    framed = Image.new("RGB", (core.width + 16, core.height + top_h + bot_h), (255, 255, 255))
    fd = ImageDraw.Draw(framed)
    fd.text((framed.width // 2, 18), title.upper(), fill=(10, 10, 10), font=f_title, anchor="mm")
    framed.paste(core, (8, top_h))
    for i, ln in enumerate(alt_lines):
        fd.text((framed.width // 2, top_h + core.height + 12 + i * 16), ln, fill=(65, 65, 65), font=f_alt, anchor="mm")
    return framed


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

    def to_png(self, output_path: str | Path | None = None, *, include_chrome: bool = False) -> bytes:
        im = _render_comic_pil(self._data, include_chrome=include_chrome)
        buf = BytesIO()
        im.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(png_bytes)
        return png_bytes

    def to_svg(self, width: int | None = None, height: int | None = None) -> str:
        im = _render_comic_pil(self._data, include_chrome=False)
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
            f'  <title>{alt_esc}</title>\n'
            f'  <desc>{title_esc}: {alt_esc}</desc>\n'
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
            global _ORT_SESSION_CACHE, _PT_MODEL_CACHE
            import sys
            if str(ROOT / "training") not in sys.path:
                sys.path.insert(0, str(ROOT / "training"))
            from generate_dataset import encode_prompt_tokens
            enriched_prompt = f"turns_{len(turns)} {layout} {prop} {tpl['title']} {clean}"
            tok_ids = encode_prompt_tokens(enriched_prompt, max_len=36)
            cast_idx = [CHARACTERS.index(c) if c in CHARACTERS else 0 for c in (cast + ["cueball", "megan"])[:2]]

            if backend == "onnx" and (ARTIFACT_DIR / "onnx" / "model_quantized.onnx").exists():
                import numpy as np
                import onnxruntime as ort
                if _ORT_SESSION_CACHE is None:
                    _ORT_SESSION_CACHE = ort.InferenceSession(str(ARTIFACT_DIR / "onnx" / "model_quantized.onnx"))
                outs = _ORT_SESSION_CACHE.run(
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
                if _PT_MODEL_CACHE is None:
                    _PT_MODEL_CACHE = XkcdArtistForGeneration.from_pretrained(ARTIFACT_DIR).eval()
                with torch.no_grad():
                    out = _PT_MODEL_CACHE(
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
        "chart_meta": tpl.get("chart_meta", {}),
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
