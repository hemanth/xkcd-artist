#!/usr/bin/env python3
"""
Regenerate all canonical & novel XKCD sample PNGs in `docs/samples/`
and compose `docs/samples/hybrid_lora_showcase.png` in authentic xkcd.com style.
"""
from __future__ import annotations

import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python" / "src"))

from xkcd_artist import xkcd_artist  # noqa: E402

SAMPLES_DIR = ROOT / "docs" / "samples"
FONT_XKCD = ROOT / "docs" / "fonts" / "xkcd-script.ttf"
OUT_SHOWCASE = SAMPLES_DIR / "hybrid_lora_showcase.png"

SAMPLE_SPECS = [
    ("sandwich.png", "Sudo make me a sandwich"),
    ("compiling.png", "Compiling code sword fight"),
    ("machine_learning.png", "Machine learning pile of linear algebra"),
    ("dependency.png", "Modern digital infrastructure maintained in Nebraska"),
    ("standards.png", "14 competing standards"),
    ("duty_calls.png", "Someone is wrong on the internet"),
    ("friday_deploy.png", "Deploying a config tweak on Friday at 5pm"),
    ("novel_docker.png", "Why my Docker container image is 18GB"),
    ("novel_pr_review.png", "Reviewing a 4200 line pull request in TypeScript"),
    ("novel_agents.png", "Autonomous AI agents automating my workflow"),
    ("novel_git_rebase.png", "Git rebase before release"),
    ("novel_chart.png", "Annotated chart of time spent tuning webpack config"),
    ("novel_flowchart.png", "Flowchart for deciding whether to touch legacy DNS"),
]


def _font(path_list: list[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in path_list:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _wrap_px(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int) -> list[str]:
    lines: list[str] = []
    words = text.split()
    if not words:
        return lines
    cur = words[0]
    for w in words[1:]:
        cand = f"{cur} {w}"
        bb = draw.textbbox((0, 0), cand, font=font)
        if (bb[2] - bb[0]) <= max_w:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    comics_by_file = {}
    for fname, prompt in SAMPLE_SPECS:
        comic = xkcd_artist(prompt, backend="onnx")
        out_path = SAMPLES_DIR / fname
        comic.to_png(out_path)
        comics_by_file[fname] = comic
        print(f"Rendered {fname:22s} | layout={comic.layout:17s} | title={comic.title}")

    # Compose 2x2 xkcd.com-authentic showcase sheet (`#96A8C8` slate-blue background + white bordered boxes)
    W, H = 1440, 980
    canvas = Image.new("RGB", (W, H), (150, 168, 200))  # #96A8C8 authentic xkcd.com slate-blue
    draw = ImageDraw.Draw(canvas)

    ui_bold = _font(
        [
            "/System/Library/Fonts/LucidaGrande.ttc",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            str(FONT_XKCD),
        ],
        20,
    )
    ui_title = _font(
        [
            "/System/Library/Fonts/LucidaGrande.ttc",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            str(FONT_XKCD),
        ],
        24,
    )
    ui_meta = _font(
        [
            "/System/Library/Fonts/LucidaGrande.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            str(FONT_XKCD),
        ],
        13,
    )

    # Top xkcd.com header bar
    draw.rounded_rectangle([28, 18, W - 28, 86], radius=10, fill=(255, 255, 255), outline=(7, 20, 25), width=2)
    draw.text(
        (W // 2, 40),
        "XKCD-ARTIST — MULTI-PANEL & DIAGRAM SYNTHESIS SHOWCASE",
        fill=(10, 10, 10),
        font=ui_title,
        anchor="mm",
    )
    draw.text(
        (W // 2, 66),
        "Pure monochrome Randall Munroe stage art · 4x anti-aliased wobbled pen · xkcd-script.ttf hand-lettering",
        fill=(65, 75, 90),
        font=ui_meta,
        anchor="mm",
    )

    cards = [
        ("novel_docker.png", 'Prompt: "Why my Docker container image is 18GB"'),
        ("novel_pr_review.png", 'Prompt: "Reviewing a 4200 line pull request in TypeScript"'),
        ("novel_agents.png", 'Prompt: "Autonomous AI agents automating my workflow"'),
        ("novel_flowchart.png", 'Prompt: "Flowchart for deciding whether to touch legacy DNS"'),
    ]

    cw, ch = 678, 418
    for idx, (fname, prompt_lbl) in enumerate(cards):
        row = idx // 2
        col = idx % 2
        x0 = 28 + col * (cw + 28)
        y0 = 104 + row * (ch + 22)
        draw.rounded_rectangle(
            [x0, y0, x0 + cw, y0 + ch],
            radius=10,
            fill=(255, 255, 255),
            outline=(7, 20, 25),
            width=2,
        )
        comic = comics_by_file[fname]
        draw.text((x0 + cw // 2, y0 + 22), comic.title.upper(), fill=(10, 10, 10), font=ui_bold, anchor="mm")
        draw.text((x0 + cw // 2, y0 + 42), prompt_lbl, fill=(95, 105, 120), font=ui_meta, anchor="mm")

        im = Image.open(SAMPLES_DIR / fname).convert("RGB")
        im.thumbnail((cw - 32, ch - 110), Image.Resampling.LANCZOS)
        px = x0 + (cw - im.width) // 2
        py = y0 + 56 + (ch - 112 - im.height) // 2
        canvas.paste(im, (px, py))

        alt_lines = _wrap_px(draw, f'Title text: "{comic.alt}"', ui_meta, cw - 36)[:2]
        for li, ln in enumerate(alt_lines):
            draw.text((x0 + cw // 2, y0 + ch - 34 + li * 16), ln, fill=(55, 62, 72), font=ui_meta, anchor="mm")

    canvas.save(OUT_SHOWCASE)
    print(f"Saved showcase to {OUT_SHOWCASE}")


if __name__ == "__main__":
    main()
