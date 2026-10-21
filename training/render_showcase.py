#!/usr/bin/env python3
"""Render a visual showcase of xkcd-artist outputs using xkcd-script.ttf and spotless XKCD stage art."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT_PATH = ROOT / "docs" / "fonts" / "xkcd-script.ttf"
OUT = ROOT / "docs" / "samples" / "hybrid_lora_showcase.png"


def get_font(size: int):
    if FONT_PATH.exists():
        try:
            return ImageFont.truetype(str(FONT_PATH), size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def main() -> None:
    W, H = 1360, 960
    canvas = Image.new("RGB", (W, H), (246, 245, 239))
    draw = ImageDraw.Draw(canvas)

    f_title = get_font(30)
    f_sub = get_font(16)
    f_card = get_font(16)

    draw.text(
        (40, 22),
        "XKCD-ARTIST // SPOTLESS STAGE EXTRACTION + NEURAL SCENE PLANNER + XKCD-SCRIPT COMPOSITOR",
        fill=(17, 17, 17),
        font=f_title,
    )
    draw.text(
        (40, 62),
        "Trained on GCP NVIDIA L4 GPU (24GB VRAM) over 891 ExplainXKCD scripts + 226 connected-component text-cleaned XKCD panels",
        fill=(80, 80, 80),
        font=f_sub,
    )

    cards = [
        ("Prompt: 'Sudo make me a sandwich'", ROOT / "docs" / "samples" / "sandwich.png"),
        ("Prompt: 'Compiling code sword fight'", ROOT / "docs" / "samples" / "compiling.png"),
        ("Prompt: '14 competing standards'", ROOT / "docs" / "samples" / "standards.png"),
        ("Prompt: 'Machine learning pile of linear algebra'", ROOT / "docs" / "samples" / "machine_learning.png"),
        ("Prompt: 'Modern digital infrastructure in Nebraska'", ROOT / "docs" / "samples" / "dependency.png"),
        ("Novel Prompt: 'Deploying to production on Friday'", ROOT / "docs" / "samples" / "friday_deploy.png"),
    ]

    cw, ch = 412, 405
    for idx, (label, path) in enumerate(cards):
        row = idx // 3
        col = idx % 3
        x0 = 40 + col * (cw + 22)
        y0 = 102 + row * (ch + 24)
        draw.rectangle([x0 + 4, y0 + 4, x0 + cw + 4, y0 + ch + 4], fill=(17, 17, 17))
        draw.rectangle([x0, y0, x0 + cw, y0 + ch], fill=(255, 255, 255), outline=(17, 17, 17), width=2)
        draw.text((x0 + 14, y0 + 10), label.upper(), fill=(17, 17, 17), font=f_card)

        if path.exists():
            im = Image.open(path).convert("RGB")
            im.thumbnail((cw - 24, ch - 48), Image.Resampling.LANCZOS)
            px = x0 + (cw - im.width) // 2
            py = y0 + 38 + (ch - 48 - im.height) // 2
            canvas.paste(im, (px, py))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print(f"Saved showcase to {OUT}")


if __name__ == "__main__":
    main()
