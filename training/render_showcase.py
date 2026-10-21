#!/usr/bin/env python3
"""Render a visual showcase of the GCP NVIDIA L4 Hybrid LoRA + Compositor pipeline."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "samples" / "hybrid_lora_showcase.png"


def get_font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Comic Sans MS Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def main() -> None:
    W, H = 1320, 860
    canvas = Image.new("RGB", (W, H), (246, 244, 238))
    draw = ImageDraw.Draw(canvas)

    f_title = get_font(28, bold=True)
    f_sub = get_font(15, bold=False)
    f_card = get_font(16, bold=True)
    f_meta = get_font(13, bold=False)

    draw.text((40, 28), "XKCD-ARTIST // GCP NVIDIA L4 GPU HYBRID LORA + VECTOR COMPOSITOR", fill=(17, 17, 17), font=f_title)
    draw.text(
        (40, 66),
        "Trained on 891 ExplainXKCD stage directions + 226 speech-bubble-masked comic panels on g2-standard-4 (NVIDIA L4 24GB VRAM)",
        fill=(85, 85, 85),
        font=f_sub,
    )

    # Top row: 3 LoRA raw panels from NVIDIA L4 GPU
    lora_cards = [
        ("L4 GPU Diffusion LoRA: Sudo Sandwich", ROOT / "artifacts" / "xkcd-artist" / "lora_samples" / "sudo_sandwich_raw_lora.png"),
        ("L4 GPU Diffusion LoRA: Compiling Code", ROOT / "artifacts" / "xkcd-artist" / "lora_samples" / "compiling_code_raw_lora.png"),
        ("L4 GPU Diffusion LoRA: Machine Learning", ROOT / "artifacts" / "xkcd-artist" / "lora_samples" / "machine_learning_raw_lora.png"),
    ]

    for idx, (label, path) in enumerate(lora_cards):
        x0 = 40 + idx * 420
        y0 = 108
        cw, ch = 400, 290
        draw.rectangle([x0 + 4, y0 + 4, x0 + cw + 4, y0 + ch + 4], fill=(17, 17, 17))
        draw.rectangle([x0, y0, x0 + cw, y0 + ch], fill=(255, 255, 255), outline=(17, 17, 17), width=2)
        draw.text((x0 + 14, y0 + 10), label, fill=(17, 17, 17), font=f_card)
        if path.exists():
            im = Image.open(path).convert("RGB").resize((230, 230), Image.Resampling.LANCZOS)
            canvas.paste(im, (x0 + (cw - 230) // 2, y0 + 42))
            draw.rectangle([x0 + (cw - 230) // 2, y0 + 42, x0 + (cw - 230) // 2 + 230, y0 + 272], outline=(17, 17, 17), width=2)

    # Bottom row: Final crisp composited XKCD strips with hand-lettered speech bubbles
    strip_path = ROOT / "docs" / "samples" / "sandwich.png"
    ml_path = ROOT / "docs" / "samples" / "machine_learning.png"

    bx0, by0, bw, bh = 40, 424, 1240, 400
    draw.rectangle([bx0 + 4, by0 + 4, bx0 + bw + 4, by0 + bh + 4], fill=(17, 17, 17))
    draw.rectangle([bx0, by0, bx0 + bw, by0 + bh], fill=(255, 255, 255), outline=(17, 17, 17), width=2)
    draw.text(
        (bx0 + 18, by0 + 12),
        "Stage 2: Crisp Hand-Lettered Speech Bubble, Leader-Line & Alt-Text Compositor (SVG + High-DPI PNG)",
        fill=(17, 17, 17),
        font=f_card,
    )

    if strip_path.exists():
        s1 = Image.open(strip_path).convert("RGB")
        s1.thumbnail((580, 330), Image.Resampling.LANCZOS)
        canvas.paste(s1, (bx0 + 24, by0 + 48))

    if ml_path.exists():
        s2 = Image.open(ml_path).convert("RGB")
        s2.thumbnail((580, 330), Image.Resampling.LANCZOS)
        canvas.paste(s2, (bx0 + 636, by0 + 48))

    draw.text(
        (bx0 + 24, by0 + 372),
        "Model: XkcdArtistForGeneration (1.59M params, 1.72MB Int8 ONNX) + segmind/tiny-sd UNet LoRA (1.73M LoRA params, r=16)",
        fill=(85, 85, 85),
        font=f_meta,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print(f"Saved showcase to {OUT}")


if __name__ == "__main__":
    main()
