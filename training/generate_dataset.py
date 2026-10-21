#!/usr/bin/env python3
"""
Build the hybrid XKCD + ExplainXKCD training, validation, and test datasets for xkcd-artist.

Ingests:
- Real XKCD + ExplainXKCD metadata (/Users/lika/labs/xkcd-bench/comics/metadata.json, 891 comics)
- Real XKCD comic strip PNGs (/Users/lika/labs/xkcd-bench/comics/images/*.png)

Performs:
1. Transcript stage-direction & dialogue parsing ([Cueball sits at desk...] + speaker turns)
2. Character cast & prop extraction (Cueball, Megan, Black Hat, Beret Guy, Ponytail, Hairbun + props)
3. Speech-bubble text masking on real XKCD PNGs (producing clean monochrome stick-figure ink patches
   for Diffusion LoRA training without garbled text hallucinations)
4. Deterministic train/val/test JSONL splits (seed = 20261020)
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CLEAN_PANELS_DIR = DATA_DIR / "clean_panels"

XKCD_BENCH_META = Path("/Users/lika/labs/xkcd-bench/comics/metadata.json")
XKCD_BENCH_IMGS = Path("/Users/lika/labs/xkcd-bench/comics/images")

SEED = 20261020

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

LAYOUT_TYPES = [
    "single_panel",
    "two_panel",
    "three_panel",
    "four_panel",
    "annotated_chart",
    "flowchart_diagram",
]

CATEGORIES = [
    "Programming & Linux",
    "Computer Science & Security",
    "Math & Physics",
    "AI & Machine Learning",
    "Science & Biology",
    "Tech Culture & Internet",
    "Language & Philosophy",
    "Romance & Everyday Life",
]

VOCAB_WORDS = [
    "<pad>", "<unk>", "sudo", "make", "me", "a", "sandwich", "regex", "git", "commit",
    "push", "merge", "conflict", "compiler", "code", "compiling", "sword", "fighting",
    "server", "cloud", "kubernetes", "docker", "dependency", "nebraska", "tower", "stack",
    "ai", "machine", "learning", "linear", "algebra", "pile", "stir", "until", "answers",
    "python", "import", "antigravity", "flying", "dynamic", "typing", "pointers", "memory",
    "leak", "segfault", "null", "timezone", "utc", "iso", "8601", "date", "format",
    "standard", "standards", "14", "15", "competing", "universal", "password", "entropy",
    "correct", "horse", "battery", "staple", "security", "crypto", "rsa", "wrench",
    "quantum", "physics", "math", "integral", "proof", "statistic", "correlation",
    "causation", "p_value", "significant", "jelly", "beans", "acne", "green", "chart",
    "graph", "axis", "exponential", "extrapolating", "husband", "yesterday", "zero",
    "dojo", "emacs", "vim", "nano", "butterfly", "cosmic", "ray", "flip", "bit",
    "cueball", "megan", "black", "hat", "beret", "guy", "ponytail", "hairbun",
    "desk", "laptop", "whiteboard", "pointer", "couch", "server", "rack", "chart",
    "someone", "is", "wrong", "on", "the", "internet", "duty", "calls", "sleep",
    "3am", "debugging", "production", "database", "sql", "injection", "bobby", "tables",
    "drop", "table", "students", "sanitize", "inputs", "xkcd", "comic", "panel",
    "dialogue", "speech", "bubble", "stick", "figure", "monochrome", "ink", "drawing",
]
WORD_TO_ID = {w: i for i, w in enumerate(VOCAB_WORDS)}


def encode_prompt_tokens(text: str, max_len: int = 36) -> list[int]:
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    ids: list[int] = []
    for tok in tokens[:max_len]:
        if tok in WORD_TO_ID:
            ids.append(WORD_TO_ID[tok])
        else:
            # Deterministic hash into [len(VOCAB_WORDS) .. 1199]
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
            ids.append(len(VOCAB_WORDS) + (h % (1200 - len(VOCAB_WORDS))))
    while len(ids) < max_len:
        ids.append(0)
    return ids


def parse_transcript(transcript: str, title: str, alt: str) -> dict[str, Any]:
    """Extract stage directions, speaker turns, cast, props, and layout from ExplainXKCD transcript."""
    raw = (transcript or "").strip()
    stage_dirs = re.findall(r"\[([^\]]+)\]", raw)
    # Remove stage directions and {{Title text:...}} to extract clean speaker lines
    cleaned = re.sub(r"\{\{.*?\}\}", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"\[[^\]]+\]", "\n", cleaned)

    turns: list[dict[str, str]] = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("title text") or line.lower().startswith("alt"):
            continue
        m = re.match(r"^([A-Za-z0-9 _\-\.]{2,22}):\s*(.+)$", line)
        if m:
            speaker = m.group(1).strip().lower()
            utterance = m.group(2).strip()
            if utterance:
                turns.append({"speaker": speaker, "text": utterance[:110]})
        elif len(line) > 3 and len(turns) < 6:
            turns.append({"speaker": "narrator", "text": line[:110]})

    if not turns:
        turns = [
            {"speaker": "cueball", "text": title},
            {"speaker": "megan", "text": (alt or "Wait, really?")[:95]},
        ]

    blob = (raw + " " + title + " " + alt).lower()
    cast: list[int] = []
    if "cueball" in blob or "man" in blob or "friend" in blob or not cast:
        cast.append(0)
    if "megan" in blob or "woman" in blob or "girl" in blob:
        cast.append(1)
    if "black hat" in blob or "hat" in blob:
        cast.append(2)
    if "beret" in blob:
        cast.append(3)
    if "ponytail" in blob:
        cast.append(4)
    if "hairbun" in blob or "mom" in blob or "roberts" in blob:
        cast.append(5)
    if len(cast) == 1 and len(turns) >= 2:
        cast.append(1)
    cast = cast[:3]

    # Detect prop
    prop_id = 0
    if any(k in blob for k in ["computer", "laptop", "desk", "typing", "screen", "terminal", "keyboard"]):
        prop_id = 1
    elif any(k in blob for k in ["whiteboard", "board", "equation", "formula", "proof"]):
        prop_id = 2
    elif any(k in blob for k in ["server", "rack", "cloud", "datacenter", "mainframe"]):
        prop_id = 3
    elif any(k in blob for k in ["pointer", "stick", "presentation", "slide", "talk"]):
        prop_id = 4
    elif any(k in blob for k in ["couch", "sofa", "chair", "sitting", "sandwich"]):
        prop_id = 5
    elif any(k in blob for k in ["chart", "graph", "plot", "axis", "bar", "line", "pie", "venn"]):
        prop_id = 6
    elif any(k in blob for k in ["pedestal", "podium", "stage", "lecture"]):
        prop_id = 7

    # Detect layout type
    if prop_id == 6 and len(turns) <= 2:
        layout_id = 4  # annotated_chart
    elif "flowchart" in blob or "diagram" in blob or "arrow" in blob:
        layout_id = 5  # flowchart_diagram
    elif len(turns) == 1:
        layout_id = 0  # single_panel
    elif len(turns) == 2:
        layout_id = 1  # two_panel
    elif len(turns) == 3:
        layout_id = 2  # three_panel
    else:
        layout_id = 3  # four_panel

    return {
        "stage_directions": stage_dirs[:4],
        "turns": turns[:4],
        "cast": cast,
        "prop_id": prop_id,
        "layout_id": layout_id,
    }


def mask_speech_bubbles_and_extract_patch(img_path: Path, size: int = 32) -> tuple[list[list[float]], dict[str, float]]:
    """
    Load a real XKCD comic PNG, mask out high-frequency speech-bubble text in the top 40%
    of the panel so only stick-figure line art & borders remain, and extract a 32x32
    ink density patch + empirical stroke style targets.
    """
    try:
        im = Image.open(img_path).convert("L")
    except Exception:
        grid = [[0.0] * size for _ in range(size)]
        return grid, {"ink_density": 0.08, "wobble_amp": 1.6, "stroke_px": 2.0, "upper_bubble_ratio": 0.28}

    arr = np.asarray(im, dtype=np.float32)
    h, w = arr.shape
    # Crop center square-ish region representing a comic panel
    side = min(h, w)
    y0 = max(0, (h - side) // 2)
    x0 = max(0, (w - side) // 2)
    crop = arr[y0 : y0 + side, x0 : x0 + side].copy()

    # Invert so ink = 1.0, white paper = 0.0
    ink = (255.0 - crop) / 255.0
    upper_h = max(1, int(side * 0.40))
    upper_bubble_ratio = float(np.mean(ink[:upper_h, :] > 0.35))

    # Mask out speech-bubble text region in the top 38% (preserving outer 4% panel border)
    margin = max(2, int(side * 0.05))
    ink[margin:upper_h, margin : side - margin] *= 0.15

    ink_density = float(np.clip(np.mean(ink > 0.25), 0.02, 0.35))

    patch_im = Image.fromarray(np.uint8(np.clip(ink * 255.0, 0, 255))).resize(
        (size, size), Image.Resampling.BILINEAR
    )
    patch_arr = (np.asarray(patch_im, dtype=np.float32) / 255.0).round(4)

    return patch_arr.tolist(), {
        "ink_density": round(ink_density, 4),
        "wobble_amp": round(1.35 + ink_density * 2.4, 4),
        "stroke_px": round(1.75 + ink_density * 1.8, 4),
        "upper_bubble_ratio": round(min(0.45, max(0.18, upper_bubble_ratio * 2.2)), 4),
    }


def synthesize_target_poses_and_bubbles(
    cast: list[int],
    prop_id: int,
    layout_id: int,
    num_turns: int,
    rng: random.Random,
) -> tuple[list[float], list[float], list[float]]:
    """
    Generate ground-truth normalized kinematic pose vector (12 floats for 2 primary characters),
    speech-bubble layout vector (12 floats for up to 3 bubbles), and style vector (4 floats).
    """
    # Character 1 (left) & Character 2 (right)
    # Each char: [x, y_foot, head_tilt, left_arm_angle, right_arm_angle, action_state]
    c1_x = 0.28 + rng.uniform(-0.03, 0.03)
    c2_x = 0.72 + rng.uniform(-0.03, 0.03)
    if prop_id == 1:  # laptop_desk
        c1_pose = [c1_x, 0.84, 0.12, -0.55, 0.65, 0.85]
        c2_pose = [c2_x, 0.84, -0.08, -0.35, 0.25, 0.40]
    elif prop_id in (2, 4, 6):  # whiteboard / pointer / chart
        c1_pose = [0.24, 0.85, 0.05, -0.25, 0.95, 0.90]
        c2_pose = [0.76, 0.85, -0.10, -0.40, 0.30, 0.30]
    else:
        c1_pose = [c1_x, 0.85, 0.04, -0.42, 0.48, 0.50]
        c2_pose = [c2_x, 0.85, -0.05, -0.45, 0.38, 0.50]

    poses = [round(v, 4) for v in (c1_pose + c2_pose)]

    # Speech bubble boxes: 3 bubbles x [cx, cy, w, h] in normalized panel coords
    # Invariants: cy + h/2 <= 0.44 (never overlaps stick figure heads at y >= 0.48)
    b1 = [0.30, 0.16, 0.46, 0.16]
    b2 = [0.68, 0.31, 0.46, 0.16]
    b3 = [0.50, 0.15, 0.56, 0.15]
    bubbles = [round(v, 4) for v in (b1 + b2 + b3)]

    style_vec = [
        round(1.65 + rng.uniform(-0.15, 0.15), 4),  # wobble_amp_px
        round(2.05 + rng.uniform(-0.12, 0.12), 4),  # stroke_width_px
        round(0.28 + rng.uniform(-0.03, 0.03), 4),  # bubble_zone_ratio
        round(0.92 + rng.uniform(-0.04, 0.04), 4),  # ink_contrast
    ]
    return poses, bubbles, style_vec


def build_datasets() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_PANELS_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    raw_comics: list[dict[str, Any]] = []
    if XKCD_BENCH_META.exists():
        with open(XKCD_BENCH_META, "r", encoding="utf-8") as f:
            raw_comics = json.load(f)

    records: list[dict[str, Any]] = []
    comic_bank: list[dict[str, Any]] = []

    for idx, item in enumerate(raw_comics):
        num = int(item.get("num", idx + 1))
        title = str(item.get("title", f"Comic #{num}")).strip()
        alt = str(item.get("alt", "")).strip()
        transcript = str(item.get("transcript", "")).strip()
        explanation = str(item.get("explanation", "")).strip()
        category = str(item.get("category", "Programming & Linux")).strip()
        concepts = item.get("key_concepts") or []
        if isinstance(concepts, str):
            try:
                concepts = json.loads(concepts.replace("'", '"'))
            except Exception:
                concepts = [title.lower()]

        parsed = parse_transcript(transcript, title, alt)
        img_file = XKCD_BENCH_IMGS / f"{num}.png"
        has_real_png = img_file.exists()

        if has_real_png:
            ink_patch, img_stats = mask_speech_bubbles_and_extract_patch(img_file, size=16)
        else:
            # Synthesize deterministic 16x16 stick-figure ink density target from layout/prop
            patch = np.zeros((16, 16), dtype=np.float32)
            # Stick figure 1 column (x=4) and stick figure 2 column (x=11)
            patch[7:14, 4] = 0.75
            patch[5:7, 3:6] = 0.65
            if len(parsed["cast"]) > 1:
                patch[7:14, 11] = 0.75
                patch[5:7, 10:13] = 0.65
            if parsed["prop_id"] == 1:  # desk
                patch[10:14, 6:9] = 0.55
            elif parsed["prop_id"] in (2, 6):  # whiteboard / chart
                patch[5:11, 7:11] = 0.50
            ink_patch = patch.round(4).tolist()
            img_stats = {"ink_density": 0.11, "wobble_amp": 1.62, "stroke_px": 2.0, "upper_bubble_ratio": 0.28}

        poses, bubbles, style_vec = synthesize_target_poses_and_bubbles(
            parsed["cast"], parsed["prop_id"], parsed["layout_id"], len(parsed["turns"]), rng
        )
        style_vec[0] = img_stats["wobble_amp"]
        style_vec[1] = img_stats["stroke_px"]

        n_turns = len(parsed["turns"])
        prop_tag = PROPS[parsed["prop_id"]]
        layout_tag = LAYOUT_TYPES[parsed["layout_id"]]
        # Mix short natural user prompts (50%) and full ExplainXKCD enriched prompts (50%)
        if idx % 2 == 0:
            prompt_text = f"turns_{n_turns} {layout_tag} {prop_tag} {title} {' '.join(str(c) for c in concepts[:4])}"
        else:
            prompt_text = (
                f"turns_{n_turns} {layout_tag} {prop_tag} {title}. "
                f"{' '.join(str(c) for c in concepts[:5])}. "
                f"{' '.join(parsed['stage_directions'][:2])} {explanation[:160]}"
            )
        token_ids = encode_prompt_tokens(prompt_text, max_len=36)

        rec = {
            "comic_num": num,
            "title": title,
            "alt": alt or "Hover text.",
            "category": category,
            "prompt": prompt_text[:220],
            "token_ids": token_ids,
            "layout_id": parsed["layout_id"],
            "prop_id": parsed["prop_id"],
            "cast_ids": (parsed["cast"] + [0, 1])[:2],
            "poses": poses,
            "bubbles": bubbles,
            "style_vec": style_vec,
            "ink_patch": ink_patch,
            "turns": parsed["turns"],
            "stage_directions": parsed["stage_directions"],
            "has_real_png": has_real_png,
            "is_invalid": 0,
        }
        records.append(rec)

        if idx < 120 or item.get("is_iconic"):
            comic_bank.append(
                {
                    "num": num,
                    "title": title,
                    "alt": alt[:160],
                    "category": category,
                    "keywords": [str(c).lower() for c in concepts[:6]] + re.findall(r"[a-z0-9]+", title.lower()),
                    "layout": LAYOUT_TYPES[parsed["layout_id"]],
                    "prop": PROPS[parsed["prop_id"]],
                    "cast": [CHARACTERS[c] for c in parsed["cast"][:2]],
                    "turns": parsed["turns"][:4],
                }
            )

    # Also add 90 negative/invalid abstention prompts (empty, binary garbage, non-visual noise)
    for neg_i in range(90):
        neg_prompts = [
            "",
            "   ",
            "\x00\xff\xfe binary payload",
            "SELECT * FROM comics WHERE id = -99999;",
            "null undefined NaN [[[ ]]]",
            "404 not found empty prompt abstention check",
        ]
        p_txt = neg_prompts[neg_i % len(neg_prompts)]
        records.append(
            {
                "comic_num": -1 - neg_i,
                "title": "Invalid Prompt",
                "alt": "",
                "category": "Invalid",
                "prompt": p_txt,
                "token_ids": encode_prompt_tokens(p_txt, max_len=36),
                "layout_id": 0,
                "prop_id": 0,
                "cast_ids": [0, 0],
                "poses": [0.0] * 12,
                "bubbles": [0.0] * 12,
                "style_vec": [0.0, 0.0, 0.0, 0.0],
                "ink_patch": [[0.0] * 16 for _ in range(16)],
                "turns": [],
                "stage_directions": [],
                "has_real_png": False,
                "is_invalid": 1,
            }
        )

    rng.shuffle(records)
    n_total = len(records)
    n_train = int(n_total * 0.76)
    n_val = int(n_total * 0.12)

    train_split = records[:n_train]
    val_split = records[n_train : n_train + n_val]
    test_split = records[n_train + n_val :]

    for split_name, split_rows in [("train", train_split), ("val", val_split), ("test", test_split)]:
        out_path = DATA_DIR / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in split_rows:
                f.write(json.dumps(r) + "\n")

    bank_path = DATA_DIR / "comic_bank.json"
    with open(bank_path, "w", encoding="utf-8") as f:
        json.dump(comic_bank[:100], f, indent=2)

    summary = {
        "total_records": n_total,
        "real_xkcd_comics": len(raw_comics),
        "real_pngs_masked": sum(1 for r in records if r["has_real_png"]),
        "train": len(train_split),
        "val": len(val_split),
        "test": len(test_split),
        "bank_size": len(comic_bank[:100]),
    }
    print(f"[generate_dataset] Built dataset splits: {json.dumps(summary)}")
    return summary


if __name__ == "__main__":
    build_datasets()
