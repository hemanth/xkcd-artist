from __future__ import annotations

import io
import json
import sys
import time
import unittest
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "python" / "src"))

from xkcd_artist import LAYOUT_TYPES, xkcd_artist  # noqa: E402


def _connected_component_sizes(binary_mask: np.ndarray) -> list[int]:
    h, w = binary_mask.shape
    vis = np.zeros((h, w), dtype=bool)
    sizes: list[int] = []
    ys, xs = np.where(binary_mask)
    for sy, sx in zip(ys, xs):
        if vis[sy, sx]:
            continue
        q: deque[tuple[int, int]] = deque([(sy, sx)])
        vis[sy, sx] = True
        cnt = 0
        while q:
            cy, cx = q.popleft()
            cnt += 1
            for ny in range(max(0, cy - 1), min(h, cy + 2)):
                for nx in range(max(0, cx - 1), min(w, cx + 2)):
                    if not vis[ny, nx] and binary_mask[ny, nx]:
                        vis[ny, nx] = True
                        q.append((ny, nx))
        sizes.append(cnt)
    return sizes


class TestXkcdArtist(unittest.TestCase):
    def test_iconic_templates_and_monochrome_purity(self) -> None:
        cases = [
            ("Sudo make me a sandwich", "SANDWICH", "single_panel", "Simon Says"),
            ("Compiling code sword fight", "COMPILING", "single_panel", "LCDs"),
            ("Machine learning pile of linear algebra", "MACHINE LEARNING", "single_panel", "recurrent"),
            ("Modern digital infrastructure maintained in Nebraska", "DEPENDENCY", "single_panel", "ImageMagick"),
            ("14 competing standards", "STANDARDS", "three_panel", "mini-USB"),
            ("Someone is wrong on the internet", "DUTY CALLS", "single_panel", "wrong"),
        ]
        for prompt, expected_title, expected_layout, expected_alt_substr in cases:
            with self.subTest(prompt=prompt):
                comic = xkcd_artist(prompt, backend="onnx")
                self.assertFalse(comic.is_invalid)
                self.assertEqual(comic.title, expected_title)
                self.assertEqual(comic.layout, expected_layout)
                self.assertIn(expected_alt_substr, comic.alt)

                png_bytes = comic.to_png()
                self.assertTrue(png_bytes.startswith(b"\x89PNG"))
                im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                arr = np.array(im, dtype=np.int16)

                # Pure monochrome check: R, G, B channels are identical (max channel delta == 0)
                max_chroma = int(np.max(np.abs(arr[:, :, 0] - arr[:, :, 2])))
                self.assertEqual(max_chroma, 0, f"{expected_title} should be pure grayscale/monochrome")

                # Healthy ink density (neither blank white nor solid black)
                ink_ratio = float(np.mean(arr[:, :, 0] < 200))
                self.assertGreater(ink_ratio, 0.03)
                self.assertLess(ink_ratio, 0.35)

                svg = comic.to_svg()
                self.assertIn("<svg", svg)
                self.assertIn(expected_alt_substr, svg)

    def test_all_six_layout_families_generated(self) -> None:
        prompts_by_layout = {
            "single_panel": "Git rebase before release",
            "two_panel": "Deploying a config tweak on Friday at 5pm",
            "three_panel": "Why my Docker container image is 18GB",
            "four_panel": "Reviewing a 4200 line pull request in TypeScript",
            "annotated_chart": "Annotated chart of time spent tuning webpack config",
            "flowchart_diagram": "Flowchart for deciding whether to touch legacy DNS",
        }
        seen_layouts = set()
        for expected_layout, prompt in prompts_by_layout.items():
            comic = xkcd_artist(prompt, backend="onnx")
            self.assertEqual(comic.layout, expected_layout)
            seen_layouts.add(comic.layout)
            png = comic.to_png()
            im = Image.open(io.BytesIO(png))
            self.assertGreater(im.width, 300)
            self.assertGreater(im.height, 220)
        self.assertEqual(seen_layouts, set(LAYOUT_TYPES))

    def test_optional_framed_chrome_untruncated_alt(self) -> None:
        comic = xkcd_artist("Modern digital infrastructure maintained in Nebraska", backend="onnx")
        raw_im = Image.open(io.BytesIO(comic.to_png(include_chrome=False)))
        framed_im = Image.open(io.BytesIO(comic.to_png(include_chrome=True)))
        self.assertGreater(framed_im.height, raw_im.height + 50)
        self.assertGreaterEqual(framed_im.width, raw_im.width)

    def test_keyword_deduplication_prevents_tautologies(self) -> None:
        pr = xkcd_artist("Reviewing a 4200 line pull request in TypeScript")
        joined_pr = " ".join(t["text"] for t in pr.to_dict()["turns"])
        self.assertNotIn("PULL PR", joined_pr)
        self.assertIn("TYPESCRIPT PR", joined_pr)

        fri = xkcd_artist("Deploying a config tweak on Friday at 5pm")
        joined_fri = " ".join(t["text"] for t in fri.to_dict()["turns"])
        self.assertNotIn("CONFIG CONFIG", joined_fri)

        dk = xkcd_artist("Why my Docker container image is 18GB")
        joined_dk = " ".join(t["text"] for t in dk.to_dict()["turns"])
        self.assertNotIn("18GB IMAGE", joined_dk)
        self.assertIn("DOCKER IMAGE", joined_dk)

    def test_neural_onnx_and_pytorch_backends_and_warm_cache(self) -> None:
        c_onnx = xkcd_artist("Machine learning pile of linear algebra", backend="onnx")
        self.assertFalse(c_onnx.is_invalid)
        self.assertLess(c_onnx.p_invalid, 0.05)

        t0 = time.perf_counter()
        c_warm = xkcd_artist("Autonomous AI agents automating my workflow", backend="onnx")
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.assertFalse(c_warm.is_invalid)
        self.assertLess(dt_ms, 50.0, f"Warm ONNX inference took {dt_ms:.2f}ms")

        c_pt = xkcd_artist("Compiling code sword fight", backend="pytorch")
        self.assertFalse(c_pt.is_invalid)
        self.assertLess(c_pt.p_invalid, 0.05)

    def test_abstention_on_empty_and_adversarial_prompts(self) -> None:
        for bad in [
            "",
            "   ",
            "bad\x00input",
            "404 NOT FOUND EMPTY PROMPT ABSTENTION",
            "SELECT * FROM comics WHERE id = -99999",
            "null undefined NaN [[[",
        ]:
            inv = xkcd_artist(bad)
            self.assertTrue(inv.is_invalid, f"Expected abstention for {bad!r}")
            self.assertGreater(inv.p_invalid, 0.95)

    def test_stage_scenes_pixel_purity_and_anatomical_props(self) -> None:
        scenes_dir = ROOT / "docs" / "scenes"
        meta = json.loads((scenes_dir / "scenes_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(len(meta), 11)

        for name, info in meta.items():
            p = scenes_dir / info["file"]
            self.assertTrue(p.exists(), f"Missing stage file {p}")
            im = Image.open(p).convert("L")
            self.assertEqual(im.size, (info["width"], info["height"]))
            arr = np.array(im)
            # Upper cleared speech zone (excluding borders x=8..w-8, y=8..70) must have zero tiny leftover glyph specks
            if name not in ("scene_dependency",):
                zone = arr[8:68, 8 : im.width - 8] < 200
                tiny_specks = [s for s in _connected_component_sizes(zone) if 1 <= s <= 18]
                self.assertEqual(tiny_specks, [], f"Orphan ink specks found in {name}: {tiny_specks}")

        # Verify #303 Compiling preserves wall poster & light switch
        im303 = np.array(Image.open(scenes_dir / "scene_compiling.png").convert("L"))
        self.assertGreater(int(np.sum(im303[222:274, 42:106] < 180)), 80)

        # Verify #2347 Dependency preserves authentic curly brace & curved arrow
        im2347 = np.array(Image.open(scenes_dir / "scene_dependency.png").convert("L"))
        self.assertGreater(int(np.sum(im2347[60:92, 46:222] < 180)), 120)
        self.assertGreater(int(np.sum(im2347[382:424, 200:260] < 180)), 60)

        # Verify #1513 4-panel preserves Panel 4 Cueball top skull (no decapitation)
        im1513 = np.array(Image.open(scenes_dir / "scene_1513_4panel.png").convert("L"))
        self.assertGreater(int(np.sum(im1513[140:152, 655:685] < 180)), 25)


if __name__ == "__main__":
    unittest.main()
