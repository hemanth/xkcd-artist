import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "python" / "src"))

from xkcd_artist import xkcd_artist


class TestXkcdArtist(unittest.TestCase):
    def test_iconic_sandwich_prompt(self) -> None:
        comic = xkcd_artist("Sudo make me a sandwich", backend="onnx")
        self.assertFalse(comic.is_invalid)
        self.assertEqual(comic.title, "SANDWICH")
        self.assertGreaterEqual(len(comic.panels), 2)
        svg = comic.to_svg()
        self.assertIn("<svg", svg)
        self.assertIn("SUDO MAKE ME A", svg)
        self.assertIn("SANDWICH.", svg)
        png = comic.to_png()
        self.assertTrue(png.startswith(b"\x89PNG"))

    def test_neural_onnx_and_pytorch_backends(self) -> None:
        c_onnx = xkcd_artist("Machine learning pile of linear algebra", backend="onnx")
        self.assertFalse(c_onnx.is_invalid)
        self.assertLess(c_onnx.p_invalid, 0.05)

        c_pt = xkcd_artist("Compiling code sword fight", backend="pytorch")
        self.assertFalse(c_pt.is_invalid)
        self.assertLess(c_pt.p_invalid, 0.05)

    def test_abstention_on_empty_prompt(self) -> None:
        inv = xkcd_artist("   ")
        self.assertTrue(inv.is_invalid)
        self.assertGreater(inv.p_invalid, 0.95)


if __name__ == "__main__":
    unittest.main()
