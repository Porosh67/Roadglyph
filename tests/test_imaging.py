import tempfile
import unittest
from pathlib import Path

import _path  # noqa: F401
import numpy as np
from PIL import Image
from roadglyph.imaging import ImageLoadError, load_image


class LoadImageTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _assert_rgb(self, img):
        self.assertEqual(img.mode, "RGB")
        self.assertGreater(img.width, 0)

    def test_png_jpeg_tiff(self):
        base = Image.new("RGB", (64, 32), (200, 30, 30))
        for name, fmt in (("a.png", "PNG"), ("a.jpg", "JPEG"), ("a.tiff", "TIFF")):
            with self.subTest(name=name):
                base.save(self.dir / name, format=fmt)
                img = load_image(self.dir / name)
                self._assert_rgb(img)
                self.assertEqual(img.size, (64, 32))

    def test_sixteen_bit_tiff_is_stretched_to_full_range(self):
        arr = (np.linspace(1000, 3000, 64 * 32).reshape(32, 64)).astype(np.uint16)
        Image.fromarray(arr).save(self.dir / "deep.tiff")
        img = load_image(self.dir / "deep.tiff")
        self._assert_rgb(img)
        gray = np.asarray(img.convert("L"))
        self.assertLess(int(gray.min()), 5)
        self.assertGreater(int(gray.max()), 250)

    def test_alpha_is_flattened_onto_white(self):
        rgba = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
        rgba.save(self.dir / "alpha.png")
        img = load_image(self.dir / "alpha.png")
        self.assertEqual(img.getpixel((5, 5)), (255, 255, 255))

    def test_palette_and_grayscale_and_cmyk(self):
        Image.new("P", (8, 8)).save(self.dir / "p.png")
        Image.new("L", (8, 8), 128).save(self.dir / "l.png")
        Image.new("CMYK", (8, 8), (0, 0, 0, 0)).save(self.dir / "c.tiff")
        for name in ("p.png", "l.png", "c.tiff"):
            with self.subTest(name=name):
                self._assert_rgb(load_image(self.dir / name))

    def test_multi_frame_tiff_uses_first_frame(self):
        frames = [Image.new("RGB", (20, 10), (255, 0, 0)), Image.new("RGB", (20, 10), (0, 0, 255))]
        frames[0].save(self.dir / "multi.tiff", save_all=True, append_images=frames[1:])
        img = load_image(self.dir / "multi.tiff")
        self.assertEqual(img.getpixel((1, 1)), (255, 0, 0))

    def test_exif_orientation_is_applied(self):
        img = Image.new("RGB", (40, 20), (10, 200, 10))
        exif = img.getexif()
        exif[0x0112] = 6  # rotate 90 degrees
        img.save(self.dir / "rot.jpg", exif=exif)
        self.assertEqual(load_image(self.dir / "rot.jpg").size, (20, 40))

    def test_missing_and_corrupt_files_raise_a_clear_error(self):
        with self.assertRaises(ImageLoadError):
            load_image(self.dir / "nope.png")
        (self.dir / "bad.png").write_bytes(b"not an image at all")
        with self.assertRaises(ImageLoadError):
            load_image(self.dir / "bad.png")


if __name__ == "__main__":
    unittest.main()
