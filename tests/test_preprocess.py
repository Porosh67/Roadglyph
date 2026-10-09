import unittest

import _path  # noqa: F401
import numpy as np
from PIL import Image
from roadglyph.preprocess import build_variants, compute_stats, fit_size
from roadglyph.synthetic import add_glare, add_noise, draw_plate, low_light


class FitSizeTests(unittest.TestCase):
    def test_downscales_large_and_upscales_small(self):
        big = fit_size(Image.new("RGB", (4000, 3000)), 448, 1792)
        self.assertEqual(max(big.size), 1792)
        small = fit_size(Image.new("RGB", (200, 120)), 448, 1792)
        self.assertGreaterEqual(min(small.size), 440)

    def test_upscaling_is_capped_at_four_times(self):
        tiny = fit_size(Image.new("RGB", (100, 60)), 448, 1792)
        self.assertEqual(tiny.size, (400, 240))

    def test_leaves_reasonable_images_alone(self):
        img = Image.new("RGB", (800, 600))
        self.assertIs(fit_size(img, 448, 1792), img)

    def test_extreme_aspect_ratio_does_not_explode(self):
        out = fit_size(Image.new("RGB", (3000, 20)), 448, 1792)
        self.assertLessEqual(max(out.size), 1792)
        self.assertGreaterEqual(min(out.size), 1)


class StatsAndVariantsTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.plate = draw_plate("7ABC123")

    def test_clean_image_gets_base_first(self):
        variants = build_variants(self.plate, compute_stats(self.plate), max_variants=3)
        self.assertEqual(variants[0][0], "base")
        self.assertEqual(len({name for name, _ in variants}), len(variants))

    def test_dark_image_triggers_lift(self):
        dark = low_light(self.plate)
        stats = compute_stats(dark)
        self.assertLess(stats.mean, 75)
        names = [n for n, _ in build_variants(dark, stats, max_variants=4)]
        self.assertIn("lift", names)

    def test_noisy_image_triggers_denoise(self):
        noisy = add_noise(self.plate, 30.0, self.rng)
        stats = compute_stats(noisy)
        self.assertGreater(stats.noise, 7.0)
        names = [n for n, _ in build_variants(noisy, stats, max_variants=4)]
        self.assertIn("denoise", names)

    def test_glare_triggers_deglare(self):
        glare = add_glare(self.plate)
        stats = compute_stats(glare)
        names = [n for n, _ in build_variants(glare, stats, max_variants=5)]
        self.assertGreater(stats.saturated_frac, 0.04)
        self.assertIn("deglare", names)

    def test_variant_count_and_sizes_respect_limits(self):
        for limit in (1, 2, 3):
            variants = build_variants(
                self.plate,
                compute_stats(self.plate),
                max_variants=limit,
                min_side=300,
                max_side=900,
            )
            self.assertEqual(len(variants), limit)
            for _, img in variants:
                self.assertEqual(img.mode, "RGB")
                self.assertLessEqual(max(img.size), 900)

    def test_tiny_image_does_not_crash(self):
        tiny = Image.new("RGB", (2, 2), (5, 5, 5))
        variants = build_variants(tiny, compute_stats(tiny), max_variants=3)
        self.assertGreaterEqual(len(variants), 1)


if __name__ == "__main__":
    unittest.main()
