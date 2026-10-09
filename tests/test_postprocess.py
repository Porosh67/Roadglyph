import unittest

import _path  # noqa: F401
from roadglyph import prompts
from roadglyph.normalize import matches
from roadglyph.postprocess import postprocess

# (category, raw model output, expected answer from the brief)
CASES = [
    (prompts.PLATE, "7ABC123", "7ABC123"),
    (prompts.PLATE, "CALIFORNIA 7ABC123", "7ABC123"),
    (prompts.PLATE, "California\n7ABC123\nThe Golden State", "7ABC123"),
    (prompts.PLATE, "京A·12345", "京 A·12345"),
    (prompts.PLATE, "京 A 12345", "京 A·12345"),
    (prompts.PLATE, "京A・12345", "京 A·12345"),
    (prompts.PLATE, "沪B·88888", "沪 B·88888"),
    (prompts.PLATE, "NEW YORK\nJHT 2951\nEMPIRE STATE", "JHT 2951"),
    (prompts.PLATE, "5XYZ891", "5XYZ891"),
    (prompts.PLATE, "The license plate number is: 7ABC123", "7ABC123"),
    (prompts.STOP, "STOP", "STOP"),
    (prompts.STOP, "Stop.", "STOP"),
    (prompts.STOP, "ST0P", "STOP"),
    (prompts.SPEED_LIMIT, "SPEED LIMIT 65", "SPEED LIMIT 65"),
    (prompts.SPEED_LIMIT, "SPEED\nLIMIT\n65", "SPEED LIMIT 65"),
    (prompts.SPEED_LIMIT, "65", "SPEED LIMIT 65"),
    (prompts.WARNING_SIGN, "ROAD\nWORK\nAHEAD", "ROAD WORK AHEAD"),
    (prompts.WARNING_SIGN, "```\nROAD WORK\nAHEAD\n```", "ROAD WORK AHEAD"),
    (prompts.ADVISORY_PLAQUE, "35", "35"),
    (prompts.ADVISORY_PLAQUE, "35 MPH", "35"),
    (prompts.ADVISORY_PLAQUE, "**35**", "35"),
]


class PostprocessTests(unittest.TestCase):
    def test_brief_conventions(self):
        for category, raw, expected in CASES:
            with self.subTest(category=category, raw=raw):
                self.assertTrue(
                    matches(postprocess(category, raw), expected),
                    f"{postprocess(category, raw)!r} != {expected!r}",
                )

    def test_empty_and_whitespace_give_empty_answer(self):
        for raw in ("", "   ", "\n\n", None):
            for category in prompts.CATEGORIES:
                with self.subTest(category=category, raw=raw):
                    self.assertEqual(postprocess(category, raw), "")

    def test_plate_with_only_a_banner_falls_back_to_the_text(self):
        self.assertTrue(postprocess(prompts.PLATE, "TEXAS"))

    def test_plate_prefers_the_number_over_a_slogan_line(self):
        out = postprocess(prompts.PLATE, "THE LONE STAR STATE\nTEXAS\nABC 1234")
        self.assertTrue(matches(out, "ABC1234"))

    def test_unknown_category_joins_lines(self):
        self.assertEqual(postprocess("OTHER", "A\nB\n C "), "A B C")


class ParseCategoryTests(unittest.TestCase):
    def test_exact_labels(self):
        for label in prompts.CATEGORIES:
            self.assertEqual(prompts.parse_category(label), label)

    def test_noisy_replies(self):
        self.assertEqual(prompts.parse_category("Plate."), prompts.PLATE)
        self.assertEqual(prompts.parse_category("**STOP**"), prompts.STOP)
        self.assertEqual(prompts.parse_category("speed limit"), prompts.SPEED_LIMIT)
        self.assertEqual(prompts.parse_category("advisory plaque"), prompts.ADVISORY_PLAQUE)
        self.assertEqual(prompts.parse_category("work zone"), prompts.WARNING_SIGN)
        self.assertEqual(prompts.parse_category(""), prompts.OTHER)
        self.assertEqual(prompts.parse_category("a banana"), prompts.OTHER)


if __name__ == "__main__":
    unittest.main()
