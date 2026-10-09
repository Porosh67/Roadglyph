import unittest

import _path  # noqa: F401
from roadglyph.normalize import matches, normalize


class NormalizeTests(unittest.TestCase):
    def test_brief_examples_that_must_pass(self):
        for answer in ("7ABC123", "7abc123", "7-ABC-123", "7 ABC 123"):
            with self.subTest(answer=answer):
                self.assertTrue(matches(answer, "7ABC123"))

    def test_brief_examples_that_must_fail(self):
        for answer in ("7ABC128", "CALIFORNIA 7ABC123", "PLATE: 7ABC123"):
            with self.subTest(answer=answer):
                self.assertFalse(matches(answer, "7ABC123"))

    def test_chinese_plate_keeps_province_and_ignores_interpunct(self):
        self.assertTrue(matches("京A12345", "京 A·12345"))
        self.assertTrue(matches("京 A·12345", "京A12345"))
        self.assertFalse(matches("12345", "京 A·12345"))

    def test_strips_all_four_separator_characters(self):
        self.assertEqual(normalize("a-b.c·d_e f"), "ABCDEF")

    def test_empty(self):
        self.assertEqual(normalize(""), "")


if __name__ == "__main__":
    unittest.main()
