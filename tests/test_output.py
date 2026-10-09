import json
import tempfile
import unittest
from pathlib import Path

import _path  # noqa: F401
from roadglyph.output import ResultWriter, output_path_for


class OutputPathTests(unittest.TestCase):
    def test_replaces_extension_with_output_json(self):
        self.assertEqual(
            output_path_for("/app/input/image_01.png", "/app/output"),
            Path("/app/output/image_01_output.json"),
        )
        self.assertEqual(output_path_for("x/a.tiff", "out").name, "a_output.json")
        self.assertEqual(output_path_for("a.b.jpg", "out").name, "a.b_output.json")


class ResultWriterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "nested" / "img_output.json"

    def tearDown(self):
        self._tmp.cleanup()

    def read(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_provisional_then_final(self):
        writer = ResultWriter(self.path)
        writer.write_provisional()
        self.assertEqual(self.read(), {"text": "", "confidence": 0.0})
        self.assertTrue(writer.finalize("7ABC123", 0.94))
        self.assertEqual(self.read(), {"text": "7ABC123", "confidence": 0.94})

    def test_first_final_write_wins(self):
        writer = ResultWriter(self.path)
        self.assertTrue(writer.finalize("A", 0.5))
        self.assertFalse(writer.finalize("B", 0.9))
        writer.write_provisional()  # must not clobber the final answer
        self.assertEqual(self.read()["text"], "A")

    def test_text_is_single_line_and_confidence_is_clamped(self):
        writer = ResultWriter(self.path)
        writer.finalize("ROAD\n WORK   AHEAD", 7)
        self.assertEqual(self.read(), {"text": "ROAD WORK AHEAD", "confidence": 1.0})

    def test_chinese_characters_are_not_escaped(self):
        ResultWriter(self.path).finalize("京 A·12345", 0.9)
        self.assertIn("京", self.path.read_text(encoding="utf-8"))

    def test_garbage_inputs_never_raise(self):
        writer = ResultWriter(self.path)
        writer.finalize(None, "not a number")
        self.assertEqual(self.read(), {"text": "", "confidence": 0.0})

    def test_no_temp_files_left_behind(self):
        writer = ResultWriter(self.path)
        writer.write_provisional()
        writer.finalize("X", 0.5)
        self.assertEqual([p.name for p in self.path.parent.iterdir()], [self.path.name])


if __name__ == "__main__":
    unittest.main()
