import json
import tempfile
import time
import unittest
from pathlib import Path

import _path  # noqa: F401
from _util import Scripted
from roadglyph import prompts
from roadglyph.backends.mock import MockBackend
from roadglyph.cli import main, run_inference
from roadglyph.config import Settings
from roadglyph.synthetic import draw_plate


def factory(responder, delay_s=0.0):
    return lambda settings: MockBackend(responder, delay_s=delay_s)


class InferenceContractTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.image = self.dir / "image_01.png"
        draw_plate("7ABC123").save(self.image)
        self.out = self.dir / "out"
        self.settings = Settings(
            backend="mock", mode="standalone", output_dir=str(self.out), client_deadline_s=20.0
        )

    def tearDown(self):
        self._tmp.cleanup()

    def result(self, name="image_01_output.json"):
        return json.loads((self.out / name).read_text(encoding="utf-8"))

    def test_writes_exactly_the_documented_json(self):
        code = run_inference(
            self.image,
            self.settings,
            backend_factory=factory(Scripted(prompts.PLATE, ["CALIFORNIA 7ABC123", "7ABC123"])),
            exit_fn=lambda c: None,
        )
        self.assertEqual(code, 0)
        data = self.result()
        self.assertEqual(set(data), {"text", "confidence"})
        self.assertEqual(data["text"], "7ABC123")
        self.assertTrue(0.0 < data["confidence"] <= 1.0)

    def test_missing_image_still_produces_a_valid_empty_answer(self):
        code = run_inference(
            self.dir / "ghost.png",
            self.settings,
            backend_factory=factory(Scripted("OTHER", ["x"])),
            exit_fn=lambda c: None,
        )
        self.assertEqual(code, 0)
        self.assertEqual(self.result("ghost_output.json"), {"text": "", "confidence": 0.0})

    def test_corrupt_image_still_produces_a_valid_empty_answer(self):
        bad = self.dir / "bad.png"
        bad.write_bytes(b"garbage")
        run_inference(
            bad,
            self.settings,
            backend_factory=factory(Scripted("OTHER", ["x"])),
            exit_fn=lambda c: None,
        )
        self.assertEqual(self.result("bad_output.json"), {"text": "", "confidence": 0.0})

    def test_backend_crash_still_produces_a_valid_answer(self):
        class Exploding(MockBackend):
            def load(self):
                raise RuntimeError("GPU on fire")

        run_inference(
            self.image, self.settings, backend_factory=lambda s: Exploding(), exit_fn=lambda c: None
        )
        self.assertEqual(self.result(), {"text": "", "confidence": 0.0})

    def test_watchdog_answers_and_exits_before_a_slow_model_finishes(self):
        exits = []
        settings = Settings(
            backend="mock", mode="standalone", output_dir=str(self.out), client_deadline_s=0.4
        )
        started = time.monotonic()
        run_inference(
            self.image,
            settings,
            backend_factory=factory(Scripted(prompts.PLATE, ["7ABC123"]), delay_s=3.0),
            exit_fn=exits.append,
            started_at=started,
        )
        self.assertEqual(exits, [0])
        # The slow model eventually returned, but the watchdog's answer is the final one.
        self.assertEqual(self.result(), {"text": "", "confidence": 0.0})

    def test_main_parses_arguments_and_runs(self):
        code = main(
            [
                "--input-image",
                str(self.image),
                "--mode",
                "standalone",
                "--backend",
                "mock",
                "--output-dir",
                str(self.out),
            ],
            settings=Settings(socket_path=str(self.dir / "none.sock")),
            exit_fn=lambda c: None,
        )
        self.assertEqual(code, 0)
        self.assertEqual(self.result()["text"], "MOCK")

    def test_main_requires_an_action(self):
        with self.assertRaises(SystemExit):
            main([])


if __name__ == "__main__":
    unittest.main()
