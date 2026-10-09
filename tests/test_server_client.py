import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import _path  # noqa: F401
from _util import Scripted
from roadglyph import client, prompts
from roadglyph.backends.mock import MockBackend
from roadglyph.cli import main, run_inference
from roadglyph.config import Settings
from roadglyph.server import OCRServer
from roadglyph.synthetic import draw_plate
from roadglyph.timing import Deadline


class ServerClientTests(unittest.TestCase):
    def setUp(self):
        # Unix socket paths are limited to ~100 chars, so keep the directory short.
        self.dir = Path(tempfile.mkdtemp(prefix="rg-", dir="/tmp"))
        self.sock = str(self.dir / "s.sock")
        self.image = self.dir / "image_01.png"
        draw_plate("7ABC123").save(self.image)
        self.settings = Settings(
            backend="mock",
            mode="server",
            socket_path=self.sock,
            output_dir=str(self.dir / "out"),
            client_deadline_s=10.0,
        )
        responder = Scripted(prompts.PLATE, ["CALIFORNIA 7ABC123", "7ABC123"])
        self.server = OCRServer(self.settings, lambda s: MockBackend(responder))
        self.server.start()

    def tearDown(self):
        self.server.stop()
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_ping_reports_loading_then_ready(self):
        self.assertEqual(client.ping(self.sock), {"ok": True, "ready": False})
        self.server.service.load()
        self.assertEqual(client.ping(self.sock), {"ok": True, "ready": True})

    def test_predict_roundtrip(self):
        self.server.service.load()
        reply = client.predict(self.sock, str(self.image), Deadline(10.0))
        self.assertEqual(reply["text"], "7ABC123")
        self.assertEqual(reply["category"], prompts.PLATE)

    def test_request_during_loading_waits_then_succeeds(self):
        import threading
        import time

        threading.Timer(0.5, self.server.service.load).start()
        started = time.monotonic()
        reply = client.predict(self.sock, str(self.image), Deadline(10.0))
        self.assertEqual(reply["text"], "7ABC123")
        self.assertGreaterEqual(time.monotonic() - started, 0.4)

    def test_model_never_ready_gives_a_clean_error(self):
        with self.assertRaises(client.ServerUnavailable) as ctx:
            client.predict(self.sock, str(self.image), Deadline(2.5))
        self.assertIn("not ready", str(ctx.exception))

    def test_bad_requests_do_not_kill_the_server(self):
        self.server.service.load()
        with self.assertRaises(client.ServerUnavailable):
            client.predict(self.sock, str(self.dir / "missing.png"), Deadline(5.0))
        self.assertEqual(self.server.service.handle({"op": "nope"})["ok"], False)
        self.assertEqual(self.server.service.handle({"op": "predict"})["ok"], False)
        self.assertTrue(client.ping(self.sock)["ready"])

    def test_cli_server_mode_end_to_end(self):
        self.server.service.load()
        run_inference(self.image, self.settings, exit_fn=lambda c: None)
        data = json.loads((self.dir / "out" / "image_01_output.json").read_text(encoding="utf-8"))
        self.assertEqual(data["text"], "7ABC123")

    def test_cli_server_mode_without_server_degrades_to_empty_answer(self):
        self.server.stop()
        settings = Settings(**{**self.settings.__dict__, "client_deadline_s": 1.0})
        run_inference(self.image, settings, exit_fn=lambda c: None)
        data = json.loads((self.dir / "out" / "image_01_output.json").read_text(encoding="utf-8"))
        self.assertEqual(data, {"text": "", "confidence": 0.0})

    def test_healthcheck_exit_codes(self):
        base = ["--healthcheck"]
        self.assertEqual(main(base, settings=self.settings), 1)  # still loading
        self.server.service.load()
        self.assertEqual(main(base, settings=self.settings), 0)

    def test_stop_removes_the_socket_file(self):
        self.server.stop()
        self.assertFalse(os.path.exists(self.sock))


if __name__ == "__main__":
    unittest.main()
