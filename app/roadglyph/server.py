"""Resident OCR server.

The grader launches a *new Python process for every image*. Loading an
8-billion-parameter model each time would blow the 30 s per-image budget, so
the container's ``CMD`` starts this server once. It loads the model, warms it
up and then answers tiny requests from ``app.py`` over a Unix socket.

Protocol: one JSON object per line, one request per connection.

    {"op": "ping"}                                   -> {"ok": true, "ready": bool}
    {"op": "predict", "image": PATH, "deadline_s": N} -> {"ok": true, "text": ..., ...}
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import socketserver
import threading
from collections.abc import Callable

from .backends import create_backend
from .backends.base import VisionBackend
from .config import Settings
from .imaging import ImageLoadError, load_image
from .pipeline import OCRPipeline
from .timing import Deadline

log = logging.getLogger("roadglyph.server")

BackendFactory = Callable[[Settings], VisionBackend]
_MAX_REQUEST_BYTES = 1 << 20


class OCRService:
    """Holds the model and turns request dicts into response dicts."""

    def __init__(self, settings: Settings, backend_factory: BackendFactory | None = None) -> None:
        self._settings = settings
        self._factory = backend_factory or create_backend
        self._lock = threading.Lock()  # one GPU job at a time
        self._pipeline: OCRPipeline | None = None
        self.ready = threading.Event()

    def load(self) -> None:
        backend = self._factory(self._settings)
        backend.load()
        backend.warmup()
        self._pipeline = OCRPipeline(backend, self._settings)
        self.ready.set()

    def handle(self, request: dict) -> dict:
        op = request.get("op")
        if op == "ping":
            return {"ok": True, "ready": self.ready.is_set()}
        if op == "predict":
            return self._predict(request)
        return {"ok": False, "error": f"unknown op: {op!r}"}

    def _predict(self, request: dict) -> dict:
        try:
            budget = float(request.get("deadline_s", self._settings.request_budget_s))
        except (TypeError, ValueError):
            budget = self._settings.request_budget_s
        deadline = Deadline(max(1.0, budget - 1.0))  # leave time to send the reply

        if not self.ready.wait(timeout=max(0.0, deadline.remaining() - 1.0)):
            return {"ok": False, "error": "model not ready yet"}
        path = request.get("image")
        if not isinstance(path, str) or not path:
            return {"ok": False, "error": "missing 'image' path"}
        try:
            image = load_image(path)
        except ImageLoadError as exc:
            return {"ok": False, "error": str(exc)}
        if not self._lock.acquire(timeout=max(0.0, deadline.remaining())):
            return {"ok": False, "error": "busy"}
        try:
            assert self._pipeline is not None
            prediction = self._pipeline.predict(image, deadline)
        finally:
            self._lock.release()
        return {"ok": True, **prediction.to_dict()}


def _make_handler(service: OCRService) -> type[socketserver.StreamRequestHandler]:
    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            try:
                line = self.rfile.readline(_MAX_REQUEST_BYTES)
                response = service.handle(json.loads(line))
            except Exception as exc:  # noqa: BLE001 - never kill the server on one bad request
                log.exception("request failed")
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            payload = (json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8")
            with contextlib.suppress(OSError):  # client gave up (deadline); nothing to do
                self.wfile.write(payload)

    return Handler


class _UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class OCRServer:
    """Unix-socket front end around :class:`OCRService`."""

    def __init__(self, settings: Settings, backend_factory: BackendFactory | None = None) -> None:
        self.settings = settings
        self.service = OCRService(settings, backend_factory)
        self._server: _UnixServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Bind the socket and start accepting (requests wait for the model)."""
        path = self.settings.socket_path
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)
        self._server = _UnixServer(path, _make_handler(self.service))
        os.chmod(path, 0o666)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="roadglyph-accept", daemon=True
        )
        self._thread.start()
        log.info("listening on %s", path)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.settings.socket_path)


def run_server(settings: Settings, backend_factory: BackendFactory | None = None) -> int:
    """Entry point for ``app.py --serve`` (the container's CMD)."""
    server = OCRServer(settings, backend_factory)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    server.start()
    try:
        server.service.load()
    except Exception:  # noqa: BLE001
        log.exception("model failed to load; exiting")
        server.stop()
        return 1
    log.info("ready")
    stop.wait()
    server.stop()
    return 0
