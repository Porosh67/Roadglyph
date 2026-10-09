"""Writing the answer file the grader reads.

The file is written *early* (an empty provisional answer) and replaced
atomically once a real answer exists, so a valid JSON file is on disk at every
moment, even if the process is killed.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
from pathlib import Path

log = logging.getLogger("roadglyph.output")


def output_path_for(input_image: str | Path, output_dir: str | Path) -> Path:
    """``image_01.png`` -> ``<output_dir>/image_01_output.json``."""
    return Path(output_dir) / f"{Path(input_image).stem}_output.json"


def _sanitize(text: object, confidence: object) -> dict:
    clean = " ".join(str(text if text is not None else "").split())
    try:
        conf = float(confidence)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        conf = 0.0
    return {"text": clean, "confidence": round(min(1.0, max(0.0, conf)), 4)}


class ResultWriter:
    """Thread-safe, first-final-write-wins writer for one answer file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._final = False

    def _atomic_write(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise

    def write_provisional(self) -> None:
        """Put an empty but valid answer on disk (never overrides a final one)."""
        with self._lock:
            if self._final:
                return
            try:
                self._atomic_write(_sanitize("", 0.0))
            except OSError as exc:
                log.error("cannot write provisional answer to %s: %s", self.path, exc)

    def finalize(self, text: object, confidence: object) -> bool:
        """Write the real answer. Returns False if a final answer already exists."""
        with self._lock:
            if self._final:
                return False
            self._final = True
            try:
                self._atomic_write(_sanitize(text, confidence))
            except OSError as exc:
                log.error("cannot write answer to %s: %s", self.path, exc)
                return False
            return True
