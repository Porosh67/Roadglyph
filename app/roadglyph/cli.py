"""Command line interface.

python3 /app/app.py --serve                       # container CMD: resident server
python3 /app/app.py --input-image /app/input/x.png  # what the grader runs per image
python3 /app/app.py --healthcheck                 # Docker HEALTHCHECK
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

from . import __version__, client
from .backends import create_backend
from .backends.base import VisionBackend
from .config import BACKENDS, MODES, Settings
from .imaging import ImageLoadError, load_image
from .output import ResultWriter, output_path_for
from .pipeline import OCRPipeline
from .server import run_server
from .timing import Deadline, Watchdog

log = logging.getLogger("roadglyph.cli")

BackendFactory = Callable[[Settings], VisionBackend]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roadglyph",
        description="Read text from licence plates and road signs.",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--input-image", type=Path, help="image to read (PNG, JPEG or TIFF)")
    action.add_argument("--serve", action="store_true", help="run the resident model server")
    action.add_argument(
        "--healthcheck", action="store_true", help="exit 0 once the server is ready"
    )
    parser.add_argument("--output-dir", type=Path, help="where to write <name>_output.json")
    parser.add_argument("--mode", choices=MODES, help="how to obtain the prediction")
    parser.add_argument("--backend", choices=BACKENDS, help="model backend")
    parser.add_argument("--deadline", type=float, help="seconds before a fallback answer is used")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("--version", action="version", version=f"roadglyph {__version__}")
    return parser


def _apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    changes: dict = {}
    if args.output_dir is not None:
        changes["output_dir"] = str(args.output_dir)
    if args.mode:
        changes["mode"] = args.mode
    if args.backend:
        changes["backend"] = args.backend
    if args.deadline is not None:
        changes["client_deadline_s"] = args.deadline
    if args.verbose:
        changes["log_level"] = "DEBUG"
    return dataclasses.replace(settings, **changes) if changes else settings


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _predict_in_process(
    image_path: Path,
    settings: Settings,
    deadline: Deadline,
    backend_factory: BackendFactory | None,
) -> dict:
    """Load the model in this process (slow: for development and fallback)."""
    image = load_image(image_path)
    backend = (backend_factory or create_backend)(settings)
    backend.load()
    return OCRPipeline(backend, settings).predict(image, deadline).to_dict()


def _predict_via_server(image_path: Path, settings: Settings, deadline: Deadline) -> dict:
    if not client.wait_for_socket(settings.socket_path, deadline):
        raise client.ServerUnavailable("server socket never appeared")
    return client.predict(settings.socket_path, str(image_path), deadline)


def _obtain_result(
    image_path: Path,
    settings: Settings,
    deadline: Deadline,
    backend_factory: BackendFactory | None,
) -> dict:
    mode = settings.mode
    if mode == "auto":
        reply = client.ping(settings.socket_path) if os.path.exists(settings.socket_path) else None
        mode = "server" if reply is not None else "standalone"
        log.debug("auto mode resolved to %s", mode)
    if mode == "server":
        return _predict_via_server(image_path, settings, deadline)
    return _predict_in_process(image_path, settings, deadline, backend_factory)


def _write_trace(settings: Settings, image_path: Path, result: dict) -> None:
    if not settings.trace_dir:
        return
    try:
        target = Path(settings.trace_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{image_path.stem}_trace.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        log.warning("could not write trace: %s", exc)


def run_inference(
    image_path: Path,
    settings: Settings,
    *,
    started_at: float | None = None,
    backend_factory: BackendFactory | None = None,
    exit_fn: Callable[[int], None] = os._exit,
) -> int:
    """Read one image and write ``<name>_output.json``. Always exits 0 with a valid file."""
    started = started_at if started_at is not None else time.monotonic()
    deadline = Deadline.from_start(started, settings.client_deadline_s)
    writer = ResultWriter(output_path_for(image_path, settings.output_dir))
    writer.write_provisional()

    def on_expire() -> None:
        log.error("hard deadline reached; answering with what we have")
        writer.finalize("", 0.0)
        exit_fn(0)

    watchdog = Watchdog(deadline.remaining(), on_expire)
    watchdog.start()
    try:
        result = _obtain_result(image_path, settings, deadline, backend_factory)
        _write_trace(settings, image_path, result)
        writer.finalize(result.get("text", ""), result.get("confidence", 0.0))
    except (client.ServerUnavailable, ImageLoadError) as exc:
        log.error("no prediction for %s: %s", image_path, exc)
        writer.finalize("", 0.0)
    except Exception:  # noqa: BLE001 - the harness must always get a valid file
        log.exception("unexpected failure for %s", image_path)
        writer.finalize("", 0.0)
    finally:
        watchdog.cancel()
    return 0


def main(
    argv: list[str] | None = None,
    *,
    started_at: float | None = None,
    backend_factory: BackendFactory | None = None,
    exit_fn: Callable[[int], None] = os._exit,
    settings: Settings | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    settings = _apply_overrides(settings or Settings.from_env(), args)
    _configure_logging(settings.log_level)

    if args.serve:
        return run_server(settings, backend_factory)
    if args.healthcheck:
        reply = client.ping(settings.socket_path)
        return 0 if reply and reply.get("ready") else 1
    return run_inference(
        args.input_image,
        settings,
        started_at=started_at,
        backend_factory=backend_factory,
        exit_fn=exit_fn,
    )
