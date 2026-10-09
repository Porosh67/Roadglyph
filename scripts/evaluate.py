#!/usr/bin/env python3
"""Score the pipeline on a folder of images exactly like the grader does.

    python scripts/evaluate.py --images samples/synthetic            # uses labels.csv
    python scripts/evaluate.py --images my_samples --labels my.csv   # filename,expected

Prints a pass/fail table and a score out of 200 (10 images x 20 points, scaled).
Runs the model in-process, so run it on the GPU notebook, not on a laptop.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from roadglyph.backends import create_backend  # noqa: E402
from roadglyph.config import Settings  # noqa: E402
from roadglyph.imaging import load_image  # noqa: E402
from roadglyph.normalize import matches  # noqa: E402
from roadglyph.pipeline import OCRPipeline  # noqa: E402
from roadglyph.timing import Deadline  # noqa: E402


def read_labels(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [(row["filename"], row["expected"]) for row in reader]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument(
        "--labels", type=Path, help="CSV with filename,expected (default: IMAGES/labels.csv)"
    )
    parser.add_argument("--backend", choices=("qwen", "mock"), default=None)
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.backend:
        settings = Settings(**{**settings.__dict__, "backend": args.backend})
    labels = read_labels(args.labels or args.images / "labels.csv")

    backend = create_backend(settings)
    print(f"loading backend {settings.backend} ...", flush=True)
    backend.load()
    backend.warmup()
    pipeline = OCRPipeline(backend, settings)

    passed, slowest = 0, 0.0
    print(f"{'file':44} {'expected':18} {'got':18} {'cat':16} {'s':>5}  result")
    for filename, expected in labels:
        started = time.monotonic()
        prediction = pipeline.predict(load_image(args.images / filename), Deadline(28.0))
        elapsed = time.monotonic() - started
        slowest = max(slowest, elapsed)
        ok = matches(prediction.text, expected)
        passed += ok
        print(
            f"{filename[:44]:44} {expected[:18]:18} {prediction.text[:18]:18} "
            f"{prediction.category[:16]:16} {elapsed:5.1f}  {'PASS' if ok else 'FAIL'}"
        )
    total = len(labels)
    score = passed / total * 200 if total else 0.0
    print(
        f"\n{passed}/{total} passed -> {score:.0f}/200 (slowest image: {slowest:.1f}s, limit 30s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
