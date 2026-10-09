#!/usr/bin/env python3
"""Harness entry point.

    python3 /app/app.py --input-image /app/input/image_01.png

Writes /app/output/image_01_output.json. See README.md for the full contract.
"""

import sys
import time

_STARTED_AT = time.monotonic()  # measured before heavy imports so the deadline is honest

from roadglyph.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], started_at=_STARTED_AT))
