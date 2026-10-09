#!/usr/bin/env python3
"""Generate synthetic plate/sign images (PNG, JPEG, TIFF) plus labels.csv.

python scripts/make_synthetic_dataset.py --out samples/synthetic --count 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from roadglyph.synthetic import generate_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("samples/synthetic"))
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    samples = generate_dataset(args.out, args.count, args.seed)
    print(f"wrote {len(samples)} images and labels.csv to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
