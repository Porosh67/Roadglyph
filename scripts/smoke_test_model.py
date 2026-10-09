#!/usr/bin/env python3
"""Run this FIRST on the AMD notebook, before building any Docker image.

It answers the three questions that decide whether the whole plan works:

  1. Does PyTorch see the GPU, and is it a ROCm build?
  2. Do ``transformers`` and the Qwen3-VL processor import on this Python/torch?
  3. Can the model read one synthetic plate, and how long does it take?

    pip install -r app/requirements.txt
    python scripts/smoke_test_model.py --model Qwen/Qwen3-VL-8B-Instruct
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct", help="HF id or local path")
    args = parser.parse_args()

    import torch

    print(f"python {sys.version.split()[0]}  torch {torch.__version__}  hip={torch.version.hip}")
    print(f"gpu visible: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: no GPU visible; this notebook/container cannot run the model")
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}")

    try:
        import transformers

        print(f"transformers {transformers.__version__}")
    except ImportError as exc:
        print(f"FAIL: transformers missing ({exc}); run: pip install -r app/requirements.txt")
        return 1

    from roadglyph.backends.qwen import QwenVLBackend
    from roadglyph.config import Settings
    from roadglyph.pipeline import OCRPipeline
    from roadglyph.synthetic import draw_plate
    from roadglyph.timing import Deadline

    settings = Settings(model_path=args.model)
    backend = QwenVLBackend(settings)
    started = time.monotonic()
    try:
        backend.load()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL while loading the model: {type(exc).__name__}: {exc}")
        if "torchvision" in str(exc).lower():
            print("hint: see README > Known risks > torchvision")
        return 1
    print(f"model loaded in {time.monotonic() - started:.1f}s")
    backend.warmup()

    started = time.monotonic()
    prediction = OCRPipeline(backend, settings).predict(draw_plate("7ABC123"), Deadline(28.0))
    print(
        f"read {prediction.text!r} (category {prediction.category}) "
        f"in {time.monotonic() - started:.1f}s; peak VRAM "
        f"{torch.cuda.max_memory_allocated() / 1024**3:.1f} GiB"
    )
    print("OK" if prediction.text else "WARN: empty answer, inspect prediction.notes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
