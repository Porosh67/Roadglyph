"""Robust image loading.

The grader sends PNG, JPEG and TIFF files of arbitrary size. TIFFs in
particular come in many flavours (16-bit, multi-frame, alpha, CMYK), so
everything is funnelled into a plain 8-bit RGB image here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# Guard against decompression bombs without rejecting legitimately large photos.
Image.MAX_IMAGE_PIXELS = 300_000_000

_HIGH_DEPTH_MODES = frozenset({"I", "F", "I;16", "I;16L", "I;16B", "I;16N"})


class ImageLoadError(RuntimeError):
    """The file could not be read as an image."""


def _stretch_to_uint8(img: Image.Image) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return Image.new("RGB", img.size, (int(np.clip(lo, 0, 255)),) * 3)
    scaled = ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return Image.fromarray(scaled, mode="L").convert("RGB")


def _flatten_alpha(img: Image.Image) -> Image.Image:
    rgba = img.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    canvas.alpha_composite(rgba)
    return canvas.convert("RGB")


def to_rgb(img: Image.Image) -> Image.Image:
    """Convert any Pillow image to 8-bit RGB."""
    if img.mode in _HIGH_DEPTH_MODES:
        return _stretch_to_uint8(img)
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        return _flatten_alpha(img)
    return img.convert("RGB")


def load_image(path: str | Path) -> Image.Image:
    """Load ``path`` as an upright, 8-bit RGB image (first frame only)."""
    try:
        with Image.open(path) as opened:
            if getattr(opened, "n_frames", 1) > 1:
                opened.seek(0)
            opened.load()
            try:
                upright = ImageOps.exif_transpose(opened) or opened
            except Exception:  # noqa: BLE001 - broken EXIF must not lose the image
                upright = opened
            rgb = to_rgb(upright)
    except FileNotFoundError as exc:
        raise ImageLoadError(f"image not found: {path}") from exc
    except Exception as exc:  # noqa: BLE001 - Pillow raises many exception types
        raise ImageLoadError(f"cannot read image {path}: {exc}") from exc
    if rgb.width < 1 or rgb.height < 1:
        raise ImageLoadError(f"image has no pixels: {path}")
    return rgb
