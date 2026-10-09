"""Image analysis and test-time-augmentation variants.

Only Pillow and NumPy are used (the mandated base image ships no OpenCV).

``build_variants`` inspects the image and proposes a short, ordered list of
enhanced copies. The first variant is always the untouched (resized) image;
the rest target whatever looks wrong: darkness, glare, sensor noise or blur.
Each variant gives the model another independent look, and the pipeline votes
across them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter, ImageOps

_STATS_SIDE = 512


@dataclass(frozen=True)
class ImageStats:
    mean: float  # average luminance, 0-255
    std: float  # luminance contrast
    saturated_frac: float  # share of near-white pixels (glare)
    noise: float  # estimated noise sigma (Immerkaer)
    sharpness: float  # variance of the Laplacian (low = blurry)


def _gray_thumbnail(img: Image.Image) -> np.ndarray:
    thumb = img.convert("L")
    longest = max(thumb.size)
    if longest > _STATS_SIDE:
        scale = _STATS_SIDE / longest
        thumb = thumb.resize(
            (max(3, round(thumb.width * scale)), max(3, round(thumb.height * scale))),
            Image.BILINEAR,
        )
    return np.asarray(thumb, dtype=np.float32)


def _native_center_crop(img: Image.Image) -> np.ndarray:
    """Grayscale centre crop at native resolution (downscaling would average the noise away)."""
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = gray.shape
    top, left = max(0, (h - _STATS_SIDE) // 2), max(0, (w - _STATS_SIDE) // 2)
    return gray[top : top + _STATS_SIDE, left : left + _STATS_SIDE]


def _estimate_noise(g: np.ndarray) -> float:
    """Immerkaer's fast noise-sigma estimate."""
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    kernel = (
        g[:-2, :-2]
        - 2 * g[:-2, 1:-1]
        + g[:-2, 2:]
        - 2 * g[1:-1, :-2]
        + 4 * g[1:-1, 1:-1]
        - 2 * g[1:-1, 2:]
        + g[2:, :-2]
        - 2 * g[2:, 1:-1]
        + g[2:, 2:]
    )
    return math.sqrt(math.pi / 2.0) * float(np.abs(kernel).mean()) / 6.0


def compute_stats(img: Image.Image) -> ImageStats:
    g = _gray_thumbnail(img)
    noise = _estimate_noise(_native_center_crop(img))
    if g.shape[0] < 3 or g.shape[1] < 3:
        return ImageStats(float(g.mean()), float(g.std()), 0.0, noise, 1e6)
    laplacian = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    return ImageStats(
        mean=float(g.mean()),
        std=float(g.std()),
        saturated_frac=float((g >= 245).mean()),
        noise=noise,
        sharpness=float(laplacian.var()),
    )


def fit_size(img: Image.Image, min_side: int, max_side: int) -> Image.Image:
    """Scale so the long side <= ``max_side`` and (if possible) short side >= ``min_side``."""
    w, h = img.size
    longest, shortest = max(w, h), min(w, h)
    scale = 1.0
    if longest > max_side:
        scale = max_side / longest
    elif shortest < min_side:
        scale = min(min_side / shortest, max_side / longest, 4.0)
    if abs(scale - 1.0) < 1e-3:
        return img
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return img.resize(size, Image.LANCZOS if scale < 1.0 else Image.BICUBIC)


def _gamma(img: Image.Image, gamma: float) -> Image.Image:
    lut = [round(255.0 * ((i / 255.0) ** gamma)) for i in range(256)]
    return img.point(lut * len(img.getbands()))


def _lift(img: Image.Image) -> Image.Image:
    """Brighten an under-exposed image."""
    return ImageOps.autocontrast(_gamma(img, 0.55), cutoff=1)


def _deglare(img: Image.Image) -> Image.Image:
    """Pull highlights down so glare stops washing out characters."""
    return ImageOps.autocontrast(_gamma(img, 1.7), cutoff=0.5)


def _denoise(img: Image.Image) -> Image.Image:
    cleaned = img.filter(ImageFilter.MedianFilter(3))
    return ImageOps.autocontrast(cleaned, cutoff=0.5).filter(
        ImageFilter.UnsharpMask(radius=1.2, percent=110, threshold=3)
    )


def _sharpen(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.UnsharpMask(radius=2.0, percent=220, threshold=2))


def _contrast(img: Image.Image) -> Image.Image:
    return ImageOps.autocontrast(img, cutoff=2)


def _zoom(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.resize((round(w * 1.5), round(h * 1.5)), Image.BICUBIC)


def build_variants(
    img: Image.Image,
    stats: ImageStats,
    *,
    max_variants: int = 3,
    min_side: int = 448,
    max_side: int = 1792,
) -> list[tuple[str, Image.Image]]:
    """Return ``[(name, image), ...]`` with the plain image first."""
    plan: list[tuple[str, object]] = [("base", None)]
    if stats.mean < 75:
        plan.append(("lift", _lift))
    if stats.saturated_frac > 0.04:
        plan.append(("deglare", _deglare))
    if stats.noise > 7.0:
        plan.append(("denoise", _denoise))
    if stats.sharpness < 60.0 and stats.noise <= 7.0:
        plan.append(("sharpen", _sharpen))
    plan.append(("contrast", _contrast))
    plan.append(("zoom", _zoom))

    variants: list[tuple[str, Image.Image]] = []
    seen: set[str] = set()
    for name, op in plan:
        if name in seen:
            continue
        seen.add(name)
        enhanced = img if op is None else op(img)  # type: ignore[operator]
        variants.append((name, fit_size(enhanced, min_side, max_side)))
        if len(variants) >= max(1, max_variants):
            break
    return variants
