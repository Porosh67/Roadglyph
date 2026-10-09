"""Synthetic test images (plates, signs) with realistic degradations.

Used by the unit tests, the local evaluation script and the pre-submission
checker. These are *not* a substitute for the real samples: they exist so the
whole container contract can be exercised without any external files.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


@dataclass(frozen=True)
class Sample:
    filename: str
    expected: str
    category: str
    degradation: str


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def _centered(draw: ImageDraw.ImageDraw, box, text: str, size: int, fill) -> None:
    font = _font(size)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    x0, y0, x1, y1 = box
    draw.text(
        (x0 + (x1 - x0 - (right - left)) / 2 - left, y0 + (y1 - y0 - (bottom - top)) / 2 - top),
        text,
        font=font,
        fill=fill,
    )


def draw_plate(number: str, banner: str = "CALIFORNIA") -> Image.Image:
    img = Image.new("RGB", (640, 340), (60, 70, 80))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(
        (20, 20, 620, 320), radius=24, fill=(245, 245, 240), outline=(20, 20, 90), width=8
    )
    _centered(d, (40, 30, 600, 100), banner, 44, (20, 40, 140))
    _centered(d, (40, 110, 600, 280), number, 120, (15, 15, 60))
    return img


def draw_stop() -> Image.Image:
    img = Image.new("RGB", (480, 480), (120, 150, 120))
    d = ImageDraw.Draw(img)
    cx = cy = 240
    r = 220
    import math

    pts = [
        (
            cx + r * math.cos(math.radians(22.5 + 45 * i)),
            cy + r * math.sin(math.radians(22.5 + 45 * i)),
        )
        for i in range(8)
    ]
    d.polygon(pts, fill=(255, 255, 255))
    pts_inner = [
        (
            cx + (r - 14) * math.cos(math.radians(22.5 + 45 * i)),
            cy + (r - 14) * math.sin(math.radians(22.5 + 45 * i)),
        )
        for i in range(8)
    ]
    d.polygon(pts_inner, fill=(190, 20, 30))
    _centered(d, (40, 170, 440, 310), "STOP", 130, (255, 255, 255))
    return img


def draw_speed_limit(speed: int) -> Image.Image:
    img = Image.new("RGB", (360, 480), (110, 140, 160))
    d = ImageDraw.Draw(img)
    d.rectangle((30, 30, 330, 450), fill=(250, 250, 250), outline=(10, 10, 10), width=8)
    _centered(d, (40, 50, 320, 130), "SPEED", 70, (10, 10, 10))
    _centered(d, (40, 130, 320, 210), "LIMIT", 70, (10, 10, 10))
    _centered(d, (40, 230, 320, 430), str(speed), 170, (10, 10, 10))
    return img


def draw_work_zone() -> Image.Image:
    img = Image.new("RGB", (480, 480), (100, 100, 100))
    d = ImageDraw.Draw(img)
    d.rectangle((30, 30, 450, 450), fill=(255, 130, 0), outline=(0, 0, 0), width=8)
    for i, word in enumerate(("ROAD", "WORK", "AHEAD")):
        _centered(d, (40, 50 + i * 130, 440, 170 + i * 130), word, 90, (0, 0, 0))
    return img


def draw_plaque(number: int) -> Image.Image:
    img = Image.new("RGB", (300, 260), (120, 140, 110))
    d = ImageDraw.Draw(img)
    d.rectangle((20, 20, 280, 240), fill=(255, 215, 0), outline=(0, 0, 0), width=6)
    _centered(d, (30, 30, 270, 230), str(number), 140, (0, 0, 0))
    return img


# ------------------------------------------------------------------ degradations
def add_noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + rng.normal(0.0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def low_light(img: Image.Image, factor: float = 0.28) -> Image.Image:
    return img.point([round(i * factor) for i in range(256)] * 3)


def add_glare(img: Image.Image) -> Image.Image:
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    blob = np.exp(-(((xx - w * 0.65) / (w * 0.22)) ** 2 + ((yy - h * 0.4) / (h * 0.25)) ** 2))
    arr = np.asarray(img, dtype=np.float32) + blob[..., None] * 230.0
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def perspective(img: Image.Image, strength: float = 0.12) -> Image.Image:
    w, h = img.size
    dx, dy = w * strength, h * strength
    quad = (dx, dy * 0.4, 0, h - dy * 0.4, w, h, w - dx, 0)
    return img.transform((w, h), Image.QUAD, quad, Image.BICUBIC, fillcolor=(40, 40, 40))


def degrade(img: Image.Image, kind: str, rng: np.random.Generator) -> Image.Image:
    if kind == "clean":
        return img
    if kind == "noise":
        return add_noise(img, 28.0, rng)
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(3.5))
    if kind == "lowlight":
        return add_noise(low_light(img), 6.0, rng)
    if kind == "glare":
        return add_glare(img)
    if kind == "angle":
        return perspective(img)
    raise ValueError(f"unknown degradation: {kind}")


DEGRADATIONS = ("clean", "noise", "blur", "lowlight", "glare", "angle")
_FORMATS = (("png", "PNG"), ("jpg", "JPEG"), ("tiff", "TIFF"))
_PLATES = ("7ABC123", "5XYZ891", "JHT 2951", "8KLM402")


def generate_dataset(out_dir: str | Path, count: int = 12, seed: int = 7) -> list[Sample]:
    """Write ``count`` images plus ``labels.csv`` into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    makers = (
        ("plate", lambda: (lambda n: (draw_plate(n), n))(py_rng.choice(_PLATES))),
        ("stop", lambda: (draw_stop(), "STOP")),
        (
            "speed_limit",
            lambda: (lambda s: (draw_speed_limit(s), f"SPEED LIMIT {s}"))(
                py_rng.choice((25, 35, 45, 65))
            ),
        ),
        ("warning", lambda: (draw_work_zone(), "ROAD WORK AHEAD")),
        ("plaque", lambda: (lambda n: (draw_plaque(n), str(n)))(py_rng.choice((15, 25, 35, 45)))),
    )
    samples: list[Sample] = []
    for index in range(count):
        category, make = makers[index % len(makers)]
        kind = DEGRADATIONS[(index // len(makers) + index) % len(DEGRADATIONS)]
        ext, pil_format = _FORMATS[index % len(_FORMATS)]
        base, expected = make()
        img = degrade(base, kind, rng)
        name = f"synthetic_{index + 1:02d}_{category}_{kind}.{ext}"
        img.save(out / name, format=pil_format)
        samples.append(Sample(name, expected, category, kind))

    with (out / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "expected", "category", "degradation"])
        for s in samples:
            writer.writerow([s.filename, s.expected, s.category, s.degradation])
    return samples
