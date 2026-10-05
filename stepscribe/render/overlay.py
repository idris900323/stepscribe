# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Label overlays with leader lines and simple collision avoidance."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

FONT_SIZE = 20
LEADER_RADIUS = (34, 62, 90)
PAD = 3
MAX_PNG_BYTES = 1_000_000


Box = tuple[float, float, float, float]
_TEXT_LOG: list[tuple[str, Box, tuple[int, int]]] | None = None


@contextmanager
def record_text_boxes() -> Iterator[list[tuple[str, Box, tuple[int, int]]]]:
    """Collect (text, box, canvas size) for every text drawn through :func:`draw_text`.

    Used by tests to prove nothing is clipped: each box must lie inside its canvas.
    """
    global _TEXT_LOG
    previous, _TEXT_LOG = _TEXT_LOG, []
    try:
        yield _TEXT_LOG
    finally:
        _TEXT_LOG = previous


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Box:
    """Draw *text* with its ink box top-left at *xy*, log the box, and return it."""
    tb = draw.textbbox((0, 0), text, font=font)
    x, y = xy
    box = (x, y, x + tb[2] - tb[0], y + tb[3] - tb[1])
    draw.text((x - tb[0], y - tb[1]), text, fill=fill, font=font)
    if _TEXT_LOG is not None:
        _TEXT_LOG.append((text, box, draw.im.size))
    return box


def text_size(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont | ImageFont.FreeTypeFont
) -> tuple[float, float]:
    """Width and height of the ink box of *text*."""
    tb = draw.textbbox((0, 0), text, font=font)
    return tb[2] - tb[0], tb[3] - tb[1]


def pad_canvas(
    image: Image.Image, left: int = 0, top: int = 0, right: int = 0, bottom: int = 0
) -> Image.Image:
    """White border added around *image* (the content shifts right / down by left / top)."""
    if not (left or top or right or bottom):
        return image
    out = Image.new(
        "RGB", (image.width + left + right, image.height + top + bottom), (255, 255, 255)
    )
    out.paste(image.convert("RGB"), (left, top))
    return out


@dataclass
class Label:
    """A text label anchored at pixel (x, y)."""

    text: str
    x: float
    y: float


def _font(size: int = FONT_SIZE) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Pillow's bundled default font (no system font dependency)."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # old Pillow without a size argument
        return ImageFont.load_default()


def _overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def place_labels(
    labels: list[Label],
    draw: ImageDraw.ImageDraw,
    size: tuple[int, int],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> list[tuple[Label, tuple[float, float, float, float]]]:
    """Choose a box for each label: try 8 directions per radius, first non-overlapping wins."""
    w, h = size
    placed: list[tuple[Label, tuple[float, float, float, float]]] = []
    boxes: list[tuple[float, float, float, float]] = []
    for lab in sorted(labels, key=lambda lb: (lb.y, lb.x, lb.text)):
        tb = draw.textbbox((0, 0), lab.text, font=font)
        tw, th = tb[2] - tb[0] + 2 * PAD, tb[3] - tb[1] + 2 * PAD
        choice: tuple[float, float, float, float] | None = None
        first: tuple[float, float, float, float] | None = None
        for radius in LEADER_RADIUS:
            for k in range(8):
                ang = math.radians(45 * k - 45)  # start up-right, go clockwise
                cx, cy = lab.x + radius * math.cos(ang), lab.y + radius * math.sin(ang)
                x0 = cx if math.cos(ang) >= -0.01 else cx - tw
                y0 = cy if math.sin(ang) >= -0.01 else cy - th
                box = (
                    min(max(x0, 0), w - tw),
                    min(max(y0, 0), h - th),
                    min(max(x0, 0), w - tw) + tw,
                    min(max(y0, 0), h - th) + th,
                )
                first = first or box
                if not any(_overlaps(box, b) for b in boxes):
                    choice = box
                    break
            if choice:
                break
        box = choice or first or (lab.x, lab.y, lab.x + tw, lab.y + th)
        boxes.append(box)
        placed.append((lab, box))
    return placed


def draw_labels(
    image: Image.Image, labels: list[Label], legend: list[str] | None = None
) -> Image.Image:
    """Draw anchor dots, leader lines, label boxes and an optional legend; returns a new image."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = _font()
    for lab, box in place_labels(labels, draw, out.size, font):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        draw.line([(lab.x, lab.y), (cx, cy)], fill=(0, 0, 0), width=2)
        draw.ellipse(
            [lab.x - 4, lab.y - 4, lab.x + 4, lab.y + 4],
            fill=(255, 255, 255),
            outline=(0, 0, 0),
            width=2,
        )
        draw.rectangle(box, fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        draw_text(draw, (box[0] + PAD, box[1] + PAD), lab.text, font)
    if legend:
        _draw_legend(draw, legend, font)
    return out


def _draw_legend(
    draw: ImageDraw.ImageDraw, lines: list[str], font: ImageFont.ImageFont | ImageFont.FreeTypeFont
) -> None:
    line_h = FONT_SIZE + 4
    width = max(int(draw.textlength(s, font=font)) for s in lines) + 16
    draw.rectangle(
        [8, 8, 8 + width, 8 + line_h * len(lines) + 10],
        fill=(255, 255, 255),
        outline=(0, 0, 0),
        width=2,
    )
    for i, s in enumerate(lines):
        draw_text(draw, (16, 12 + i * line_h), s, font)


def save_png(image: Image.Image, path: str) -> None:
    """Save a PNG under 1 MB, quantising to a palette only if needed."""
    import os

    image.save(path, optimize=True)
    if os.path.getsize(path) > MAX_PNG_BYTES:
        for colors in (128, 64, 32):
            image.convert("RGB").quantize(colors=colors).save(path, optimize=True)
            if os.path.getsize(path) <= MAX_PNG_BYTES:
                break
