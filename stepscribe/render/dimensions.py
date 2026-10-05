# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Dimension annotations for renders: overall size lines (orthographic views) and a size caption."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from stepscribe.assembly.spatial import frame
from stepscribe.describe.phrases import fmt
from stepscribe.geometry.occ_utils import Vec
from stepscribe.render.overlay import FONT_SIZE, _font, draw_text, pad_canvas, text_size
from stepscribe.render.raster import Rendered
from stepscribe.render.scene import CameraSpec

OFFSET = 34  # px between the model and its dimension line
TICK = 9
INK = (170, 20, 20)  # dimension red: stands out from part colours and black edges
MIN_SPAN_PX = 40  # skip a dimension line when the model is too small in the image


def size_in_frame(bounds: tuple[Vec, Vec], up: str, front: str) -> tuple[float, float, float]:
    """(width, depth, height) of an axis-aligned box using the up/front axis labels."""
    lo, hi = bounds
    size = hi - lo
    u, right, f = frame(up, front)

    def pick(v: Vec) -> float:
        return float(abs(np.dot(size, np.abs(v))))

    return pick(right), pick(f), pick(u)


def size_caption(bounds: tuple[Vec, Vec], up: str, front: str) -> str:
    """'Overall 110.9 (width) x 168.3 (depth) x 211.8 (height) mm'."""
    w, d, h = size_in_frame(bounds, up, front)
    return f"Overall {fmt(w)} (width) x {fmt(d)} (depth) x {fmt(h)} (height) mm"


def _corners(bounds: tuple[Vec, Vec]) -> np.ndarray:
    lo, hi = bounds
    return np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )


def view_spans(camera: CameraSpec, bounds: tuple[Vec, Vec]) -> tuple[float, float, np.ndarray]:
    """(width mm, height mm, corners) of *bounds* as seen along the camera's right and up axes."""
    corners = _corners(bounds)
    direction = camera.position - camera.focal
    direction = direction / np.linalg.norm(direction)
    cam_right = np.cross(camera.view_up, direction)
    cam_right = cam_right / np.linalg.norm(cam_right)
    centre = corners.mean(axis=0)
    return (
        float(np.ptp((corners - centre) @ cam_right)),
        float(np.ptp((corners - centre) @ camera.view_up)),
        corners,
    )


def annotate(
    rendered: Rendered,
    camera: CameraSpec,
    bounds: tuple[Vec, Vec],
    caption: str | None = None,
    lines: bool = True,
) -> None:
    """Draw overall dimension lines (if *lines*) and a caption on ``rendered.image`` in place.

    The views are orthographic, so pixel spans along the camera's right and up axes are
    proportional to millimetres; the values printed are the true spans of *bounds*.
    """
    img = rendered.image.convert("RGB")
    font = _font(FONT_SIZE)
    corners = _corners(bounds)
    px = rendered.project_many(corners)
    x0, y0 = float(px[:, 0].min()), float(px[:, 1].min())
    x1, y1 = float(px[:, 0].max()), float(px[:, 1].max())
    w_mm = h_mm = 0.0
    if lines:
        direction = camera.position - camera.focal
        direction = direction / np.linalg.norm(direction)
        cam_right = np.cross(camera.view_up, direction)
        cam_right = cam_right / np.linalg.norm(cam_right)
        centre = corners.mean(axis=0)
        w_mm = float(np.ptp((corners - centre) @ cam_right))
        h_mm = float(np.ptp((corners - centre) @ camera.view_up))
    # Measure everything first and grow the canvas (right / bottom only, so projections stay valid)
    # instead of letting text run off the image.
    probe = ImageDraw.Draw(img)
    w_text, h_text = f"{fmt(w_mm)} mm", f"{fmt(h_mm)} mm"
    cap_h = text_size(probe, caption, font)[1] + 12 if caption else 0
    cap_w = text_size(probe, caption, font)[0] + 16 + 10 if caption else 0
    need_bottom = (y1 + OFFSET + TICK + 6 if lines and x1 - x0 > MIN_SPAN_PX else y1 + 12) + (
        cap_h + 12 if caption else 0
    )
    need_right = max(
        cap_w,
        x1 + OFFSET + 12 + text_size(probe, h_text, font)[0] + 12
        if lines and y1 - y0 > MIN_SPAN_PX
        else 0,
    )
    img = pad_canvas(
        img,
        right=max(0, math.ceil(need_right - img.width)),
        bottom=max(0, math.ceil(need_bottom - img.height)),
    )
    draw = ImageDraw.Draw(img)
    if lines:
        if x1 - x0 > MIN_SPAN_PX:
            _hline(draw, x0, x1, y1 + OFFSET, w_text, font)
        if y1 - y0 > MIN_SPAN_PX:
            _vline(draw, x1 + OFFSET, y0, y1, h_text, font)
    if caption:
        _caption(draw, img.size, caption, font)
    rendered.image = img


def _hline(draw: ImageDraw.ImageDraw, x0: float, x1: float, y: float, text: str, font) -> None:  # type: ignore[no-untyped-def]
    draw.line([(x0, y), (x1, y)], fill=INK, width=2)
    for x in (x0, x1):
        draw.line([(x, y - TICK), (x, y + TICK)], fill=INK, width=2)
    tw, th = text_size(draw, text, font)
    cx = max((x0 + x1) / 2, tw / 2 + 6)
    draw.rectangle([cx - tw / 2 - 4, y - 24, cx + tw / 2 + 4, y - 2], fill=(255, 255, 255))
    draw_text(draw, (cx - tw / 2, max(y - 22, 2)), text, font, INK)


def _vline(draw: ImageDraw.ImageDraw, x: float, y0: float, y1: float, text: str, font) -> None:  # type: ignore[no-untyped-def]
    draw.line([(x, y0), (x, y1)], fill=INK, width=2)
    for y in (y0, y1):
        draw.line([(x - TICK, y), (x + TICK, y)], fill=INK, width=2)
    tw, th = text_size(draw, text, font)
    cy = (y0 + y1) / 2
    draw.rectangle([x + 8, cy - 12, x + 8 + tw + 8, cy + 10], fill=(255, 255, 255))
    draw_text(draw, (x + 12, cy - 10), text, font, INK)


def _caption(draw: ImageDraw.ImageDraw, size: tuple[int, int], text: str, font) -> None:  # type: ignore[no-untyped-def]
    tw, th = text_size(draw, text, font)
    w, h = tw + 16, th + 12
    x, y = 10, size[1] - h - 10
    draw.rectangle([x, y, x + w, y + h], fill=(255, 255, 255), outline=INK, width=2)
    draw_text(draw, (x + 8, y + 6), text, font, INK)


__all__ = ["Image", "annotate", "size_caption", "size_in_frame"]
