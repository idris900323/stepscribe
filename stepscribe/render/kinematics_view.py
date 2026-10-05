# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Joint axes as arrows on the assembly view, and a link-colour-coded view."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

from stepscribe.models.schema import KinematicModel
from stepscribe.render.overlay import FONT_SIZE, _font, draw_text, text_size
from stepscribe.render.raster import Rendered
from stepscribe.render.scene import SceneItem

if TYPE_CHECKING:
    pass

ARROW = (190, 20, 100)  # magenta: unlike part colours, black edges and the red dimension lines
LINK_PALETTE = [
    (0.55, 0.55, 0.60),  # L0 ground: grey
    (0.90, 0.35, 0.25),
    (0.25, 0.55, 0.90),
    (0.95, 0.75, 0.20),
    (0.35, 0.75, 0.40),
    (0.65, 0.40, 0.85),
    (0.20, 0.75, 0.75),
    (0.90, 0.55, 0.70),
]
ARROW_FRACTION = 0.45  # arrow length as a fraction of the assembly diagonal


def link_colors(model: KinematicModel) -> dict[str, tuple[float, float, float]]:
    return {g.id: LINK_PALETTE[i % len(LINK_PALETTE)] for i, g in enumerate(model.links)}


def recolor_by_link(items: list[SceneItem], model: KinematicModel) -> list[SceneItem]:
    """Copies of *items* coloured by rigid group; the ground link is grey."""
    colors = link_colors(model)
    link_of = {i: g.id for g in model.links for i in g.instance_ids}
    out = []
    for it in items:
        link = link_of.get(it.key)
        out.append(
            SceneItem(
                it.key,
                it.part_id,
                it.name,
                it.mesh,
                it.matrix,
                colors.get(link or "", it.color),
                it.group,
            )
        )
    return out


def link_labels(model: KinematicModel) -> dict[str, str]:
    return {i: g.id for g in model.links for i in g.instance_ids}


def link_legend(model: KinematicModel) -> list[str]:
    return [
        f"{g.id}  {'ground: ' if g.is_ground else ''}{g.description}"[:70] for g in model.links
    ][:30]


def draw_joint_arrows(
    rendered: Rendered, model: KinematicModel, diagonal: float, image: Image.Image | None = None
) -> None:
    """Draw each joint axis (double-headed for prismatic) with its KJ label, in place."""
    img = (image or rendered.image).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font(FONT_SIZE)
    length = ARROW_FRACTION * diagonal
    for j in model.joints:
        o = np.array([j.axis.origin.x, j.axis.origin.y, j.axis.origin.z])
        d = np.array([j.axis.direction.x, j.axis.direction.y, j.axis.direction.z])
        d = d / np.linalg.norm(d)
        a, b = o - d * length / 2, o + d * length / 2
        xa, ya, _ = rendered.project(a)
        xb, yb, _ = rendered.project(b)
        xo, yo, _ = rendered.project(o)
        draw.line([(xa, ya), (xb, yb)], fill=ARROW, width=5)
        _head(draw, (xa, ya), (xb, yb))
        if j.kind == "prismatic":
            _head(draw, (xb, yb), (xa, ya))
        text = f"{j.id} {j.kind[:3]}"
        tw, th = text_size(draw, text, font)
        w, h = tw + 10, th + 8
        bx = min(max(xo - w / 2, 2), img.width - w - 2)  # keep the whole label inside the image
        by = min(max(yo - h / 2, 2), img.height - h - 2)
        draw.rectangle([bx, by, bx + w, by + h], fill=(255, 255, 255), outline=ARROW, width=2)
        draw_text(draw, (bx + 5, by + 4), text, font, ARROW)
    rendered.image = img


def _head(draw: ImageDraw.ImageDraw, tail: tuple[float, float], tip: tuple[float, float]) -> None:
    ang = math.atan2(tip[1] - tail[1], tip[0] - tail[0])
    size = 22
    pts = [
        tip,
        (tip[0] - size * math.cos(ang - 0.4), tip[1] - size * math.sin(ang - 0.4)),
        (tip[0] - size * math.cos(ang + 0.4), tip[1] - size * math.sin(ang + 0.4)),
    ]
    draw.polygon(pts, fill=ARROW)
