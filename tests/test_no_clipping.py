# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""No label, dimension value or panel row may be cut off at the edge of any image."""

from __future__ import annotations

from pathlib import Path

import pytest

from stepscribe.render import render_pack_images
from stepscribe.render.overlay import record_text_boxes

NAMES = [
    "plate_three_stepped",
    "plate_edge_notch",
    "plate_rounded_rect_thru",
    "block_side_recess",
    "plate_4xM3",
    "bolt_circle",
    "nema17_plate",
    "two_plates_assembly",
]


@pytest.mark.parametrize("name", NAMES)
def test_every_text_box_is_inside_its_image(name: str, analyses, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    analysis = analyses(name)
    before = [list(ap.part.images) for ap in analysis.parts]  # the analysis is shared by tests
    try:
        with record_text_boxes() as log:
            render_pack_images(analysis, tmp_path)
    finally:
        for ap, imgs in zip(analysis.parts, before, strict=True):
            ap.part.images[:] = imgs
    assert log, "nothing was recorded"
    bad = [
        (text, box, size)
        for text, box, size in log
        if box[0] < 0 or box[1] < 0 or box[2] > size[0] or box[3] > size[1]
    ]
    assert not bad, bad[:5]
