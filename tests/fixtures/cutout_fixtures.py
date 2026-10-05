"""Cut-out fixtures with known ground truth.

Every plate is 100 x 60 x 4 (z from -2 to +2); openings are cut from the top face (z = +2).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from build123d import (
    Box,
    Cylinder,
    Plane,
    Polyline,
    Pos,
    RectangleRounded,
    Rot,
    Shape,
    extrude,
    make_face,
)
from build123d import export_step as _export

FIXTURE_DIR = Path(__file__).parent / "step" / "cutouts"


def _plate() -> Shape:
    return Box(100, 60, 4)


def plate_rect_thru() -> Shape:
    """Sharp 20 x 10 window through the plate, centred at (10, 5)."""
    return _plate() - Pos(10, 5, 0) * Box(20, 10, 10)


def plate_rounded_rect_thru() -> Shape:
    """20 x 10 window with R2 corners, centred at (10, 5)."""
    return _plate() - Pos(10, 5, -3) * extrude(RectangleRounded(20, 10, 2), amount=8)


def plate_rect_recess() -> Shape:
    """20 x 10 recess, 2 deep, from the top face."""
    return _plate() - Pos(10, 5, 2 - 1 + 0.5) * Box(20, 10, 3)


def plate_stepped_cutout() -> Shape:
    """10 x 18 recess 2 deep, then a 10 x 6 window through the rest of the plate."""
    return _plate() - Pos(0, 0, 1.5) * Box(10, 18, 3) - Pos(0, 0, 0) * Box(10, 6, 10)


def plate_stepped_offset() -> Shape:
    """12 x 20 recess 2 deep with a narrower 6 x 8 window inside it (a ring-shaped floor)."""
    return _plate() - Pos(0, 0, 1.5) * Box(12, 20, 3) - Pos(0, 0, 0) * Box(6, 8, 10)


def plate_three_stepped() -> Shape:
    """Three stepped cutouts (10 x 18 recess 2 deep, 10 x 6 through) in a row, pitch 30 along x."""
    s = _plate()
    for x in (-30, 0, 30):
        s = s - Pos(x, 0, 1.5) * Box(10, 18, 3) - Pos(x, 0, 0) * Box(10, 6, 10)
    return s


def plate_edge_notch() -> Shape:
    """8 wide x 5 deep notch on the +x edge, centred at y = 0."""
    return _plate() - Pos(50 - 2.5, 0, 0) * Box(5, 8, 10)


def plate_l_cutout() -> Shape:
    """L-shaped window: area 20x10 + 10x10 = 300 (a 6-sided polygon)."""
    pts = [(0, 0), (20, 0), (20, 10), (10, 10), (10, 20), (0, 20), (0, 0)]
    face = make_face(Plane.XY * Polyline(*pts))
    return _plate() - Pos(-10, -10, -3) * extrude(face, amount=8)


def plate_rotated_window() -> Shape:
    """20 x 10 window rotated 30 degrees about the plate normal, centred at (0, 0)."""
    return _plate() - Rot(0, 0, 30) * Box(20, 10, 10)


def block_side_recess() -> Shape:
    """40 x 40 x 30 block with a 16 x 10 recess 3 deep in the +x side face."""
    b = Box(40, 40, 30)
    return b - Pos(20 - 1.5 + 0.5, 0, 0) * Box(4, 16, 10)


def plate_close_cutouts() -> Shape:
    """Two 10 x 10 windows with a 1.2 mm web between them (thin bridge)."""
    return _plate() - Pos(-5.6, 0, 0) * Box(10, 10, 10) - Pos(5.6, 0, 0) * Box(10, 10, 10)


def plate_cutout_and_holes() -> Shape:
    """A 20 x 10 window next to two Ø3 holes (holes stay holes)."""
    s = _plate() - Pos(0, 0, 0) * Box(20, 10, 10)
    for x in (-30, 30):
        s = s - Pos(x, 0, 0) * Cylinder(1.5, 10)
    return s


FIXTURES: dict[str, Callable[[], Shape]] = {
    "plate_rect_thru": plate_rect_thru,
    "plate_rounded_rect_thru": plate_rounded_rect_thru,
    "plate_rect_recess": plate_rect_recess,
    "plate_stepped_cutout": plate_stepped_cutout,
    "plate_stepped_offset": plate_stepped_offset,
    "plate_three_stepped": plate_three_stepped,
    "plate_edge_notch": plate_edge_notch,
    "plate_L_cutout": plate_l_cutout,
    "plate_rotated_window": plate_rotated_window,
    "block_side_recess": block_side_recess,
    "plate_close_cutouts": plate_close_cutouts,
    "plate_cutout_and_holes": plate_cutout_and_holes,
}


def generate_cutouts(out: Path = FIXTURE_DIR) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, fn in FIXTURES.items():
        p = out / f"{name}.step"
        _export(fn(), str(p))
        paths[name] = p
    return paths
