"""Single-part fixtures for the understanding layer."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from build123d import (
    Box,
    Cylinder,
    Plane,
    Polyline,
    Pos,
    RegularPolygon,
    Rot,
    Shape,
    extrude,
    fillet,
    make_face,
)
from build123d import export_step as _export

FIXTURE_DIR = Path(__file__).parent.parent / "step" / "understanding"


def ribbed_plate() -> Shape:
    """80 x 60 x 4 plate with three ribs 2 mm thick, 12 mm high, 50 mm long on top."""
    s = Pos(0, 0, 2) * Box(80, 60, 4)
    for x in (-25, 0, 25):
        s = s + Pos(x, 0, 4 + 6) * Box(2, 50, 12)
    return s


def gusseted_l_bracket() -> Shape:
    """3 mm L bracket (flanges 40 x 30 and 40 x 30) with one 3 mm triangular gusset, legs 20."""
    base = Pos(0, 0, 1.5) * Box(40, 30, 3)
    wall = Pos(0, -15 + 1.5, 15) * Box(40, 3, 30)
    s = base + wall
    # triangle in the YZ plane at x = 0, between the flange tops: legs 20 mm along +Y and +Z
    y0, z0 = -15 + 3, 3
    tri = make_face(Polyline((y0, z0), (y0 + 20, z0), (y0, z0 + 20), close=True))
    gusset = extrude(Plane.YZ.offset(-1.5) * tri, 3)
    return s + gusset


def stepped_block() -> Shape:
    """Block 60 x 40 with two steps: heights 6, 12 and 18 mm from the long left to the right."""
    s = Pos(-20, 0, 3) * Box(20, 40, 6)
    s = s + Pos(0, 0, 6) * Box(20, 40, 12)
    s = s + Pos(20, 0, 9) * Box(20, 40, 18)
    return s


def sheet_metal_u() -> Shape:
    """U channel from 2 mm sheet, inner bend radius 3 mm, 50 mm long, 40 wide, 25 high."""
    t, ri, length, width, height = 2.0, 3.0, 50.0, 40.0, 25.0
    outer = Pos(0, 0, height / 2) * Box(width, length, height)
    outer = _fillet_bottom_long_edges(outer, ri + t)
    cavity = Pos(0, 0, t + height / 2) * Box(width - 2 * t, length + 2, height)
    cavity = _fillet_bottom_long_edges(cavity, ri)
    return outer - cavity


def _fillet_bottom_long_edges(solid: Shape, radius: float) -> Shape:
    zmin = solid.bounding_box().min.Z
    edges = [
        e
        for e in solid.edges()
        if abs(e.center().Z - zmin) < 1e-6 and abs(e.start_point().Y - e.end_point().Y) > 1.0
    ]
    return fillet(edges, radius)


def lightened_plate() -> Shape:
    """100 x 60 x 4 plate with four 20 x 14 rectangular cutouts and four M3 clearance holes."""
    s = Pos(0, 0, 2) * Box(100, 60, 4)
    for x in (-30, 30):
        for y in (-12, 12):
            s = s - Pos(x, y, 2) * Box(20, 14, 8)
    for x in (-45, 45):
        for y in (-25, 25):
            s = s - Pos(x, y, 2) * Cylinder(1.7, 8)
    return s


def symmetric_bracket() -> Shape:
    """L bracket (base 40 x 30 x 3, wall 40 x 3 x 30) with holes mirrored about x = 0: one plane."""
    base = Pos(0, 0, 1.5) * Box(40, 30, 3)
    wall = Pos(0, -15 + 1.5, 15) * Box(40, 3, 30)
    s = base + wall
    for x in (-12, 12):
        s = s - Pos(x, 5, 1.5) * Cylinder(2.0, 8)
    s = s - Pos(0, -13.5, 20) * (Rot(90, 0, 0) * Cylinder(2.5, 8))
    return s


def hex_flange() -> Shape:
    """Hexagonal prism (across corners 40) 6 mm thick with a 10 mm bore: 6-fold symmetry."""
    hexagon = extrude(RegularPolygon(20, 6), 6)
    return Pos(0, 0, 3) * hexagon - Pos(0, 0, 3) * Cylinder(5, 10)


def laser_plate() -> Shape:
    """3 mm plate with an L-shaped outline and four through holes: a 2D profile extruded."""
    outline = make_face(
        Polyline((0, 0), (60, 0), (60, 20), (25, 20), (25, 50), (0, 50), close=True)
    )
    s = extrude(outline, 3)
    for x, y in ((8, 8), (52, 8), (8, 42), (17, 25)):
        s = s - Pos(x, y, 1.5) * Cylinder(2.1, 6)
    return s


def turned_shaft() -> Shape:
    """Stepped shaft: dia 12 x 20, dia 8 x 30, dia 6 x 10 with a chamfer-free end and a 3 mm bore."""
    s = Pos(0, 0, 10) * Cylinder(6, 20)
    s = s + Pos(0, 0, 35) * Cylinder(4, 30)
    s = s + Pos(0, 0, 55) * Cylinder(3, 10)
    return s - Pos(0, 0, 5) * Cylinder(1.5, 10)


def milled_pocket_block() -> Shape:
    """Block 50 x 40 x 20 with a 30 x 20 x 10 pocket with R3 vertical corners and two holes."""
    s = Pos(0, 0, 10) * Box(50, 40, 20)
    pocket = Pos(0, 0, 15 + 0.5) * Box(30, 20, 11)
    verticals = [e for e in pocket.edges() if abs(e.start_point().Z - e.end_point().Z) > 1.0]
    s = s - fillet(verticals, 3)
    for x in (-20, 20):
        s = s - Pos(x, 0, 10) * Cylinder(2.2, 30)
    return s


def printed_like_part() -> Shape:
    """Printed-looking bracket: sharp internal corners, an overhanging bar and holes on 4 sides."""
    s = Pos(0, 0, 3) * Box(40, 30, 6)
    s = s + Pos(-17, 0, 16) * Box(6, 30, 20)  # upright wall: sharp internal corner at its base
    s = s + Pos(-5, 0, 24) * Box(30, 14, 4)  # overhanging arm: nothing under it
    s = s - Pos(10, 0, 3) * Cylinder(2.0, 10)
    s = s - Pos(0, 0, 24) * Cylinder(2.0, 10)
    s = s - Pos(-17, 0, 14) * (Rot(0, 90, 0) * Cylinder(2.0, 10))
    s = s - Pos(0, 0, 3) * (Rot(90, 0, 0) * Cylinder(1.5, 40))
    return s


FIXTURES: dict[str, Callable[[], Shape]] = {
    "thin_printed_bracket": lambda: thin_printed_bracket(),
    "chiral_part": lambda: chiral_part(),
    "symmetric_bracket": symmetric_bracket,
    "hex_flange": hex_flange,
    "laser_plate": laser_plate,
    "turned_shaft": turned_shaft,
    "milled_pocket_block": milled_pocket_block,
    "printed_like_part": printed_like_part,
    "ribbed_plate": ribbed_plate,
    "gusseted_l_bracket": gusseted_l_bracket,
    "stepped_block": stepped_block,
    "sheet_metal_u": sheet_metal_u,
    "lightened_plate": lightened_plate,
}


def generate_parts(out: Path = FIXTURE_DIR) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, fn in FIXTURES.items():
        p = out / f"{name}.step"
        _export(fn(), str(p))
        paths[name] = p
    return paths


def chiral_part() -> Shape:
    """Block with a boss and a hole at unrelated positions: no mirror symmetry of its own."""
    s = Pos(0, 0, 5) * Box(30, 20, 10)
    s = s + Pos(8, 5, 12) * Cylinder(3, 4)
    s = s - Pos(-6, -4, 5) * Cylinder(2, 12)
    s = s - Pos(0, 10, 7) * (Rot(90, 0, 0) * Cylinder(1.5, 8))
    return s


def thin_printed_bracket() -> Shape:
    """L bracket with 0.6 mm walls, 40 x 30 base and 40 x 30 wall, one hole 1 mm from the edge."""
    base = Pos(0, 0, 0.3) * Box(40, 30, 0.6)
    wall = Pos(0, -15 + 0.3, 15) * Box(40, 0.6, 30)
    s = base + wall
    return s - Pos(-17, 5, 0.3) * Cylinder(2.0, 3)
