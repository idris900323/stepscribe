"""Generate STEP fixtures with known ground truth using build123d."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from build123d import (
    Box,
    Compound,
    Cone,
    Cylinder,
    Pos,
    RegularPolygon,
    Rot,
    Shape,
    chamfer,
    export_step,
    extrude,
    fillet,
)
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

FIXTURE_DIR = Path(__file__).parent / "step"


def cube_10() -> Shape:
    return Box(10, 10, 10)


def plate_4xm3() -> Shape:
    p = Box(60, 60, 5)
    for x in (-20, 20):
        for y in (-20, 20):
            p = p - Pos(x, y, 0) * Cylinder(1.7, 6)
    return p


def nema17_plate() -> Shape:
    p = Box(60, 60, 4)
    for x in (-15.5, 15.5):
        for y in (-15.5, 15.5):
            p = p - Pos(x, y, 0) * Cylinder(1.7, 6)
    return p - Cylinder(11, 6)


def blind_holes() -> Shape:
    """Flat-bottom Ø4.2 x 10 and drill-point Ø4.2 x 12 (cylinder depth) in a 30 mm block."""
    b = Box(40, 30, 30)  # top face at z = +15
    flat = Pos(-10, 0, 15 - 5) * Cylinder(2.1, 10)
    tip_h = 2.1 / math.tan(math.radians(59))  # 118 deg point
    drill = Pos(10, 0, 15 - 6) * Cylinder(2.1, 12) + Pos(10, 0, 15 - 12 - tip_h / 2) * Cone(
        0.0001, 2.1, tip_h
    )
    return b - flat - drill


def cbore() -> Shape:
    """M5 counterbored hole: Ø5.5 through, Ø9.5 x 5.4 counterbore, plate 12 mm."""
    p = Box(30, 30, 12)
    through = Cylinder(2.75, 14)
    cb = Pos(0, 0, 6 - 2.7) * Cylinder(4.75, 5.4)
    return p - through - cb


def csk() -> Shape:
    """90 deg countersink to Ø9 on a Ø5.5 hole, plate 8 mm."""
    p = Box(30, 30, 8)
    through = Cylinder(2.75, 10)
    h = (9 - 5.5) / 2  # 45 deg => depth = radius difference
    cone = Pos(0, 0, 4 - h / 2) * Cone(2.75, 4.5, h)
    return p - through - cone


def boss_and_shaft() -> Shape:
    """Base plate with a Ø8 boss carrying a Ø2.5 hole; a Ø6 shaft stub on the other side."""
    base = Box(40, 40, 4)
    boss = Pos(0, 0, 2 + 3) * Cylinder(4, 6)
    shaft = Pos(12, 0, -2 - 5) * Cylinder(3, 10)
    s = base + boss + shaft
    return s - Pos(0, 0, 2 + 3) * Cylinder(1.25, 6.5) - Pos(0, 0, 1) * Cylinder(1.25, 4.1)


def slot_and_fillets() -> Shape:
    """Plate with one w=6 x l=20 stadium slot and R2 vertical-edge fillets; no holes."""
    p = Box(50, 40, 6)
    p = fillet(p.edges().filter_by(lambda e: abs(e.length - 6) < 1e-6), 2)
    slot_len, w = 20.0, 6.0
    slot = (
        Box(slot_len - w, w, 8)
        + Pos(-(slot_len - w) / 2, 0, 0) * Cylinder(w / 2, 8)
        + Pos((slot_len - w) / 2, 0, 0) * Cylinder(w / 2, 8)
    )
    return p - slot


def pocket() -> Shape:
    """40x40x10 block with a 20x14 pocket, 4 deep, corner radius 2."""
    b = Box(40, 40, 10)
    pk = Box(20, 14, 4 + 1)
    pk = fillet(pk.edges().filter_by(lambda e: abs(e.length - 5) < 1e-6), 2)
    return b - Pos(0, 0, 5 - 2 + 0.5) * pk


def split_cylinder() -> Shape:
    """Hole Ø6 cut as two separate half-cylinders so the bore is two 180 deg faces."""
    p = Box(30, 30, 10)
    cyl = Cylinder(3, 12)
    upper = cyl & (Pos(0, 5, 0) * Box(10, 10, 12))
    lower = cyl & (Pos(0, -5, 0) * Box(10, 10, 12))
    return (p - upper) - lower


def bolt_circle() -> Shape:
    """Six Ø4 holes on a PCD 60 circle (60 deg pitch) in a Ø80 disc, 6 thick."""
    d = Cylinder(40, 6)
    for i in range(6):
        a = math.radians(60 * i)
        d = d - Pos(30 * math.cos(a), 30 * math.sin(a), 0) * Cylinder(2, 8)
    return d


def linear_row() -> Shape:
    """Five Ø4 holes in a row, pitch 15."""
    p = Box(100, 20, 5)
    for i in range(5):
        p = p - Pos(-30 + 15 * i, 0, 0) * Cylinder(2, 7)
    return p


def rotated_plate() -> Shape:
    return Rot(37, 20, 11) * plate_4xm3()


def l_bracket() -> Shape:
    """3 mm thick L bracket, flanges 40 x 30 and 40 x 30, with two holes."""
    a = Pos(0, 0, 1.5) * Box(40, 30, 3)
    b = Pos(0, -15 + 1.5, 15) * Box(40, 3, 30)
    s = a + b
    s = s - Pos(-10, 0, 1.5) * Cylinder(2.2, 5) - Pos(10, 0, 1.5) * Cylinder(2.2, 5)
    return s


def extrusion_2020_like() -> Shape:
    """200 mm long 20x20 profile: constant section with a Ø4.2 centre bore and 4 T-slot grooves."""
    s = Box(20, 20, 200)
    s = s - Cylinder(2.1, 210)
    for ang in (0, 90, 180, 270):
        s = s - Rot(0, 0, ang) * (Pos(10 - 2.5, 0, 0) * Box(5, 6.2, 210))
    return s


def tube() -> Shape:
    return Cylinder(10, 60) - Cylinder(7, 62)


def bearing_block() -> Shape:
    """Block with a 608 seat: Ø22 x 7 pocket from the top, Ø12 bore through, shoulder at 7."""
    b = Box(40, 40, 14)
    seat = Pos(0, 0, 7 - 3.5) * Cylinder(11, 7)
    bore = Cylinder(6, 16)
    return b - seat - bore


def thin_wall_box() -> Shape:
    """Hollow box with 1.2 mm walls and an open top."""
    outer = Box(40, 30, 20)
    inner = Pos(0, 0, 1.2) * Box(40 - 2.4, 30 - 2.4, 20)
    return outer - inner


def chamfer_block() -> Shape:
    """20 x 20 x 10 block with 1.5 mm chamfers on the four top edges."""
    b = Box(20, 20, 10)
    return chamfer(b.edges().filter_by_position(__import__("build123d").Axis.Z, 4.9, 5.1), 1.5)


def washer_m3() -> Shape:
    """ISO 7089-like M3 washer: Ø7 x Ø3.2 x 0.5."""
    return Cylinder(3.5, 0.5) - Cylinder(1.6, 1.0)


def nut_m3() -> Shape:
    """M3 hex nut: 5.5 across flats, 2.4 high, Ø2.5 tapped bore."""
    hexa = extrude(RegularPolygon(5.5 / math.sqrt(3), 6), 2.4)
    return Pos(0, 0, -1.2) * hexa - Cylinder(1.25, 4)


def screw_m3x10() -> Shape:
    """M3 socket head cap screw: Ø5.5 x 3 head, Ø3 x 10 shank, hex socket."""
    head = Pos(0, 0, 1.5) * Cylinder(2.75, 3)
    shank = Pos(0, 0, -5) * Cylinder(1.5, 10)
    socket = Pos(0, 0, 3 - 1.25) * extrude(
        RegularPolygon(1.5 / math.cos(math.radians(30)) * 0.9, 6), 2.5
    )
    return head + shank - socket


def multi_solid() -> Shape:
    return Compound([Box(10, 10, 10), Pos(30, 0, 0) * Box(5, 5, 5)])


FIXTURES: dict[str, Callable[[], Shape]] = {
    "cube_10": cube_10,
    "plate_4xM3": plate_4xm3,
    "nema17_plate": nema17_plate,
    "blind_holes": blind_holes,
    "cbore": cbore,
    "csk": csk,
    "boss_and_shaft": boss_and_shaft,
    "slot_and_fillets": slot_and_fillets,
    "pocket": pocket,
    "split_cylinder": split_cylinder,
    "bolt_circle": bolt_circle,
    "linear_row": linear_row,
    "rotated_plate": rotated_plate,
    "l_bracket": l_bracket,
    "extrusion_2020_like": extrusion_2020_like,
    "tube": tube,
    "bearing_block": bearing_block,
    "thin_wall_box": thin_wall_box,
    "chamfer_block": chamfer_block,
    "washer_m3": washer_m3,
    "nut_m3": nut_m3,
    "screw_m3x10": screw_m3x10,
    "multi_solid": multi_solid,
}


def write_inch_cube(path: Path) -> None:
    """A 1-inch cube written with INCH as the STEP length unit."""
    Interface_Static.SetCVal_s("write.step.unit", "INCH")
    try:
        writer = STEPControl_Writer()
        writer.Transfer(Box(25.4, 25.4, 25.4).wrapped, STEPControl_AsIs)
        writer.Write(str(path))
    finally:
        Interface_Static.SetCVal_s("write.step.unit", "MM")


def generate_all(out: Path = FIXTURE_DIR) -> dict[str, Path]:
    """Write every fixture to *out*; returns name -> path."""
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, fn in FIXTURES.items():
        p = out / f"{name}.step"
        export_step(fn(), str(p))
        paths[name] = p
    inch = out / "cube_1in.step"
    write_inch_cube(inch)
    paths["cube_1in"] = inch
    return paths


if __name__ == "__main__":
    for n, p in generate_all().items():
        print(n, p)
