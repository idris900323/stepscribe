"""Load path, weak spot and stability fixtures.

Ground truth:
- cantilever_bracket_assembly: a 120 mm shelf bolted by two screws at x = -35, payload at the far end.
- single_screw_motor_mount: a NEMA 17 on a bracket that is held to the base by ONE screw.
- short_screw_assembly: a 10 mm stack held by an M3 x 10 screw (too short for grip + nut).
- tippy_mast_robot: 100 x 100 x 10 base plus a 20 x 20 x 500 mast at x = 40; with density 2.7 g/cm3
  the centre of mass is at x = 26.67, z = 175 mm: margin 23.33 mm, tipping angle 7.6 degrees.
- thin_printed_bracket (a single part, in part_fixtures): L bracket with 0.6 mm walls.
"""

from __future__ import annotations

from pathlib import Path

from assembly_fixtures import Node, write_assembly
from assembly_fixtures_u import FIXTURE_DIR
from build123d import Box, Cylinder, Location, Pos, Rot
from generate_fixtures import screw_m3x10
from mechanism_fixtures import nema17_motor


def cantilever_bracket_assembly(out: Path = FIXTURE_DIR) -> Path:
    base = Pos(0, 0, 3) * Box(100, 100, 6)
    shelf = Pos(0, 0, 2) * Box(120, 40, 4)
    for y in (-10, 10):
        base = base - Pos(-35, y, 3) * Cylinder(1.7, 8)
        shelf = shelf - Pos(-35, y, 2) * Cylinder(1.7, 8)
    payload = Box(30, 30, 30)
    protos = {"BasePlate": base, "Shelf": shelf, "Payload": payload}
    root = Node(
        "Cantilever",
        children=[
            Node("BasePlate", "BasePlate", Location()),
            Node("Shelf", "Shelf", Pos(0, 0, 6)),
            Node("Payload", "Payload", Pos(40, 0, 6 + 4 + 15)),
        ],
    )
    return write_assembly(out / "cantilever_bracket_assembly.step", protos, root)


def single_screw_motor_mount(out: Path = FIXTURE_DIR) -> Path:
    base = Pos(-25, 0, 3) * Box(50, 100, 6) - Pos(-40, 0, 3) * Cylinder(1.7, 8)
    bracket = Pos(0, 0, 2.5) * Box(100, 60, 5) - Pos(-40, 0, 2.5) * Cylinder(1.7, 8)
    bracket = bracket - Pos(20, 0, 2.5) * Cylinder(11, 7)
    for dx in (-15.5, 15.5):
        for dy in (-15.5, 15.5):
            bracket = bracket - Pos(20 + dx, dy, 2.5) * Cylinder(1.7, 7)
    protos = {"BasePlate": base, "Bracket": bracket, "Stepper": nema17_motor(shaft=5)}
    root = Node(
        "SingleScrew",
        children=[
            Node("BasePlate", "BasePlate", Location()),
            Node("Bracket", "Bracket", Pos(0, 0, 6)),
            # motor above the bracket, shaft pointing down through the bore (clear of the base)
            Node("Motor", "Stepper", Pos(20, 0, 11) * Rot(180, 0, 0)),
        ],
    )
    return write_assembly(out / "single_screw_motor_mount.step", protos, root)


def short_screw_assembly(out: Path = FIXTURE_DIR) -> Path:
    plate = Pos(0, 0, 2.5) * Box(40, 40, 5) - Pos(0, 0, 2.5) * Cylinder(1.7, 7)
    protos = {"Plate": plate, "ScrewM3x10": screw_m3x10()}
    root = Node(
        "ShortScrew",
        children=[
            Node("PlateBottom", "Plate", Location()),
            Node("PlateTop", "Plate", Pos(0, 0, 5)),
            Node("Screw", "ScrewM3x10", Pos(0, 0, 10 + 0)),
        ],
    )
    return write_assembly(out / "short_screw_assembly.step", protos, root)


def tippy_mast_robot(out: Path = FIXTURE_DIR) -> Path:
    protos = {
        "BasePlate": Pos(0, 0, 5) * Box(100, 100, 10),
        "Mast": Pos(0, 0, 250) * Box(20, 20, 500),
    }
    root = Node(
        "TippyMast",
        children=[
            Node("BasePlate", "BasePlate", Location()),
            Node("Mast", "Mast", Pos(40, 0, 10)),
        ],
    )
    return write_assembly(out / "tippy_mast_robot.step", protos, root)


BUILDERS = (
    cantilever_bracket_assembly,
    single_screw_motor_mount,
    short_screw_assembly,
    tippy_mast_robot,
)


def generate_risks(out: Path = FIXTURE_DIR) -> dict[str, Path]:
    return {fn.__name__: fn(out) for fn in BUILDERS}
