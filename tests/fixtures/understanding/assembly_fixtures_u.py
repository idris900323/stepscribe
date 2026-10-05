"""Kinematic assembly fixtures for the understanding layer.

Ground truth (global frame, mm; up = +Z):
- two_link_arm: revolute about X through z = 30 (base to arm 1) and z = 95 (arm 1 to arm 2).
- linear_slide: prismatic along X (carriage on two rods with LM8UU bearings).
- four_bar_linkage: four revolute joints about Z at (0,0), (0,30), (60,30), (100,0); closed loop.
- wheeled_base: four revolute joints about Y at x = +-40, wheel centres z = 15.
- floating_part_assembly: one part that touches nothing; one that only rests on the base.
"""

from __future__ import annotations

import math
from pathlib import Path

from assembly_fixtures import Node, write_assembly
from build123d import Box, Cylinder, Location, Pos, Rot, Shape

FIXTURE_DIR = Path(__file__).parent.parent / "step" / "understanding"

X_AXIS = Rot(0, 90, 0)  # a Z cylinder becomes an X cylinder
Y_AXIS = Rot(90, 0, 0)  # a Z cylinder becomes a Y cylinder


def _generic(protos: dict[str, Shape], root: Node) -> tuple[dict[str, Shape], Node]:
    """Rename every prototype and instance to Part1, Part2... (tests name-free inference)."""
    mapping = {name: f"Part{n}" for n, name in enumerate(protos, 1)}
    count = [0]

    def walk(node: Node) -> Node:
        count[0] += 1
        return Node(
            f"Inst{count[0]}" if node.proto or node.children else node.name,
            mapping.get(node.proto or "", None),
            node.loc,
            [walk(c) for c in node.children],
        )

    new_root = walk(root)
    new_root.name = "Assembly"
    return {mapping[k]: v for k, v in protos.items()}, new_root


def bearing_608() -> Shape:
    return Cylinder(11, 7) - Cylinder(4, 8)


def two_link_arm(out: Path = FIXTURE_DIR) -> Path:
    base = Pos(0, 0, 22.5) * Box(40, 30, 45) - Pos(0, 0, 30) * X_AXIS * Cylinder(11, 50)
    arm1 = (
        Box(10, 30, 90)
        - Pos(0, 0, -37) * X_AXIS * Cylinder(4, 12)
        - Pos(0, 0, 28) * X_AXIS * Cylinder(11, 12)
    )
    arm2 = Box(10, 30, 63) - Pos(0, 0, -26) * X_AXIS * Cylinder(4, 12)
    protos = {
        "Base": base,
        "Bearing608": bearing_608(),
        "ShaftA": Cylinder(4, 42),
        "ShaftB": Cylinder(4, 20),
        "ArmLink1": arm1,
        "ArmLink2": arm2,
    }
    root = Node(
        "TwoLinkArm",
        children=[
            Node("Base", "Base", Location()),
            Node("Bearing1", "Bearing608", Pos(0, 0, 30) * X_AXIS),
            Node("Shaft1", "ShaftA", Pos(11, 0, 30) * X_AXIS),
            Node("Arm1", "ArmLink1", Pos(27, 0, 67)),
            Node("Bearing2", "Bearing608", Pos(27, 0, 95) * X_AXIS),
            Node("Shaft2", "ShaftB", Pos(34, 0, 95) * X_AXIS),
            Node("Arm2", "ArmLink2", Pos(39, 0, 121)),
        ],
    )
    named = write_assembly(out / "two_link_arm.step", protos, root)
    gp, groot = _generic(protos, root)
    write_assembly(out / "two_link_arm_generic.step", gp, groot)
    return named


def linear_slide(out: Path = FIXTURE_DIR) -> Path:
    plate = (
        Box(10, 50, 30)
        - Pos(0, 15, 0) * X_AXIS * Cylinder(4, 12)
        - Pos(0, -15, 0) * X_AXIS * Cylinder(4, 12)
    )
    carriage = (
        Box(40, 50, 24)
        - Pos(0, 15, 0) * X_AXIS * Cylinder(7.5, 42)
        - Pos(0, -15, 0) * X_AXIS * Cylinder(7.5, 42)
    )
    lm8uu = Cylinder(7.5, 24) - Cylinder(4, 25)
    protos = {"EndPlate": plate, "Rod": Cylinder(4, 220), "Carriage": carriage, "LM8UU": lm8uu}
    root = Node(
        "LinearSlide",
        children=[
            Node("EndPlateLeft", "EndPlate", Pos(-105, 0, 20)),
            Node("EndPlateRight", "EndPlate", Pos(105, 0, 20)),
            Node("RodFront", "Rod", Pos(0, -15, 20) * X_AXIS),
            Node("RodBack", "Rod", Pos(0, 15, 20) * X_AXIS),
            Node("Carriage", "Carriage", Pos(0, 0, 20)),
            Node("BushingFront", "LM8UU", Pos(0, -15, 20) * X_AXIS),
            Node("BushingBack", "LM8UU", Pos(0, 15, 20) * X_AXIS),
        ],
    )
    return write_assembly(out / "linear_slide.step", protos, root)


def _bar(length: float, free_end: str, hole_free: float = 6.2, hole_press: float = 6.0) -> Shape:
    """Flat link 4 mm thick, 12 wide, with a pivot hole at each end (local x = +-length/2)."""
    bar = Box(length + 12, 12, 4)
    left = hole_free if free_end == "left" else hole_press
    right = hole_press if free_end == "left" else hole_free
    bar = bar - Pos(-length / 2, 0, 0) * Cylinder(left / 2, 6)
    return bar - Pos(length / 2, 0, 0) * Cylinder(right / 2, 6)


def _place(p0: tuple[float, float], p1: tuple[float, float], z: float) -> Location:
    ang = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
    return Pos((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, z) * Rot(0, 0, ang)


def four_bar_linkage(out: Path = FIXTURE_DIR) -> Path:
    A, B, C, D = (0.0, 0.0), (0.0, 30.0), (60.0, 30.0), (100.0, 0.0)
    ground = Pos(50, 0, 3) * Box(120, 30, 6)
    ground = ground - Pos(0, 0, 3) * Cylinder(3, 8) - Pos(100, 0, 3) * Cylinder(3, 8)
    pin = Pos(0, 0, 1) * Cylinder(4.5, 2) + Pos(0, 0, 8) * Cylinder(3, 12)  # head 0..2, shaft 2..14
    protos = {
        "Ground": ground,
        "Crank": _bar(30, "left"),
        "Coupler": _bar(60, "left", 6.2, 6.2),
        "Rocker": _bar(50, "right"),
        "Pin": pin,
    }
    root = Node(
        "FourBar",
        children=[
            Node("Ground", "Ground", Location()),
            Node("PinA", "Pin", Pos(A[0], A[1], -2)),
            Node("PinD", "Pin", Pos(D[0], D[1], -2)),
            Node("Crank", "Crank", _place(A, B, 8)),
            Node("Rocker", "Rocker", _place(C, D, 8)),
            Node("PinB", "Pin", Pos(B[0], B[1], 4)),
            Node("PinC", "Pin", Pos(C[0], C[1], 4)),
            Node("Coupler", "Coupler", _place(B, C, 12)),
        ],
    )
    return write_assembly(out / "four_bar_linkage.step", protos, root)


def wheeled_base(out: Path = FIXTURE_DIR) -> Path:
    chassis = Pos(0, 0, 15) * Box(120, 60, 26)
    for x in (-40, 40):
        for y in (-1, 1):
            chassis = chassis - Pos(x, y * 25, 15) * Y_AXIS * Cylinder(11, 10)
    wheel = Cylinder(15, 10) - Cylinder(4, 11)
    protos = {
        "Chassis": chassis,
        "Bearing608": bearing_608(),
        "Axle": Cylinder(4, 30),
        "Wheel": wheel,
    }
    kids = [Node("Chassis", "Chassis", Location())]
    n = 0
    for x in (-40, 40):
        for y in (-1, 1):
            n += 1
            kids += [
                Node(f"Bearing{n}", "Bearing608", Pos(x, y * 26.5, 15) * Y_AXIS),
                Node(f"Axle{n}", "Axle", Pos(x, y * 30, 15) * Y_AXIS),
                Node(f"Wheel{n}", "Wheel", Pos(x, y * 40, 15) * Y_AXIS),
            ]
    root = Node("WheeledBase", children=kids)
    named = write_assembly(out / "wheeled_base.step", protos, root)
    gp, groot = _generic(protos, root)
    write_assembly(out / "wheeled_base_generic.step", gp, groot)
    return named


def floating_part_assembly(out: Path = FIXTURE_DIR) -> Path:
    plate = Box(60, 60, 5) - Pos(-20, -20, 0) * Cylinder(1.7, 6) - Pos(20, 20, 0) * Cylinder(1.7, 6)
    protos = {"Plate": plate, "Lid": Box(30, 30, 4), "Loose": Box(20, 20, 20)}
    root = Node(
        "Floating",
        children=[
            Node("BasePlate", "Plate", Location()),
            Node("RestingLid", "Lid", Pos(0, 0, 4.5)),
            Node("LooseBlock", "Loose", Pos(120, 0, 10)),
        ],
    )
    return write_assembly(out / "floating_part_assembly.step", protos, root)


BUILDERS = (two_link_arm, linear_slide, four_bar_linkage, wheeled_base, floating_part_assembly)


def generate_kinematic(out: Path = FIXTURE_DIR) -> dict[str, Path]:
    paths = {fn.__name__: fn(out) for fn in BUILDERS}
    for name in ("two_link_arm", "wheeled_base"):
        paths[name + "_generic"] = out / f"{name}_generic.step"
    return paths
