"""Mechanism fixtures for the understanding layer (U4): gears, belts, motors, lead screw.

Ground truth:
- gear_pair_3to1: 16T and 48T spur gears, module 1.5, centre distance 48, ratio 3.0.
- belt_drive_gt2: 20T and 60T GT2 pulleys, centre distance 60, ratio 3.0, belt length ~202.7 mm.
- direct_drive_joint: a NEMA 17 whose shaft is pressed into an arm: one revolute, direct drive.
- leadscrew_axis: a "T8x8" screw through the carriage nut bore: lead 8 mm drives a prismatic joint.
- arm_with_belt: the two-link arm with a motor and a 20T/60T GT2 pair on the shoulder.
"""

from __future__ import annotations

import math
from pathlib import Path

from assembly_fixtures import Node, write_assembly
from assembly_fixtures_u import FIXTURE_DIR, X_AXIS, bearing_608
from build123d import Box, Cylinder, Location, Polyline, Pos, Rot, Shape, extrude, make_face


def toothed_wheel(n: int, root_r: float, outer_r: float, thick: float, bore_r: float) -> Shape:
    """Spur wheel with n trapezoid teeth (flat lands) extruded from z = 0 to thick."""
    step = 2 * math.pi / n
    pts = []
    for k in range(n):
        t = k * step
        for r, a in (
            (root_r, -0.30 * step),
            (outer_r, -0.17 * step),
            (outer_r, 0.17 * step),
            (root_r, 0.30 * step),
        ):
            pts.append((r * math.cos(t + a), r * math.sin(t + a)))
    wheel = extrude(make_face(Polyline(*pts, close=True)), thick)
    return wheel - Pos(0, 0, thick / 2) * Cylinder(bore_r, thick + 2)


def spur_gear(n: int, module: float = 1.5, thick: float = 8.0, bore: float = 6.2) -> Shape:
    d = module * n
    return toothed_wheel(n, d / 2 - 1.25 * module, d / 2 + module, thick, bore / 2)


def gt2_pulley(n: int, thick: float = 6.0, bore: float = 6.2) -> Shape:
    pd = n * 2.0 / math.pi
    ro = pd / 2 - 0.38
    return toothed_wheel(n, ro - 0.75, ro, thick, bore / 2)


def _plate_with_pins(centres: list[float], width: float = 80.0) -> Shape:
    x0, x1 = min(centres) - 40, max(centres) + 40
    plate = Pos((x0 + x1) / 2, 0, 3) * Box(x1 - x0, width, 6)
    for x in centres:
        plate = plate - Pos(x, 0, 3) * Cylinder(3.0, 8)
    return plate


def gear_pair_3to1(out: Path = FIXTURE_DIR) -> Path:
    protos = {
        "Plate": _plate_with_pins([0, 48]),
        "Pin": Cylinder(3, 16),
        "GearSmall": spur_gear(16),
        "GearLarge": spur_gear(48),
    }
    root = Node(
        "GearPair",
        children=[
            Node("Plate", "Plate", Location()),
            Node("Pin1", "Pin", Pos(0, 0, 8)),
            Node("Pin2", "Pin", Pos(48, 0, 8)),
            Node("GearA", "GearSmall", Pos(0, 0, 6)),
            Node("GearB", "GearLarge", Pos(48, 0, 6)),
        ],
    )
    return write_assembly(out / "gear_pair_3to1.step", protos, root)


def belt_drive_gt2(out: Path = FIXTURE_DIR) -> Path:
    protos = {
        "Plate": _plate_with_pins([0, 60]),
        "Pin": Cylinder(3, 16),
        "Pulley20": gt2_pulley(20),
        "Pulley60": gt2_pulley(60),
    }
    root = Node(
        "BeltDrive",
        children=[
            Node("Plate", "Plate", Location()),
            Node("Pin1", "Pin", Pos(0, 0, 8)),
            Node("Pin2", "Pin", Pos(60, 0, 8)),
            Node("PulleyA", "Pulley20", Pos(0, 0, 6)),
            Node("PulleyB", "Pulley60", Pos(60, 0, 6)),
        ],
    )
    return write_assembly(out / "belt_drive_gt2.step", protos, root)


def nema17_motor(shaft: float = 24.0) -> Shape:
    """NEMA 17 body 42.3 x 42.3 x 40 below z = 0, 22 mm pilot boss, 5 mm shaft (default 24 long)."""
    body = Pos(0, 0, -20) * Box(42.3, 42.3, 40)
    body = body + Pos(0, 0, 1) * Cylinder(11, 2) + Pos(0, 0, shaft / 2) * Cylinder(2.5, shaft)
    for x in (-15.5, 15.5):
        for y in (-15.5, 15.5):
            body = body - Pos(x, y, -2.25) * Cylinder(1.25, 4.5)
    return body


def motor_plate() -> Shape:
    plate = Pos(0, 0, 2.5) * Box(60, 60, 5) - Pos(0, 0, 2.5) * Cylinder(11, 7)
    for x in (-15.5, 15.5):
        for y in (-15.5, 15.5):
            plate = plate - Pos(x, y, 2.5) * Cylinder(1.7, 7)
    return plate


def direct_drive_joint(out: Path = FIXTURE_DIR) -> Path:
    arm = Box(60, 12, 6) - Pos(-24, 0, 0) * Cylinder(2.5, 8)
    protos = {"MotorPlate": motor_plate(), "Stepper": nema17_motor(), "Arm": arm}
    root = Node(
        "DirectDrive",
        children=[
            Node("MotorPlate", "MotorPlate", Location()),
            Node("Motor", "Stepper", Location()),
            Node("Arm", "Arm", Pos(24, 0, 11)),
        ],
    )
    return write_assembly(out / "direct_drive_joint.step", protos, root)


def leadscrew_axis(out: Path = FIXTURE_DIR) -> Path:
    plate = (
        Box(10, 50, 30)
        - Pos(0, 15, 0) * X_AXIS * Cylinder(4, 12)
        - Pos(0, -15, 0) * X_AXIS * Cylinder(4, 12)
        - X_AXIS * Cylinder(4.1, 12)
    )
    carriage = (
        Box(40, 50, 24)
        - Pos(0, 15, 0) * X_AXIS * Cylinder(7.5, 42)
        - Pos(0, -15, 0) * X_AXIS * Cylinder(7.5, 42)
        - X_AXIS * Cylinder(4.1, 42)
    )
    protos = {
        "EndPlate": plate,
        "Rod": Cylinder(4, 220),
        "Carriage": carriage,
        "LM8UU": Cylinder(7.5, 24) - Cylinder(4, 25),
        "T8x8_Leadscrew": Cylinder(4, 220),
    }
    root = Node(
        "LeadScrewAxis",
        children=[
            Node("EndPlateLeft", "EndPlate", Pos(-105, 0, 20)),
            Node("EndPlateRight", "EndPlate", Pos(105, 0, 20)),
            Node("RodFront", "Rod", Pos(0, -15, 20) * X_AXIS),
            Node("RodBack", "Rod", Pos(0, 15, 20) * X_AXIS),
            Node("Carriage", "Carriage", Pos(0, 0, 20)),
            Node("BushingFront", "LM8UU", Pos(0, -15, 20) * X_AXIS),
            Node("BushingBack", "LM8UU", Pos(0, 15, 20) * X_AXIS),
            Node("Screw", "T8x8_Leadscrew", Pos(0, 0, 20) * X_AXIS),
        ],
    )
    return write_assembly(out / "leadscrew_axis.step", protos, root)


def arm_with_belt(out: Path = FIXTURE_DIR) -> Path:
    """The two-link arm with a NEMA 17 driving the shoulder shaft through 20T/60T GT2 pulleys.

    Shoulder shaft axis: +X through (y = 0, z = 30); motor shaft axis: +X through (y = -60, z = 30)
    with the shaft end toward +X. Pulleys sit at x = -31..-25, outside the base. Reduction 3:1.
    """
    base = Pos(0, 0, 22.5) * Box(40, 30, 45) - Pos(0, 0, 30) * X_AXIS * Cylinder(11, 50)
    # one solid: a bridge and a motor plate (x = -47..-42, bore for the 22 mm pilot, 4 screw holes)
    base = base + Pos(-31, -30, 5) * Box(22, 80, 10)
    mount = Pos(-44.5, -60, 30) * Box(5, 60, 60) - Pos(-44.5, -60, 30) * X_AXIS * Cylinder(11, 7)
    for dy in (-15.5, 15.5):
        for dz in (-15.5, 15.5):
            mount = mount - Pos(-44.5, -60 + dy, 30 + dz) * X_AXIS * Cylinder(1.7, 7)
    base = base + mount
    arm1 = (
        Box(10, 30, 90)
        - Pos(0, 0, -37) * X_AXIS * Cylinder(4, 12)
        - Pos(0, 0, 28) * X_AXIS * Cylinder(11, 12)
    )
    arm2 = Box(10, 30, 63) - Pos(0, 0, -26) * X_AXIS * Cylinder(4, 12)
    protos = {
        "Base": base,
        "Bearing608": bearing_608(),
        "ShaftA": Cylinder(4, 70),
        "ShaftB": Cylinder(4, 20),
        "ArmLink1": arm1,
        "ArmLink2": arm2,
        "Stepper": nema17_motor(),
        "PulleySmall": gt2_pulley(20, thick=6, bore=5.0),
        "PulleyBig": gt2_pulley(60, thick=6, bore=8.0),
    }
    root = Node(
        "ArmWithBelt",
        children=[
            Node("Base", "Base", Location()),
            Node("Bearing1", "Bearing608", Pos(0, 0, 30) * X_AXIS),
            Node("Shaft1", "ShaftA", Pos(3, 0, 30) * X_AXIS),
            Node("Arm1", "ArmLink1", Pos(27, 0, 67)),
            Node("Bearing2", "Bearing608", Pos(27, 0, 95) * X_AXIS),
            Node("Shaft2", "ShaftB", Pos(34, 0, 95) * X_AXIS),
            Node("Arm2", "ArmLink2", Pos(39, 0, 121)),
            Node("Motor", "Stepper", Pos(-47, -60, 30) * Rot(0, 90, 0)),
            Node("PulleyMotor", "PulleySmall", Pos(-31, -60, 30) * X_AXIS),
            Node("PulleyShoulder", "PulleyBig", Pos(-31, 0, 30) * X_AXIS),
        ],
    )
    return write_assembly(out / "arm_with_belt.step", protos, root)


def generate_mechanisms(out: Path = FIXTURE_DIR) -> dict[str, Path]:
    paths = {fn.__name__: fn(out) for fn in BUILDERS}
    paths["mini_mobile_manipulator_generic"] = out / "mini_mobile_manipulator_generic.step"
    return paths


def mini_mobile_manipulator(out: Path = FIXTURE_DIR) -> Path:
    """Wheeled base (4 wheels on axles in bearings) carrying the belt-driven two-link arm.

    End-to-end ground truth: topology mobile_manipulator, 4 wheel joints, a 2-DOF arm whose
    shoulder is driven by a motor through a 3:1 GT2 belt, one planar rest without fasteners
    (arm mount on the chassis) and no floating parts.
    """
    from assembly_fixtures_u import Y_AXIS, _generic

    chassis = Pos(0, 0, 15) * Box(120, 60, 26)
    for x in (-40, 40):
        for y in (-1, 1):
            chassis = chassis - Pos(x, y * 25, 15) * Y_AXIS * Cylinder(11, 10)
    base = Pos(0, 0, 22.5) * Box(40, 30, 45) - Pos(0, 0, 30) * X_AXIS * Cylinder(11, 50)
    base = base + Pos(-31, -30, 5) * Box(22, 80, 10)
    mount = Pos(-44.5, -60, 30) * Box(5, 60, 60) - Pos(-44.5, -60, 30) * X_AXIS * Cylinder(11, 7)
    for dy in (-15.5, 15.5):
        for dz in (-15.5, 15.5):
            mount = mount - Pos(-44.5, -60 + dy, 30 + dz) * X_AXIS * Cylinder(1.7, 7)
    base = base + mount
    arm1 = (
        Box(10, 30, 90)
        - Pos(0, 0, -37) * X_AXIS * Cylinder(4, 12)
        - Pos(0, 0, 28) * X_AXIS * Cylinder(11, 12)
    )
    arm2 = Box(10, 30, 63) - Pos(0, 0, -26) * X_AXIS * Cylinder(4, 12)
    protos = {
        "Chassis": chassis,
        "Bearing608": bearing_608(),
        "Axle": Cylinder(4, 30),
        "Wheel": Cylinder(15, 10) - Cylinder(4, 11),
        "ArmMount": Pos(0, 0, 2) * Box(60, 40, 4),
        "ArmBase": base,
        "ShaftA": Cylinder(4, 70),
        "ShaftB": Cylinder(4, 20),
        "ArmLink1": arm1,
        "ArmLink2": arm2,
        "Stepper": nema17_motor(),
        "PulleySmall": gt2_pulley(20, thick=6, bore=5.0),
        "PulleyBig": gt2_pulley(60, thick=6, bore=8.0),
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
    lift = Pos(0, 0, 32)
    kids += [
        Node("ArmMount", "ArmMount", Pos(0, 0, 28)),
        Node("ArmBase", "ArmBase", lift),
        Node("ShoulderBearing", "Bearing608", lift * Pos(0, 0, 30) * X_AXIS),
        Node("ShoulderShaft", "ShaftA", lift * Pos(3, 0, 30) * X_AXIS),
        Node("UpperArm", "ArmLink1", lift * Pos(27, 0, 67)),
        Node("ElbowBearing", "Bearing608", lift * Pos(27, 0, 95) * X_AXIS),
        Node("ElbowShaft", "ShaftB", lift * Pos(34, 0, 95) * X_AXIS),
        Node("Forearm", "ArmLink2", lift * Pos(39, 0, 121)),
        Node("ShoulderMotor", "Stepper", lift * Pos(-47, -60, 30) * Rot(0, 90, 0)),
        Node("MotorPulley", "PulleySmall", lift * Pos(-31, -60, 30) * X_AXIS),
        Node("ShoulderPulley", "PulleyBig", lift * Pos(-31, 0, 30) * X_AXIS),
    ]
    root = Node("MiniMobileManipulator", children=kids)
    named = write_assembly(out / "mini_mobile_manipulator.step", protos, root)
    gp, groot = _generic(protos, root)
    write_assembly(out / "mini_mobile_manipulator_generic.step", gp, groot)
    return named


BUILDERS = (
    gear_pair_3to1,
    belt_drive_gt2,
    direct_drive_joint,
    leadscrew_axis,
    arm_with_belt,
    mini_mobile_manipulator,
)


def limited_rotation_joint(out: Path = FIXTURE_DIR) -> Path:
    """An arm turning on a pin with a stop pin at 90 degrees: free to about 85, blocked at +90.

    The arm lies along +X from the pivot at (0, 0); the 2 mm stop pin stands at (0, 30), so the
    arm hits it when it has turned 90 degrees counter-clockwise (the 5 degree check steps first
    see the collision at +90).
    """
    plate = Pos(0, 0, 3) * Box(80, 80, 6)
    plate = plate - Pos(0, 0, 3) * Cylinder(3.0, 8) - Pos(0, 30, 3) * Cylinder(1.0, 8)
    arm = Cylinder(6, 4) + Pos(32.5, 0, 0) * Box(55, 2, 4)  # hub plus a thin bar overlapping it
    arm = arm - Cylinder(3.1, 6)
    protos = {"Plate": plate, "Pivot": Cylinder(3, 14), "Stop": Cylinder(1, 14), "Arm": arm}
    root = Node(
        "LimitedRotation",
        children=[
            Node("Plate", "Plate", Location()),
            Node("Pivot", "Pivot", Pos(0, 0, 7)),
            Node("Stop", "Stop", Pos(0, 30, 7)),
            Node("Arm", "Arm", Pos(0, 0, 8)),
        ],
    )
    return write_assembly(out / "limited_rotation_joint.step", protos, root)


BUILDERS = (*BUILDERS, limited_rotation_joint)


def servo_horn_arm(out: Path = FIXTURE_DIR) -> Path:
    """A bus servo (named like the Feetech STS3215) on a base plate with a link on its horn.

    Ground truth: one revolute joint about +Z between the servo and the link (the link has four
    screw holes on the horn circle, 7 mm from the output axis); the base plate has none.
    """
    servo = Pos(0, 0, 17.5) * Box(45, 25, 35) + Pos(0, 0, 37) * Cylinder(10, 4)
    link = Box(70, 20, 4)
    for sx in (-5, 5):
        for sy in (-5, 5):
            link = link - Pos(sx, sy, 0) * Cylinder(1.25, 6)
    base = Pos(0, 0, -3) * Box(80, 60, 6)
    protos = {"BasePlate": base, "STS3215": servo, "Link": link}
    root = Node(
        "ServoArm",
        children=[
            Node("BasePlate", "BasePlate", Location()),
            Node("STS3215", "STS3215", Location()),
            Node("Link", "Link", Pos(0, 0, 41)),
        ],
    )
    return write_assembly(out / "servo_horn_arm.step", protos, root)


BUILDERS = (*BUILDERS, servo_horn_arm)
