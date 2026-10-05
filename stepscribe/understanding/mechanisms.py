# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Mechanisms: direct drive, gear pairs, belt drives, lead screws, rack and pinion.

Detected from tooth counts (rotational repeats of equal faces), axes and centre distances; the
catalogue data (standard modules, belt pitches, lead screws) lives in sourced YAML files.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from stepscribe.assembly.tree import InstanceData
from stepscribe.describe.phrases import fmt
from stepscribe.features.hole_standards import load_table
from stepscribe.features.shape_class import GEAR_MIN_TEETH, _signed_normal_angle
from stepscribe.geometry.occ_utils import Vec, canonical_dir
from stepscribe.geometry.part_geom import PartGeom
from stepscribe.models.schema import Evidence, KinematicModel, KinJoint, Mechanism, Part
from stepscribe.understanding.kinematics import KinContext, refresh_description, retopology

if TYPE_CHECKING:
    from stepscribe.api import Analysis

AXIS_PARALLEL_COS = math.cos(math.radians(0.5))
MIN_AXIAL_OVERLAP_FRACTION = 0.5
BELT_CLEARANCE_MM = 1.0  # centre distance beyond the tooth tips for a belt (else the teeth mesh)
PULLEY_OD_BELOW_PITCH = 1.5  # tip circle may be this far under the pitch circle (belt tooth depth)
PULLEY_OD_ABOVE_PITCH = 0.3


@dataclass
class Wheel:
    """A toothed rotating part: teeth around one axis, placed in the global frame."""

    inst: InstanceData
    teeth: int
    axis: Vec  # canonical unit direction (global)
    centre: Vec  # point on the axis (global)
    od: float
    width: float
    link: str

    # filled by classify()
    module: float | None = None  # snapped standard module
    module_raw: float = 0.0
    pulley: tuple[str, float] | None = None  # (profile, pitch mm)

    @property
    def pitch_diameter(self) -> float | None:
        return self.module * self.teeth if self.module else None


def gears() -> dict:  # type: ignore[type-arg]
    return load_table("gears.yaml")


def belts() -> dict:  # type: ignore[type-arg]
    return load_table("belts.yaml")


def find_teeth(part: Part, geom: PartGeom) -> tuple[np.ndarray, int, int] | None:
    """(axis in the part frame, tooth count, OBB axis index) for a rotationally repeating part."""
    ob = part.obb
    axes = [np.array([a.x, a.y, a.z]) for a in ob.axes]
    centre = np.array([ob.center.x, ob.center.y, ob.center.z])
    skip = {fid for h in part.holes for s in h.segments for fid in s.face_ids}
    faces = [f for f in geom.table.faces if f.id not in skip]
    for idx, axis in enumerate(axes):
        groups: dict[tuple[str, float, float], list] = {}  # type: ignore[type-arg]
        for f in faces:
            rel = f.centroid - centre
            rel = rel - axis * float(np.dot(rel, axis))
            if np.linalg.norm(rel) < 1e-3:
                continue
            groups.setdefault(
                (f.kind, round(f.area, 2), _signed_normal_angle(f, rel, axis)), []
            ).append(f)
        best: tuple[int, int] | None = None
        for _key, members in sorted(groups.items()):
            if len(members) < GEAR_MIN_TEETH:
                continue
            rel = np.array([m.centroid - centre for m in members])
            rel = rel - np.outer(rel @ axis, axis)
            radii = np.linalg.norm(rel, axis=1)
            if radii.std() > 0.02 * radii.mean():
                continue
            e1 = rel[0] / np.linalg.norm(rel[0])
            e2 = np.cross(axis, e1)
            ang = np.sort(np.degrees(np.arctan2(rel @ e2, rel @ e1)) % 360.0)
            gaps = np.diff(np.r_[ang, ang[0] + 360.0])
            if float(np.std(gaps)) < 1.0 and (best is None or len(members) > best[0]):
                best = (len(members), idx)
        if best is not None:
            return axes[best[1]], best[0], best[1]
    return None


def toothed_wheels(ctx: KinContext) -> list[Wheel]:
    out: list[Wheel] = []
    for inst in ctx.insts:
        part = inst.ap.part
        if part.shape_class.label != "gear_like":
            continue
        found = find_teeth(part, inst.ap.geom)
        if found is None:
            continue
        axis_local, n, idx = found
        sizes = [2 * part.obb.half_sizes.x, 2 * part.obb.half_sizes.y, 2 * part.obb.half_sizes.z]
        others = [s for i, s in enumerate(sizes) if i != idx]
        centre_local = np.array([part.obb.center.x, part.obb.center.y, part.obb.center.z])
        out.append(
            Wheel(
                inst=inst,
                teeth=n,
                axis=canonical_dir(inst.to_global_dir(axis_local)),
                centre=inst.to_global_point(centre_local),
                od=float(max(others)),
                width=float(sizes[idx]),
                link=ctx.link_of[inst.id],
            )
        )
    return out


def classify_wheel(w: Wheel) -> None:
    """Snap to a standard module (gear) and/or match a belt pitch (pulley)."""
    g = gears()
    raw = w.od / (w.teeth + 2)
    w.module_raw = raw
    for m in g["standard_modules_mm"]:
        if abs(raw - float(m)) <= float(g["module_snap_tolerance"]) * float(m):
            w.module = float(m)
            break
    best: tuple[float, str, float] | None = None
    for name, row in belts()["profiles"].items():
        p = float(row["pitch_mm"])
        pd = w.teeth * p / math.pi
        if pd - PULLEY_OD_BELOW_PITCH <= w.od <= pd + PULLEY_OD_ABOVE_PITCH:
            err = abs(w.od - (pd - 0.38 * 2))
            if best is None or err < best[0]:
                best = (err, name, p)
    if best:
        w.pulley = (best[1], best[2])


def _ev(code: str, refs: list[str], weight: float, text: str) -> Evidence:
    return Evidence(code=code, refs=refs, weight=weight, text=text)


def _axis_distance(c1: Vec, c2: Vec, d: Vec) -> float:
    off = c2 - c1
    return float(np.linalg.norm(off - d * float(np.dot(off, d))))


def _axial_overlap(a: Wheel, b: Wheel) -> float:
    d = a.axis
    a0 = float(np.dot(a.centre, d))
    b0 = float(np.dot(b.centre, d))
    lo = max(a0 - a.width / 2, b0 - b.width / 2)
    hi = min(a0 + a.width / 2, b0 + b.width / 2)
    return hi - lo


def _joint_for_wheel(
    model: KinematicModel, w: Wheel, exclude_motor: bool = True
) -> KinJoint | None:
    for j in model.joints:
        if w.link not in (j.parent_link, j.child_link) or j.kind != "revolute":
            continue
        if exclude_motor and any(e.code == "motor:shaft" for e in j.evidence):
            continue
        o = np.array([j.axis.origin.x, j.axis.origin.y, j.axis.origin.z])
        d = np.array([j.axis.direction.x, j.axis.direction.y, j.axis.direction.z])
        d = d / np.linalg.norm(d)
        if abs(float(np.dot(d, w.axis))) < AXIS_PARALLEL_COS:
            continue
        if _axis_distance(o, w.centre, d) <= 0.5:
            return j
    return None


def belt_length(d1: float, d2: float, c: float) -> float:
    """Open-belt length from pitch diameters and centre distance."""
    return 2 * c + math.pi * (d1 + d2) / 2 + (d2 - d1) ** 2 / (4 * c)


def _motors(model: KinematicModel, ctx: KinContext) -> dict[str, tuple[str, KinJoint]]:
    """driven link -> (motor instance id, the motor-shaft joint)."""
    out: dict[str, tuple[str, KinJoint]] = {}
    for j in model.joints:
        for e in j.evidence:
            if e.code == "motor:shaft" and len(e.refs) >= 2:
                out[ctx.link_of[e.refs[1]]] = (e.refs[0], j)
    return out


def _servo_horn_drives(model: KinematicModel, ctx: KinContext) -> list[Mechanism]:
    """A servo whose output horn carries a link drives that joint directly (ratio 1:1)."""
    out: list[Mechanism] = []
    for joint in model.joints:
        if joint.driven_by:
            continue
        for e in joint.evidence:
            if e.code != "servo:horn" or len(e.refs) < 2:
                continue
            servo_id, partner = e.refs[0], e.refs[1]
            out.append(
                Mechanism(
                    id="",
                    kind="direct_drive",
                    instance_ids=[servo_id, partner],
                    ratio=1.0,
                    parameters={"driven_link": ctx.link_of[partner], "joint": joint.id},
                    input_instance_id=servo_id,
                    output_joint_id=joint.id,
                    confidence=0.7,
                    evidence=[
                        _ev(
                            "servo:horn",
                            [servo_id, partner, joint.id],
                            0.7,
                            f"{partner} is clamped to the output horn of servo {servo_id}, coaxial with joint {joint.id}",
                        )
                    ],
                    description=f"{joint.id} is driven directly by servo {servo_id} (ratio 1:1)",
                )
            )
            break
    return out


def detect_mechanisms(analysis: Analysis, model: KinematicModel, ctx: KinContext) -> None:
    """Append mechanisms to *model* and set ``driven_by`` on the joints they drive."""
    del analysis
    wheels = toothed_wheels(ctx)
    for w in wheels:
        classify_wheel(w)
    motor_of_link = _motors(model, ctx)
    new: list[Mechanism] = []
    wheel_links = {w.link for w in wheels}
    # direct drive: a motor shaft whose partner link carries no gear or pulley
    for link, (motor_id, joint) in sorted(motor_of_link.items()):
        if link in wheel_links:
            joint.driven_by = motor_id
            continue
        partner = next(e.refs[1] for e in joint.evidence if e.code == "motor:shaft")
        new.append(
            Mechanism(
                id="",
                kind="direct_drive",
                instance_ids=[motor_id, partner],
                ratio=1.0,
                parameters={"driven_link": link, "joint": joint.id},
                input_instance_id=motor_id,
                output_joint_id=joint.id,
                confidence=0.85,
                evidence=[
                    _ev(
                        "motor:shaft",
                        [motor_id, partner, joint.id],
                        0.85,
                        f"the shaft of motor {motor_id} is fixed in {partner}, coaxial with joint {joint.id}",
                    )
                ],
                description=f"{joint.id} is driven directly by motor {motor_id} (ratio 1:1)",
            )
        )
        joint.driven_by = motor_id
    new += _servo_horn_drives(model, ctx)
    new += _wheel_pairs(model, ctx, wheels, motor_of_link)
    new += _lead_screws(model, ctx, motor_of_link)
    order = {
        "direct_drive": 0,
        "gear_pair": 1,
        "belt_drive": 2,
        "lead_screw": 3,
        "rack_and_pinion": 4,
    }
    new.sort(key=lambda m: (order.get(m.kind, 9), m.instance_ids))
    start = len(model.mechanisms)
    for k, m in enumerate(new, start + 1):
        m.id = f"M{k}"
    by_joint = {j.id: j for j in model.joints}
    for m in new:
        if m.output_joint_id in by_joint:
            by_joint[m.output_joint_id].driven_by = m.id
    model.mechanisms.extend(new)
    # a lead screw's turn and its nut's travel, or two wheels in mesh or on one belt, are one
    # coupled motion, not two freedoms
    coupled = sum(m.kind == "lead_screw" for m in new) + sum(
        1
        for m in new
        if m.kind in ("gear_pair", "belt_drive")
        and m.parameters.get("input_joint")
        and m.output_joint_id
    )
    model.dof = max(0, model.dof - coupled)
    _train_ratios(model, motor_of_link)
    retopology(model, ctx)
    refresh_description(model, ctx)
    for j in model.joints:
        if j.driven_by:
            j.description += f", driven by {j.driven_by}"


def _wheel_pairs(
    model: KinematicModel,
    ctx: KinContext,
    wheels: list[Wheel],
    motor_of_link: dict[str, tuple[str, KinJoint]],
) -> list[Mechanism]:
    g = gears()
    out: list[Mechanism] = []
    for i, a in enumerate(wheels):
        for b in wheels[i + 1 :]:
            if a.link == b.link:
                continue
            parallel = abs(float(np.dot(a.axis, b.axis))) >= AXIS_PARALLEL_COS
            c = _axis_distance(a.centre, b.centre, a.axis) if parallel else None
            if not parallel:
                gap = float(np.linalg.norm(a.centre - b.centre)) - (a.od + b.od) / 2
                if gap < 2.0:
                    out.append(_unknown(a, b))
                continue
            assert c is not None
            if _axial_overlap(a, b) < MIN_AXIAL_OVERLAP_FRACTION * min(a.width, b.width):
                continue
            mech = None
            if a.module and b.module and abs(a.module - b.module) < 1e-9:
                pa, pb = a.pitch_diameter, b.pitch_diameter
                assert pa is not None and pb is not None
                if (
                    abs(c - (pa + pb) / 2)
                    <= float(g["mesh_centre_distance_tolerance_modules"]) * a.module
                ):
                    mech = _gear_pair(model, motor_of_link, a, b, c, g)
            if mech is None and a.pulley and b.pulley and a.pulley[0] == b.pulley[0]:
                if c >= (a.od + b.od) / 2 + BELT_CLEARANCE_MM:
                    mech = _belt_drive(model, ctx, motor_of_link, a, b, c)
            if mech is not None:
                out.append(mech)
    return out


def _driver(
    a: Wheel, b: Wheel, motor_of_link: dict[str, tuple[str, KinJoint]]
) -> tuple[Wheel, Wheel, str | None, bool]:
    """(driver, driven, motor id, assumed) - the wheel on a motor shaft drives; else the smaller."""
    if a.link in motor_of_link:
        return a, b, motor_of_link[a.link][0], False
    if b.link in motor_of_link:
        return b, a, motor_of_link[b.link][0], False
    return (a, b, None, True) if a.teeth <= b.teeth else (b, a, None, True)


def _gear_pair(
    model: KinematicModel,
    motor_of_link: dict[str, tuple[str, KinJoint]],
    a: Wheel,
    b: Wheel,
    c: float,
    g: dict,  # type: ignore[type-arg]
) -> Mechanism:
    drv, dvn, motor, assumed = _driver(a, b, motor_of_link)
    assert drv.module is not None
    ratio = dvn.teeth / drv.teeth
    out_joint = _joint_for_wheel(model, dvn)
    conf = 0.85 if not assumed else 0.65
    evidence = [
        _ev(
            "gear:teeth",
            [drv.inst.id, dvn.inst.id],
            0.5,
            f"{drv.inst.id} has {drv.teeth} teeth and {dvn.inst.id} has {dvn.teeth}",
        ),
        _ev(
            "gear:module",
            [drv.inst.id, dvn.inst.id],
            0.3,
            f"outside diameters {fmt(drv.od, 2)} and {fmt(dvn.od, 2)} mm give module {fmt(drv.module_raw, 3)} "
            f"and {fmt(dvn.module_raw, 3)}, both snap to the standard module {fmt(drv.module, 2)} "
            f"({g['formulas']['outside_diameter']})",
        ),
        _ev(
            "gear:centre_distance",
            [drv.inst.id, dvn.inst.id],
            0.4,
            f"axes {fmt(c, 2)} mm apart; (d1 + d2)/2 = {fmt(((drv.pitch_diameter or 0) + (dvn.pitch_diameter or 0)) / 2, 2)} mm",
        ),
    ]
    if assumed:
        evidence.append(
            _ev(
                "assumed_driver",
                [drv.inst.id],
                0.0,
                "no motor found: the smaller gear is assumed to drive",
            )
        )
    return Mechanism(
        id="",
        kind="gear_pair",
        instance_ids=[drv.inst.id, dvn.inst.id],
        ratio=round(ratio, 4),
        parameters={
            "module_mm": drv.module,
            "teeth_driver": drv.teeth,
            "teeth_driven": dvn.teeth,
            "centre_distance_mm": round(c, 3),
            "pressure_angle_deg": g["pressure_angle_deg"]["value"],
            "driver_link": drv.link,
            "driven_link": dvn.link,
            "input_joint": getattr(_joint_for_wheel(model, drv, False), "id", None),
        },
        input_instance_id=motor,
        output_joint_id=out_joint.id if out_joint else None,
        confidence=conf,
        evidence=evidence,
        description=(
            f"gear pair {drv.inst.id} ({drv.teeth}T) drives {dvn.inst.id} ({dvn.teeth}T), module "
            f"{fmt(drv.module, 2)}, reduction {fmt(ratio, 2)}:1"
            + (" (driver assumed)" if assumed else "")
        ),
    )


def _belt_drive(
    model: KinematicModel,
    ctx: KinContext,
    motor_of_link: dict[str, tuple[str, KinJoint]],
    a: Wheel,
    b: Wheel,
    c: float,
) -> Mechanism:
    assert a.pulley and b.pulley
    name, pitch = a.pulley
    drv, dvn, motor, assumed = _driver(a, b, motor_of_link)
    d1, d2 = drv.teeth * pitch / math.pi, dvn.teeth * pitch / math.pi
    length = belt_length(d1, d2, c)
    out_joint = _joint_for_wheel(model, dvn)
    belt_inst = next(
        (
            i
            for i in ctx.insts
            if "belt" in f"{i.name} {i.ap.part.name}".lower() and i.id not in (a.inst.id, b.inst.id)
        ),
        None,
    )
    ids = [drv.inst.id, dvn.inst.id] + ([belt_inst.id] if belt_inst else [])
    evidence = [
        _ev(
            "belt:pitch",
            [drv.inst.id, dvn.inst.id],
            0.5,
            f"{drv.teeth} and {dvn.teeth} teeth at {fmt(pitch, 1)} mm pitch ({name}); tip diameters "
            f"{fmt(drv.od, 2)} and {fmt(dvn.od, 2)} mm match pitch circles {fmt(d1, 2)} and {fmt(d2, 2)} mm",
        ),
        _ev("belt:centre_distance", ids[:2], 0.3, f"axes {fmt(c, 2)} mm apart, in the same plane"),
    ]
    if assumed:
        evidence.append(
            _ev(
                "assumed_driver",
                [drv.inst.id],
                0.0,
                "no motor found: the smaller pulley is assumed to drive",
            )
        )
    ratio = dvn.teeth / drv.teeth
    return Mechanism(
        id="",
        kind="belt_drive",
        instance_ids=ids,
        ratio=round(ratio, 4),
        parameters={
            "profile": name,
            "pitch_mm": pitch,
            "teeth_driver": drv.teeth,
            "teeth_driven": dvn.teeth,
            "centre_distance_mm": round(c, 3),
            "belt_length_mm": round(length, 1),
            "belt_teeth": math.ceil(length / pitch),
            "driver_link": drv.link,
            "driven_link": dvn.link,
            "input_joint": getattr(_joint_for_wheel(model, drv, False), "id", None),
        },
        input_instance_id=motor,
        output_joint_id=out_joint.id if out_joint else None,
        confidence=0.8 if not assumed else 0.6,
        evidence=evidence,
        description=(
            f"{name} belt drive {drv.inst.id} ({drv.teeth}T) to {dvn.inst.id} ({dvn.teeth}T), reduction "
            f"{fmt(ratio, 2)}:1, centre distance {fmt(c, 1)} mm, belt about {fmt(length, 0)} mm "
            f"({math.ceil(length / pitch)} teeth)"
        ),
    )


def _unknown(a: Wheel, b: Wheel) -> Mechanism:
    return Mechanism(
        id="",
        kind="unknown",
        instance_ids=[a.inst.id, b.inst.id],
        parameters={"teeth": [a.teeth, b.teeth]},
        confidence=0.3,
        evidence=[
            _ev(
                "gear:non_parallel_axes",
                [a.inst.id, b.inst.id],
                0.3,
                "two toothed parts close together on non-parallel axes",
            )
        ],
        description=(
            f"{a.inst.id} and {b.inst.id} look like a bevel or worm gear pair (axes not parallel): "
            "not supported yet"
        ),
    )


LEAD_RE = re.compile(r"t[r]?8\s*x\s*(\d+(?:\.\d+)?)", re.I)


def _lead_screws(
    model: KinematicModel, ctx: KinContext, motor_of_link: dict[str, tuple[str, KinJoint]]
) -> list[Mechanism]:
    table = load_table("lead_screws.yaml")
    out: list[Mechanism] = []
    for spec in ctx.screws:
        inst = ctx.by_id[spec.screw_id]
        out_id = spec.prismatic.final_id
        text = f"{inst.name} {inst.ap.part.name}"
        m = LEAD_RE.search(text)
        lead = float(m.group(1)) if m else None
        row = table["screws"]["T8"]
        motor = motor_of_link.get(ctx.link_of[spec.screw_id], (None, None))[0]
        support = next(
            (
                j.id
                for j in model.joints
                if j.kind == "revolute"
                and ctx.link_of[spec.screw_id] in (j.parent_link, j.child_link)
            ),
            None,
        )
        params: dict[str, object] = {
            "input_joint": support,
            "screw_instance": spec.screw_id,
            "nut_link": ctx.names.get(spec.nut_link_root, spec.nut_link_root),
        }
        if lead is not None:
            params["lead_mm"] = lead
            conf, why = 0.8, f"lead {fmt(lead, 1)} mm read from the name '{inst.name}'"
        else:
            params["pitch_mm"] = row["pitch_mm"]
            params["common_leads_mm"] = row["common_leads_mm"]
            conf = float(table["default_lead_confidence"])
            why = "lead not in the name: T8 screws come in 2, 4 or 8 mm leads"
        out.append(
            Mechanism(
                id="",
                kind="lead_screw",
                instance_ids=[spec.screw_id],
                ratio=None,
                parameters=params,
                input_instance_id=motor,
                output_joint_id=out_id,
                confidence=conf,
                evidence=[
                    _ev(
                        "lead_screw:name",
                        [spec.screw_id],
                        0.5,
                        f"{spec.screw_id} is named like a lead screw ({inst.name}); {why}",
                    ),
                    _ev(
                        "lead_screw:nut",
                        [spec.screw_id, out_id],
                        0.4,
                        f"it runs through a bore on the link that slides as {out_id}, parallel to the screw",
                    ),
                ],
                description=f"lead screw {spec.screw_id} drives prismatic joint {out_id} ({why})",
            )
        )
    return out


def _train_ratios(model: KinematicModel, motor_of_link: dict[str, tuple[str, KinJoint]]) -> None:
    """Multiply ratios along gear/belt chains that start at a motor-driven link."""
    pairs = [m for m in model.mechanisms if m.kind in ("gear_pair", "belt_drive")]
    memo: dict[str, float] = {link: 1.0 for link in motor_of_link}

    def total(link: str, seen: frozenset[str] = frozenset()) -> float | None:
        if link in memo:
            return memo[link]
        if link in seen:
            return None
        for m in pairs:
            if m.parameters["driven_link"] == link and m.ratio:
                up = total(str(m.parameters["driver_link"]), seen | {link})
                if up is not None:
                    memo[link] = up * m.ratio
                    return memo[link]
        return None

    for m in pairs:
        t = total(str(m.parameters["driven_link"]))
        if t is not None and abs(t - (m.ratio or 0)) > 1e-6:
            m.parameters["train_ratio"] = round(t, 4)
            m.description += f"; overall {fmt(t, 2)}:1 from the motor"
