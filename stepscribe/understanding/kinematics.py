# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Kinematic model: connections, rigid groups (links), ground, joints and DOF.

Connections come from the assembly's measured geometry (fastener joints, coaxial cylinders,
planar contacts). Every joint and every ground choice carries evidence that refers to
instance, contact and joint IDs. Hypotheses, never certainties.
"""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from stepscribe import config
from stepscribe.assembly import build_instances
from stepscribe.assembly.contacts import PartFaces, part_faces
from stepscribe.assembly.tree import InstanceData
from stepscribe.describe.phrases import fmt
from stepscribe.features.hole_standards import load_table
from stepscribe.geometry.occ_utils import Vec, canonical_dir
from stepscribe.geometry.properties import vec3
from stepscribe.models.schema import (
    AssemblyInfo,
    Axis,
    Evidence,
    KinematicModel,
    KinJoint,
    Mechanism,
    RigidGroup,
)

if TYPE_CHECKING:
    from stepscribe.api import Analysis

FASTENER_WORDS = ("screw", "bolt", "nut", "washer", "standoff")


def rules() -> dict:  # type: ignore[type-arg]
    return load_table("roles.yaml")


def axis_vector(label: str) -> Vec:
    v = np.zeros(3)
    v["XYZ".index(label[1])] = 1.0 if label[0] == "+" else -1.0
    return v


def axis_label(d: Vec) -> str:
    """'+X' style label when *d* is along a principal axis, else the unit vector."""
    d = d / np.linalg.norm(d)
    k = int(np.argmax(np.abs(d)))
    if abs(abs(float(d[k])) - 1.0) < 1e-3:
        return ("+" if d[k] > 0 else "-") + "XYZ"[k]
    return f"({fmt(d[0], 2)}, {fmt(d[1], 2)}, {fmt(d[2], 2)})"


LEAD_SCREW_RE = re.compile(r"(lead ?screw|leadscrew|trapezoidal|acme|t8(x\d+)?|tr8)", re.I)


def is_lead_screw(inst: InstanceData) -> bool:
    """Named like a lead screw (keywords from lead_screws.yaml: leadscrew, T8, trapezoidal...)."""
    text = f"{inst.name} {inst.ap.part.name}".replace("_", " ")
    return LEAD_SCREW_RE.search(text) is not None


def hardware_kind(inst: InstanceData) -> str | None:
    """'bearing', 'linear_bearing', 'fastener' or None (from the Phase 5 hardware guess)."""
    p = inst.ap.part
    if not p.likely_purchased_hardware or not p.hardware_guess:
        return None
    g = p.hardware_guess.lower()
    if "linear bearing" in g or "linear ball" in g:
        return "linear_bearing"
    if "bearing" in g:
        return "bearing"
    if "motor" in g or "servo" in g:
        return "motor"
    if any(w in g for w in FASTENER_WORDS):
        return "fastener"
    return None


@dataclass
class CylPair:
    """A convex cylinder (shaft) of one instance inside a concave one (bore) of another."""

    shaft: InstanceData
    bore: InstanceData
    origin: Vec  # global point on the shared axis
    direction: Vec  # canonical unit
    shaft_r: float
    bore_r: float
    shaft_len: float
    bore_len: float
    overlap: float

    @property
    def diametral(self) -> float:
        return 2.0 * (self.bore_r - self.shaft_r)


def cylinder_pairs(
    a: InstanceData, b: InstanceData, fa: PartFaces, fb: PartFaces, tol: float
) -> list[CylPair]:
    """All coaxial shaft/bore pairs that overlap axially between two instances."""
    out: list[CylPair] = []
    for shaft_i, shaft_f, bore_i, bore_f in ((a, fa, b, fb), (b, fb, a, fa)):
        for s in (c for c in shaft_f.cylinders if c.convex):
            so, sd = shaft_i.to_global_point(s.origin), shaft_i.to_global_dir(s.direction)
            for h in (c for c in bore_f.cylinders if not c.convex):
                ho, hd = bore_i.to_global_point(h.origin), bore_i.to_global_dir(h.direction)
                if abs(abs(float(np.dot(sd, hd))) - 1.0) > 1e-4:
                    continue
                off = ho - so
                if float(np.linalg.norm(off - sd * np.dot(off, sd))) > tol:
                    continue
                s0, s1 = sorted((float(np.dot(so, sd)) + s.t0, float(np.dot(so, sd)) + s.t1))
                sign = 1.0 if float(np.dot(hd, sd)) > 0 else -1.0
                h0, h1 = sorted(
                    (float(np.dot(ho, sd)) + sign * h.t0, float(np.dot(ho, sd)) + sign * h.t1)
                )
                overlap = min(s1, h1) - max(s0, h0)
                if overlap <= config.LINEAR_TOL:
                    continue
                if abs(2 * (h.radius - s.radius)) > 2 * config.NEAR_MISS_MAX:
                    continue
                mid = 0.5 * (max(s0, h0) + min(s1, h1))
                centre = so + sd * (mid - float(np.dot(so, sd)))
                out.append(
                    CylPair(
                        shaft_i,
                        bore_i,
                        centre,
                        canonical_dir(sd),
                        s.radius,
                        h.radius,
                        s1 - s0,
                        h1 - h0,
                        overlap,
                    )
                )
    out.sort(key=lambda p: (-p.overlap, p.shaft.id, p.bore.id))
    return out


@dataclass
class Conn:
    """A classified connection between two instances."""

    a: str
    b: str
    kind: str  # fixed | revolute | prismatic | cylindrical | soft
    confidence: float
    axis_origin: Vec | None = None
    axis_dir: Vec | None = None
    evidence: list[Evidence] = field(default_factory=list)
    area: float = 0.0


def _ev(code: str, refs: list[str], weight: float, text: str) -> Evidence:
    return Evidence(code=code, refs=refs, weight=weight, text=text)


def _servo_output_axis(inst: InstanceData) -> tuple[np.ndarray, np.ndarray, float] | None:
    """(point, direction, radius) of a servo's output axis: its widest full convex cylinder.

    A servo body is a box with a round output boss (and often a matching idle boss on the other
    side); the widest cylinder is that boss, so its axis is the output axis.
    """
    p = inst.ap.part
    if not p.likely_purchased_hardware or "servo" not in (p.hardware_guess or "").lower():
        return None
    cyls = [c for c in part_faces(inst.ap.geom).cylinders if c.convex and c.radius >= 4.0]
    if not cyls:
        return None
    c = max(cyls, key=lambda x: x.radius)
    return inst.to_global_point(c.origin), inst.to_global_dir(c.direction), float(c.radius)


def _servo_horn_conn(
    servo: InstanceData,
    other: InstanceData,
    faces: Callable[[InstanceData], PartFaces],
    refs: list[str],
) -> Conn | None:
    """A part that touches a servo and has screw holes on the servo's horn circle is clamped to
    the output horn, so it turns with the output shaft: a revolute joint about the output axis.

    The bracket that holds the servo case has no holes on that circle, so it stays fixed.
    """
    got = _servo_output_axis(servo)
    if got is None:
        return None
    o, d, radius = got
    d = d / np.linalg.norm(d)
    n = 0
    for c in faces(other).cylinders:
        if c.convex or c.radius > 2.5:
            continue
        co, cd = other.to_global_point(c.origin), other.to_global_dir(c.direction)
        if abs(abs(float(np.dot(cd, d))) - 1.0) > 1e-2:
            continue
        off = co - o
        dist = float(np.linalg.norm(off - d * float(np.dot(off, d))))
        if 3.0 <= dist <= radius - 1.0:
            n += 1
    if n < 2:
        return None
    return Conn(
        servo.id,
        other.id,
        "revolute",
        0.7,
        o,
        canonical_dir(d),
        [
            _ev(
                "servo:horn",
                refs,
                0.7,
                f"{other.id} has {n} screw holes on the output horn circle of servo {servo.id}, so it "
                f"turns with the output shaft (about {axis_label(canonical_dir(d))})",
            )
        ],
    )


def classify_connections(
    analysis: Analysis, insts: list[InstanceData]
) -> tuple[list[Conn], dict[tuple[str, str], list[str]]]:
    """Connections for every touching instance pair; also pair -> contact IDs for evidence."""
    asm = analysis.report.assembly
    assert asm is not None
    r = rules()["kinematics"]
    conf = r["confidence"]
    by_id = {i.id: i for i in insts}
    pf: dict[str, PartFaces] = {}

    def faces(i: InstanceData) -> PartFaces:
        if i.part_id not in pf:
            pf[i.part_id] = part_faces(i.ap.geom)
        return pf[i.part_id]

    contacts_of: dict[tuple[str, str], list[str]] = {}
    planar: dict[tuple[str, str], float] = {}
    for c in asm.contacts:
        if c.kind == "near_miss":
            continue
        if c.kind != "cylindrical_fit" and c.min_distance_mm >= config.CONTACT_TOL:
            continue  # a pin in a clearance hole is not "touching" but is still a connection
        lo_id, hi_id = sorted((c.instance_a, c.instance_b))
        key = (lo_id, hi_id)
        contacts_of.setdefault(key, []).append(c.id)
        if c.kind == "planar_face":
            planar[key] = planar.get(key, 0.0) + (c.contact_area_mm2 or 0.0)
    conns: list[Conn] = []
    fastened: set[tuple[str, str]] = set()
    for j in asm.fastener_joints:
        existing = by_id.get(j.existing_fastener_instance_id or "") or _pin_in_joint(
            j, insts, faces, float(r["coaxial_tol_mm"])
        )
        if existing is not None and hardware_kind(existing) != "fastener":
            continue  # a pin that is not a screw/bolt: judged by its fit below (pivot or press)
        ids = list(j.instance_ids) + (
            [j.existing_fastener_instance_id] if j.existing_fastener_instance_id else []
        )
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                lo_id, hi_id = sorted((ids[x], ids[y]))
                key = (lo_id, hi_id)
                if key in fastened:
                    continue
                fastened.add(key)
                conns.append(
                    Conn(
                        key[0],
                        key[1],
                        "fixed",
                        conf["fastener_joint"],
                        evidence=[
                            _ev(
                                "fastener_joint",
                                [j.id, *key],
                                1.0,
                                f"{key[0]} and {key[1]} are clamped by fastener joint {j.id}",
                            )
                        ],
                    )
                )
    for key in sorted(contacts_of):
        if key in fastened:
            continue
        a, b = by_id[key[0]], by_id[key[1]]
        refs = [a.id, b.id, *contacts_of[key]]
        ka, kb = hardware_kind(a), hardware_kind(b)
        if ka == "fastener" or kb == "fastener":
            conns.append(
                Conn(
                    a.id,
                    b.id,
                    "fixed",
                    conf["fastener_part"],
                    evidence=[
                        _ev("fastener_part", refs, 0.85, "touches a screw, nut, washer or standoff")
                    ],
                )
            )
            continue
        horn = None
        for servo, other in ((a, b), (b, a)):
            horn = horn or _servo_horn_conn(servo, other, faces, [servo.id, other.id, *refs[2:]])
        if horn is not None:
            conns.append(horn)
            continue
        pairs = cylinder_pairs(a, b, faces(a), faces(b), float(r["coaxial_tol_mm"]))
        if pairs:
            conns.append(
                _cyl_conn(a, b, pairs, key in planar, refs, conf, float(r["loose_clearance_mm"]))
            )
        elif key in planar:
            conns.append(
                Conn(
                    a.id,
                    b.id,
                    "soft",
                    0.3,
                    area=planar[key],
                    evidence=[
                        _ev(
                            "planar_only",
                            refs,
                            0.3,
                            f"only a planar contact of {fmt(planar[key], 0)} mm² and no fasteners",
                        )
                    ],
                )
            )
    return conns, contacts_of


def _pin_in_joint(j, insts, faces, tol: float):  # type: ignore[no-untyped-def]
    """An instance outside the joint with a shaft coaxial with the joint axis (a pin, not a bolt)."""
    o = np.array([j.axis.origin.x, j.axis.origin.y, j.axis.origin.z])
    d = np.array([j.axis.direction.x, j.axis.direction.y, j.axis.direction.z])
    d = d / np.linalg.norm(d)
    for inst in insts:
        if inst.id in j.instance_ids:
            continue
        for c in faces(inst).cylinders:
            if not c.convex or c.radius * 2 > j.common_diameter_mm + 1.0:
                continue
            co, cd = inst.to_global_point(c.origin), inst.to_global_dir(c.direction)
            if abs(abs(float(np.dot(cd, d))) - 1.0) > 1e-3:
                continue
            off = co - o
            if float(np.linalg.norm(off - d * float(np.dot(off, d)))) <= tol:
                return inst
    return None


def _motor_shaft_radius(inst: InstanceData) -> float:
    """Radius of the output shaft: the smallest full convex cylinder of the motor part."""
    radii = [c.radius for c in part_faces(inst.ap.geom).cylinders if c.convex and c.radius >= 1.0]
    return min(radii) if radii else 0.0


def _cyl_conn(
    a: InstanceData,
    b: InstanceData,
    pairs: list[CylPair],
    has_planar: bool,
    refs: list[str],
    conf: dict[str, float],
    loose: float,
) -> Conn:
    p = pairs[0]
    ax = (p.origin, p.direction)
    for inst, other in ((a, b), (b, a)):
        if hardware_kind(inst) != "motor":
            continue
        mine = [q for q in pairs if q.shaft is inst]
        if mine and mine[0].shaft_r <= _motor_shaft_radius(inst) + 1e-6:
            q = mine[0]
            return Conn(
                a.id,
                b.id,
                "revolute",
                0.85,
                q.origin,
                q.direction,
                [
                    _ev(
                        "motor:shaft",
                        [inst.id, other.id],
                        0.85,
                        f"{other.id} sits on the output shaft of motor {inst.id} (about {axis_label(q.direction)})",
                    )
                ],
            )
        return Conn(
            a.id,
            b.id,
            "fixed",
            0.8,
            evidence=[
                _ev(
                    "motor:body", refs, 0.8, f"{other.id} is mounted on the body of motor {inst.id}"
                )
            ],
        )
    if is_lead_screw(a) or is_lead_screw(b):
        screw = a if is_lead_screw(a) else b
        return Conn(
            a.id,
            b.id,
            "revolute",
            0.6,
            p.origin,
            p.direction,
            [
                _ev(
                    "lead_screw:fit",
                    [screw.id, (b if screw is a else a).id],
                    0.6,
                    f"{screw.id} is named like a lead screw and runs through {(b if screw is a else a).id}",
                )
            ],
        )
    for inst, other in ((a, b), (b, a)):
        kind = hardware_kind(inst)
        if kind not in ("bearing", "linear_bearing"):
            continue
        mine = [q for q in pairs if q.shaft is inst or q.bore is inst]
        q = mine[0]
        if q.shaft is inst:  # the bearing's outer surface sits in a bore
            return Conn(
                a.id,
                b.id,
                "fixed",
                conf["bearing_outer"],
                evidence=[
                    _ev(
                        "bearing:outer",
                        refs,
                        0.85,
                        f"{inst.id} ({inst.ap.part.hardware_guess}) sits in the bore of {other.id}",
                    )
                ],
            )
        if kind == "linear_bearing":
            return Conn(
                a.id,
                b.id,
                "prismatic",
                conf["linear_bearing"],
                q.origin,
                q.direction,
                [
                    _ev(
                        "bearing:linear",
                        refs,
                        0.8,
                        f"linear bearing {inst.id} slides on {other.id} along {axis_label(q.direction)}",
                    )
                ],
            )
        return Conn(
            a.id,
            b.id,
            "revolute",
            conf["bearing_inner"],
            q.origin,
            q.direction,
            [
                _ev(
                    "bearing:inner",
                    refs,
                    0.9,
                    f"{other.id} runs in the bore of bearing {inst.id} about {axis_label(q.direction)}",
                )
            ],
        )
    if abs(p.diametral) <= 0.011 or p.diametral < 0:
        word = "interference" if p.diametral < -0.011 else "line-to-line"
        return Conn(
            a.id,
            b.id,
            "fixed",
            conf["press_fit"],
            evidence=[
                _ev(
                    "press_fit",
                    refs,
                    0.6,
                    f"{word} fit ({fmt(p.diametral, 3)} mm diametral) between {p.shaft.id} and {p.bore.id}",
                )
            ],
        )
    if p.diametral > loose and not has_planar:
        return Conn(
            a.id,
            b.id,
            "cylindrical",
            conf["loose_pin"],
            ax[0],
            ax[1],
            [
                _ev(
                    "loose_pin",
                    refs,
                    0.4,
                    f"loose pin ({fmt(p.diametral, 2)} mm clearance) with no stop: can rotate and slide",
                )
            ],
        )
    return Conn(
        a.id,
        b.id,
        "revolute",
        conf["clearance_pin"],
        ax[0],
        ax[1],
        [
            _ev(
                "clearance_pin",
                refs,
                0.6,
                f"{p.shaft.id} turns in the bore of {p.bore.id} (clearance {fmt(p.diametral, 2)} mm) about {axis_label(p.direction)}",
            )
        ],
    )


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.p = {i: i for i in items}

    def find(self, x: str) -> str:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            keep, drop = sorted((ra, rb))
            self.p[drop] = keep


@dataclass
class RawJoint:
    kind: str
    la: str  # link roots
    lb: str
    origin: Vec
    direction: Vec
    confidence: float
    evidence: list[Evidence]
    final_id: str = ""


def _parallel(d1: Vec, d2: Vec, tol_deg: float) -> bool:
    c = abs(float(np.dot(d1, d2)))
    return c >= math.cos(math.radians(tol_deg))


def _merge_joints(raw: list[RawJoint], tol_deg: float) -> list[RawJoint]:
    """One joint per link pair and axis direction (rods, paired bearings)."""
    out: list[RawJoint] = []
    for j in sorted(raw, key=lambda j: (-j.confidence, j.la, j.lb)):
        pair = tuple(sorted((j.la, j.lb)))
        for k in out:
            if tuple(sorted((k.la, k.lb))) == pair and _parallel(k.direction, j.direction, tol_deg):
                k.evidence.extend(j.evidence)
                if j.kind == "prismatic" and k.kind != "prismatic" and j.confidence > k.confidence:
                    k.kind, k.confidence = "prismatic", j.confidence
                break
        else:
            out.append(j)
    return out


def _link_score(
    link: str,
    members: list[InstanceData],
    stats: dict[str, float],
    up: Vec,
    low: float,
    r: dict,  # type: ignore[type-arg]
) -> tuple[float, list[Evidence]]:
    g = r["ground"]
    score, ev = 0.0, []
    words = [w for w in g["keywords"]]
    hit = next(
        (m for m in members for w in words if w in m.ap.part.name.lower() or w in m.name.lower()),
        None,
    )
    if hit is not None:
        score += float(g["name_keyword_weight"])
        ev.append(
            _ev(
                "name:ground_word",
                [hit.id],
                float(g["name_keyword_weight"]),
                f"{hit.id} is named '{hit.name}'",
            )
        )
    tol = float(g["lowest_plane_tolerance_mm"])
    touching = [m for m in members if _low(m, up) <= low + tol]
    if touching:
        score += float(g["lowest_plane_weight"])
        ev.append(
            _ev(
                "lowest_plane",
                [t.id for t in touching][:4],
                float(g["lowest_plane_weight"]),
                "touches the lowest plane of the assembly",
            )
        )
    fp, ms = stats["footprint"], stats["mass"]
    if fp > 0:
        w = float(g["footprint_weight"]) * stats["footprint_share"]
        score += w
        ev.append(
            _ev(
                "footprint",
                [m.id for m in members][:4],
                w,
                f"{100 * stats['footprint_share']:.0f}% of the largest footprint",
            )
        )
    if ms > 0:
        w = float(g["mass_weight"]) * stats["mass_share"]
        score += w
        ev.append(
            _ev(
                "mass",
                [m.id for m in members][:4],
                w,
                f"{100 * stats['mass_share']:.0f}% of the largest mass",
            )
        )
    return score, ev


def _low(inst: InstanceData, up: Vec) -> float:
    lo, hi = inst.bounds
    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )
    return float((corners @ up).min())


def _footprint(members: list[InstanceData], up: Vec) -> float:
    lo = np.min([m.bounds[0] for m in members], axis=0)
    hi = np.max([m.bounds[1] for m in members], axis=0)
    size = hi - lo
    mask = 1.0 - np.abs(up)
    dims = size * mask
    dims = dims[dims > 1e-9]
    return float(np.prod(dims)) if len(dims) else 0.0


@dataclass
class ScrewSpec:
    """A lead screw whose nut interface was removed from the joint list (it is a transmission)."""

    screw_id: str
    nut_link_root: str
    prismatic: RawJoint


@dataclass
class KinContext:
    """Working data shared with the mechanism detector."""

    insts: list[InstanceData]
    by_id: dict[str, InstanceData]
    names: dict[str, str]  # link root -> "L0"...
    link_of: dict[str, str]  # instance id -> "L0"...
    screws: list[ScrewSpec]
    asm: AssemblyInfo
    conns: list[Conn]
    contacts_of: dict[tuple[str, str], list[str]]
    soft_used: list[Conn] = field(default_factory=list)


def build_kinematics(analysis: Analysis) -> KinematicModel:
    """Links, joints, ground, DOF and topology of an assembly."""
    return build_kinematics_ex(analysis)[0]


def _joints_for(moving: list[Conn], uf: UnionFind, tol: float) -> list[RawJoint]:
    return _merge_joints(_raw_joints(moving, uf), tol)


def _drop_screw_nuts(
    merged: list[RawJoint], uf: UnionFind, ground: str
) -> tuple[list[RawJoint], list[ScrewSpec]]:
    """Remove lead-screw nut interfaces (a transmission, not a joint).

    The screw turns in its supports and its nut translates: the revolute between the screw and
    the nut would close a loop and wrongly lock the motion. The nut link is the one (never the
    ground link) that slides on another link which also supports the screw.
    """
    specs: list[ScrewSpec] = []
    drop: set[int] = set()
    for idx, rj in enumerate(merged):
        screw = next((e.refs[0] for e in rj.evidence if e.code == "lead_screw:fit"), None)
        if screw is None or rj.kind != "revolute":
            continue
        ls = uf.find(screw)
        nut = rj.lb if rj.la == ls else rj.la
        if nut == ground:
            continue
        supports = {
            (o.lb if o.la == ls else o.la)
            for o in merged
            if o is not rj and ls in (o.la, o.lb) and o.kind == "revolute"
        }
        pri = next(
            (
                p
                for p in merged
                if p.kind == "prismatic"
                and nut in (p.la, p.lb)
                and (p.lb if p.la == nut else p.la) in supports
                and _parallel(p.direction, rj.direction, 2.0)
            ),
            None,
        )
        if pri is not None:
            specs.append(ScrewSpec(screw, nut, pri))
            drop.add(idx)
    return [j for i, j in enumerate(merged) if i not in drop], specs


def build_kinematics_ex(analysis: Analysis) -> tuple[KinematicModel, KinContext]:
    """Like :func:`build_kinematics`, also returning what the mechanism detector needs."""
    asm = analysis.report.assembly
    assert asm is not None
    r = rules()
    insts = build_instances(analysis)
    by_id = {i.id: i for i in insts}
    conns, contacts_of = classify_connections(analysis, insts)
    up = axis_vector(asm.up_axis)
    ids = [i.id for i in insts]

    hard = [c for c in conns if c.kind == "fixed"]
    soft = sorted((c for c in conns if c.kind == "soft"), key=lambda c: (c.a, c.b))
    moving = [c for c in conns if c.kind in ("revolute", "prismatic", "cylindrical")]
    uf = UnionFind(ids)
    for c in hard:
        uf.union(c.a, c.b)
    soft_used: list[Conn] = []
    ground = ""
    ground_ev: list[Evidence] = []
    for _round in range(len(ids) + 1):
        tol = float(r["kinematics"]["parallel_tol_deg"])
        merged = _joints_for(moving, uf, tol)
        links = _links(ids, uf, by_id)
        ground, ground_ev = _pick_ground(links, by_id, up, r)
        reach = _reachable(ground, merged, links)
        pending = [c for c in soft if uf.find(c.a) != uf.find(c.b)]
        attach = None
        for c in pending:  # attach an unreachable component through a planar rest
            ca, cb = uf.find(c.a), uf.find(c.b)
            if (ca in reach) != (cb in reach):
                attach = c
                break
        if attach is None:
            break
        uf.union(attach.a, attach.b)
        soft_used.append(attach)
    links = _links(ids, uf, by_id)
    merged = _joints_for(moving, uf, float(r["kinematics"]["parallel_tol_deg"]))
    ground, ground_ev = _pick_ground(links, by_id, up, r)
    merged, screws = _drop_screw_nuts(merged, uf, ground)
    reach = _reachable(ground, merged, links)

    order = _order_links(ground, merged, links)
    names = {root: f"L{k}" for k, root in enumerate(order)}
    groups: list[RigidGroup] = []
    for root in order:
        members = links[root]
        mass = [m.ap.part.mass.mass_g for m in members]
        groups.append(
            RigidGroup(
                id=names[root],
                instance_ids=[m.id for m in members],
                is_ground=root == ground,
                ground_evidence=ground_ev if root == ground else [],
                mass_g=float(sum(mass)) if all(x is not None for x in mass) else None,  # type: ignore[arg-type]
                description=_link_description(members),
            )
        )
    floating = [names[x] for x in order if x not in reach]
    joints = _final_joints(ground, merged, links, names, soft_used)
    mech = _loops(joints, names, links)
    dof, loops = _dof(joints, [x for x in order if x in reach], names)
    topo = _topology(joints, groups, by_id, up, asm, r, floating, loops)
    model = KinematicModel(
        links=groups,
        joints=joints,
        mechanisms=mech,
        dof=dof,
        topology=topo,
        floating_groups=floating,
        description="",
        mermaid="",
    )
    model.description = describe_kinematics(model, by_id, soft_used)
    model.mermaid = mermaid(model)
    link_of = {i: names[root] for root, ms in links.items() for i in (m.id for m in ms)}
    return model, KinContext(
        insts, by_id, names, link_of, screws, asm, conns, contacts_of, soft_used
    )


def _raw_joints(moving: list[Conn], uf: UnionFind) -> list[RawJoint]:
    out = []
    for c in moving:
        la, lb = uf.find(c.a), uf.find(c.b)
        if la == lb or c.axis_origin is None or c.axis_dir is None:
            continue
        out.append(
            RawJoint(c.kind, la, lb, c.axis_origin, c.axis_dir, c.confidence, list(c.evidence))
        )
    return out


def _links(
    ids: list[str], uf: UnionFind, by_id: dict[str, InstanceData]
) -> dict[str, list[InstanceData]]:
    out: dict[str, list[InstanceData]] = {}
    for i in ids:
        out.setdefault(uf.find(i), []).append(by_id[i])
    return out


def _pick_ground(
    links: dict[str, list[InstanceData]],
    by_id: dict[str, InstanceData],
    up: Vec,
    r: dict,  # type: ignore[type-arg]
) -> tuple[str, list[Evidence]]:
    low = min(_low(i, up) for i in by_id.values())
    fps = {k: _footprint(v, up) for k, v in links.items()}
    masses = {
        k: sum((m.ap.part.mass.mass_g or m.ap.part.mass.volume_mm3 * 1e-3) for m in v)
        for k, v in links.items()
    }
    best: tuple[float, str, list[Evidence]] | None = None
    for k in sorted(links):
        stats = {
            "footprint": fps[k],
            "footprint_share": fps[k] / (max(fps.values()) or 1.0),
            "mass": masses[k],
            "mass_share": masses[k] / (max(masses.values()) or 1.0),
        }
        s, ev = _link_score(k, links[k], stats, up, low, r)
        if best is None or s > best[0] + 1e-9:
            best = (s, k, ev)
    assert best is not None
    return best[1], best[2]


def _reachable(
    ground: str, joints: list[RawJoint], links: dict[str, list[InstanceData]]
) -> set[str]:
    adj: dict[str, set[str]] = {k: set() for k in links}
    for j in joints:
        adj[j.la].add(j.lb)
        adj[j.lb].add(j.la)
    seen, q = {ground}, deque([ground])
    while q:
        cur = q.popleft()
        for n in sorted(adj[cur]):
            if n not in seen:
                seen.add(n)
                q.append(n)
    return seen


def _order_links(
    ground: str, joints: list[RawJoint], links: dict[str, list[InstanceData]]
) -> list[str]:
    adj: dict[str, list[str]] = {k: [] for k in links}
    for j in joints:
        adj[j.la].append(j.lb)
        adj[j.lb].append(j.la)
    order, seen = [ground], {ground}
    q = deque([ground])
    while q:
        cur = q.popleft()
        for n in sorted(set(adj[cur]), key=lambda x: min(i.id for i in links[x])):
            if n not in seen:
                seen.add(n)
                order.append(n)
                q.append(n)
    rest = sorted((k for k in links if k not in seen), key=lambda k: min(i.id for i in links[k]))
    return order + rest


def _link_description(members: list[InstanceData]) -> str:
    big = sorted(members, key=lambda m: -m.ap.part.mass.volume_mm3)
    names = [m.name for m in big[:3]]
    more = f" + {len(members) - 3} more" if len(members) > 3 else ""
    return " + ".join(names) + more + f" ({len(members)} instance(s))"


def _final_joints(
    ground: str,
    merged: list[RawJoint],
    links: dict[str, list[InstanceData]],
    names: dict[str, str],
    soft_used: list[Conn],
) -> list[KinJoint]:
    depth: dict[str, int] = {ground: 0}
    adj: dict[str, list[RawJoint]] = {k: [] for k in links}
    for j in merged:
        adj[j.la].append(j)
        adj[j.lb].append(j)
    q = deque([ground])
    while q:
        cur = q.popleft()
        for j in adj[cur]:
            nxt = j.lb if j.la == cur else j.la
            if nxt not in depth:
                depth[nxt] = depth[cur] + 1
                q.append(nxt)
    rows = []
    for j in merged:
        if j.la not in depth and j.lb not in depth:
            continue
        # parent = the link closer to ground
        if depth.get(j.la, 99) <= depth.get(j.lb, 99):
            parent, child = j.la, j.lb
        else:
            parent, child = j.lb, j.la
        rows.append((depth.get(parent, 99), names[child], names[parent], j, parent, child))
    rows.sort(key=lambda t: (t[0], t[1], t[2], t[3].origin.round(3).tolist()))
    out = []
    for n, (_d, _c, _p, j, parent, child) in enumerate(rows, 1):
        d = canonical_dir(j.direction)
        kind = j.kind
        j.final_id = f"KJ{n}"
        out.append(
            KinJoint(
                id=f"KJ{n}",
                kind=kind,
                parent_link=names[parent],
                child_link=names[child],
                axis=Axis(origin=vec3(j.origin), direction=vec3(d)),
                evidence=j.evidence,
                confidence=round(min(0.95, j.confidence + 0.05 * (len(j.evidence) - 1)), 2),
                description=(
                    f"KJ{n} {kind} about {axis_label(d)} through "
                    f"({fmt(j.origin[0], 1)}, {fmt(j.origin[1], 1)}, {fmt(j.origin[2], 1)}) "
                    f"between {names[parent]} (parent) and {names[child]} (child)"
                ),
            )
        )
    del soft_used
    return out


def _loops(
    joints: list[KinJoint], names: dict[str, str], links: dict[str, list[InstanceData]]
) -> list[Mechanism]:
    """A cycle in the link graph is a closed-loop linkage: one mechanism per independent loop."""
    adj: dict[str, list[tuple[str, str]]] = {}
    for j in joints:
        adj.setdefault(j.parent_link, []).append((j.child_link, j.id))
        adj.setdefault(j.child_link, []).append((j.parent_link, j.id))
    seen: set[str] = set()
    comps: list[list[str]] = []
    for start in sorted(adj):
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for n, _jid in adj[cur]:
                if n not in seen:
                    seen.add(n)
                    stack.append(n)
        comps.append(sorted(comp))
    out: list[Mechanism] = []
    reverse = {v: k for k, v in names.items()}
    for comp in comps:
        js = [j for j in joints if j.parent_link in comp and j.child_link in comp]
        if len(js) >= len(comp):  # edges >= nodes: at least one cycle
            inst = sorted({m.id for ln in comp for m in links[reverse[ln]]})
            out.append(
                Mechanism(
                    id="",
                    kind="linkage_loop",
                    instance_ids=inst,
                    parameters={"links": comp, "joints": [j.id for j in js]},
                    confidence=0.6,
                    evidence=[
                        Evidence(
                            code="cycle",
                            refs=[j.id for j in js],
                            weight=1.0,
                            text=f"joints {', '.join(j.id for j in js)} connect links {', '.join(comp)} in a closed loop",
                        )
                    ],
                    description=f"closed-loop linkage through {', '.join(comp)} (joints {', '.join(j.id for j in js)})",
                )
            )
    for k, mech in enumerate(out, 1):
        mech.id = f"M{k}"
    return out


def _dof(joints: list[KinJoint], reach_order: list[str], names: dict[str, str]) -> tuple[int, int]:
    """(degrees of freedom estimate, independent loops). Planar loops use 3(n-1)-2J."""
    f = {"revolute": 1, "prismatic": 1, "cylindrical": 2, "fixed_uncertain": 0}
    n_links = len(reach_order)
    nj = len(joints)
    loops = max(0, nj - (n_links - 1))
    if loops == 0:
        return sum(f[j.kind] for j in joints), 0
    axes = [np.array([j.axis.direction.x, j.axis.direction.y, j.axis.direction.z]) for j in joints]
    planar = all(j.kind == "revolute" for j in joints) and all(
        abs(abs(float(np.dot(axes[0], a))) - 1.0) < 1e-3 for a in axes
    )
    if planar:
        m = 3 * (n_links - 1) - 2 * nj
    else:
        m = 6 * (n_links - 1) - sum(6 - f[j.kind] for j in joints)
    return max(m, 0), loops


def _topology(
    joints: list[KinJoint],
    groups: list[RigidGroup],
    by_id: dict[str, InstanceData],
    up: Vec,
    asm: AssemblyInfo,
    r: dict,  # type: ignore[type-arg]
    floating: list[str],
    loops: int,
) -> str:
    if not joints:
        return "static"
    if loops:
        return "closed_loop"
    wheels = _wheel_joints(joints, groups, by_id, up, asm, r)
    others = [j for j in joints if j.id not in {w.id for w in wheels}]
    children: dict[str, int] = {}
    for j in joints:
        children[j.parent_link] = children.get(j.parent_link, 0) + 1
    if len(wheels) >= int(r["kinematics"]["wheel"]["min_wheels"]):
        chain = _longest_chain(others, groups)
        return "mobile_manipulator" if chain >= 2 else "mobile_base"
    if max(children.values()) <= 1:
        return "serial_chain"
    return "tree"


def _longest_chain(joints: list[KinJoint], groups: list[RigidGroup]) -> int:
    child_of: dict[str, list[str]] = {}
    for j in joints:
        child_of.setdefault(j.parent_link, []).append(j.child_link)

    def depth(n: str) -> int:
        return 1 + max((depth(c) for c in child_of.get(n, [])), default=0) if n in child_of else 0

    starts = {j.parent_link for j in joints} - {j.child_link for j in joints}
    return max((depth(s) for s in starts), default=0)


def _wheel_joints(
    joints: list[KinJoint],
    groups: list[RigidGroup],
    by_id: dict[str, InstanceData],
    up: Vec,
    asm: AssemblyInfo,
    r: dict,  # type: ignore[type-arg]
) -> list[KinJoint]:
    w = r["kinematics"]["wheel"]
    gbox = asm.global_bbox
    lo = np.array([gbox.min.x, gbox.min.y, gbox.min.z])
    hi = np.array([gbox.max.x, gbox.max.y, gbox.max.z])
    height = float(np.dot(hi - lo, np.abs(up)))
    floor = min(_low(i, up) for i in by_id.values())
    tol = float(w["lowest_plane_tolerance_fraction"]) * height
    link_inst = {g.id: [by_id[i] for i in g.instance_ids] for g in groups}
    out = []
    for j in joints:
        if j.kind != "revolute":
            continue
        d = np.array([j.axis.direction.x, j.axis.direction.y, j.axis.direction.z])
        if abs(float(np.dot(d, up))) > float(w["horizontal_axis_max_dot_with_up"]):
            continue
        for inst in link_inst[j.child_link]:
            a, b, c = inst.ap.part.obb.size_sorted
            discish = c <= float(w["disc_ratio"]) * a and b >= 0.9 * a
            if discish and _low(inst, up) <= floor + tol:
                out.append(j)
                break
    return out


def retopology(model: KinematicModel, ctx: KinContext) -> None:
    """Re-classify the topology without the joints that only feed a transmission.

    A motor turning a pulley, or the support of a lead screw, is a drive train input, not a
    branch of the mechanism it drives.
    """
    if model.topology == "closed_loop":
        return
    feeders = {
        str(m.parameters["input_joint"])
        for m in model.mechanisms
        if m.kind in ("gear_pair", "belt_drive", "lead_screw") and m.parameters.get("input_joint")
    }
    if not feeders:
        return
    kept = [j for j in model.joints if j.id not in feeders]
    r = rules()
    up = axis_vector(ctx.asm.up_axis)
    model.topology = _topology(
        kept, model.links, ctx.by_id, up, ctx.asm, r, model.floating_groups, 0
    )  # type: ignore[assignment]


def wheel_joint_ids(model: KinematicModel, ctx: KinContext) -> set[str]:
    """IDs of the revolute joints that carry wheels (horizontal axis, disc on the lowest plane)."""
    up = axis_vector(ctx.asm.up_axis)
    return {j.id for j in _wheel_joints(model.joints, model.links, ctx.by_id, up, ctx.asm, rules())}


def refresh_description(model: KinematicModel, ctx: KinContext) -> None:
    """Rebuild the text and diagram after mechanisms changed the DOF and the topology."""
    model.description = describe_kinematics(model, ctx.by_id, ctx.soft_used)
    model.mermaid = mermaid(model)


def describe_kinematics(
    model: KinematicModel, by_id: dict[str, InstanceData], soft_used: list[Conn]
) -> str:
    n_rev = sum(j.kind == "revolute" for j in model.joints)
    n_pri = sum(j.kind == "prismatic" for j in model.joints)
    bits = [
        f"{len(model.links)} rigid link(s), {len(model.joints)} joint(s) "
        f"({n_rev} revolute, {n_pri} prismatic), estimated {model.dof} degree(s) of freedom, "
        f"topology {model.topology.replace('_', ' ')}"
    ]
    ground = next((g for g in model.links if g.is_ground), None)
    if ground:
        bits.append(f"ground link {ground.id} ({ground.description})")
    if model.floating_groups:
        bits.append(f"floating (not connected to ground): {', '.join(model.floating_groups)}")
    if soft_used:
        bits.append(
            "attached only by a planar rest (no fasteners): "
            + ", ".join(f"{c.a}-{c.b}" for c in soft_used)
        )
    del by_id
    return "; ".join(bits)


def mermaid(model: KinematicModel) -> str:
    """Mermaid flowchart: links are nodes, joints labelled edges, mechanisms dashed."""

    def esc(s: str) -> str:
        return s.replace('"', "'")

    lines = ["flowchart LR"]
    for g in model.links:
        role = "Ground: " if g.is_ground else ""
        lines.append(f'  {g.id}["{esc(g.id)} {role}{esc(g.description)}"]')
    for j in model.joints:
        lab = f"{j.id} {j.kind} {axis_label(np.array([j.axis.direction.x, j.axis.direction.y, j.axis.direction.z]))}"
        lines.append(f'  {j.parent_link} -- "{esc(lab)}" --> {j.child_link}')
    by_joint = {j.id: j for j in model.joints}
    for m in model.mechanisms:
        out = by_joint.get(m.output_joint_id or "")
        if m.input_instance_id and out is not None:
            ratio = f" {m.ratio:g}:1" if m.ratio else ""
            lines.append(f'  {m.input_instance_id}(("motor {m.input_instance_id}"))')
            lines.append(
                f'  {m.input_instance_id} -. "{m.id} {m.kind.replace("_", " ")}{ratio}" .-> {out.child_link}'
            )
    return "\n".join(lines)
