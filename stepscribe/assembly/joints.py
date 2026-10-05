# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Fastener joints: aligned holes across instances become one joint."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from stepscribe import config
from stepscribe.assembly.tree import InstanceData
from stepscribe.describe.phrases import fmt
from stepscribe.features.hole_standards import load_table
from stepscribe.geometry.occ_utils import Vec, canonical_dir
from stepscribe.geometry.properties import vec3
from stepscribe.models.schema import Axis, FastenerJoint, Hole

HOLE_GAP_MAX = 0.5  # mm between consecutive holes in one stack
STANDARD_LENGTHS = (4, 5, 6, 8, 10, 12, 14, 16, 20, 25, 30, 35, 40, 45, 50)
NUT_THREAD_PITCHES = 1.5  # extra screw length beyond the nut, in pitches
JOINT_CONF_FACTOR = 0.9
EXISTING_FASTENER_RADIAL = 0.1  # mm
EXISTING_FASTENER_ANGLE = 1.0  # degrees


@dataclass
class GlobalHole:
    """A hole transformed into the assembly frame."""

    inst: InstanceData
    hole: Hole
    origin: Vec
    direction: Vec  # into the material
    t0: float = 0.0
    t1: float = 0.0


def _global_holes(instances: list[InstanceData]) -> list[GlobalHole]:
    out: list[GlobalHole] = []
    for inst in instances:
        if inst.ap.part.likely_purchased_hardware:
            continue
        for h in inst.ap.part.holes:
            if h.depth_mm is None:
                continue
            o = np.array([h.axis.origin.x, h.axis.origin.y, h.axis.origin.z])
            d = np.array([h.axis.direction.x, h.axis.direction.y, h.axis.direction.z])
            out.append(GlobalHole(inst, h, inst.to_global_point(o), inst.to_global_dir(d)))
    return out


def _coaxial_groups(holes: list[GlobalHole]) -> list[list[GlobalHole]]:
    groups: list[list[GlobalHole]] = []
    for gh in holes:
        d = canonical_dir(gh.direction)
        for g in groups:
            ref = g[0]
            rd = canonical_dir(ref.direction)
            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(d, rd))))))
            off = gh.origin - ref.origin
            dist = float(np.linalg.norm(off - rd * np.dot(off, rd)))
            if ang <= config.ANGULAR_TOL_DEG and dist <= config.CONTACT_TOL:
                g.append(gh)
                break
        else:
            groups.append([gh])
    return groups


def _chains(group: list[GlobalHole]) -> list[list[GlobalHole]]:
    """Split a coaxial group into stacks whose consecutive holes are <= 0.5 mm apart."""
    ref = group[0]
    d = canonical_dir(ref.direction)
    for gh in group:
        t_entry = float(np.dot(gh.origin - ref.origin, d))
        depth = gh.hole.depth_mm or 0.0
        along = float(np.dot(gh.direction, d))
        gh.t0, gh.t1 = (t_entry, t_entry + depth) if along > 0 else (t_entry - depth, t_entry)
    chains: list[list[GlobalHole]] = []
    end = -math.inf
    for gh in sorted(group, key=lambda g: (g.t0, g.inst.id)):
        if chains and gh.t0 - end <= HOLE_GAP_MAX:
            chains[-1].append(gh)
        else:
            chains.append([gh])
        end = max(end, gh.t1)
    return chains


def _round_up_length(x: float) -> int:
    for s in STANDARD_LENGTHS:
        if s >= x - 1e-9:
            return s
    return int(math.ceil(x / 10.0) * 10)


def _round_down_length(x: float) -> int | None:
    fits = [s for s in STANDARD_LENGTHS if s <= x + 1e-9]
    return fits[-1] if fits else None


def _designation(members: list[GlobalHole]) -> tuple[str | None, float]:
    """Screw size consistent with every hole in the stack.

    Each hole lists its top standard matches; a designation scores the sum of its best
    confidence in each hole (0 where a hole does not match it), so M3 beats M2.5 when one
    plate has Ø3.4 clearance and the other a Ø2.5 tap drill (which is also M2.5's nominal).
    """
    scores: dict[str, float] = {}
    for gh in members:
        best: dict[str, float] = {}
        for m in gh.hole.standard_matches:
            best[m.designation] = max(best.get(m.designation, 0.0), m.confidence)
        for des, c in best.items():
            scores[des] = scores.get(des, 0.0) + c
    if not scores:
        return None, 0.0
    des, total = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return des, total / len(members)


def _suggest(designation: str | None, grip: float, threaded: bool) -> tuple[str | None, str]:
    """(suggested fastener text, reasoning) following the length rule."""
    if designation is None:
        return None, "no standard screw size matches the hole diameter"
    row = load_table("screws_iso_metric.yaml")["sizes"].get(designation)
    if row is None:
        return f"{designation} screw", "length not suggested for non-metric sizes"
    if threaded:
        length = _round_down_length(grip)
        if length is None:
            return (
                f"{designation} screw",
                f"grip {fmt(grip)} mm is shorter than the shortest standard screw",
            )
        return (
            f"{designation} × {length} SHCS",
            f"threaded end: longest standard length not exceeding the {fmt(grip)} mm available depth",
        )
    extra = float(row["nut_m"]) + NUT_THREAD_PITCHES * float(row["pitch"])
    length = _round_up_length(grip + extra)
    return (
        f"{designation} × {length} SHCS",
        f"through stack: grip {fmt(grip)} + nut {fmt(float(row['nut_m']))} + 1.5 × pitch {fmt(float(row['pitch']), 2)}, rounded up",
    )


def _existing_fastener(instances: list[InstanceData], origin: Vec, d: Vec) -> str | None:
    for inst in instances:
        part = inst.ap.part
        if not part.likely_purchased_hardware or part.shape_class.label != "fastener":
            continue
        order = np.argsort([-part.obb.half_sizes.x, -part.obb.half_sizes.y, -part.obb.half_sizes.z])
        ax = part.obb.axes[int(order[0])]
        axis = inst.to_global_dir(np.array([ax.x, ax.y, ax.z]))
        c = part.obb.center
        centre = inst.to_global_point(np.array([c.x, c.y, c.z]))
        ang = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(np.dot(axis, d)))))))
        off = centre - origin
        radial = float(np.linalg.norm(off - d * np.dot(off, d)))
        if ang <= EXISTING_FASTENER_ANGLE and radial <= EXISTING_FASTENER_RADIAL:
            return inst.id
    return None


def detect_joints(instances: list[InstanceData]) -> list[FastenerJoint]:
    """Joints between instances; IDs J001... sorted by first instance and position."""
    joints: list[tuple[tuple[object, ...], FastenerJoint]] = []
    for group in _coaxial_groups(_global_holes(instances)):
        for chain in _chains(group):
            ids = list(dict.fromkeys(gh.inst.id for gh in chain))
            if len(ids) < 2:
                continue
            ref = group[0]
            d = canonical_dir(ref.direction)
            t_lo, t_hi = min(g.t0 for g in chain), max(g.t1 for g in chain)
            grip = t_hi - t_lo
            dia = Counter(round(g.hole.diameter_mm, 2) for g in chain)
            common = sorted(dia.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            threaded = any(g.hole.likely_threaded for g in chain)
            desig, conf = _designation(chain)
            text, why = _suggest(desig, grip, threaded)
            origin = ref.origin + d * t_lo
            existing = _existing_fastener(instances, origin, d)
            key = (ids[0], *(round(float(c), 2) for c in origin))
            joints.append(
                (
                    key,
                    FastenerJoint(
                        id="",
                        confidence=round(conf * JOINT_CONF_FACTOR, 2),
                        evidence=(
                            f"{len(chain)} coaxial holes across {len(ids)} parts (Ø{', Ø'.join(str(k) for k in sorted(dia))}), "
                            f"grip {fmt(grip)} mm; {why}"
                        ),
                        instance_ids=ids,
                        hole_refs=[
                            (g.inst.id, g.hole.id) for g in sorted(chain, key=lambda g: g.t0)
                        ],
                        axis=Axis(origin=vec3(origin), direction=vec3(d)),
                        common_diameter_mm=float(common),
                        stack_thickness_mm=grip,
                        suggested_fastener=text,
                        has_threaded_end=threaded,
                        existing_fastener_instance_id=existing,
                    ),
                )
            )
    joints.sort(key=lambda kv: kv[0])
    out = [j for _k, j in joints]
    for i, j in enumerate(out, 1):
        j.id = f"J{i:03d}"
    return out


def shopping_list(joints: list[FastenerJoint]) -> list[dict[str, object]]:
    """Aggregate screws (and nuts for through stacks) from joints that have no fastener modelled."""
    counts: Counter[str] = Counter()
    for j in joints:
        if j.existing_fastener_instance_id or not j.suggested_fastener:
            continue
        counts[j.suggested_fastener] += 1
        if not j.has_threaded_end:
            size = j.suggested_fastener.split(" ")[0]
            counts[f"{size} nut"] += 1
    return [{"item": item, "qty": qty} for item, qty in sorted(counts.items())]
