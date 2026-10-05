# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Optional motion check: move the links behind each joint and look for collisions. Opt-in (``--check-motion``) and time-budgeted.

For a revolute joint the downstream links turn about the joint axis in 5 degree steps up to
+-180 degrees; for a prismatic joint they slide in 1 mm steps along the axis. At each step the
moved instances are tested against everything that does not move: an axis-aligned box test
first, then the volume of the boolean intersection (> 0.01 mm3 is a collision). Pairs that
already overlap at the start pose are ignored. The result is the last free position in each
direction and what blocks it.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepGProp import BRepGProp
from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from OCP.GProp import GProp_GProps

from stepscribe.assembly.tree import InstanceData
from stepscribe.describe.phrases import fmt
from stepscribe.geometry.occ_utils import Vec, bbox_of, transform_shape
from stepscribe.models.schema import KinematicModel, KinJoint
from stepscribe.understanding.kinematics import KinContext

if TYPE_CHECKING:
    from stepscribe.api import Analysis

ANGLE_STEP_DEG = 5.0
MAX_ANGLE_DEG = 180.0
LINEAR_STEP_MM = 1.0
MAX_TRAVEL_MM = 400.0
COLLISION_VOLUME_MM3 = 0.01
BOX_MARGIN_MM = 0.01


def _volume(shape) -> float:  # type: ignore[no-untyped-def]
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return float(props.Mass())


def _overlap_volume(a, b) -> float:  # type: ignore[no-untyped-def]
    op = BRepAlgoAPI_Common(a, b)
    if not op.IsDone():
        return 0.0
    return _volume(op.Shape())


def _boxes_touch(a: tuple[Vec, Vec], b: tuple[Vec, Vec]) -> bool:
    return bool(np.all(a[0] <= b[1] + BOX_MARGIN_MM) and np.all(b[0] <= a[1] + BOX_MARGIN_MM))


def _trsf(joint: KinJoint, amount: float) -> gp_Trsf:
    o = joint.axis.origin
    d = joint.axis.direction
    t = gp_Trsf()
    if joint.kind == "prismatic":
        v = np.array([d.x, d.y, d.z])
        v = v / np.linalg.norm(v) * amount
        t.SetTranslation(gp_Vec(float(v[0]), float(v[1]), float(v[2])))
    else:
        t.SetRotation(gp_Ax1(gp_Pnt(o.x, o.y, o.z), gp_Dir(d.x, d.y, d.z)), math.radians(amount))
    return t


def _downstream(model: KinematicModel, start: str) -> set[str]:
    out, stack = {start}, [start]
    while stack:
        cur = stack.pop()
        for j in model.joints:
            if j.parent_link == cur and j.child_link not in out:
                out.add(j.child_link)
                stack.append(j.child_link)
    return out


def _colliding(
    moving: list[InstanceData],
    static: list[InstanceData],
    trsf: gp_Trsf,
    skip: set[tuple[str, str]],
    cache: dict[str, tuple[Vec, Vec]],
) -> list[str]:
    hits: list[str] = []
    for m in moving:
        moved = transform_shape(m.shape, trsf)
        mb = bbox_of(moved)
        for s in static:
            if (m.id, s.id) in skip:
                continue
            sb = cache.setdefault(s.id, s.bounds)
            if not _boxes_touch(mb, sb):
                continue
            if _overlap_volume(moved, s.shape) > COLLISION_VOLUME_MM3:
                hits.append(s.id)
    return sorted(set(hits))


def _sweep(
    joint: KinJoint,
    moving: list[InstanceData],
    static: list[InstanceData],
    skip: set[tuple[str, str]],
    deadline: float,
    travel: float,
) -> tuple[float, float, list[str]] | None:
    """(lowest free value, highest free value, blockers) or None when out of time."""
    step = LINEAR_STEP_MM if joint.kind == "prismatic" else ANGLE_STEP_DEG
    limit = min(travel, MAX_TRAVEL_MM) if joint.kind == "prismatic" else MAX_ANGLE_DEG
    cache: dict[str, tuple[Vec, Vec]] = {}
    free = [0.0, 0.0]
    blockers: list[str] = []
    for side, sign in enumerate((-1.0, 1.0)):
        k = 1
        while k * step <= limit + 1e-9:
            if time.time() > deadline:
                return None
            hits = _colliding(moving, static, _trsf(joint, sign * k * step), skip, cache)
            if hits:
                blockers += hits
                break
            free[side] = sign * k * step
            k += 1
    return free[0], free[1], sorted(set(blockers))


def check_motion(
    analysis: Analysis, model: KinematicModel, ctx: KinContext, budget_s: float = 120.0
) -> None:
    """Fill ``range_deg_or_mm`` and ``blocked_by`` on the revolute and prismatic joints."""
    del analysis
    deadline = time.time() + budget_s
    link_inst = {g.id: [ctx.by_id[i] for i in g.instance_ids] for g in model.links}
    for joint in model.joints:
        if joint.kind not in ("revolute", "prismatic"):
            continue
        moving_links = _downstream(model, joint.child_link)
        moving = [i for ln in moving_links for i in link_inst[ln]]
        static = [i for ln, ins in link_inst.items() if ln not in moving_links for i in ins]
        skip = _baseline(moving, static)
        d = joint.axis.direction
        axis = np.array([d.x, d.y, d.z])
        axis = axis / np.linalg.norm(axis)
        lo = np.min([s.bounds[0] for s in static], axis=0)
        hi = np.max([s.bounds[1] for s in static], axis=0)
        travel = float(np.abs(axis) @ (hi - lo))
        got = _sweep(joint, moving, static, skip, deadline, travel)
        if got is None:
            joint.description += ", motion not checked (time budget used up)"
            continue
        low, high, blockers = got
        joint.range_deg_or_mm = (low, high)
        joint.blocked_by = blockers
        unit = "mm" if joint.kind == "prismatic" else "deg"
        full = MAX_ANGLE_DEG if joint.kind == "revolute" else min(travel, MAX_TRAVEL_MM)
        if blockers:
            joint.description += f"; free range {fmt(low, 0)} to {fmt(high, 0)} {unit}, blocked by {', '.join(blockers)}"
        elif joint.kind == "revolute" and low <= -full and high >= full:
            joint.description += "; turns freely through a full circle"
        else:
            joint.description += f"; free range {fmt(low, 0)} to {fmt(high, 0)} {unit}"


def _baseline(moving: list[InstanceData], static: list[InstanceData]) -> set[tuple[str, str]]:
    """Pairs that already overlap in the start pose (press fits, shafts in bearings)."""
    out: set[tuple[str, str]] = set()
    for m in moving:
        mb = m.bounds
        for s in static:
            if (
                _boxes_touch(mb, s.bounds)
                and _overlap_volume(m.shape, s.shape) > COLLISION_VOLUME_MM3
            ):
                out.add((m.id, s.id))
    return out
