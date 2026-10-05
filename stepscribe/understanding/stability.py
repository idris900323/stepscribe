# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Stability: centre of mass against the support polygon, and the tipping angle."""

from __future__ import annotations

import math

import numpy as np

from stepscribe.assembly.tree import InstanceData
from stepscribe.describe.phrases import fmt
from stepscribe.features.hole_standards import load_table
from stepscribe.geometry.occ_utils import Vec
from stepscribe.geometry.properties import vec3
from stepscribe.models.schema import KinematicModel, StabilityInfo
from stepscribe.understanding.kinematics import KinContext, axis_vector, wheel_joint_ids


def convex_hull_2d(pts: np.ndarray) -> np.ndarray:
    """Counter-clockwise hull (Andrew's monotone chain) of an (n, 2) array."""
    p = sorted({(round(float(x), 6), round(float(y), 6)) for x, y in pts})
    if len(p) < 3:
        return np.array(p)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper: list[tuple[float, float]] = []
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])


def polygon_margin(hull: np.ndarray, point: np.ndarray) -> tuple[float, np.ndarray]:
    """(signed distance from *point* to the nearest hull edge, outward normal of that edge)."""
    best = (float("inf"), np.zeros(2))
    n = len(hull)
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        e = b - a
        length = float(np.linalg.norm(e))
        if length < 1e-9:
            continue
        outward = np.array([e[1], -e[0]]) / length  # CCW hull: the right-hand normal points out
        d = -float(np.dot(point - a, outward))  # positive inside
        if d < best[0]:
            best = (d, outward)
    return best


def _mass(i: InstanceData) -> float | None:
    return i.ap.part.mass.mass_g


def _centroid(i: InstanceData) -> Vec:
    return i.to_global_point(
        np.array([i.ap.part.mass.centroid.x, i.ap.part.mass.centroid.y, i.ap.part.mass.centroid.z])
    )


def _low(i: InstanceData, up: Vec) -> float:
    lo, hi = i.bounds
    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )
    return float((corners @ up).min())


def _basis(up: Vec) -> tuple[Vec, Vec]:
    ref = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(up, ref)
    e1 = e1 / np.linalg.norm(e1)
    return e1, np.cross(up, e1)


def support_points(ctx: KinContext, up: Vec, tol: float) -> tuple[list[Vec], float]:
    """Contact points with the ground plane and the height of that plane along *up*."""
    low = min(_low(i, up) for i in ctx.insts)
    pts: list[Vec] = []
    for inst in ctx.insts:
        if _low(inst, up) > low + tol:
            continue
        a, b, c = inst.ap.part.obb.size_sorted
        lo, hi = inst.bounds
        wheel = (
            c <= 0.35 * a
            and b >= 0.9 * a
            and inst.ap.part.shape_class.label in ("disc", "ring", "gear_like", "other")
        )
        centre = 0.5 * (lo + hi)
        if wheel:
            pts.append(centre - up * (float(np.dot(centre, up)) - low))
        else:
            e1, e2 = _basis(up)
            for x in (lo[0], hi[0]):
                for y in (lo[1], hi[1]):
                    for z in (lo[2], hi[2]):
                        q = np.array([x, y, z])
                        if float(np.dot(q, up)) <= low + tol:
                            pts.append(q - up * (float(np.dot(q, up)) - low))
            del e1, e2
    return pts, low


def stability(
    ctx: KinContext,
    model: KinematicModel,
    up_label: str,
    payload: tuple[float, Vec] | None = None,
) -> StabilityInfo | None:
    """None when masses are unknown or fewer than three support points exist."""
    masses = [_mass(i) for i in ctx.insts]
    if any(m is None for m in masses):
        return None
    cfg = load_table("weak_spot_rules.yaml")["stability"]
    up = axis_vector(up_label)
    m = np.array(masses, dtype=float)
    cents = np.array([_centroid(i) for i in ctx.insts])
    total = float(m.sum())
    if total <= 0:
        return None
    if payload is not None:
        m = np.append(m, payload[0])
        cents = np.vstack([cents, payload[1]])
        total = float(m.sum())
    com = (m[:, None] * cents).sum(axis=0) / total
    pts, low = support_points(ctx, up, float(cfg["support_tolerance_mm"]))
    if len(pts) < 3:
        return None
    e1, e2 = _basis(up)
    flat = np.array([[float(np.dot(p, e1)), float(np.dot(p, e2))] for p in pts])
    hull = convex_hull_2d(flat)
    if len(hull) < 3:
        return None
    area = 0.5 * abs(
        sum(
            hull[i][0] * hull[(i + 1) % len(hull)][1] - hull[(i + 1) % len(hull)][0] * hull[i][1]
            for i in range(len(hull))
        )
    )
    c2 = np.array([float(np.dot(com, e1)), float(np.dot(com, e2))])
    margin, outward = polygon_margin(hull, c2)
    height = float(np.dot(com, up)) - low
    angle = math.degrees(math.atan2(margin, height)) if height > 0 else None
    direction = outward[0] * e1 + outward[1] * e2
    ext = _extended(ctx, model, up, e1, e2, hull, low, wheel_joint_ids(model, ctx), payload)
    text = (
        f"centre of mass {fmt(height, 1)} mm above the ground plane; support polygon "
        f"{fmt(area, 0)} mm² from {len(pts)} support point(s); margin {fmt(margin, 1)} mm to the nearest edge; "
        f"tips at {fmt(angle or 0.0, 1)} deg toward ({fmt(direction[0], 2)}, {fmt(direction[1], 2)}, {fmt(direction[2], 2)})"
    )
    if ext:
        text += (
            f". With the arm fully extended toward that edge the margin is {fmt(ext[1], 1)} mm "
            f"and the tipping angle {fmt(ext[2], 1)} deg"
        )
    return StabilityInfo(
        center_of_mass=vec3(com),
        support_points=[vec3(p) for p in pts],
        support_polygon_area_mm2=area,
        com_height_mm=height,
        min_margin_mm=margin,
        tipping_angle_deg=angle,
        tipping_direction=vec3(direction),
        extended_center_of_mass=vec3(ext[0]) if ext else None,
        extended_margin_mm=ext[1] if ext else None,
        extended_tipping_angle_deg=ext[2] if ext else None,
        description=text,
    )


def _extended(
    ctx: KinContext,
    model: KinematicModel,
    up: Vec,
    e1: Vec,
    e2: Vec,
    hull: np.ndarray,
    low: float,
    wheel_ids: set[str],
    payload: tuple[float, Vec] | None = None,
) -> tuple[Vec, float, float] | None:
    """Worst-case tipping with the arm stretched out horizontally toward the weakest edge.

    The arm is every link reached from ground through joints that are not wheels. Its centre
    of mass is moved to its farthest reach (distance from the first arm joint, kept in the
    ground plane) in the direction of the nearest hull edge. An approximation, stated as such.
    """
    ground = next((g for g in model.links if g.is_ground), None)
    if ground is None:
        return None
    wheel_children = {j.child_link for j in model.joints if j.id in wheel_ids}
    arm_links: set[str] = set()
    first_joint = None
    frontier = [ground.id]
    seen = {ground.id}
    while frontier:
        cur = frontier.pop()
        for j in model.joints:
            if (
                j.parent_link == cur
                and j.child_link not in seen
                and j.child_link not in wheel_children
            ):
                seen.add(j.child_link)
                arm_links.add(j.child_link)
                frontier.append(j.child_link)
                if cur == ground.id and first_joint is None:
                    first_joint = j
    if len(arm_links) < 2 or first_joint is None:
        return None
    members = [ctx.by_id[i] for g in model.links if g.id in arm_links for i in g.instance_ids]
    ms = np.array([_mass(i) or 0.0 for i in members])
    arm_pts = [_centroid(i) for i in members]
    if payload is not None:
        ms = np.append(ms, payload[0])
        arm_pts.append(payload[1])
    if ms.sum() <= 0:
        return None
    arm_com = (ms[:, None] * np.array(arm_pts)).sum(axis=0) / ms.sum()
    anchor = np.array(
        [first_joint.axis.origin.x, first_joint.axis.origin.y, first_joint.axis.origin.z]
    )
    reach = float(np.linalg.norm(arm_com - anchor))
    all_m = np.array([_mass(i) or 0.0 for i in ctx.insts])
    cents = np.array([_centroid(i) for i in ctx.insts])
    rest_mask = np.array([i not in members for i in ctx.insts])
    rest_com = (all_m[rest_mask, None] * cents[rest_mask]).sum(axis=0) / max(
        all_m[rest_mask].sum(), 1e-9
    )
    rest_mass = max(all_m[rest_mask].sum(), 1e-9)
    base_c = (rest_com * rest_mass + arm_com * ms.sum()) / (rest_mass + ms.sum())
    c2 = np.array([float(np.dot(base_c, e1)), float(np.dot(base_c, e2))])
    _margin, outward = polygon_margin(hull, c2)
    direction = outward[0] * e1 + outward[1] * e2
    anchor_h = anchor - up * float(np.dot(anchor, up))
    new_arm = anchor_h + direction * reach + up * float(np.dot(arm_com, up))
    com = (rest_com * rest_mass + new_arm * ms.sum()) / (rest_mass + ms.sum())
    c2n = np.array([float(np.dot(com, e1)), float(np.dot(com, e2))])
    margin, _ = polygon_margin(hull, c2n)
    height = float(np.dot(com, up)) - low
    angle = math.degrees(math.atan2(margin, height)) if height > 0 else 0.0
    return com, margin, angle
