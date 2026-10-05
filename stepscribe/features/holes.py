# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Hole recognition. Each step is its own tested function."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from OCP.BRep import BRep_Builder
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepTools import BRepTools
from OCP.TopoDS import TopoDS_Compound, TopoDS_Face

from stepscribe import config
from stepscribe.features.hole_standards import is_likely_threaded, match_standards
from stepscribe.geometry.occ_utils import (
    Vec,
    canonical_dir,
    interior_uv,
    make_vertex,
    outward_normal,
    surface_point,
)
from stepscribe.geometry.part_geom import PartGeom
from stepscribe.geometry.properties import vec3
from stepscribe.geometry.surfaces import cone_params, cylinder_params, plane_normal_origin
from stepscribe.models.schema import Axis, Hole, HoleSegment


# ---------------------------------------------------------------- 7.1 / 7.2 candidates
@dataclass
class BoreFace:
    """A concave cylinder/cone face that may belong to a hole."""

    idx: int  # index into the face table
    kind: str  # cylinder | cone
    origin: Vec  # surface axis origin
    direction: Vec  # surface axis direction
    span_deg: float
    v0: float
    v1: float
    radius_ref: float
    semi_angle: float = 0.0


def _uv_mid(face: TopoDS_Face) -> tuple[float, float, float, float, float, float]:
    """UV bounds plus a (u, v) sample that lies on the trimmed face."""
    u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
    um, vm = interior_uv(face)
    return u0, u1, v0, v1, um, vm


def _axis_point(p: Vec, origin: Vec, direction: Vec) -> Vec:
    d = direction / np.linalg.norm(direction)
    q: Vec = origin + d * float(np.dot(p - origin, d))
    return q


def is_concave(part: PartGeom, idx: int, kind: str, origin: Vec, direction: Vec) -> bool | None:
    """Concave (hole-like) test for a cylinder/cone face (7.2).

    The analytic test uses the solid-outward normal (flipped for REVERSED faces); the solid
    classifier on a point offset toward the axis confirms it. On disagreement the classifier wins.
    """
    face = part.table.faces[idx].face
    _u0, _u1, _v0, _v1, um, vm = _uv_mid(face)
    p = surface_point(face, um, vm)
    n = outward_normal(face, um, vm)
    q = _axis_point(p, origin, direction)
    radial = p - q
    r = float(np.linalg.norm(radial))
    if n is None or r < 1e-9:
        return None
    analytic = float(np.dot(n, radial)) < 0.0
    offset = min(config.BORE_PROBE_REL * r, config.BORE_PROBE_MAX)
    probe = p + (q - p) / r * offset
    state = part.classifier.state(probe)
    if state == "on":
        return analytic
    by_classifier = state == "out"
    if by_classifier != analytic:
        part.warnings.append(
            f"{part.table.faces[idx].id}: normal test and solid classifier disagree; trusting classifier"
        )
    return by_classifier


def find_bore_faces(part: PartGeom) -> list[BoreFace]:
    """All concave cylindrical / conical faces (7.1 + 7.2); cached on the part."""
    if part.bores_cache is not None:
        cached: list[BoreFace] = part.bores_cache
        return cached
    out: list[BoreFace] = []
    slivers = 0
    for idx, fi in enumerate(part.table.faces):
        if fi.kind not in ("cylinder", "cone"):
            continue
        if fi.area < config.MIN_FEATURE_SIZE**2:
            slivers += 1
            continue
        u0, u1, v0, v1, _um, _vm = _uv_mid(fi.face)
        if fi.kind == "cylinder":
            cp = cylinder_params(fi.face)
            origin, direction, radius, semi = cp.origin, cp.direction, cp.radius, 0.0
        else:
            cn = cone_params(fi.face)
            origin, direction, radius, semi = cn.origin, cn.direction, cn.ref_radius, cn.semi_angle
        if is_concave(part, idx, fi.kind, origin, direction):
            out.append(
                BoreFace(
                    idx, fi.kind, origin, direction, math.degrees(u1 - u0), v0, v1, radius, semi
                )
            )
    if slivers:
        part.warnings.append(
            f"{slivers} cylindrical/conical face(s) smaller than {config.MIN_FEATURE_SIZE} mm ignored"
        )
    part.bores_cache = out
    return out


# ---------------------------------------------------------------- 7.3 grouping
@dataclass
class AxisGroup:
    """Bore faces sharing one axis line, parametrised by a canonical direction."""

    origin: Vec
    direction: Vec  # canonical
    faces: list[BoreFace] = field(default_factory=list)


def group_by_axis(bores: list[BoreFace]) -> list[AxisGroup]:
    """Group faces whose axes are parallel (ANGULAR_TOL_DEG) and within COAXIAL_TOL."""
    groups: list[AxisGroup] = []
    for bf in bores:
        d = canonical_dir(bf.direction)
        for g in groups:
            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(d, g.direction))))))
            off = bf.origin - g.origin
            dist = float(np.linalg.norm(off - g.direction * np.dot(off, g.direction)))
            if ang < config.ANGULAR_TOL_DEG and dist < config.COAXIAL_TOL:
                g.faces.append(bf)
                break
        else:
            groups.append(AxisGroup(bf.origin, d, [bf]))
    return groups


@dataclass
class Segment:
    """Coaxial faces of one radius (cylinder) or one half-angle (cone), merged."""

    kind: str
    t0: float
    t1: float
    r_at_t0: float
    r_at_t1: float
    semi_angle_deg: float
    span_deg: float
    face_idx: list[int]


def _face_interval(bf: BoreFace, g: AxisGroup) -> tuple[float, float, float, float]:
    """Axial interval [t0, t1] along the group axis and the radii at both ends."""
    sign = 1.0 if float(np.dot(bf.direction, g.direction)) >= 0 else -1.0
    base = float(np.dot(bf.origin - g.origin, g.direction))
    if bf.kind == "cylinder":
        ta, tb = base + sign * bf.v0, base + sign * bf.v1
        ra = rb = bf.radius_ref
    else:
        c, s = math.cos(bf.semi_angle), math.sin(bf.semi_angle)
        ta, tb = base + sign * bf.v0 * c, base + sign * bf.v1 * c
        ra, rb = bf.radius_ref + bf.v0 * s, bf.radius_ref + bf.v1 * s
    if ta <= tb:
        return ta, tb, ra, rb
    return tb, ta, rb, ra


def build_segments(g: AxisGroup) -> list[Segment]:
    """Merge faces of equal radius/half-angle with touching axial ranges (7.3).

    Angular coverage is the length-weighted mean span, so two 180 deg halves on the same
    range give 360 deg while two 180 deg halves end-to-end stay at 180 deg.
    """
    pieces = []
    for bf in g.faces:
        t0, t1, ra, rb = _face_interval(bf, g)
        key_val = bf.radius_ref if bf.kind == "cylinder" else math.degrees(bf.semi_angle)
        pieces.append((bf.kind, round(key_val / config.LINEAR_TOL), t0, t1, ra, rb, bf))
    pieces.sort(key=lambda p: (p[0], p[1], p[2], p[3]))
    segs: list[Segment] = []
    cur: list[tuple[Any, ...]] = []

    def flush() -> None:
        if not cur:
            return
        t0 = min(p[2] for p in cur)
        t1 = max(p[3] for p in cur)
        length = max(t1 - t0, 1e-9)
        span = sum(p[6].span_deg * max(p[3] - p[2], 1e-9) for p in cur) / length
        first = min(cur, key=lambda p: p[2])
        last = max(cur, key=lambda p: p[3])
        semi = math.degrees(first[6].semi_angle)
        segs.append(
            Segment(first[0], t0, t1, first[4], last[5], semi, span, sorted(p[6].idx for p in cur))
        )
        cur.clear()

    for p in pieces:
        if (
            cur
            and (p[0], p[1]) == (cur[-1][0], cur[-1][1])
            and p[2] <= max(c[3] for c in cur) + config.LINEAR_TOL
        ):
            cur.append(p)
        else:
            flush()
            cur.append(p)
    flush()
    return segs


def stack_segments(segs: list[Segment]) -> list[list[Segment]]:
    """Bores (>= 359 deg, or > 190 deg when another feature cuts into them), stacked (7.3/7.4)."""
    full = sorted(
        (s for s in segs if s.span_deg >= config.PARTIAL_BORE_MIN_SPAN_DEG), key=lambda s: s.t0
    )
    stacks: list[list[Segment]] = []
    end = -math.inf
    for s in full:
        if stacks and s.t0 - end < config.LINEAR_TOL:
            stacks[-1].append(s)
        else:
            stacks.append([s])
        end = max(end, s.t1)
    return stacks


# ---------------------------------------------------------------- 7.4 / 7.5 build holes
def _axis_pt(g: AxisGroup, t: float) -> Vec:
    return g.origin + g.direction * t


def _classify_ends(part: PartGeom, g: AxisGroup, stack: list[Segment]) -> tuple[str, str]:
    t_lo = min(s.t0 for s in stack)
    t_hi = max(s.t1 for s in stack)
    d_main = 2 * min(min(s.r_at_t0, s.r_at_t1) for s in stack)
    off = max(config.HOLE_CLASSIFY_OFFSET_MIN, config.HOLE_CLASSIFY_OFFSET_REL * d_main)
    return (
        part.classifier.state(_axis_pt(g, t_lo - off)),
        part.classifier.state(_axis_pt(g, t_hi + off)),
    )


def _end_radius(stack: list[Segment], low_end: bool) -> float:
    if low_end:
        s = min(stack, key=lambda x: x.t0)
        return s.r_at_t0
    s = max(stack, key=lambda x: x.t1)
    return s.r_at_t1


def _pick_entry(lo: str, hi: str, stack: list[Segment]) -> tuple[bool, bool] | None:
    """Return (entry_is_low_end, is_through) or None for an internal void."""
    if lo == "in" and hi == "in":
        return None
    if lo != "in" and hi != "in":
        r_lo, r_hi = _end_radius(stack, True), _end_radius(stack, False)
        if abs(r_lo - r_hi) > config.LINEAR_TOL:
            return r_lo > r_hi, True
        return False, True  # larger canonical axis parameter
    return lo != "in", False


def _snap_csk(angle: float) -> float:
    for a in config.CSK_SNAP_ANGLES:
        if abs(angle - a) <= config.CSK_SNAP_TOL_DEG:
            return a
    return round(angle, 2)


def _ordered_segments(
    stack: list[Segment], entry_low: bool, t_entry: float, part: PartGeom
) -> list[HoleSegment]:
    ordered = sorted(stack, key=lambda s: s.t0, reverse=not entry_low)
    result = []
    for s in ordered:
        a, b = (abs(s.t0 - t_entry), abs(s.t1 - t_entry))
        d0, d1 = min(a, b), max(a, b)
        ra, rb = (s.r_at_t0, s.r_at_t1) if entry_low else (s.r_at_t1, s.r_at_t0)
        ids = [part.table.faces[i].id for i in s.face_idx]
        if s.kind == "cylinder":
            result.append(
                HoleSegment(
                    kind="cylinder",
                    diameter_mm=2 * ra,
                    start_depth_mm=d0,
                    end_depth_mm=d1,
                    face_ids=ids,
                )
            )
        else:
            big, small = max(ra, rb), min(ra, rb)
            result.append(
                HoleSegment(
                    kind="cone",
                    diameter_mm=2 * big,
                    diameter_small_mm=2 * small,
                    cone_angle_deg=2 * abs(s.semi_angle_deg),
                    start_depth_mm=d0,
                    end_depth_mm=d1,
                    face_ids=ids,
                )
            )
    return result


def _entry_and_bottom(
    segs: list[HoleSegment], main_d: float
) -> tuple[str, float | None, float | None, float | None, float | None]:
    """Entry type and counterbore / countersink sizes from the segments ordered from the entry."""
    first = segs[0]
    csk = cbore = False
    csk_d = csk_angle = cb_d = cb_depth = None
    idx = 0
    if (
        first.kind == "cone"
        and (first.diameter_mm - main_d) > config.LINEAR_TOL
        and first.diameter_small_mm is not None
    ):
        # widening toward the entry (large end first from the entry)
        csk = True
        csk_d = first.diameter_mm
        csk_angle = first.cone_angle_deg
        idx = 1
    if (
        len(segs) > idx
        and segs[idx].kind == "cylinder"
        and segs[idx].diameter_mm - main_d > config.LINEAR_TOL
    ):
        cbore = True
        cb_d = segs[idx].diameter_mm
        cb_depth = segs[idx].end_depth_mm
    if csk and cbore:
        kind = "counterbore_and_countersink"
    elif csk:
        kind = "countersink"
    elif cbore:
        kind = "counterbore"
    else:
        kind = "plain"
    return kind, cb_d, cb_depth, csk_d, csk_angle


def _flat_bottom(part: PartGeom, g: AxisGroup, t_end: float, radius: float) -> bool:
    """Is there a planar face perpendicular to the axis closing the hole at ``t_end``?"""
    centre = make_vertex(_axis_pt(g, t_end))
    for fi in part.table.faces:
        if fi.kind != "plane":
            continue
        n, _o = plane_normal_origin(fi.face)
        if abs(abs(float(np.dot(n, g.direction))) - 1.0) > 1e-3:
            continue
        dss = BRepExtrema_DistShapeShape(centre, fi.face)
        if dss.IsDone() and dss.Value() < max(10 * config.LINEAR_TOL, 0.01 * radius):
            return True
    return False


def _bottom_type(
    part: PartGeom, g: AxisGroup, segs: list[HoleSegment], t_end: float, radius: float
) -> str:
    last = segs[-1]
    if (
        last.kind == "cone"
        and last.diameter_small_mm is not None
        and last.cone_angle_deg is not None
    ):
        narrowing = last.diameter_small_mm < last.diameter_mm
        if (
            narrowing
            and abs(last.cone_angle_deg - config.DRILL_POINT_ANGLE) <= config.DRILL_POINT_TOL_DEG
        ):
            return "drill_point"
    if _flat_bottom(part, g, t_end, radius):
        return "flat"
    return "other"


def _entry_face_id(part: PartGeom, g: AxisGroup, bore_idx: list[int], t_entry: float) -> str | None:
    best: str | None = None
    for i in bore_idx:
        for j in sorted(part.table.faces[i].neighbors):
            fj = part.table.faces[j]
            if fj.kind != "plane":
                continue
            n, o = plane_normal_origin(fj.face)
            if abs(abs(float(np.dot(n, g.direction))) - 1.0) > 1e-3:
                continue
            if abs(float(np.dot(o - g.origin, g.direction)) - t_entry) <= 10 * config.LINEAR_TOL:
                best = fj.id if best is None else min(best, fj.id)
    return best


def build_hole(part: PartGeom, g: AxisGroup, stack: list[Segment]) -> Hole | None:
    """Turn one coaxial stack into a Hole (7.4-7.6); None for internal voids."""
    lo, hi = _classify_ends(part, g, stack)
    pick = _pick_entry(lo, hi, stack)
    t_lo, t_hi = min(s.t0 for s in stack), max(s.t1 for s in stack)
    if pick is None:
        part.warnings.append(
            f"internal void along axis at {np.round(_axis_pt(g, 0.5 * (t_lo + t_hi)), 3).tolist()} skipped"
        )
        return None
    entry_low, through = pick
    t_entry = t_lo if entry_low else t_hi
    direction = g.direction if entry_low else -g.direction
    segs = _ordered_segments(stack, entry_low, t_entry, part)
    cylinders = [s for s in segs if s.kind == "cylinder"]
    main_d = min(
        (s.diameter_mm for s in cylinders),
        default=min(s.diameter_small_mm or s.diameter_mm for s in segs),
    )
    entry_type, cb_d, cb_depth, csk_d, csk_angle = _entry_and_bottom(segs, main_d)
    total = t_hi - t_lo
    if through:
        bottom, depth = "through", total
    else:
        t_end = t_hi if entry_low else t_lo
        bottom = _bottom_type(part, g, segs, t_end, main_d / 2)
        depth = max((s.end_depth_mm for s in cylinders), default=segs[-1].end_depth_mm)
    bore_idx = sorted({i for s in stack for i in s.face_idx})
    matches = match_standards(main_d, cb_d)
    warnings: list[str] = []
    partial = [s for s in stack if s.span_deg < config.FULL_BORE_MIN_SPAN_DEG]
    threaded = is_likely_threaded(matches, main_d)
    if partial:
        warnings.append(
            f"{len(partial)} bore segment(s) wrap only {min(s.span_deg for s in partial):.0f} deg: "
            "another feature cuts into this hole"
        )
    if threaded:
        warnings.append(
            "STEP files rarely carry thread data; threaded status is inferred from the diameter only"
        )
    return Hole(
        id="",
        axis=Axis(origin=vec3(_axis_pt(g, t_entry)), direction=vec3(direction)),
        diameter_mm=main_d,
        depth_mm=depth,
        is_through=through,
        entry_type=entry_type,
        counterbore_diameter_mm=cb_d,
        counterbore_depth_mm=cb_depth,
        countersink_diameter_mm=csk_d,
        countersink_angle_deg=_snap_csk(csk_angle) if csk_angle is not None else None,
        bottom_type=bottom,
        segments=segs,
        standard_matches=matches,
        likely_threaded=threaded,
        entry_face_id=_entry_face_id(part, g, bore_idx, t_entry),
        edge_distance_mm=None,
        warnings=warnings,
    )


# ---------------------------------------------------------------- 7.7 edge distance
def _compound(faces: list) -> TopoDS_Compound:  # type: ignore[type-arg]
    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)
    for f in faces:
        builder.Add(comp, f)
    return comp


def compute_edge_distances(part: PartGeom, holes: list[Hole]) -> None:
    """Wall left around each hole: min distance from the bore to non-hole, non-adjacent faces."""
    table = part.table
    if len(table.faces) > config.EDGE_DISTANCE_MAX_FACES:
        part.warnings.append(
            f"edge distance skipped: more than {config.EDGE_DISTANCE_MAX_FACES} faces"
        )
        return
    hole_faces: dict[str, set[int]] = {}
    all_hole_idx: set[int] = set()
    for h in holes:
        idx = {int(fid[1:]) - 1 for s in h.segments for fid in s.face_ids}
        hole_faces[h.id] = idx
        all_hole_idx |= idx
    for h in holes:
        mine = hole_faces[h.id]
        adjacent = set().union(*(table.faces[i].neighbors for i in mine)) if mine else set()
        outer = [
            table.faces[i].face
            for i in range(len(table.faces))
            if i not in all_hole_idx and i not in adjacent
        ]
        if not outer:
            continue
        dss = BRepExtrema_DistShapeShape(
            _compound([table.faces[i].face for i in sorted(mine)]), _compound(outer)
        )
        if dss.IsDone():
            h.edge_distance_mm = float(dss.Value())


# ---------------------------------------------------------------- public entry point
def assign_ids(holes: list[Hole]) -> list[Hole]:
    """Stable IDs H001... sorted by (axis, rounded entry point, diameter): Rule 6.4."""

    def key(h: Hole) -> tuple[float, ...]:
        d = canonical_dir(np.array([h.axis.direction.x, h.axis.direction.y, h.axis.direction.z]))
        o = h.axis.origin
        return (
            *(round(float(c), 3) for c in d),
            round(o.x, 3),
            round(o.y, 3),
            round(o.z, 3),
            round(h.diameter_mm, 3),
        )

    ordered = sorted(holes, key=key)
    for i, h in enumerate(ordered, 1):
        h.id = f"H{i:03d}"
    return ordered


def detect_holes(part: PartGeom) -> list[Hole]:
    """Full hole recognition for one solid."""
    holes: list[Hole] = []
    for g in group_by_axis(find_bore_faces(part)):
        for stack in stack_segments(build_segments(g)):
            h = build_hole(part, g, stack)
            if h is not None:
                holes.append(h)
    holes = assign_ids(holes)
    compute_edge_distances(part, holes)
    return holes
