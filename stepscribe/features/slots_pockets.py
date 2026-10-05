# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Slots (two concave half-cylinders + walls) and pockets (concave planar floors)."""

from __future__ import annotations

import math

import numpy as np
from OCP.Bnd import Bnd_OBB
from OCP.BRepBndLib import BRepBndLib
from OCP.TopoDS import TopoDS_Face

from stepscribe import config
from stepscribe.features.holes import BoreFace, find_bore_faces
from stepscribe.geometry.occ_utils import canonical_dir, outward_normal, to_np, uv_bounds
from stepscribe.geometry.part_geom import PartGeom
from stepscribe.geometry.properties import vec3
from stepscribe.geometry.surfaces import cylinder_params, plane_normal_origin
from stepscribe.models.schema import Hole, Pocket, Slot


def _half_cylinders(part: PartGeom) -> list[BoreFace]:
    lo = config.SLOT_SPAN_DEG - config.SLOT_SPAN_TOL_DEG
    hi = config.SLOT_SPAN_DEG + config.SLOT_SPAN_TOL_DEG
    return [b for b in find_bore_faces(part) if b.kind == "cylinder" and lo <= b.span_deg <= hi]


def _common_walls(part: PartGeom, a: int, b: int) -> list[int]:
    common = part.table.faces[a].neighbors & part.table.faces[b].neighbors
    return sorted(j for j in common if part.table.faces[j].kind == "plane")


def _walls_parallel(
    part: PartGeom,
    walls: list[int],
    axis_dir: np.ndarray,
    cache: dict[int, np.ndarray] | None = None,
) -> bool:
    """True if exactly two walls parallel to the slot axis exist and are parallel to each other.

    ``cache`` maps a face index to its plane normal; a part with hundreds of half-cylinders asks
    for the same few planes over and over, and each lookup builds an OCC surface adaptor.
    """
    if cache is None:
        cache = {}
    normals = []
    for j in walls:
        if j not in cache:
            cache[j] = plane_normal_origin(part.table.faces[j].face)[0]
        normals.append(cache[j])
    side = [n for n in normals if abs(float(np.dot(n, axis_dir))) < 1e-3]
    if len(side) != 2:
        return False
    return abs(abs(float(np.dot(side[0], side[1]))) - 1.0) < 1e-3


def detect_slots(part: PartGeom) -> tuple[list[Slot], set[str]]:
    """Slots and the IDs of their end faces (so fillet detection can skip them)."""
    halves = _half_cylinders(part)
    slots: list[tuple[tuple[float, ...], Slot]] = []
    used: set[str] = set()
    normal_cache: dict[int, np.ndarray] = {}
    for i, a in enumerate(halves):
        for b in halves[i + 1 :]:
            da, db = canonical_dir(a.direction), canonical_dir(b.direction)
            if (
                abs(a.radius_ref - b.radius_ref) > config.LINEAR_TOL
                or abs(float(np.dot(da, db))) < 1 - 1e-6
            ):
                continue
            walls = _common_walls(part, a.idx, b.idx)
            if not _walls_parallel(part, walls, da, normal_cache):
                continue
            slot = _make_slot(part, a, b, da)
            if slot is None:
                continue
            used.update(slot.face_ids)
            slots.append(
                (
                    (
                        round(slot.width_mm, 3),
                        *(round(c, 3) for c in (slot.center.x, slot.center.y, slot.center.z)),
                    ),
                    slot,
                )
            )
    slots.sort(key=lambda kv: kv[0])
    out = [s for _k, s in slots]
    for n, s in enumerate(out, 1):
        s.id = f"S{n:03d}"
    return out, used


def _axis_point(b: BoreFace, d: np.ndarray, t: float) -> np.ndarray:
    return b.origin + d * (t - float(np.dot(b.origin, d)))


def _make_slot(part: PartGeom, a: BoreFace, b: BoreFace, d: np.ndarray) -> Slot | None:
    ca = cylinder_params(part.table.faces[a.idx].face)
    cb = cylinder_params(part.table.faces[b.idx].face)
    off = cb.origin - ca.origin
    sep = off - d * float(np.dot(off, d))  # separation between the two axes, perpendicular to d
    dist = float(np.linalg.norm(sep))
    if dist < config.LINEAR_TOL:
        return None
    r = a.radius_ref
    ta0 = float(np.dot(ca.origin, d)) + min(a.v0, a.v1) * np.sign(
        float(np.dot(ca.direction, d)) or 1.0
    )
    ta1 = float(np.dot(ca.origin, d)) + max(a.v0, a.v1) * np.sign(
        float(np.dot(ca.direction, d)) or 1.0
    )
    t_lo, t_hi = min(ta0, ta1), max(ta0, ta1)
    centre_axis = ca.origin + sep / 2
    centre_axis = centre_axis - d * float(np.dot(centre_axis, d))
    p_lo = centre_axis + d * (t_lo - config.HOLE_CLASSIFY_OFFSET_MIN)
    p_hi = centre_axis + d * (t_hi + config.HOLE_CLASSIFY_OFFSET_MIN)
    s_lo, s_hi = part.classifier.state(p_lo), part.classifier.state(p_hi)
    through = s_lo != "in" and s_hi != "in"
    mid = centre_axis + d * (t_lo + t_hi) / 2
    return Slot(
        id="",
        width_mm=2 * r,
        length_mm=dist + 2 * r,
        depth_mm=None if through else t_hi - t_lo,
        is_through=through,
        axis_dir=vec3(sep / dist),
        center=vec3(mid),
        face_ids=sorted([part.table.faces[a.idx].id, part.table.faces[b.idx].id]),
    )


def detect_pockets(part: PartGeom, holes: list[Hole]) -> list[Pocket]:
    """Planar floors whose boundary walls are all concave and perpendicular to the floor."""
    table = part.table
    hole_face_idx = {int(fid[1:]) - 1 for h in holes for s in h.segments for fid in s.face_ids}
    slot_halves = {b.idx for b in _half_cylinders(part)}
    pockets: list[tuple[tuple[float, ...], Pocket]] = []
    for idx, fi in enumerate(table.faces):
        if fi.kind != "plane" or not fi.neighbors or fi.neighbors & hole_face_idx:
            continue
        floor_n = _plane_outward(part, idx)
        if floor_n is None:
            continue
        walls = [table.faces[j] for j in sorted(fi.neighbors)]
        if not all(_is_concave_wall(part, fi, w, floor_n) for w in walls):
            continue
        top = _top_plane(part, fi, floor_n)
        if top is None:
            continue
        pocket = _make_pocket(part, fi, floor_n, top, walls)
        if pocket is not None and idx not in slot_halves:
            pockets.append(
                ((round(pocket.depth_mm, 3), *(round(float(c), 3) for c in fi.centroid)), pocket)
            )
    pockets.sort(key=lambda kv: kv[0])
    out = [p for _k, p in pockets]
    for i, p in enumerate(out, 1):
        p.id = f"PK{i:03d}"
    return out


def _plane_outward(part: PartGeom, idx: int) -> np.ndarray | None:
    fi = part.table.faces[idx]
    u0, u1, v0, v1 = uv_bounds(fi.face)
    return outward_normal(fi.face, 0.5 * (u0 + u1), 0.5 * (v0 + v1))


def _is_concave_wall(part: PartGeom, floor, wall, floor_n: np.ndarray) -> bool:  # type: ignore[no-untyped-def]
    """Wall perpendicular to the floor with its outward normal pointing toward the floor centre."""
    if wall.kind == "plane":
        wn, _o = plane_normal_origin(wall.face)
        if abs(float(np.dot(wn, floor_n))) > 1e-3:
            return False
        out = outward_normal(wall.face, *_mid(wall.face))
        return out is not None and float(np.dot(out, floor.centroid - wall.centroid)) > 0
    if wall.kind == "cylinder":
        cp = cylinder_params(wall.face)
        if (
            abs(abs(float(np.dot(cp.direction / np.linalg.norm(cp.direction), floor_n))) - 1.0)
            > 1e-3
        ):
            return False
        out = outward_normal(wall.face, *_mid(wall.face))
        return out is not None and float(np.dot(out, floor.centroid - wall.centroid)) > 0
    return False


def _mid(face: TopoDS_Face) -> tuple[float, float]:
    u0, u1, v0, v1 = uv_bounds(face)
    return 0.5 * (u0 + u1), 0.5 * (v0 + v1)


def _top_plane(part: PartGeom, floor, floor_n: np.ndarray):  # type: ignore[no-untyped-def]
    """A planar face parallel to the floor, facing the same way, reached through a wall."""
    for j in sorted(floor.neighbors):
        for k in sorted(part.table.faces[j].neighbors):
            cand = part.table.faces[k]
            if cand.kind != "plane" or cand is floor:
                continue
            n, _o = plane_normal_origin(cand.face)
            out = _plane_outward(part, k)
            if (
                out is not None
                and float(np.dot(out, floor_n)) > 1 - 1e-3
                and abs(abs(float(np.dot(n, floor_n))) - 1) < 1e-3
            ):
                return cand
    return None


def _make_pocket(part: PartGeom, floor, floor_n: np.ndarray, top, walls) -> Pocket | None:  # type: ignore[no-untyped-def]
    n, o_top = plane_normal_origin(top.face)
    _n2, o_floor = plane_normal_origin(floor.face)
    depth = float(np.dot(o_top - o_floor, floor_n))
    if depth <= config.LINEAR_TOL:
        return None
    box = Bnd_OBB()
    BRepBndLib.AddOBB_s(floor.face, box, False, True, False)
    sizes = sorted([2 * box.XHSize(), 2 * box.YHSize(), 2 * box.ZHSize()], reverse=True)
    radii = [cylinder_params(w.face).radius for w in walls if w.kind == "cylinder"]
    return Pocket(
        id="",
        depth_mm=depth,
        outline_size=(float(sizes[0]), float(sizes[1])),
        floor_face_id=floor.id,
        corner_radius_mm=float(min(radii)) if radii else None,
    )


__all__ = ["detect_pockets", "detect_slots", "math", "to_np"]
