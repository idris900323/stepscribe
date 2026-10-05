# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Bosses: convex full cylinders standing on a larger planar face."""

from __future__ import annotations

import math

import numpy as np

from stepscribe import config
from stepscribe.features.holes import find_bore_faces
from stepscribe.geometry.occ_utils import canonical_dir, outward_normal, surface_point, uv_bounds
from stepscribe.geometry.part_geom import PartGeom
from stepscribe.geometry.properties import vec3
from stepscribe.geometry.surfaces import cylinder_params, plane_normal_origin
from stepscribe.models.schema import Axis, Boss, Hole


def _is_convex_full_cylinder(part: PartGeom, idx: int) -> bool:
    fi = part.table.faces[idx]
    u0, u1, v0, v1 = uv_bounds(fi.face)
    if fi.kind != "cylinder" or math.degrees(u1 - u0) < config.FULL_BORE_MIN_SPAN_DEG:
        return False
    cp = cylinder_params(fi.face)
    um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    p = surface_point(fi.face, um, vm)
    n = outward_normal(fi.face, um, vm)
    d = cp.direction / np.linalg.norm(cp.direction)
    radial = p - (cp.origin + d * float(np.dot(p - cp.origin, d)))
    return n is not None and float(np.dot(n, radial)) > 0


def detect_bosses(part: PartGeom, holes: list[Hole]) -> list[Boss]:
    """Convex full cylinders (ratio height/diameter <= 1.5) whose base meets a larger plane."""
    concave_idx = {b.idx for b in find_bore_faces(part)}
    found: list[tuple[tuple[float, ...], Boss]] = []
    for idx, fi in enumerate(part.table.faces):
        if idx in concave_idx or not _is_convex_full_cylinder(part, idx):
            continue
        cp = cylinder_params(fi.face)
        d = canonical_dir(cp.direction)
        _u0, _u1, v0, v1 = uv_bounds(fi.face)
        height = abs(v1 - v0)
        diameter = 2 * cp.radius
        if height > config.BOSS_MAX_HEIGHT_RATIO * diameter:
            continue
        planes = []
        for j in sorted(fi.neighbors):
            nb = part.table.faces[j]
            if nb.kind != "plane":
                continue
            n, o = plane_normal_origin(nb.face)
            if abs(abs(float(np.dot(n, d))) - 1.0) < 1e-3:
                planes.append((nb, float(np.dot(o - cp.origin, d))))
        footprint = math.pi * cp.radius**2
        bases = [(nb, t) for nb, t in planes if nb.area >= config.BOSS_BASE_AREA_FACTOR * footprint]
        tops = [(nb, t) for nb, t in planes if nb.area < config.BOSS_BASE_AREA_FACTOR * footprint]
        if len(bases) != 1 or not tops:
            continue
        t_base, t_top = bases[0][1], tops[0][1]
        direction = d if t_top > t_base else -d
        origin = cp.origin + d * t_base
        has_hole = _coaxial_hole(holes, origin, d, min(t_base, t_top), max(t_base, t_top))
        key = (round(diameter, 3), *(round(float(c), 3) for c in fi.centroid))
        found.append(
            (
                key,
                Boss(
                    id="",
                    axis=Axis(origin=vec3(origin), direction=vec3(direction)),
                    diameter_mm=diameter,
                    height_mm=abs(t_top - t_base),
                    face_ids=[fi.id],
                    has_hole_id=has_hole,
                ),
            )
        )
    found.sort(key=lambda kv: kv[0])
    bosses = [b for _k, b in found]
    for i, b in enumerate(bosses, 1):
        b.id = f"B{i:03d}"
    return bosses


def _coaxial_hole(
    holes: list[Hole], origin: np.ndarray, d: np.ndarray, t0: float, t1: float
) -> str | None:
    for h in holes:
        e = np.array([h.axis.origin.x, h.axis.origin.y, h.axis.origin.z])
        off = e - origin
        dist = float(np.linalg.norm(off - d * np.dot(off, d)))
        t = float(np.dot(off, d))
        if (
            dist < config.COAXIAL_TOL
            and t0 - 10 * config.LINEAR_TOL <= t <= t1 + 10 * config.LINEAR_TOL
        ):
            return h.id
    return None
