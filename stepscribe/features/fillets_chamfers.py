# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Fillets and chamfers. Conservative: a miss is better than a false hit."""

from __future__ import annotations

import math

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.TopoDS import TopoDS_Face

from stepscribe import config
from stepscribe.geometry.occ_utils import (
    angle_between,
    edge_midpoint_and_length,
    normal_at_point,
    outward_normal,
    shared_edges,
    surface_point,
    to_np,
    uv_bounds,
)
from stepscribe.geometry.part_geom import PartGeom
from stepscribe.geometry.surfaces import cylinder_params, plane_normal_origin
from stepscribe.models.schema import Chamfer, Fillet


def tangent_neighbors(part: PartGeom, idx: int) -> int:
    """How many neighbours of face *idx* are G1 with it along a shared edge."""
    fi = part.table.faces[idx]
    count = 0
    for j in sorted(fi.neighbors):
        other = part.table.faces[j]
        for e in shared_edges(fi.face, other.face):
            p, _ln = edge_midpoint_and_length(e)
            na, nb = normal_at_point(fi.face, p), normal_at_point(other.face, p)
            if (
                na is not None
                and nb is not None
                and angle_between(na, nb) <= config.TANGENT_TOL_DEG
            ):
                count += 1
                break
    return count


def _round_face_geometry(part: PartGeom, idx: int) -> tuple[float, float, bool] | None:
    """(radius, span_rad, convex) for a cylinder/torus face, else None."""
    fi = part.table.faces[idx]
    surf = BRepAdaptor_Surface(fi.face)
    u0, u1, v0, v1 = uv_bounds(fi.face)
    um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    p = surface_point(fi.face, um, vm)
    n = outward_normal(fi.face, um, vm)
    if n is None:
        return None
    if fi.kind == "cylinder":
        cp = cylinder_params(fi.face)
        d = cp.direction / np.linalg.norm(cp.direction)
        centre = cp.origin + d * float(np.dot(p - cp.origin, d))
        return cp.radius, u1 - u0, float(np.dot(n, p - centre)) > 0
    if fi.kind == "torus":
        t = surf.Torus()
        ax = t.Axis()
        d = to_np(ax.Direction())
        o = to_np(ax.Location())
        radial = (p - o) - d * float(np.dot(p - o, d))
        if np.linalg.norm(radial) < 1e-9:
            return None
        minor_centre = (
            o
            + d * float(np.dot(p - o, d))
            + radial / np.linalg.norm(radial) * float(t.MajorRadius())
        )
        return float(t.MinorRadius()), v1 - v0, float(np.dot(n, p - minor_centre)) > 0
    return None


def detect_fillets(part: PartGeom, exclude_ids: set[str]) -> list[Fillet]:
    """Cylinders/tori with span < 359 deg that are G1 with at least two neighbours."""
    found: list[tuple[tuple[float, ...], Fillet]] = []
    for idx, fi in enumerate(part.table.faces):
        if fi.kind not in ("cylinder", "torus") or fi.id in exclude_ids:
            continue
        geo = _round_face_geometry(part, idx)
        if geo is None:
            continue
        radius, span, convex = geo
        if math.degrees(span) >= config.FULL_BORE_MIN_SPAN_DEG or radius < config.MIN_FEATURE_SIZE:
            continue
        if tangent_neighbors(part, idx) < 2:
            continue
        length = fi.area / max(radius * span, 1e-9)
        key = (round(radius, 3), *(round(float(c), 3) for c in fi.centroid))
        found.append(
            (
                key,
                Fillet(
                    id="",
                    radius_mm=radius,
                    convex=convex,
                    face_ids=[fi.id],
                    approx_length_mm=length,
                ),
            )
        )
    found.sort(key=lambda kv: kv[0])
    fillets = [f for _k, f in found]
    for i, f in enumerate(fillets, 1):
        f.id = f"FL{i:03d}"
    return fillets


def detect_chamfers(part: PartGeom) -> list[Chamfer]:
    """Narrow planar strips whose two long edges meet neighbours at an angle (not tangent)."""
    chamfers: list[tuple[tuple[float, ...], Chamfer]] = []
    for fi in part.table.faces:
        if fi.kind != "plane":
            continue
        longest = 0.0
        edges: list[tuple[int, float]] = []
        for j in sorted(fi.neighbors):
            lens = [
                edge_midpoint_and_length(e)[1]
                for e in shared_edges(fi.face, part.table.faces[j].face)
            ]
            if lens:
                edges.append((j, max(lens)))
                longest = max(longest, max(lens))
        if longest <= 0:
            continue
        width = fi.area / longest
        if (
            width > config.CHAMFER_MAX_ASPECT * longest
            or width > config.CHAMFER_MAX_WIDTH_REL * part.diagonal
        ):
            continue
        long_nb = [j for j, ln in edges if ln >= config.CHAMFER_LONG_EDGE_REL * longest]
        if len(long_nb) != 2:
            continue
        nc = outward_normal(fi.face, *_mid_uv(fi.face))
        if nc is None:
            continue
        angles = []
        for j in long_nb:
            mid, _ = edge_midpoint_and_length(shared_edges(fi.face, part.table.faces[j].face)[0])
            nn = normal_at_point(part.table.faces[j].face, mid)
            if nn is None:
                break
            angles.append(angle_between(nc, nn))
        else:
            if all(
                config.CHAMFER_MIN_ANGLE_DEG <= a <= config.CHAMFER_MAX_ANGLE_DEG for a in angles
            ):
                alpha = float(np.mean(angles))
                mean_len = float(np.mean([ln for j, ln in edges if j in long_nb]))
                dist = (fi.area / mean_len) / (2 * math.cos(math.radians(alpha)))
                key = (round(dist, 3), *(round(float(c), 3) for c in fi.centroid))
                chamfers.append(
                    (
                        key,
                        Chamfer(
                            id="",
                            distance_mm=dist,
                            angle_deg=alpha,
                            face_ids=[fi.id],
                            approx_length_mm=longest,
                        ),
                    )
                )
    chamfers.sort(key=lambda kv: kv[0])
    out = [c for _k, c in chamfers]
    for i, c in enumerate(out, 1):
        c.id = f"CH{i:03d}"
    return out


def _mid_uv(face: TopoDS_Face) -> tuple[float, float]:
    u0, u1, v0, v1 = uv_bounds(face)
    return 0.5 * (u0 + u1), 0.5 * (v0 + v1)


__all__ = ["detect_chamfers", "detect_fillets", "plane_normal_origin", "tangent_neighbors"]
