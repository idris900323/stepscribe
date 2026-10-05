# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Deterministic shape classification. Every result carries evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from stepscribe import config
from stepscribe.features.holes import find_bore_faces
from stepscribe.geometry.occ_utils import to_np
from stepscribe.geometry.part_geom import PartGeom
from stepscribe.geometry.raycast import TriMesh
from stepscribe.geometry.sections import section_area
from stepscribe.geometry.surfaces import cylinder_params, plane_normal_origin
from stepscribe.models.schema import Hole, OrientedBBox, ShapeClass

THIN_RATIO = 0.12  # thin-walled if 2V/A <= this * middle OBB size
PLATE_C_OVER_B = 0.2
PLATE_AREA_SHARE = 0.7
CYL_AREA_SHARE = 0.9
BAR_ASPECT = 4.0
HOUSING_FILL = 0.35
GEAR_MIN_TEETH = 8
FREEFORM_SHARE = 0.5
SECTION_AREA_TOL = 0.01
GRID_N = 4


@dataclass
class Ctx:
    """Everything the classification rules look at."""

    part: PartGeom
    obb: OrientedBBox
    volume: float
    area: float
    hole_face_ids: set[str]
    wall: tuple[float, float] | None

    @property
    def sizes(self) -> tuple[float, float, float]:
        a, b, c = self.obb.size_sorted
        return a, b, c

    @property
    def axes(self) -> list[np.ndarray]:
        return [np.array([v.x, v.y, v.z]) for v in self.obb.axes]

    @property
    def half(self) -> np.ndarray:
        h = self.obb.half_sizes
        return np.array([h.x, h.y, h.z])

    @property
    def centre(self) -> np.ndarray:
        c = self.obb.center
        return np.array([c.x, c.y, c.z])


def _axis_of_size(ctx: Ctx, rank: int) -> np.ndarray:
    """OBB axis whose extent is the rank-th largest (0 = longest)."""
    order = np.argsort(-ctx.half)
    return ctx.axes[int(order[rank])]


def _total_area(ctx: Ctx) -> float:
    return sum(f.area for f in ctx.part.table.faces) or 1.0


# ------------------------------------------------------------------ individual rules
def rule_gear(ctx: Ctx) -> ShapeClass | None:
    """>= 8-fold rotational repetition of equal faces around one OBB axis."""
    faces = [f for f in ctx.part.table.faces if f.id not in ctx.hole_face_ids]
    for axis in ctx.axes:
        groups: dict[tuple[str, float, float], list] = {}  # type: ignore[type-arg]
        for f in faces:
            rel = f.centroid - ctx.centre
            rel = rel - axis * float(np.dot(rel, axis))
            if np.linalg.norm(rel) < config.LINEAR_TOL:
                continue
            groups.setdefault(
                (f.kind, round(f.area, 2), _signed_normal_angle(f, rel, axis)), []
            ).append(f)
        for (kind, area, _a), members in sorted(groups.items()):
            if len(members) < GEAR_MIN_TEETH:
                continue
            rel = np.array([m.centroid - ctx.centre for m in members])
            rel = rel - np.outer(rel @ axis, axis)
            radii = np.linalg.norm(rel, axis=1)
            if radii.std() > 0.02 * radii.mean():
                continue
            e1 = rel[0] / np.linalg.norm(rel[0])
            e2 = np.cross(axis, e1)
            ang = np.sort(np.degrees(np.arctan2(rel @ e2, rel @ e1)) % 360.0)
            gaps = np.diff(np.r_[ang, ang[0] + 360.0])
            if float(np.std(gaps)) < 1.0:
                return ShapeClass(
                    confidence=0.7,
                    evidence=f"{len(members)} equal {kind} faces (area {area}) repeat every {gaps.mean():.1f}° around one axis",
                    label="gear_like",
                )
    return None


def _signed_normal_angle(f, rel: np.ndarray, axis: np.ndarray) -> float:  # type: ignore[no-untyped-def]
    """Signed angle (deg, 1 deg bins) between a planar face normal and its radial direction.

    Mirror-image faces (e.g. the two walls of a slot) get different values, so only true
    rotational repeats group together. Non-planar faces return 0.
    """
    if f.kind != "plane":
        return 0.0
    n, _o = plane_normal_origin(f.face)
    n = n - axis * float(np.dot(n, axis))
    if np.linalg.norm(n) < 1e-9:
        return 999.0
    r = rel / np.linalg.norm(rel)
    ang = math.degrees(
        math.atan2(
            float(np.dot(np.cross(r, n / np.linalg.norm(n)), axis)),
            float(np.dot(r, n / np.linalg.norm(n))),
        )
    )
    if ang > 90:
        ang -= 180
    elif ang < -90:
        ang += 180
    return float(round(ang))


def rule_freeform(ctx: Ctx) -> ShapeClass | None:
    """B-spline faces dominate the surface area."""
    total = _total_area(ctx)
    spline = sum(f.area for f in ctx.part.table.faces if f.kind in ("bspline", "bezier"))
    if spline / total > FREEFORM_SHARE:
        return ShapeClass(
            confidence=0.8,
            evidence=f"{100 * spline / total:.0f}% of the surface area is B-spline",
            label="freeform",
        )
    return None


def rule_revolved(ctx: Ctx) -> ShapeClass | None:
    """Cylinder-dominated bodies: tube / ring / shaft / disc."""
    table = ctx.part.table
    cyls = [f for f in table.faces if f.kind == "cylinder"]
    if not cyls:
        return None
    big = max(cyls, key=lambda f: f.area)
    cp = cylinder_params(big.face)
    axis = cp.direction / np.linalg.norm(cp.direction)
    total = _total_area(ctx)
    share = 0.0
    radii: list[float] = []
    for f in table.faces:
        if f.kind == "cylinder":
            c = cylinder_params(f.face)
            off = c.origin - cp.origin
            coax = float(np.linalg.norm(off - axis * np.dot(off, axis))) < config.COAXIAL_TOL
            par = (
                abs(abs(float(np.dot(c.direction / np.linalg.norm(c.direction), axis))) - 1) < 1e-4
            )
            if coax and par:
                share += f.area
                radii.append(c.radius)
        elif f.kind in ("plane", "cone"):
            if f.kind == "plane":
                n, _o = plane_normal_origin(f.face)
                if abs(abs(float(np.dot(n, axis))) - 1) < 1e-4:
                    share += f.area
            else:
                share += f.area
    if share / total < CYL_AREA_SHARE:
        return None
    proj = [abs(float(np.dot(ax, axis))) for ax in ctx.axes]
    length = 2 * float(ctx.half[int(np.argmax(proj))])
    d_out = 2 * max(radii)
    hollow = len(radii) >= 2 and min(radii) < max(radii) * 0.98 and _has_through_bore(ctx, axis)
    ev = f"{100 * share / total:.0f}% of surface area is coaxial about one axis; Ø{d_out:.1f} × {length:.1f} long"
    if hollow:
        label = "tube" if length / d_out >= 0.5 else "ring"
        return ShapeClass(
            confidence=0.9, evidence=ev + f"; bore Ø{2 * min(radii):.1f}", label=label
        )
    label = "shaft" if length / d_out >= 1.0 else "disc"
    return ShapeClass(confidence=0.85, evidence=ev, label=label)


def _has_through_bore(ctx: Ctx, axis: np.ndarray) -> bool:
    for b in find_bore_faces(ctx.part):
        d = b.direction / np.linalg.norm(b.direction)
        if (
            abs(abs(float(np.dot(d, axis))) - 1) < 1e-4
            and b.span_deg >= config.FULL_BORE_MIN_SPAN_DEG
        ):
            return True
    return False


def rule_plate(ctx: Ctx) -> ShapeClass | None:
    """c < 0.2 b and > 70 % of area on two parallel planes."""
    _a, b, c = ctx.sizes
    if c >= PLATE_C_OVER_B * b:
        return None
    n = _axis_of_size(ctx, 2)
    share = 0.0
    for f in ctx.part.table.faces:
        if f.kind == "plane":
            pn, _o = plane_normal_origin(f.face)
            if abs(abs(float(np.dot(pn, n))) - 1) < 1e-3:
                share += f.area
    total = _total_area(ctx)
    if share / total < PLATE_AREA_SHARE:
        return None
    return ShapeClass(
        confidence=0.9,
        evidence=f"thickness {c:.1f} < 0.2 × {b:.1f}; {100 * share / total:.0f}% of area on two parallel planes",
        label="plate",
        thickness_mm=c,
    )


def _section_areas(ctx: Ctx, axis: np.ndarray) -> list[float]:
    half = float(ctx.half[int(np.argmax([abs(float(np.dot(ax, axis))) for ax in ctx.axes]))])
    extent = ctx.part.diagonal
    return [
        section_area(ctx.part.shape, ctx.centre + axis * t * half, axis, extent)
        for t in (-0.5, 0.0, 0.5)
    ]


def rule_bar(ctx: Ctx) -> ShapeClass | None:
    """a > 4 b with a constant cross-section (3 sections within 1 %)."""
    a, b, c = ctx.sizes
    if a <= BAR_ASPECT * b:
        return None
    axis = _axis_of_size(ctx, 0)
    areas = _section_areas(ctx, axis)
    if min(areas) <= 0 or (max(areas) - min(areas)) / max(areas) > SECTION_AREA_TOL:
        return None
    fill = areas[1] / (b * c)
    label = "bar" if fill >= 0.98 or abs(fill - math.pi / 4) < 0.02 else "extrusion_profile"
    return ShapeClass(
        confidence=0.85,
        evidence=f"length {a:.1f} > 4 × {b:.1f}; cross-section {areas[1]:.1f} mm² constant along the length ({100 * fill:.0f}% of its bounding rectangle)",
        label=label,
        thickness_mm=None,
        notes=[f"cross-section outer size {b:.1f} × {c:.1f} mm"],
    )


def _thin_constant(ctx: Ctx) -> float | None:
    """Wall thickness if the body is thin-walled with constant thickness, else None."""
    _a, b, _c = ctx.sizes
    t_est = 2 * ctx.volume / ctx.area if ctx.area else 0.0
    if t_est <= 0 or t_est > THIN_RATIO * b or ctx.wall is None:
        return None
    if abs(ctx.wall[0] - t_est) > 0.25 * t_est:
        return None
    return float(ctx.wall[0])


def _orientation_groups(ctx: Ctx) -> list[tuple[np.ndarray, float, list[np.ndarray]]]:
    """Planar faces grouped by normal direction (sign ignored): (normal, area, centroids)."""
    groups: list[tuple[np.ndarray, float, list[np.ndarray]]] = []
    for f in ctx.part.table.faces:
        if f.kind != "plane":
            continue
        n, _o = plane_normal_origin(f.face)
        for i, (gn, ga, gc) in enumerate(groups):
            if abs(abs(float(np.dot(n, gn))) - 1) < 1e-3:
                groups[i] = (gn, ga + f.area, [*gc, f.centroid])
                break
        else:
            groups.append((n, f.area, [f.centroid]))
    return groups


def rule_sheet_or_bracket(ctx: Ctx) -> ShapeClass | None:
    """Constant-thickness bodies: sheet_metal (with bends) or L / U / Z / angle brackets."""
    t = _thin_constant(ctx)
    if t is None:
        return None
    total_planar = sum(a for _n, a, _c in _orientation_groups(ctx)) or 1.0
    groups = sorted(_orientation_groups(ctx), key=lambda g: -g[1])
    big = [g for g in groups if g[1] / total_planar >= 0.1]
    bends = [
        f
        for f in ctx.part.table.faces
        if f.kind == "cylinder"
        and cylinder_params(f.face).radius <= 3 * t + config.LINEAR_TOL
        and f.id not in ctx.hole_face_ids
    ]
    base_ev = f"constant thickness {t:.1f} mm (2V/A and sampled minimum agree)"
    if len(bends) >= 2:
        return ShapeClass(
            confidence=0.75,
            evidence=base_ev + f"; {len(bends)} bend faces of radius ≈ thickness",
            label="sheet_metal",
            thickness_mm=t,
        )
    if len(big) == 2 and abs(float(np.dot(big[0][0], big[1][0]))) < 1e-3:
        a = ctx.sizes[0]
        leg = ctx.sizes[1]
        label = "angle" if a > 4 * leg else "l_bracket"
        return ShapeClass(
            confidence=0.85,
            evidence=base_ev + "; large planar faces fall into 2 perpendicular orientation groups",
            label=label,
            thickness_mm=t,
        )
    if len(big) == 3:
        return _u_or_z(ctx, big, base_ev, t)
    return None


def _u_or_z(ctx: Ctx, big, base_ev: str, t: float) -> ShapeClass | None:  # type: ignore[no-untyped-def]
    normals = [g[0] for g in big]
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(abs(float(np.dot(normals[i], normals[j]))) - 1) < 1e-3:
                web = next(k for k in range(3) if k not in (i, j))
                wn = normals[web]
                web_c = np.mean(big[web][2], axis=0)
                sides = [float(np.dot(np.mean(big[k][2], axis=0) - web_c, wn)) for k in (i, j)]
                if abs(sides[0]) < config.LINEAR_TOL or abs(sides[1]) < config.LINEAR_TOL:
                    return None
                same = (sides[0] > 0) == (sides[1] > 0)
                return ShapeClass(
                    confidence=0.75,
                    evidence=base_ev
                    + f"; 3 orientation groups with two parallel flanges on the {'same' if same else 'opposite'} side of the web",
                    label="u_bracket" if same else "z_bracket",
                    thickness_mm=t,
                )
    return None


def _is_enclosed(ctx: Ctx, mesh: TriMesh, p: np.ndarray) -> bool:
    """A void point is enclosed if rays in >= 5 of the 6 OBB-axis directions hit material."""
    hits = 0
    for ax in ctx.axes:
        for s in (1.0, -1.0):
            if mesh.first_hit(p, ax * s) is not None:
                hits += 1
    return hits >= 5


def rule_housing(ctx: Ctx) -> ShapeClass | None:
    """Volume / OBB volume < 0.35 and a large internal cavity (ray casting on the triangulation)."""
    a, b, c = ctx.sizes
    fill = ctx.volume / (a * b * c) if a * b * c else 1.0
    if fill >= HOUSING_FILL:
        return None
    mesh = ctx.part.trimesh
    if mesh is None:
        return None
    enclosed = 0
    for i in range(GRID_N):
        for j in range(GRID_N):
            for k in range(GRID_N):
                f = (
                    np.array([i, j, k], dtype=float) / (GRID_N - 1) * 1.6 - 0.8
                )  # stay inside the OBB
                p = ctx.centre + sum(f[m] * ctx.half[m] * ctx.axes[m] for m in range(3))
                if not mesh.is_inside(p) and _is_enclosed(ctx, mesh, p):
                    enclosed += 1
    if enclosed >= max(3, 0.05 * GRID_N**3):
        return ShapeClass(
            confidence=0.7,
            evidence=(
                f"volume is {100 * fill:.0f}% of the oriented bounding box and {enclosed} of "
                f"{GRID_N**3} sample points lie in an enclosed cavity"
            ),
            label="housing",
        )
    return None


def rule_block(ctx: Ctx) -> ShapeClass:
    """Fallback: block when it fills most of its OBB, else other."""
    a, b, c = ctx.sizes
    fill = ctx.volume / (a * b * c) if a * b * c else 0.0
    planar = sum(f.area for f in ctx.part.table.faces if f.kind == "plane") / _total_area(ctx)
    if fill >= 0.5 and planar >= 0.5:
        return ShapeClass(
            confidence=0.6,
            evidence=f"fills {100 * fill:.0f}% of its {a:.1f} × {b:.1f} × {c:.1f} oriented box; {100 * planar:.0f}% planar faces",
            label="block",
        )
    return ShapeClass(
        confidence=0.3,
        evidence=f"no rule matched; fills {100 * fill:.0f}% of its oriented box",
        label="other",
    )


def classify_shape(
    part: PartGeom,
    obb: OrientedBBox,
    volume: float,
    area: float,
    holes: list[Hole],
    wall: tuple[float, float] | None,
) -> ShapeClass:
    """Apply the rules in order; the first match wins."""
    hole_faces = {fid for h in holes for s in h.segments for fid in s.face_ids}
    ctx = Ctx(part, obb, volume, area, hole_faces, wall)
    for rule in (
        rule_gear,
        rule_freeform,
        rule_revolved,
        rule_plate,
        rule_bar,
        rule_sheet_or_bracket,
        rule_housing,
    ):
        result = rule(ctx)
        if result is not None:
            return result
    return rule_block(ctx)


__all__ = ["classify_shape", "to_np"]
