# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Hole patterns: circular, rectangular grid, linear."""

from __future__ import annotations

import math

import numpy as np

from stepscribe import config
from stepscribe.describe.phrases import fmt, hole_callout
from stepscribe.geometry.occ_utils import Vec, canonical_dir
from stepscribe.geometry.properties import vec3
from stepscribe.models.schema import Hole, HolePattern


def _entry(h: Hole) -> Vec:
    o = h.axis.origin
    return np.array([o.x, o.y, o.z])


def _axis(h: Hole) -> Vec:
    d = h.axis.direction
    return canonical_dir(np.array([d.x, d.y, d.z]))


def group_holes(holes: list[Hole]) -> list[list[Hole]]:
    """Group by (axis, diameter +/- 0.01, through/blind, entry type)."""
    groups: list[list[Hole]] = []
    for h in holes:
        for g in groups:
            r = g[0]
            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(_axis(h), _axis(r)))))))
            if (
                ang < config.ANGULAR_TOL_DEG
                and abs(h.diameter_mm - r.diameter_mm) <= config.PATTERN_DIAMETER_TOL
                and h.is_through == r.is_through
                and h.entry_type == r.entry_type
            ):
                g.append(h)
                break
        else:
            groups.append([h])
    return groups


def _basis(n: Vec) -> tuple[Vec, Vec]:
    ref = np.eye(3)[int(np.argmin(np.abs(n)))]
    e1 = np.cross(n, ref)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def _fit_circle(pts: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Kasa circle fit: (centre, radius, max residual)."""
    a = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
    b = -(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    centre = np.array([-sol[0] / 2, -sol[1] / 2])
    r2 = centre @ centre - sol[2]
    radius = math.sqrt(max(r2, 0.0))
    resid = float(np.max(np.abs(np.linalg.norm(pts - centre, axis=1) - radius)))
    return centre, radius, resid


def _circular(pts: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    """(centre, radius, angular pitch deg) if points sit on a circle with uniform spacing."""
    if len(pts) < 3:
        return None
    centre, radius, resid = _fit_circle(pts)
    if radius < config.LINEAR_TOL or resid > config.PATTERN_FIT_TOL:
        return None
    ang = np.sort(np.degrees(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])) % 360.0)
    gaps = np.diff(np.r_[ang, ang[0] + 360.0])
    k = int(np.argmax(gaps))
    rest = np.delete(gaps, k)
    pitch = float(np.mean(rest)) if len(rest) else float(gaps[0])
    if len(rest) and float(np.max(np.abs(rest - pitch))) > config.PATTERN_ANGLE_TOL_DEG:
        return None
    if gaps[k] < pitch - config.PATTERN_ANGLE_TOL_DEG:
        return None
    return centre, radius, pitch


def _cluster(values: np.ndarray) -> list[float]:
    vals = sorted(float(v) for v in values)
    out: list[list[float]] = [[vals[0]]]
    for v in vals[1:]:
        if v - out[-1][-1] <= config.PATTERN_FIT_TOL:
            out[-1].append(v)
        else:
            out.append([v])
    return [float(np.mean(c)) for c in out]


def _uniform_pitch(levels: list[float]) -> float | None:
    if len(levels) < 2:
        return None
    d = np.diff(levels)
    if float(np.max(np.abs(d - d.mean()))) > config.PATTERN_FIT_TOL:
        return None
    return float(d.mean())


def _grid(pts: np.ndarray) -> tuple[int, int, float, float] | None:
    """(rows, cols, row_pitch, col_pitch) for a full rectangular lattice, else None."""
    n = len(pts)
    if n < 4:
        return None
    d = np.linalg.norm(pts - pts[0], axis=1)
    d[0] = np.inf
    e1 = pts[int(np.argmin(d))] - pts[0]
    if float(np.linalg.norm(e1)) < config.LINEAR_TOL:
        return None  # coincident holes: no lattice direction
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.array([-e1[1], e1[0]])
    u, v = pts @ e1, pts @ e2
    ul, vl = _cluster(u), _cluster(v)
    if len(ul) < 2 or len(vl) < 2 or len(ul) * len(vl) != n:
        return None
    cells = {
        (int(np.argmin([abs(a - x) for x in ul])), int(np.argmin([abs(b - y) for y in vl])))
        for a, b in zip(u, v, strict=True)
    }
    pu, pv = _uniform_pitch(ul), _uniform_pitch(vl)
    if len(cells) != n or pu is None or pv is None:
        return None
    return len(vl), len(ul), abs(pv), abs(pu)


def _linear(pts: np.ndarray) -> float | None:
    """Pitch if all points are collinear and evenly spaced."""
    if len(pts) < 2:
        return None
    c = pts.mean(axis=0)
    _u, s, vt = np.linalg.svd(pts - c)
    direction = vt[0]
    perp = np.abs((pts - c) @ vt[1]) if len(vt) > 1 else np.zeros(len(pts))
    if float(perp.max()) > config.PATTERN_FIT_TOL:
        return None
    t = np.sort((pts - c) @ direction)
    d = np.diff(t)
    if float(np.max(np.abs(d - d.mean()))) > config.PATTERN_FIT_TOL:
        return None
    return float(d.mean())


def _describe(kind: str, count: int, callout: str, extra: str) -> str:
    return f"{count} × {callout}, {extra}"


def pattern_for_group(group: list[Hole]) -> HolePattern:
    """Classify one group of >= 2 equal holes (circular, then grid, then linear)."""
    n_axis = _axis(group[0])
    e1, e2 = _basis(n_axis)
    pts3 = np.array([_entry(h) for h in group])
    pts = np.c_[pts3 @ e1, pts3 @ e2]
    height = float(np.mean(pts3 @ n_axis))
    callout = hole_callout(group[0])
    count = len(group)
    common = {
        "hole_ids": [h.id for h in group],
        "count": count,
        "diameter_mm": float(np.mean([h.diameter_mm for h in group])),
        "normal": vec3(n_axis),
    }
    grid = _grid(pts) if count == 4 else None
    circ = None if grid else _circular(pts)
    if circ is not None:
        c2, radius, pitch = circ
        centre = e1 * c2[0] + e2 * c2[1] + n_axis * height
        desc = _describe(
            "circular", count, callout, f"on Ø{fmt(2 * radius)} bolt circle, {fmt(pitch)}° pitch"
        )
        return HolePattern(
            id="",
            kind="circular",
            center=vec3(centre),
            pitch_circle_diameter_mm=2 * radius,
            angular_pitch_deg=pitch,
            description=desc,
            **common,
        )
    grid = grid or _grid(pts)
    if grid is not None:
        rows, cols, rp, cp = grid
        centre = e1 * pts[:, 0].mean() + e2 * pts[:, 1].mean() + n_axis * height
        desc = _describe(
            "grid", count, callout, f"rectangular {fmt(cp)} × {fmt(rp)} ({cols} × {rows})"
        )
        return HolePattern(
            id="",
            kind="rectangular_grid",
            grid_rows=rows,
            grid_cols=cols,
            row_pitch_mm=rp,
            col_pitch_mm=cp,
            center=vec3(centre),
            description=desc,
            **common,
        )
    lin_pitch = _linear(pts)
    centre = e1 * pts[:, 0].mean() + e2 * pts[:, 1].mean() + n_axis * height
    if lin_pitch is not None:
        desc = _describe("linear", count, callout, f"in a row, {fmt(abs(lin_pitch))} pitch")
        return HolePattern(
            id="",
            kind="linear",
            pitch_mm=abs(lin_pitch),
            center=vec3(centre),
            description=desc,
            **common,
        )
    desc = _describe("irregular", count, callout, "irregular group")
    return HolePattern(
        id="", kind="irregular_group", center=vec3(centre), description=desc, **common
    )


def detect_patterns(holes: list[Hole]) -> list[HolePattern]:
    """All patterns for a part's holes; singleton holes make no pattern."""
    patterns = [pattern_for_group(g) for g in group_holes(holes) if len(g) >= 2]
    patterns.sort(key=lambda p: (round(p.diameter_mm, 3), p.hole_ids[0]))
    for i, p in enumerate(patterns, 1):
        p.id = f"P{i:03d}"
    return patterns
