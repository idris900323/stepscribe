# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Planar outline maths for cut-outs: chaining section edges into loops, shape class, size.

An outline is an ordered closed chain of segments in the host plane's (u, v) coordinates:
straight lines and circular arcs (a signed sweep: positive = counter-clockwise).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from stepscribe import config

Pt = np.ndarray  # shape (2,)


@dataclass
class Seg:
    """A line or an arc from ``p0`` to ``p1`` in (u, v); arcs carry centre, radius, signed sweep."""

    kind: str  # "line" | "arc"
    p0: Pt
    p1: Pt
    centre: Pt | None = None
    radius: float = 0.0
    sweep: float = 0.0  # radians, signed; +-2 pi for a full circle

    def reversed(self) -> Seg:
        return Seg(self.kind, self.p1, self.p0, self.centre, self.radius, -self.sweep)

    def length(self) -> float:
        if self.kind == "line":
            return float(np.linalg.norm(self.p1 - self.p0))
        return self.radius * abs(self.sweep)

    def sample(self, step_deg: float = config.CUTOUT_ARC_STEP_DEG) -> np.ndarray:
        """Points from p0 to p1 inclusive (arcs sampled every *step_deg*)."""
        if self.kind == "line" or self.centre is None:
            return np.array([self.p0, self.p1])
        n = max(2, int(math.ceil(abs(math.degrees(self.sweep)) / step_deg)) + 1)
        a0 = math.atan2(*(self.p0 - self.centre)[::-1])
        ang = a0 + np.linspace(0.0, self.sweep, n)
        pts = self.centre + self.radius * np.c_[np.cos(ang), np.sin(ang)]
        pts[0], pts[-1] = self.p0, self.p1
        return np.asarray(pts)


def arc_sweep(p0: Pt, pm: Pt, p1: Pt, centre: Pt) -> float:
    """Signed sweep p0 -> p1 passing through the mid point *pm* (2 pi if p0 coincides with p1)."""

    def ang(p: Pt) -> float:
        return math.atan2(p[1] - centre[1], p[0] - centre[0])

    two_pi = 2 * math.pi
    if float(np.linalg.norm(p1 - p0)) < config.CUTOUT_CHAIN_TOL:
        return two_pi if (ang(pm) - ang(p0)) % two_pi < math.pi else -two_pi
    ccw = (ang(p1) - ang(p0)) % two_pi
    mid = (ang(pm) - ang(p0)) % two_pi
    return ccw if mid < ccw else -(two_pi - ccw)


def chain_loops(segs: list[Seg]) -> tuple[list[list[Seg]], list[list[Seg]]]:
    """Join unordered segments end to end: (closed loops, open chains)."""
    tol = config.CUTOUT_CHAIN_TOL * 5
    left = list(segs)
    closed: list[list[Seg]] = []
    opened: list[list[Seg]] = []
    while left:
        chain = [left.pop(0)]
        grew = True
        while grew:
            grew = False
            if float(np.linalg.norm(chain[-1].p1 - chain[0].p0)) < tol and len(chain) > 0:
                if len(chain) > 1 or chain[0].kind == "arc":
                    break
            for k, s in enumerate(left):
                for cand in (s, s.reversed()):
                    if float(np.linalg.norm(cand.p0 - chain[-1].p1)) < tol:
                        chain.append(cand)
                        left.pop(k)
                        grew = True
                        break
                else:
                    continue
                break
            if grew:
                continue
            for k, s in enumerate(left):
                for cand in (s, s.reversed()):
                    if float(np.linalg.norm(cand.p1 - chain[0].p0)) < tol:
                        chain.insert(0, cand)
                        left.pop(k)
                        grew = True
                        break
                else:
                    continue
                break
        if float(np.linalg.norm(chain[-1].p1 - chain[0].p0)) < tol and (
            len(chain) > 1 or chain[0].kind == "arc"
        ):
            closed.append(chain)
        else:
            opened.append(chain)
    return closed, opened


def merge_collinear(loop: list[Seg]) -> list[Seg]:
    """Merge consecutive collinear lines and consecutive arcs on one circle."""
    out: list[Seg] = []
    for s in loop:
        if out and _joinable(out[-1], s):
            prev = out.pop()
            out.append(_join(prev, s))
        else:
            out.append(s)
    if len(out) > 1 and _joinable(out[-1], out[0]):
        last = out.pop()
        out[0] = _join(last, out[0])
    return out


def _joinable(a: Seg, b: Seg) -> bool:
    if a.kind != b.kind:
        return False
    if a.kind == "line":
        da, db = a.p1 - a.p0, b.p1 - b.p0
        na, nb = float(np.linalg.norm(da)), float(np.linalg.norm(db))
        if na < 1e-9 or nb < 1e-9:
            return False
        cross = abs(float(da[0] * db[1] - da[1] * db[0])) / (na * nb)
        return cross < 1e-6 and float(np.dot(da, db)) > 0
    if a.centre is None or b.centre is None:
        return False
    same = (
        float(np.linalg.norm(a.centre - b.centre)) < config.CUTOUT_RADIUS_TOL
        and abs(a.radius - b.radius) < config.CUTOUT_RADIUS_TOL
    )
    return same and a.sweep * b.sweep > 0 and abs(a.sweep) + abs(b.sweep) < 2 * math.pi + 1e-6


def _join(a: Seg, b: Seg) -> Seg:
    if a.kind == "line":
        return Seg("line", a.p0, b.p1)
    return Seg("arc", a.p0, b.p1, a.centre, a.radius, a.sweep + b.sweep)


def sample_loop(loop: list[Seg]) -> np.ndarray:
    """Closed polyline of the loop (first point not repeated)."""
    pts = [s.sample()[:-1] for s in loop]
    return np.vstack(pts)


def signed_area(loop: list[Seg]) -> float:
    """Exact signed area (counter-clockwise positive): chord polygon plus circular segments."""
    verts = np.array([s.p0 for s in loop])
    x, y = verts[:, 0], verts[:, 1]
    chord = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    seg = sum(0.5 * s.radius**2 * (s.sweep - math.sin(s.sweep)) for s in loop if s.kind == "arc")
    return chord + seg


def polygon_centroid(pts: np.ndarray) -> np.ndarray:
    """Area centroid of a closed sampled polygon."""
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    a = cr.sum() / 2
    if abs(a) < 1e-12:
        return np.asarray(pts.mean(axis=0))
    return np.array([((x + x1) * cr).sum() / (6 * a), ((y + y1) * cr).sum() / (6 * a)])


def convex_hull(pts: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain; counter-clockwise hull without the repeated first point."""
    p = sorted({(round(float(a), 9), round(float(b), 9)) for a, b in pts})
    if len(p) <= 2:
        return np.array(p)

    def build(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        h: list[tuple[float, float]] = []
        for q in points:
            while (
                len(h) >= 2
                and (
                    (h[-1][0] - h[-2][0]) * (q[1] - h[-2][1])
                    - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0])
                )
                <= 0
            ):
                h.pop()
            h.append(q)
        return h

    lower, upper = build(p), build(p[::-1])
    return np.array(lower[:-1] + upper[:-1])


def min_area_rect(pts: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """(width, length, rotation_deg, centre) of the minimum-area bounding rectangle.

    *length* >= *width*; rotation is the angle of the length axis from +u, in (-90, 90].
    """
    hull = convex_hull(pts)
    if len(hull) < 3:
        return 0.0, 0.0, 0.0, np.asarray(pts.mean(axis=0))
    best: tuple[tuple[float, float], float, float, float, np.ndarray] | None = None
    for i in range(len(hull)):
        d = hull[(i + 1) % len(hull)] - hull[i]
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            continue
        e1 = d / n
        e2 = np.array([-e1[1], e1[0]])
        a, b = hull @ e1, hull @ e2
        w, h = float(a.max() - a.min()), float(b.max() - b.min())
        ang = math.atan2(e1[1], e1[0])
        key = (round(w * h, 7), round(ang % (math.pi / 2), 7))
        if best is not None and key >= best[0]:
            continue
        centre = e1 * float(a.max() + a.min()) / 2 + e2 * float(b.max() + b.min()) / 2
        best = (key, max(w, h), min(w, h), ang if w >= h else ang + math.pi / 2, centre)
    if best is None:
        return 0.0, 0.0, 0.0, np.asarray(pts.mean(axis=0))
    _key, length, width, angle, centre = best
    deg = math.degrees(angle) % 180.0
    if deg > 90.0:
        deg -= 180.0
    if abs(deg) < 1e-6:
        deg = 0.0
    return width, length, deg, centre


def classify(loop: list[Seg]) -> tuple[str, float | None, int | None]:
    """(shape class, corner radius or None, polygon sides or None) of a merged closed loop."""
    lines = [s for s in loop if s.kind == "line"]
    arcs = [s for s in loop if s.kind == "arc"]
    if arcs and not lines and _one_circle(arcs):
        return "circle", None, None
    if arcs and len(loop) == 2 and not lines:
        return "freeform", None, None
    if not arcs and len(lines) >= 3:
        if len(lines) == 4 and _right_angles(loop):
            return "rectangle", None, None
        return "polygon", None, len(lines)
    if len(lines) == 4 and len(arcs) == 4 and _alternates(loop) and _equal_radii(arcs):
        if _right_angles_between_lines(loop):
            return "rounded_rectangle", arcs[0].radius, None
    if len(lines) == 2 and len(arcs) == 2 and _alternates(loop) and _equal_radii(arcs):
        if all(abs(abs(a.sweep) - math.pi) < math.radians(1.0) for a in arcs):
            return "obround", arcs[0].radius, None
    return "freeform", _min_corner_radius(loop), None


def _one_circle(arcs: list[Seg]) -> bool:
    c0, r0 = arcs[0].centre, arcs[0].radius
    if c0 is None:
        return False
    total = sum(abs(a.sweep) for a in arcs)
    return (
        all(
            a.centre is not None
            and float(np.linalg.norm(a.centre - c0)) < config.CUTOUT_RADIUS_TOL
            and abs(a.radius - r0) < config.CUTOUT_RADIUS_TOL
            for a in arcs
        )
        and abs(total - 2 * math.pi) < 1e-3
    )


def _turn_deg(a: Seg, b: Seg) -> float:
    """Absolute direction change at the joint between *a* and *b* (tangent directions)."""
    return abs(math.degrees(_wrap(_end_dir(b, False) - _end_dir(a, True))))


def _wrap(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi


def _end_dir(s: Seg, at_end: bool) -> float:
    if s.kind == "line":
        d = s.p1 - s.p0
        return math.atan2(d[1], d[0])
    assert s.centre is not None
    p = s.p1 if at_end else s.p0
    r = p - s.centre
    a = math.atan2(r[1], r[0])
    return a + (math.pi / 2 if s.sweep > 0 else -math.pi / 2)


def _right_angles(loop: list[Seg]) -> bool:
    tol = config.CUTOUT_RIGHT_ANGLE_TOL_DEG
    return all(abs(_turn_deg(loop[i], loop[(i + 1) % len(loop)]) - 90.0) <= tol for i in range(4))


def _alternates(loop: list[Seg]) -> bool:
    kinds = [s.kind for s in loop]
    return all(kinds[i] != kinds[(i + 1) % len(kinds)] for i in range(len(kinds)))


def _equal_radii(arcs: list[Seg]) -> bool:
    return all(abs(a.radius - arcs[0].radius) < config.CUTOUT_RADIUS_TOL for a in arcs)


def _right_angles_between_lines(loop: list[Seg]) -> bool:
    lines = [s for s in loop if s.kind == "line"]
    tol = config.CUTOUT_RIGHT_ANGLE_TOL_DEG
    for i in range(4):
        d1 = lines[i].p1 - lines[i].p0
        d2 = lines[(i + 1) % 4].p1 - lines[(i + 1) % 4].p0
        cosang = float(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)))
        if abs(math.degrees(math.acos(max(-1.0, min(1.0, cosang)))) - 90.0) > tol:
            return False
    return True


def _min_corner_radius(loop: list[Seg]) -> float | None:
    rs = [s.radius for s in loop if s.kind == "arc" and abs(s.sweep) < 2 * math.pi - 1e-6]
    return min(rs) if rs else None


def corner_radius(loop: list[Seg]) -> float | None:
    """Smallest arc radius joining two lines (inside corners), else None."""
    n = len(loop)
    rs = [
        s.radius
        for i, s in enumerate(loop)
        if s.kind == "arc" and loop[i - 1].kind == "line" and loop[(i + 1) % n].kind == "line"
    ]
    return min(rs) if rs else None


def outline_dicts(loop: list[Seg], origin: Pt) -> list[dict[str, object]]:
    """Ordered lines / arcs relative to *origin* in (u, v), same idea as the Reconstruction text."""

    def p(q: Pt) -> list[float]:
        return [float(q[0] - origin[0]), float(q[1] - origin[1])]

    out: list[dict[str, object]] = []
    for s in loop:
        if s.kind == "line":
            out.append({"type": "line", "from": p(s.p0), "to": p(s.p1)})
        else:
            assert s.centre is not None
            out.append(
                {
                    "type": "arc",
                    "from": p(s.p0),
                    "to": p(s.p1),
                    "centre": p(s.centre),
                    "radius": float(s.radius),
                    "sweep_deg": float(math.degrees(s.sweep)),
                }
            )
    return out


def min_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest distance between two closed polylines (point-to-segment, both directions)."""
    return min(_pts_to_segs(a, b), _pts_to_segs(b, a))


def _pts_to_segs(pts: np.ndarray, poly: np.ndarray) -> float:
    p0, p1 = poly, np.roll(poly, -1, axis=0)
    d = p1 - p0
    dd = (d * d).sum(axis=1)
    dd[dd < 1e-18] = 1e-18
    best = math.inf
    for q in pts:
        t = np.clip(((q - p0) * d).sum(axis=1) / dd, 0.0, 1.0)
        proj = p0 + d * t[:, None]
        best = min(best, float(np.linalg.norm(proj - q, axis=1).min()))
    return best
