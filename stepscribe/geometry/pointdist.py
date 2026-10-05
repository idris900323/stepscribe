# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Point-to-surface distance tests against a triangulation, using a uniform grid and numpy.

Used by the symmetry checks: "is every mirrored sample point within *tol* of the surface?".
Only triangles that can be within *tol* of a query point are examined.
"""

from __future__ import annotations

import numpy as np

from stepscribe.geometry.occ_utils import Vec


def sample_surface(
    vertices: np.ndarray, triangles: np.ndarray, n: int, seed: int = 12345
) -> np.ndarray:
    """Area-weighted, deterministic surface samples (n, 3)."""
    tri = vertices[triangles]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    total = float(area.sum())
    if total <= 0 or n <= 0:
        return np.zeros((0, 3))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(tri), size=n, p=area / total)
    r1, r2 = rng.random_sample(n), rng.random_sample(n)
    s = np.sqrt(r1)
    a, b, c = 1 - s, s * (1 - r2), s * r2
    t = tri[idx]
    return np.asarray(a[:, None] * t[:, 0] + b[:, None] * t[:, 1] + c[:, None] * t[:, 2])


def _point_triangles_distance(p: Vec, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Distance from *p* to each triangle (a, b, c are (m, 3)); Ericson's closest point."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    m = len(a)
    q = np.empty((m, 3))
    done = np.zeros(m, dtype=bool)

    def put(mask: np.ndarray, pts: np.ndarray) -> None:
        sel = mask & ~done
        q[sel] = pts[sel]
        done[sel] = True

    put((d1 <= 0) & (d2 <= 0), a)
    put((d3 >= 0) & (d4 <= d3), b)
    put((d6 >= 0) & (d5 <= d6), c)
    with np.errstate(divide="ignore", invalid="ignore"):
        v_ab = d1 / (d1 - d3)
        put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + v_ab[:, None] * ab)
        w_ac = d2 / (d2 - d6)
        put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + w_ac[:, None] * ac)
        w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + w_bc[:, None] * (c - b))
        denom = 1.0 / (va + vb + vc)
        v, w = vb * denom, vc * denom
        inside = a + ab * v[:, None] + ac * w[:, None]
    put(np.ones(m, dtype=bool), inside)
    return np.asarray(np.linalg.norm(q - p, axis=1))


class SurfaceGrid:
    """Grid over triangles; ``distance_within`` returns the distance to the surface if < tol."""

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray, tol: float) -> None:
        self.tol = tol
        self.tri = vertices[triangles]
        lo = self.tri.reshape(-1, 3).min(axis=0) - tol
        hi = self.tri.reshape(-1, 3).max(axis=0) + tol
        self.lo = lo
        diag = float(np.linalg.norm(hi - lo))
        self.cell = max(4.0 * tol, diag / 48.0)
        self.cells: dict[tuple[int, int, int], list[int]] = {}
        tmin = (self.tri.min(axis=1) - tol - lo) / self.cell
        tmax = (self.tri.max(axis=1) + tol - lo) / self.cell
        for k in range(len(self.tri)):
            i0, j0, k0 = np.floor(tmin[k]).astype(int)
            i1, j1, k1 = np.floor(tmax[k]).astype(int)
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    for m in range(k0, k1 + 1):
                        self.cells.setdefault((i, j, m), []).append(k)
        self._arr = {key: np.array(v) for key, v in self.cells.items()}

    def distance_within(self, p: Vec) -> float | None:
        """Distance from *p* to the surface when it is below the tolerance, else None."""
        key = tuple(np.floor((p - self.lo) / self.cell).astype(int))
        cand = self._arr.get(key)
        if cand is None:
            return None
        t = self.tri[cand]
        d = float(_point_triangles_distance(p, t[:, 0], t[:, 1], t[:, 2]).min())
        return d if d < self.tol else None

    def all_within(self, pts: np.ndarray) -> tuple[bool, float]:
        """(every point within tol, largest distance found); stops at the first miss."""
        worst = 0.0
        for p in pts:
            d = self.distance_within(p)
            if d is None:
                return False, float("inf")
            worst = max(worst, d)
        return True, worst
