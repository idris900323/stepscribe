# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Fast ray casting against a part's triangulation (vectorised Moller-Trumbore).

Used for enclosure tests and sampled wall thickness. Distances carry the mesh deflection error
(0.1 % of the bounding-box diagonal at most), which is fine for the sampled estimates they feed.
"""

from __future__ import annotations

import numpy as np

from stepscribe.geometry.occ_utils import Vec

EPS = 1e-9
MAX_TRIANGLES = 400_000  # beyond this, ray casting is skipped (callers must handle None)
PARITY_DIRECTION = np.array(
    [0.5773502, 0.3141592, 0.7548776]
)  # irrational-ish: avoids grazing edges


class TriMesh:
    """Triangle soup with ray queries."""

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray) -> None:
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        self.v0 = v0
        self.e1 = v1 - v0
        self.e2 = v2 - v0
        self.count = len(triangles)

    def hits(self, origin: Vec, direction: Vec) -> np.ndarray:
        """Sorted ray parameters t > 0 of all triangle hits (direction should be a unit vector)."""
        d = direction / np.linalg.norm(direction)
        p = np.cross(d, self.e2)
        det = np.einsum("ij,ij->i", self.e1, p)
        ok = np.abs(det) > EPS
        inv = np.zeros_like(det)
        inv[ok] = 1.0 / det[ok]
        tvec = origin - self.v0
        u = np.einsum("ij,ij->i", tvec, p) * inv
        q = np.cross(tvec, self.e1)
        v = (q @ d) * inv
        t = np.einsum("ij,ij->i", self.e2, q) * inv
        good = ok & (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7) & (t > EPS)
        return np.sort(t[good])

    def first_hit(self, origin: Vec, direction: Vec) -> float | None:
        """Distance to the nearest hit along the ray, or None."""
        t = self.hits(origin, direction)
        return float(t[0]) if len(t) else None

    def is_inside(self, point: Vec) -> bool:
        """Parity test: an odd number of crossings along a fixed oblique ray means inside."""
        t = self.hits(point, PARITY_DIRECTION)
        if len(t) > 1:  # collapse hits that are one crossing seen through a shared edge
            t = t[np.r_[True, np.diff(t) > 1e-6]]
        return len(t) % 2 == 1
