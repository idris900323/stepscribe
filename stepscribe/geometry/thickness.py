# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Sampled minimum wall thickness. A sampled estimate, never exact."""

from __future__ import annotations

import numpy as np
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.gp import gp_Pnt2d
from OCP.TopAbs import TopAbs_IN

from stepscribe import config
from stepscribe.geometry.occ_utils import outward_normal, surface_point, uv_bounds
from stepscribe.geometry.part_geom import PartGeom

_RAY_EPS = 1e-4
_MAX_ATTEMPTS = 8
WALL_MIN_SUPPORT = 3  # samples that must agree on the reported minimum wall


def _face_samples(
    part: PartGeom, idx: int, count: int, rng: np.random.RandomState
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(point, outward normal) samples inside a face, using rejection sampling in UV."""
    face = part.table.faces[idx].face
    u0, u1, v0, v1 = uv_bounds(face)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(count * _MAX_ATTEMPTS):
        if len(out) >= count:
            break
        u, v = rng.uniform(u0, u1), rng.uniform(v0, v1)
        if BRepClass_FaceClassifier(face, gp_Pnt2d(u, v), 1e-6).State() != TopAbs_IN:
            continue
        n = outward_normal(face, u, v)
        if n is not None:
            out.append((surface_point(face, u, v), n))
    return out


def sample_wall_thickness(
    part: PartGeom, n_samples: int = config.WALL_SAMPLES
) -> tuple[float, float] | None:
    """(minimum, 5th percentile) of ray-cast distances through the material, or None."""
    table = part.table
    mesh = part.trimesh
    cand = [i for i, f in enumerate(table.faces) if f.kind in ("plane", "cylinder") and f.area > 0]
    if not cand or mesh is None:
        return None
    areas = np.array([table.faces[i].area for i in cand])
    weights = areas / areas.sum()
    rng = np.random.RandomState(config.WALL_SEED)
    per_face = np.maximum(1, np.round(weights * n_samples)).astype(int)
    dists: list[float] = []
    for i, cnt in zip(cand, per_face, strict=True):
        for p, n in _face_samples(part, i, int(cnt), rng):
            t = mesh.first_hit(p - n * _RAY_EPS, -n)
            if t is not None:
                dists.append(t + _RAY_EPS)
    if not dists:
        return None
    # Hits closer than the smallest feature size come from rays leaving a face next to a
    # fillet or edge and grazing the neighbouring (tessellated) face; they are not walls.
    arr = np.array([d for d in dists if d >= config.MIN_FEATURE_SIZE])
    if arr.size == 0:
        return None
    # A real thin wall is hit by several samples; one stray ray near a fillet or a seam is not a
    # wall. Report the third-smallest hit (the smallest when there are fewer than three).
    k = min(WALL_MIN_SUPPORT - 1, arr.size - 1)
    return float(np.partition(arr, k)[k]), float(np.percentile(arr, config.PERCENTILE_WALL))
