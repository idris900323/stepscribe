# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Part symmetry and mirror-pair parts.

Sample points on the surface are mirrored (or rotated) and every image must land on the
surface within a tolerance. Deterministic: fixed seed, fixed candidate order.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

from stepscribe.geometry.occ_utils import Vec, to_np
from stepscribe.geometry.part_geom import PartGeom
from stepscribe.geometry.pointdist import SurfaceGrid, sample_surface
from stepscribe.models.schema import Axis, SymmetryInfo, Vec3
from stepscribe.render.tessellate import tessellate

SYM_TOL = 0.05  # mm (raised to 0.2 % of the diagonal for large parts: mesh error)
SAMPLES = 300
MAX_TRIANGLES = 120_000
MAX_FOLD = 24
EIGEN_TOL = 0.005  # relative: "equal" principal moments
DEDUPE_COS = math.cos(math.radians(1.0))


def _v3(v: Vec) -> Vec3:
    return Vec3(x=float(v[0]), y=float(v[1]), z=float(v[2]))


def inertia_frame(geom: PartGeom) -> tuple[Vec, np.ndarray, np.ndarray]:
    """(centroid, principal moments ascending, axes as columns) of the solid (density 1)."""
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(geom.shape, props)
    m = props.MatrixOfInertia()
    mat = np.array([[m.Value(i, j) for j in range(1, 4)] for i in range(1, 4)])
    w, vecs = np.linalg.eigh(mat)
    return to_np(props.CentreOfMass()), w, vecs


class SymmetryTester:
    """Mesh, sample points and surface grid for one part."""

    def __init__(self, geom: PartGeom, seed: int = 12345) -> None:
        self.geom = geom
        mesh = tessellate(geom, with_edges=False)
        self.ok = 0 < len(mesh.triangles) <= MAX_TRIANGLES
        self.tol = max(SYM_TOL, 0.002 * geom.diagonal)
        if not self.ok:
            return
        self.pts = sample_surface(mesh.vertices, mesh.triangles, SAMPLES, seed)
        # shuffle once so early exit finds a failure quickly (fixed seed)
        order = np.random.RandomState(seed + 1).permutation(len(self.pts))
        self.pts = self.pts[order]
        self.grid = SurfaceGrid(mesh.vertices, mesh.triangles, self.tol)
        self.centroid, self.moments, self.axes = inertia_frame(geom)

    def mirror_ok(self, point: Vec, normal: Vec) -> tuple[bool, float]:
        n = normal / np.linalg.norm(normal)
        d = (self.pts - point) @ n
        return self.grid.all_within(self.pts - 2.0 * d[:, None] * n)

    def rotation_ok(self, point: Vec, axis: Vec, fold: int) -> tuple[bool, float]:
        a = axis / np.linalg.norm(axis)
        th = 2.0 * math.pi / fold
        rel = self.pts - point
        par = (rel @ a)[:, None] * a
        perp = rel - par
        rot = par + perp * math.cos(th) + np.cross(a, perp) * math.sin(th)
        return self.grid.all_within(point + rot)


def _unique_dirs(dirs: list[Vec]) -> list[Vec]:
    out: list[Vec] = []
    for d in dirs:
        d = d / np.linalg.norm(d)
        if not any(abs(float(np.dot(d, o))) > DEDUPE_COS for o in out):
            out.append(d)
    return out


def part_symmetry(geom: PartGeom, obb_axes: list[Vec] | None = None) -> SymmetryInfo:
    """Mirror planes and rotational symmetry of a single part."""
    st = SymmetryTester(geom)
    if not st.ok:
        return SymmetryInfo(description="symmetry not checked (mesh too large or empty)")
    cands = [st.axes[:, i] for i in range(3)] + list(obb_axes or [])
    planes: list[dict[str, object]] = []
    for n in _unique_dirs(cands):
        ok, dev = st.mirror_ok(st.centroid, n)
        if ok:
            planes.append(
                {
                    "point": _v3(st.centroid).model_dump(),
                    "normal": _v3(n).model_dump(),
                    "max_dev_mm": round(dev, 4),
                }
            )
    rots: list[dict[str, object]] = []
    for ax in _unique_dirs([st.axes[:, k] for k in range(3)] + list(obb_axes or [])):
        others = [float(v) for v in st.moments]
        # test fold 2 always; folds >= 3 need two equal moments about perpendicular axes
        eq = _two_equal(others)
        best, best_dev = 0, 0.0
        # highest fold first: the first that passes is the answer (one expensive pass at most)
        for fold in ([*range(MAX_FOLD, 2, -1)] if eq else []) + [2]:
            ok, dev = st.rotation_ok(st.centroid, ax, fold)
            if ok:
                best, best_dev = fold, dev
                break
        if best:
            rots.append(
                {
                    "axis": Axis(origin=_v3(st.centroid), direction=_v3(ax)).model_dump(),
                    "fold": best,
                    "max_dev_mm": round(best_dev, 4),
                }
            )
    rots.sort(key=lambda r: (-int(r["fold"]), str(r["axis"])))  # type: ignore[call-overload]
    return SymmetryInfo(mirror_planes=planes, rotational=rots, description=_describe(planes, rots))


def _two_equal(m: list[float]) -> bool:
    return any(abs(m[a] - m[b]) <= EIGEN_TOL * max(m[a], m[b]) for a, b in ((0, 1), (1, 2), (0, 2)))


def _describe(planes: list[dict[str, object]], rots: list[dict[str, object]]) -> str:
    bits = []
    if planes:
        bits.append(f"{len(planes)} mirror plane(s)")
    if rots:
        bits.append(f"{rots[0]['fold']}-fold rotational symmetry")
    return "symmetric: " + " and ".join(bits) if bits else "no mirror or rotational symmetry found"


# ---------- mirror-pair parts ----------


def _signed_permutations() -> list[np.ndarray]:
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = np.zeros((3, 3))
            for r, (c, s) in enumerate(zip(perm, signs, strict=True)):
                m[r, c] = s
            mats.append(m)
    return mats


_SIGNED = _signed_permutations()
_SIMILAR_REL = 1e-3


def similar_invariants(ga: PartGeom, gb: PartGeom) -> bool:
    """Equal volume, area and principal moments (0.1 %): a necessary condition."""
    (_ca, wa, _), (_cb, wb, _) = inertia_frame(ga), inertia_frame(gb)
    props_a, props_b = GProp_GProps(), GProp_GProps()
    BRepGProp.VolumeProperties_s(ga.shape, props_a)
    BRepGProp.VolumeProperties_s(gb.shape, props_b)
    va, vb = props_a.Mass(), props_b.Mass()
    if abs(va - vb) > _SIMILAR_REL * max(va, vb):
        return False
    sa, sb = GProp_GProps(), GProp_GProps()
    BRepGProp.SurfaceProperties_s(ga.shape, sa)
    BRepGProp.SurfaceProperties_s(gb.shape, sb)
    if abs(sa.Mass() - sb.Mass()) > _SIMILAR_REL * max(sa.Mass(), sb.Mass()):
        return False
    return bool(np.all(np.abs(wa - wb) <= _SIMILAR_REL * np.maximum(wa, wb)))


def relation(ga: PartGeom, gb: PartGeom) -> str | None:
    """'mirror' (only reflections align A and B), 'same' (a rotation aligns them) or None."""
    if not similar_invariants(ga, gb):
        return None
    ta, tb = SymmetryTester(ga), SymmetryTester(gb)
    if not (ta.ok and tb.ok):
        return None
    ra = tb.pts - tb.centroid
    proper = improper = False
    for m in _SIGNED:
        # map B's principal frame onto A's: p_a = Va * M * Vb^T * (p_b - c_b) + c_a
        r = ta.axes @ m @ tb.axes.T
        pts = ra @ r.T + ta.centroid
        # B's sampled points transformed must lie on A's surface within tol
        ok, _dev = ta.grid.all_within(pts)
        if ok:
            if np.linalg.det(r) > 0:
                proper = True
                break
            improper = True
    if proper:
        return "same"
    return "mirror" if improper else None


def find_mirror_pairs(
    geoms: dict[str, PartGeom], hashes: dict[str, str]
) -> dict[str, tuple[str, str]]:
    """part_id -> (other_part_id, 'mirror' | 'same') for parts that match another part."""
    ids = sorted(geoms)
    out: dict[str, tuple[str, str]] = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if hashes.get(a) == hashes.get(b):
                continue  # identical content: already one unique part
            rel = relation(geoms[a], geoms[b])
            if rel:
                out.setdefault(a, (b, rel))
                out.setdefault(b, (a, rel))
    return out
