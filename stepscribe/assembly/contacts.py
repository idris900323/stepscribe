# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Contacts between instances: planar faces, cylindrical fits, line/point, near misses (10.2)."""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np
from OCP.BRep import BRep_Builder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopoDS import TopoDS_Compound, TopoDS_Shape

from stepscribe import config
from stepscribe.assembly.tree import InstanceData, to_trsf
from stepscribe.features.bosses import _is_convex_full_cylinder
from stepscribe.features.holes import find_bore_faces
from stepscribe.geometry.occ_utils import (
    Vec,
    canonical_dir,
    interior_uv,
    outward_normal,
    surface_point,
    to_np,
    transform_shape,
    uv_bounds,
)
from stepscribe.geometry.part_geom import PartGeom
from stepscribe.geometry.properties import vec3
from stepscribe.geometry.surfaces import cylinder_params
from stepscribe.models.schema import Contact
from stepscribe.progress import Tracker

PLANE_NORMAL_TOL_DEG = 0.5
MAX_FACE_PAIRS = 12
MIN_CONTACT_AREA = 0.01  # mm2
LINE_TO_LINE_TOL = 0.01  # mm, diametral
CYL_COAXIAL_TOL = 0.1  # mm: axis lines closer than this are coaxial for fit purposes
MULTITHREAD = True  # OpenCASCADE's own thread pool inside one distance computation


@dataclass
class PlanarFace:
    """A planar face in the part's local frame."""

    index: int
    normal: Vec  # outward
    point: Vec
    area: float


@dataclass
class CylFace:
    """A full cylinder (convex shaft/boss or concave bore) in the part's local frame."""

    index: int
    origin: Vec
    direction: Vec  # canonical
    radius: float
    t0: float  # axial interval along ``direction`` from ``origin``
    t1: float
    convex: bool


@dataclass
class PartFaces:
    """Per-prototype face data used for contact tests (computed once per part)."""

    planes: list[PlanarFace] = field(default_factory=list)
    cylinders: list[CylFace] = field(default_factory=list)
    face_boxes: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2, 3))
    )  # local, conservative


def _local_face_boxes(geom: PartGeom) -> np.ndarray:
    """Per-face (min, max) boxes from the triangulation, inflated by the meshing error.

    Conservative: a face's true surface lies inside its box, so box tests never discard a
    face that could be the nearest one.
    """
    from stepscribe.render.tessellate import tessellate

    mesh = tessellate(geom, with_edges=False)
    n = len(geom.table.faces)
    boxes = np.zeros((n, 2, 3))
    boxes[:, 0] = np.inf
    boxes[:, 1] = -np.inf
    if len(mesh.triangles):
        for idx in np.unique(mesh.face_of_triangle):
            verts = mesh.vertices[np.unique(mesh.triangles[mesh.face_of_triangle == idx])]
            boxes[idx, 0], boxes[idx, 1] = verts.min(axis=0), verts.max(axis=0)
    missing = ~np.isfinite(boxes[:, 0, 0])
    if missing.any():  # faces without triangles: fall back to the part's bounds
        lo, hi = geom.bounds
        boxes[missing, 0], boxes[missing, 1] = lo, hi
    eps = config.MESH_DEFLECTION_REL * geom.diagonal + config.LINEAR_TOL
    boxes[:, 0] -= eps
    boxes[:, 1] += eps
    return boxes


def part_faces(geom: PartGeom) -> PartFaces:
    """Collect planar faces and full cylinders (convex and concave) of a part."""
    pf = PartFaces(face_boxes=_local_face_boxes(geom))
    concave = {b.idx for b in find_bore_faces(geom) if b.kind == "cylinder" and b.span_deg >= 190}
    for idx, fi in enumerate(geom.table.faces):
        if fi.kind == "plane":
            u, v = interior_uv(fi.face)
            n = outward_normal(fi.face, u, v)
            if n is not None:
                pf.planes.append(PlanarFace(idx, n, surface_point(fi.face, u, v), fi.area))
        elif fi.kind == "cylinder":
            is_concave = idx in concave
            if not is_concave and not _is_convex_full_cylinder(geom, idx):
                continue
            cp = cylinder_params(fi.face)
            d = canonical_dir(cp.direction)
            u0, u1, v0, v1 = uv_bounds(fi.face)
            sign = (
                1.0 if float(np.dot(cp.direction / np.linalg.norm(cp.direction), d)) > 0 else -1.0
            )
            ta, tb = sign * v0, sign * v1
            pf.cylinders.append(
                CylFace(idx, cp.origin, d, cp.radius, min(ta, tb), max(ta, tb), not is_concave)
            )
    return pf


def _overlap_area(a: InstanceData, b: InstanceData, fa: int, fb: int) -> float:
    face_a = transform_shape(a.ap.geom.table.faces[fa].face, to_trsf(a.matrix))
    face_b = transform_shape(b.ap.geom.table.faces[fb].face, to_trsf(b.matrix))
    common = BRepAlgoAPI_Common(face_a, face_b)
    if not common.IsDone():
        return 0.0
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(common.Shape(), props)
    return float(props.Mass())


def planar_contact(
    a: InstanceData, b: InstanceData, fa: PartFaces, fb: PartFaces
) -> tuple[float, Vec] | None:
    """(total overlap area, normal from A to B) of opposing coplanar faces, or None."""
    if not fa.planes or not fb.planes:
        return None
    na = np.array([a.rotation @ p.normal for p in fa.planes])
    pa = np.array([a.to_global_point(p.point) for p in fa.planes])
    nb = np.array([b.rotation @ p.normal for p in fb.planes])
    pb = np.array([b.to_global_point(p.point) for p in fb.planes])
    anti = na @ nb.T <= -math.cos(math.radians(PLANE_NORMAL_TOL_DEG))
    off = np.einsum("ik,ijk->ij", na, pb[None, :, :] - pa[:, None, :])
    close = np.abs(off) < config.CONTACT_TOL
    cand = np.argwhere(anti & close)
    if len(cand) == 0:
        return None
    pairs = sorted(cand.tolist(), key=lambda ij: -min(fa.planes[ij[0]].area, fb.planes[ij[1]].area))
    total = 0.0
    best: tuple[float, Vec] | None = None
    for i, j in pairs[:MAX_FACE_PAIRS]:
        area = _overlap_area(a, b, fa.planes[i].index, fb.planes[j].index)
        if area > MIN_CONTACT_AREA:
            total += area
            if best is None or area > best[0]:
                best = (area, na[i])
    if best is None:
        return None
    return total, best[1]


def _global_cyl(inst: InstanceData, c: CylFace) -> tuple[Vec, Vec]:
    return inst.to_global_point(c.origin), inst.to_global_dir(c.direction)


def cylindrical_fit(
    a: InstanceData, b: InstanceData, fa: PartFaces, fb: PartFaces
) -> tuple[str, float, InstanceData, InstanceData] | None:
    """(fit, diametral value, shaft instance, bore instance) for a coaxial convex/concave pair."""
    best: tuple[str, float, InstanceData, InstanceData] | None = None
    for shaft_inst, shaft_faces, bore_inst, bore_faces in ((a, fa, b, fb), (b, fb, a, fa)):
        for s in (c for c in shaft_faces.cylinders if c.convex):
            so, sd = _global_cyl(shaft_inst, s)
            for h in (c for c in bore_faces.cylinders if not c.convex):
                ho, hd = _global_cyl(bore_inst, h)
                if abs(abs(float(np.dot(sd, hd))) - 1.0) > 1e-4:
                    continue
                off = ho - so
                if float(np.linalg.norm(off - sd * np.dot(off, sd))) > CYL_COAXIAL_TOL:
                    continue
                s0, s1 = sorted((float(np.dot(so, sd)) + s.t0, float(np.dot(so, sd)) + s.t1))
                sign = 1.0 if float(np.dot(hd, sd)) > 0 else -1.0
                h0, h1 = sorted(
                    (float(np.dot(ho, sd)) + sign * h.t0, float(np.dot(ho, sd)) + sign * h.t1)
                )
                if min(s1, h1) - max(s0, h0) <= config.LINEAR_TOL:
                    continue
                value = 2 * (h.radius - s.radius)
                if abs(value) > 2 * config.NEAR_MISS_MAX:
                    continue
                fit = (
                    "line_to_line"
                    if abs(value) <= LINE_TO_LINE_TOL
                    else ("clearance" if value > 0 else "interference")
                )
                if best is None or abs(value) < abs(best[1]):
                    best = (fit, value, shaft_inst, bore_inst)
    return best


def detect_contacts(instances: list[InstanceData], tracker: Tracker | None = None) -> list[Contact]:
    """All contacts and near misses between instance pairs."""
    if len(instances) < 2:
        return []
    cache = {i.part_id: part_faces(i.ap.geom) for i in instances}
    lo = np.array([i.bounds[0] for i in instances]) - config.NEAR_MISS_MAX
    hi = np.array([i.bounds[1] for i in instances]) + config.NEAR_MISS_MAX
    candidates = [
        (i, j)
        for i in range(len(instances))
        for j in range(i + 1, len(instances))
        if not (np.any(lo[i] > hi[j]) or np.any(lo[j] > hi[i]))
    ]
    if tracker is not None:
        tracker.plan_pairs(len(candidates))
    raw: list[tuple[str, str, Contact]] = []
    for i, j in candidates:
        started = time.time()
        raw.extend(
            (instances[i].id, instances[j].id, c)
            for c in _pair_contacts(instances[i], instances[j], cache)
        )
        if tracker is not None:
            tracker.pair_done(time.time() - started)
    raw.sort(key=lambda t: (t[0], t[1], t[2].kind))
    contacts = []
    for n, (_a, _b, c) in enumerate(raw, 1):
        contacts.append(c.model_copy(update={"id": f"C{n:03d}"}))
    return contacts


def _global_boxes(inst: InstanceData, pf: PartFaces) -> np.ndarray:
    """Face boxes in the assembly frame (corner transform, conservative for rotations)."""
    lo, hi = pf.face_boxes[:, 0], pf.face_boxes[:, 1]
    corners = np.stack(
        [np.where(np.array(c), hi, lo) for c in itertools.product([False, True], repeat=3)], axis=1
    )  # (n, 8, 3)
    g = corners @ inst.rotation.T + inst.translation
    return np.stack([g.min(axis=1), g.max(axis=1)], axis=1)


def _overlapping(boxes: np.ndarray, lo: np.ndarray, hi: np.ndarray, margin: float) -> np.ndarray:
    return np.flatnonzero(
        np.all(boxes[:, 0] <= hi + margin, axis=1) & np.all(boxes[:, 1] >= lo - margin, axis=1)
    )


def _contains(outer: InstanceData, inner: InstanceData) -> bool:
    return bool(
        np.all(inner.bounds[0] >= outer.bounds[0] - 1e-6)
        and np.all(inner.bounds[1] <= outer.bounds[1] + 1e-6)
    )


def _candidate_shapes(
    a: InstanceData, b: InstanceData, fa: PartFaces, fb: PartFaces
) -> tuple[TopoDS_Shape, TopoDS_Shape] | None:
    """Shapes to measure the distance between, or None when the parts are farther than 1 mm.

    Exact for distances below ``NEAR_MISS_MAX``: the closest pair of faces must lie within that
    distance of the other part's bounds, so faces outside it cannot matter. Nested bounds keep
    the full solids (a solid inside another has distance 0 via the inner-solution test).
    """
    if _contains(a, b) or _contains(b, a):
        return a.shape, b.shape
    box_a, box_b = _global_boxes(a, fa), _global_boxes(b, fb)
    m = config.NEAR_MISS_MAX
    ia = _overlapping(box_a, b.bounds[0], b.bounds[1], m)
    for _ in range(2):  # tighten: B against the surviving A faces, then A against those B faces
        if len(ia) == 0:
            return None
        ib = _overlapping(box_b, box_a[ia, 0].min(axis=0), box_a[ia, 1].max(axis=0), m)
        if len(ib) == 0:
            return None
        ia = _overlapping(box_a, box_b[ib, 0].min(axis=0), box_b[ib, 1].max(axis=0), m)
    if len(ia) == 0:
        return None
    return _sub_shape(a, ia), _sub_shape(b, ib)


def _sub_shape(inst: InstanceData, indices: np.ndarray) -> TopoDS_Shape:
    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)
    for i in indices:
        builder.Add(comp, inst.ap.geom.table.faces[int(i)].face)
    if np.allclose(inst.matrix, np.eye(4)):
        return comp
    return transform_shape(comp, to_trsf(inst.matrix))


def _line_or_point(dss: BRepExtrema_DistShapeShape) -> str:
    """'line' if the closest points spread over more than the contact tolerance, else 'point'.

    Geometric and deterministic: it does not depend on how many extrema solutions OCCT reports,
    which varies with how much geometry is passed in.
    """
    pts = np.array([to_np(dss.PointOnShape1(i)) for i in range(1, dss.NbSolution() + 1)])
    if len(pts) < 2:
        return "point"
    spread = float(np.max(np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)))
    return "line" if spread > config.CONTACT_TOL else "point"


def _pair_contacts(a: InstanceData, b: InstanceData, cache: dict[str, PartFaces]) -> list[Contact]:
    shapes = _candidate_shapes(a, b, cache[a.part_id], cache[b.part_id])
    if shapes is None:
        return []
    dss = BRepExtrema_DistShapeShape(*shapes)
    dss.SetMultiThread(MULTITHREAD)
    dss.Perform()
    if not dss.IsDone():
        return []
    dist = float(dss.Value())
    if dist >= config.NEAR_MISS_MAX:
        return []
    fa, fb = cache[a.part_id], cache[b.part_id]
    found: list[Contact] = []
    touching = dist < config.CONTACT_TOL
    if touching:
        planar = planar_contact(a, b, fa, fb)
        if planar is not None:
            area, normal = planar
            found.append(
                Contact(
                    id="",
                    instance_a=a.id,
                    instance_b=b.id,
                    kind="planar_face",
                    min_distance_mm=dist,
                    contact_area_mm2=area,
                    normal=vec3(normal),
                )
            )
    cyl = cylindrical_fit(a, b, fa, fb)
    if cyl is not None:
        fit, value, shaft, bore = cyl
        # normalise orientation: instance_a is the shaft side, instance_b the bore side
        found.append(
            Contact(
                id="",
                instance_a=shaft.id,
                instance_b=bore.id,
                kind="cylindrical_fit",
                min_distance_mm=dist,
                fit=fit,
                fit_value_mm=value,
            )
        )
    if not found:
        if not touching:
            found.append(
                Contact(
                    id="", instance_a=a.id, instance_b=b.id, kind="near_miss", min_distance_mm=dist
                )
            )
        else:
            kind = _line_or_point(dss)
            found.append(
                Contact(id="", instance_a=a.id, instance_b=b.id, kind=kind, min_distance_mm=dist)
            )
    return found


__all__ = ["detect_contacts", "to_np"]
