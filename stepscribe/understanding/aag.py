# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Attributed face adjacency graph.

Nodes are the part's faces (stable face IDs from the face table); edges are the B-Rep edges
shared by two different faces, classified as ``convex``, ``concave`` or ``smooth`` (tangent).
A small internal structure, no graph library.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.GCPnts import GCPnts_AbscissaPoint
from OCP.gp import gp_Pnt2d, gp_Vec
from OCP.OCP.collections import (
    IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher as EdgeFaceMap,
)
from OCP.ShapeAnalysis import ShapeAnalysis_Surface
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_IN
from OCP.TopExp import TopExp

from stepscribe.geometry.occ_utils import (
    Vec,
    angle_between,
    as_edge,
    as_face,
    outward_normal,
    to_np,
    to_pnt,
)
from stepscribe.geometry.part_geom import PartGeom
from stepscribe.geometry.surfaces import cylinder_params, plane_normal_origin
from stepscribe.models.schema import AAGSummary

SMOOTH_TOL_DEG = 1.0  # normals closer than this across an edge are tangent
EPS_MM = 0.01  # step into the face when testing which side is material
EDGE_KINDS = ("convex", "concave", "smooth")


@dataclass
class AAGEdge:
    """One shared B-Rep edge between face indices ``a`` and ``b`` (indices into the face table)."""

    a: int
    b: int
    kind: str  # convex | concave | smooth | unknown
    angle_deg: float  # angle between the two outward normals
    dihedral_deg: float  # interior angle of the material across the edge (180 for smooth)
    length: float
    midpoint: Vec
    tangent: Vec

    def other(self, i: int) -> int:
        return self.b if i == self.a else self.a


@dataclass
class FaceAttr:
    """Cached attributes of a face used by the structural rules."""

    index: int
    id: str
    kind: str
    area: float
    normal: Vec | None = None  # planes: solid-outward unit normal
    offset: float | None = None  # planes: dot(normal, point on plane)
    axis_dir: Vec | None = None  # cylinders: unit axis
    axis_point: Vec | None = None
    radius: float | None = None
    convex: bool | None = None  # curved faces: material is inside the surface
    feature: str | None = None  # id of a recognised feature that owns the face


@dataclass
class AAG:
    """The graph: face attributes, edge list and adjacency."""

    geom: PartGeom
    attrs: list[FaceAttr]
    edges: list[AAGEdge]
    adj: dict[int, list[AAGEdge]] = field(default_factory=dict)

    def neighbors(self, i: int) -> list[int]:
        """Sorted unique neighbour indices of face *i*."""
        return sorted({e.other(i) for e in self.adj.get(i, [])})

    def edges_of_type(self, i: int, kind: str) -> list[AAGEdge]:
        return [e for e in self.adj.get(i, []) if e.kind == kind]

    def connected_components(
        self,
        face_ok: Callable[[FaceAttr], bool] = lambda _a: True,
        edge_ok: Callable[[AAGEdge], bool] = lambda _e: True,
    ) -> list[list[int]]:
        """Components of the subgraph of faces passing *face_ok* joined by edges passing *edge_ok*."""
        seen: set[int] = set()
        out: list[list[int]] = []
        for start in range(len(self.attrs)):
            if start in seen or not face_ok(self.attrs[start]):
                continue
            comp, stack = [], [start]
            seen.add(start)
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for e in self.adj.get(cur, []):
                    nxt = e.other(cur)
                    if nxt not in seen and edge_ok(e) and face_ok(self.attrs[nxt]):
                        seen.add(nxt)
                        stack.append(nxt)
            out.append(sorted(comp))
        return out

    @cached_property
    def counts(self) -> dict[str, int]:
        c = dict.fromkeys(EDGE_KINDS, 0)
        for e in self.edges:
            if e.kind in c:
                c[e.kind] += 1
        return c

    def summary(self, base_body: str | None) -> AAGSummary:
        return AAGSummary(
            nodes=len(self.attrs),
            edges=len(self.edges),
            edge_types=dict(self.counts),
            base_body=base_body,
        )

    def planes(self) -> list[FaceAttr]:
        return [a for a in self.attrs if a.kind == "plane" and a.normal is not None]


def _face_attr(geom: PartGeom, idx: int) -> FaceAttr:
    fi = geom.table.faces[idx]
    attr = FaceAttr(idx, fi.id, fi.kind, fi.area)
    from stepscribe.geometry.occ_utils import interior_uv, surface_point

    try:
        u, v = interior_uv(fi.face)
        n = outward_normal(fi.face, u, v)
        p = surface_point(fi.face, u, v)
    except Exception:  # noqa: BLE001 - degenerate faces keep their bare attributes
        return attr
    if fi.kind == "plane":
        if n is None:
            _n, _o = plane_normal_origin(fi.face)
            n = _n
        n = n / np.linalg.norm(n)
        attr.normal, attr.offset = n, float(np.dot(n, p))
    elif fi.kind == "cylinder":
        cp = cylinder_params(fi.face)
        d = cp.direction / np.linalg.norm(cp.direction)
        attr.axis_dir, attr.axis_point, attr.radius = d, cp.origin, cp.radius
        if n is not None:
            centre = cp.origin + d * float(np.dot(p - cp.origin, d))
            attr.convex = float(np.dot(n, p - centre)) > 0
    return attr


def _into_face_direction(  # type: ignore[no-untyped-def]
    face, p: Vec, t: Vec, projectors: dict[int, ShapeAnalysis_Surface] | None = None
) -> Vec | None:
    """Whichever of +t / -t from *p* enters the (trimmed) face, or None.

    ``projectors`` caches the surface projector per face: building one for a free-form face is
    slow and a face with hundreds of edges would otherwise build it hundreds of times.
    """
    if projectors is None:
        surf = ShapeAnalysis_Surface(BRep_Tool.Surface_s(face))
    else:
        surf = projectors.get(id(face))
        if surf is None:
            surf = projectors[id(face)] = ShapeAnalysis_Surface(BRep_Tool.Surface_s(face))
    for sign in (1.0, -1.0):
        q = p + sign * EPS_MM * t
        uv = surf.ValueOfUV(to_pnt(q), 1e-4)
        if BRepClass_FaceClassifier(face, gp_Pnt2d(uv.X(), uv.Y()), 1e-7).State() == TopAbs_IN:
            return sign * t
    return None


def _classify(  # type: ignore[no-untyped-def]
    f1, f2, edge, projectors: dict[int, ShapeAnalysis_Surface] | None = None
) -> tuple[str, float, float, Vec, Vec, float]:
    c = BRepAdaptor_Curve(edge)
    t_mid = 0.5 * (c.FirstParameter() + c.LastParameter())
    p = to_np(c.Value(t_mid))
    d1 = gp_Vec()
    from OCP.gp import gp_Pnt

    pnt = gp_Pnt()
    c.D1(t_mid, pnt, d1)
    tan = to_np(d1)
    nt = float(np.linalg.norm(tan))
    length = float(GCPnts_AbscissaPoint.Length_s(c))
    tan = tan / nt if nt > 1e-12 else np.array([1.0, 0.0, 0.0])
    from stepscribe.geometry.occ_utils import normal_at_point

    n1, n2 = normal_at_point(f1, p), normal_at_point(f2, p)
    if n1 is None or n2 is None:
        return "unknown", 0.0, 180.0, p, tan, length
    ang = angle_between(n1, n2)
    if ang < SMOOTH_TOL_DEG:
        return "smooth", ang, 180.0, p, tan, length
    t1 = np.cross(tan, n1)
    nrm = float(np.linalg.norm(t1))
    if nrm < 1e-9:
        return "unknown", ang, 180.0, p, tan, length
    into = _into_face_direction(f1, p, t1 / nrm, projectors)
    if into is None:
        return "unknown", ang, 180.0, p, tan, length
    if float(np.dot(n2, into)) > 0:
        return "concave", ang, 180.0 + ang, p, tan, length
    return "convex", ang, 180.0 - ang, p, tan, length


def build_aag(geom: PartGeom, feature_faces: dict[str, str] | None = None) -> AAG:
    """Build the graph for *geom*; *feature_faces* maps face ID -> owning feature ID."""
    table = geom.table
    attrs = [_face_attr(geom, i) for i in range(len(table.faces))]
    for a in attrs:
        if feature_faces and a.id in feature_faces:
            a.feature = feature_faces[a.id]
    emap = EdgeFaceMap()
    TopExp.MapShapesAndAncestors_s(geom.shape, TopAbs_EDGE, TopAbs_FACE, emap)
    edges: list[AAGEdge] = []
    projectors: dict[int, ShapeAnalysis_Surface] = {}
    for k in range(1, emap.Extent() + 1):
        edge = as_edge(emap.FindKey(k))
        faces = list(emap.FindFromIndex(k))
        idx = []
        for f in faces:
            j = table.index_of(as_face(f))
            if j >= 0 and j not in idx:
                idx.append(j)
        if len(idx) != 2:  # seam edge (same face twice) or non-manifold: skip
            continue
        ia, ib = sorted(idx)
        f1, f2 = table.faces[ia].face, table.faces[ib].face
        try:
            kind, ang, dih, mid, tan, ln = _classify(f1, f2, edge, projectors)
        except Exception:  # noqa: BLE001
            continue
        edges.append(AAGEdge(ia, ib, kind, ang, dih, ln, mid, tan))
    edges.sort(
        key=lambda e: (
            e.a,
            e.b,
            round(float(e.midpoint[0]), 4),
            round(float(e.midpoint[1]), 4),
            round(float(e.midpoint[2]), 4),
        )
    )
    adj: dict[int, list[AAGEdge]] = {}
    for e in edges:
        adj.setdefault(e.a, []).append(e)
        adj.setdefault(e.b, []).append(e)
    return AAG(geom, attrs, edges, adj)


def aag_to_dict(aag: AAG) -> dict[str, object]:
    """JSON-friendly dump of the graph (written only with ``--debug-graphs``)."""

    def vec(v: Vec | None) -> list[float] | None:
        return None if v is None else [round(float(x), 4) for x in v]

    return {
        "nodes": [
            {
                "id": a.id,
                "kind": a.kind,
                "area": round(a.area, 4),
                "normal": vec(a.normal),
                "offset": None if a.offset is None else round(a.offset, 4),
                "axis": vec(a.axis_dir),
                "radius": None if a.radius is None else round(a.radius, 4),
                "convex": a.convex,
                "feature": a.feature,
            }
            for a in aag.attrs
        ],
        "edges": [
            {
                "a": aag.attrs[e.a].id,
                "b": aag.attrs[e.b].id,
                "kind": e.kind,
                "dihedral_deg": round(e.dihedral_deg, 2),
                "length": round(e.length, 4),
            }
            for e in aag.edges
        ],
    }


def plane_key(a: FaceAttr, nd: int = 2) -> tuple[float, ...]:
    """Hashable identity of a plane (outward normal and offset)."""
    assert a.normal is not None and a.offset is not None
    return (*(round(float(c), nd) + 0.0 for c in a.normal), round(float(a.offset), nd) + 0.0)


def parallel(n1: Vec, n2: Vec, tol: float = 0.02) -> bool:
    return abs(abs(float(np.dot(n1, n2))) - 1.0) < tol


def perpendicular(n1: Vec, n2: Vec, tol: float = 0.05) -> bool:
    return abs(float(np.dot(n1, n2))) < tol


def angle_deg(n1: Vec, n2: Vec) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(n1, n2))))))


def face_vertices(geom: PartGeom, idx: int) -> np.ndarray:
    """Vertices of face *idx* as an (n, 3) array."""
    from OCP.BRep import BRep_Tool as BT
    from OCP.TopAbs import TopAbs_VERTEX

    from stepscribe.geometry.occ_utils import as_vertex, unique_subshapes

    pts = [
        to_np(BT.Pnt_s(as_vertex(v)))
        for v in unique_subshapes(geom.table.faces[idx].face, TopAbs_VERTEX)
    ]
    return np.array(pts) if pts else np.zeros((0, 3))
