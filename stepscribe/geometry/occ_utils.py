# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Thin wrappers around OCP. Every OCP name used elsewhere is verified here (Rule 1).

OCCT 8 note: ``TopoDS`` downcasts are ``TopoDS.Face(shape)`` (no ``_s`` suffix) in the
cadquery-ocp 8.x build, while 7.x builds use ``TopoDS.Face_s``. ``_downcast`` handles both.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import OCP
from OCP.Bnd import Bnd_Box
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.gp import gp_Pnt, gp_Trsf, gp_Vec
from OCP.OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as ShapeMap
from OCP.TopAbs import TopAbs_IN, TopAbs_OUT, TopAbs_REVERSED, TopAbs_ShapeEnum
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS, TopoDS_Face, TopoDS_Shape

Vec = np.ndarray  # shape (3,), float64


def occt_version() -> str:
    """OCCT version as reported by the OCP wheel (``OCP.__version__`` tracks OCCT)."""
    return str(OCP.__version__)


def _downcast(kind: str, shape: TopoDS_Shape) -> Any:
    fn = getattr(TopoDS, kind + "_s", None) or getattr(TopoDS, kind)
    return fn(shape)


def as_face(shape: TopoDS_Shape) -> TopoDS_Face:
    """Downcast a generic shape to a face."""
    return _downcast("Face", shape)


def as_edge(shape: TopoDS_Shape) -> Any:
    """Downcast a generic shape to an edge."""
    return _downcast("Edge", shape)


def as_vertex(shape: TopoDS_Shape) -> Any:
    """Downcast a generic shape to a vertex."""
    return _downcast("Vertex", shape)


def as_wire(shape: TopoDS_Shape) -> Any:
    """Downcast a generic shape to a wire."""
    return _downcast("Wire", shape)


def unique_subshapes(shape: TopoDS_Shape, kind: TopAbs_ShapeEnum) -> list[TopoDS_Shape]:
    """Unique sub-shapes (by ``IsSame``) in deterministic traversal order."""
    m = ShapeMap()
    TopExp.MapShapes_s(shape, kind, m)
    return [m.FindKey(i) for i in range(1, m.Extent() + 1)]


def to_np(p: gp_Pnt | gp_Vec | Any) -> Vec:
    """gp_Pnt / gp_Vec / gp_Dir -> numpy vector."""
    return np.array([p.X(), p.Y(), p.Z()], dtype=float)


def to_pnt(v: Vec) -> gp_Pnt:
    """numpy vector -> gp_Pnt."""
    return gp_Pnt(float(v[0]), float(v[1]), float(v[2]))


def canonical_dir(d: Vec) -> Vec:
    """Rule 6.3: largest-magnitude component positive; ties prefer +Z, +Y, +X."""
    d = d / np.linalg.norm(d)
    mags = np.abs(d)
    mx = mags.max()
    for idx in (2, 1, 0):  # tie preference Z, Y, X
        if mags[idx] >= mx - 1e-9:
            return d if d[idx] >= 0 else -d
    return d  # pragma: no cover


def face_surface(face: TopoDS_Face) -> BRepAdaptor_Surface:
    """Adaptor for a face's surface (placement transforms already applied)."""
    return BRepAdaptor_Surface(face)


def face_is_reversed(face: TopoDS_Face) -> bool:
    """True if the face orientation flips its surface normal."""
    return bool(face.Orientation() == TopAbs_REVERSED)


def outward_normal(face: TopoDS_Face, u: float, v: float) -> Vec | None:
    """Solid-outward normal at (u, v): geometric normal flipped for REVERSED faces (#1 bug)."""
    props = BRepLProp_SLProps(face_surface(face), u, v, 1, 1e-6)
    if not props.IsNormalDefined():
        return None
    n = to_np(props.Normal())
    return -n if face_is_reversed(face) else n


def surface_point(face: TopoDS_Face, u: float, v: float) -> Vec:
    """3D point at surface parameters (u, v)."""
    return to_np(face_surface(face).Value(u, v))


class SolidClassifier:
    """Reusable point-in-solid classifier."""

    def __init__(self, shape: TopoDS_Shape, tol: float = 1e-6) -> None:
        self._c = BRepClass3d_SolidClassifier(shape)
        self._tol = tol

    def state(self, p: Vec) -> str:
        """Return ``'in'``, ``'out'`` or ``'on'`` for point *p*."""
        self._c.Perform(to_pnt(p), self._tol)
        s = self._c.State()
        if s == TopAbs_IN:
            return "in"
        if s == TopAbs_OUT:
            return "out"
        return "on"


def bbox_of(shape: TopoDS_Shape) -> tuple[Vec, Vec]:
    """Tight axis-aligned bounding box (min, max), no tolerance gap."""
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    return to_np(box.CornerMin()), to_np(box.CornerMax())


def make_vertex(p: Vec) -> TopoDS_Shape:
    """Vertex shape at point *p*."""
    return BRepBuilderAPI_MakeVertex(to_pnt(p)).Vertex()


def transform_shape(shape: TopoDS_Shape, trsf: gp_Trsf) -> TopoDS_Shape:
    """Apply a rigid transform to a shape (copy)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform

    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def angle_between(a: Vec, b: Vec) -> float:
    """Unsigned angle between vectors, degrees."""
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def normal_at_point(face: TopoDS_Face, p: Vec) -> Vec | None:
    """Solid-outward normal of *face* at the surface point closest to *p*."""
    from OCP.BRep import BRep_Tool
    from OCP.ShapeAnalysis import ShapeAnalysis_Surface

    surf = BRep_Tool.Surface_s(face)
    uv = ShapeAnalysis_Surface(surf).ValueOfUV(to_pnt(p), 1e-4)
    return outward_normal(face, uv.X(), uv.Y())


def edges_of(shape: TopoDS_Shape) -> list[TopoDS_Shape]:
    """Unique edges of a shape."""
    from OCP.TopAbs import TopAbs_EDGE

    return unique_subshapes(shape, TopAbs_EDGE)


def edge_midpoint_and_length(edge: TopoDS_Shape) -> tuple[Vec, float]:
    """Point at the middle parameter of an edge and its length."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_AbscissaPoint

    c = BRepAdaptor_Curve(as_edge(edge))
    mid = to_np(c.Value(0.5 * (c.FirstParameter() + c.LastParameter())))
    return mid, float(GCPnts_AbscissaPoint.Length_s(c))


def shared_edges(a: TopoDS_Shape, b: TopoDS_Shape) -> list[TopoDS_Shape]:
    """Edges present in both shapes (by ``IsSame``)."""
    from OCP.TopAbs import TopAbs_EDGE

    in_b = ShapeMap()  # hashed lookup: a big plane has hundreds of edges and many neighbours
    TopExp.MapShapes_s(b, TopAbs_EDGE, in_b)
    return [e for e in edges_of(a) if in_b.Contains(e)]


def uv_bounds(face: TopoDS_Face) -> tuple[float, float, float, float]:
    """(umin, umax, vmin, vmax) of a face."""
    from OCP.BRepTools import BRepTools

    u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
    return float(u0), float(u1), float(v0), float(v1)


def interior_uv(face: TopoDS_Face) -> tuple[float, float]:
    """A (u, v) that lies on the trimmed face: the middle of the UV box if inside, else a grid hit."""
    from OCP.BRepClass import BRepClass_FaceClassifier
    from OCP.gp import gp_Pnt2d

    u0, u1, v0, v1 = uv_bounds(face)
    candidates = [(0.5, 0.5)] + [(i / 6, j / 6) for i in range(1, 6) for j in range(1, 6)]
    for fu, fv in candidates:
        u, v = u0 + fu * (u1 - u0), v0 + fv * (v1 - v0)
        if BRepClass_FaceClassifier(face, gp_Pnt2d(u, v), 1e-6).State() == TopAbs_IN:
            return u, v
    return 0.5 * (u0 + u1), 0.5 * (v0 + v1)
