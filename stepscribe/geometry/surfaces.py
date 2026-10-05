# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Surface type names and parameters for faces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.TopoDS import TopoDS_Face

from stepscribe.geometry.occ_utils import Vec, face_surface, to_np

_NAMES = {
    "GeomAbs_Plane": "plane",
    "GeomAbs_Cylinder": "cylinder",
    "GeomAbs_Cone": "cone",
    "GeomAbs_Sphere": "sphere",
    "GeomAbs_Torus": "torus",
    "GeomAbs_BezierSurface": "bezier",
    "GeomAbs_BSplineSurface": "bspline",
    "GeomAbs_SurfaceOfRevolution": "revolution",
    "GeomAbs_SurfaceOfExtrusion": "extrusion",
    "GeomAbs_OffsetSurface": "offset",
    "GeomAbs_OtherSurface": "other",
}


def surface_kind(face: TopoDS_Face) -> str:
    """Lower-case surface type name of *face* (``plane``, ``cylinder``, ...)."""
    t = face_surface(face).GetType()
    return _NAMES.get(t.name, "other")


@dataclass(frozen=True)
class CylinderParams:
    """Cylinder surface: axis line + radius."""

    origin: Vec
    direction: Vec  # surface axis direction (not canonicalised)
    radius: float


@dataclass(frozen=True)
class ConeParams:
    """Cone surface: axis line, radius at the reference plane, half angle (rad)."""

    origin: Vec
    direction: Vec
    ref_radius: float
    semi_angle: float
    apex: Vec


def cylinder_params(face: TopoDS_Face) -> CylinderParams:
    """Parameters of a cylindrical face."""
    cyl = BRepAdaptor_Surface(face).Cylinder()
    ax = cyl.Axis()
    return CylinderParams(to_np(ax.Location()), to_np(ax.Direction()), float(cyl.Radius()))


def cone_params(face: TopoDS_Face) -> ConeParams:
    """Parameters of a conical face."""
    cone = BRepAdaptor_Surface(face).Cone()
    ax = cone.Axis()
    return ConeParams(
        to_np(ax.Location()),
        to_np(ax.Direction()),
        float(cone.RefRadius()),
        float(cone.SemiAngle()),
        to_np(cone.Apex()),
    )


def plane_normal_origin(face: TopoDS_Face) -> tuple[Vec, Vec]:
    """(unit normal, point) of a planar face; normal is the plane's own axis (not flipped)."""
    pl = BRepAdaptor_Surface(face).Plane()
    n = to_np(pl.Axis().Direction())
    return n / np.linalg.norm(n), to_np(pl.Location())
