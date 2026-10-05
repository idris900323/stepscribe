# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Planar cross-sections of a solid."""

from __future__ import annotations

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.BRepGProp import BRepGProp
from OCP.gp import gp_Ax3, gp_Dir, gp_Pln
from OCP.GProp import GProp_GProps
from OCP.TopoDS import TopoDS_Shape

from stepscribe.geometry.occ_utils import Vec, to_pnt


def section_shape(
    shape: TopoDS_Shape, origin: Vec, normal: Vec, extent: float
) -> TopoDS_Shape | None:
    """The part of the plane through *origin* with *normal* that lies inside *shape*.

    ``extent`` is the half-size of the cutting square; use at least the part's diagonal.
    """
    n = normal / np.linalg.norm(normal)
    plane = gp_Pln(gp_Ax3(to_pnt(origin), gp_Dir(float(n[0]), float(n[1]), float(n[2]))))
    face = BRepBuilderAPI_MakeFace(plane, -extent, extent, -extent, extent).Face()
    common = BRepAlgoAPI_Common(shape, face)
    if not common.IsDone():
        return None
    result: TopoDS_Shape = common.Shape()
    return result


def section_area(shape: TopoDS_Shape, origin: Vec, normal: Vec, extent: float) -> float:
    """Area of the planar cross-section (0.0 if the plane misses the solid)."""
    sec = section_shape(shape, origin, normal, extent)
    if sec is None or sec.IsNull():
        return 0.0
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(sec, props)
    return float(props.Mass())
