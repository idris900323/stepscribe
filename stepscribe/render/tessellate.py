# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""B-Rep -> triangles with face IDs, plus B-Rep edges as polylines."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GCPnts import GCPnts_QuasiUniformDeflection
from OCP.TopLoc import TopLoc_Location

from stepscribe import config
from stepscribe.geometry.occ_utils import as_edge, edges_of, face_is_reversed, to_np
from stepscribe.geometry.part_geom import PartGeom


@dataclass
class Mesh:
    """Triangle mesh in the part's local frame."""

    vertices: np.ndarray  # (N, 3)
    triangles: np.ndarray  # (M, 3) int
    face_of_triangle: np.ndarray  # (M,) index into the part's face table
    edges: list[np.ndarray] = field(default_factory=list)  # polylines, each (K, 3)


def _face_triangles(geom: PartGeom, idx: int) -> tuple[np.ndarray, np.ndarray] | None:
    fi = geom.table.faces[idx]
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(fi.face, loc)
    if tri is None:
        return None
    trsf = loc.Transformation()
    pts = np.array([to_np(tri.Node(i).Transformed(trsf)) for i in range(1, tri.NbNodes() + 1)])
    rev = face_is_reversed(fi.face)
    tris = []
    for k in range(1, tri.NbTriangles() + 1):
        a, b, c = tri.Triangle(k).Get()
        tris.append((a - 1, c - 1, b - 1) if rev else (a - 1, b - 1, c - 1))
    return pts, np.array(tris, dtype=int).reshape(-1, 3)


def _edge_polylines(geom: PartGeom, deflection: float) -> list[np.ndarray]:
    lines: list[np.ndarray] = []
    for e in edges_of(geom.shape):
        curve = BRepAdaptor_Curve(as_edge(e))
        sampler = GCPnts_QuasiUniformDeflection(curve, deflection)
        if sampler.IsDone() and sampler.NbPoints() >= 2:
            pts = [to_np(sampler.Value(i)) for i in range(1, sampler.NbPoints() + 1)]
        else:
            pts = [
                to_np(curve.Value(curve.FirstParameter())),
                to_np(curve.Value(curve.LastParameter())),
            ]
        lines.append(np.array(pts))
    return lines


def tessellate(geom: PartGeom, with_edges: bool = True) -> Mesh:
    """Mesh a part at 0.1 % of its bounding-box diagonal (angular 0.3 rad)."""
    deflection = max(geom.diagonal * config.MESH_DEFLECTION_REL, 1e-4)
    BRepMesh_IncrementalMesh(geom.shape, deflection, False, config.MESH_ANGULAR_DEFLECTION, True)
    verts: list[np.ndarray] = []
    tris: list[np.ndarray] = []
    owner: list[np.ndarray] = []
    offset = 0
    for idx in range(len(geom.table.faces)):
        res = _face_triangles(geom, idx)
        if res is None or len(res[1]) == 0:
            continue
        pts, t = res
        verts.append(pts)
        tris.append(t + offset)
        owner.append(np.full(len(t), idx, dtype=int))
        offset += len(pts)
    if not verts:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=int), np.zeros(0, dtype=int))
    return Mesh(
        np.vstack(verts),
        np.vstack(tris),
        np.concatenate(owner),
        _edge_polylines(geom, deflection) if with_edges else [],
    )
