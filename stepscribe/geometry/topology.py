# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Topology: stable face table, adjacency graph and counts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.OCP.collections import (
    IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher as EdgeFaceMap,
)
from OCP.OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as ShapeMap
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID, TopAbs_VERTEX
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS_Face, TopoDS_Shape

from stepscribe.geometry.occ_utils import as_face, to_np, unique_subshapes
from stepscribe.geometry.surfaces import surface_kind
from stepscribe.models.schema import TopologySummary


@dataclass
class FaceInfo:
    """One face with its stable ID and cached facts."""

    id: str
    face: TopoDS_Face
    kind: str
    area: float
    centroid: np.ndarray
    neighbors: set[int] = field(default_factory=set)  # indices into FaceTable.faces


@dataclass
class FaceTable:
    """All faces of a part, sorted by (kind, rounded centroid, area): Rule 6.4."""

    faces: list[FaceInfo]
    _map: ShapeMap

    def index_of(self, face: TopoDS_Face) -> int:
        """Table index of a face (via IsSame); -1 if unknown."""
        return int(self._map.FindIndex(face)) - 1

    def by_id(self, face_id: str) -> FaceInfo:
        """Lookup by ID such as ``F0003``."""
        return self.faces[int(face_id[1:]) - 1]


def _face_facts(face: TopoDS_Face) -> tuple[float, np.ndarray]:
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    return float(props.Mass()), to_np(props.CentreOfMass())


def build_face_table(shape: TopoDS_Shape) -> FaceTable:
    """Collect faces, assign stable IDs and compute face adjacency."""
    raw = [as_face(f) for f in unique_subshapes(shape, TopAbs_FACE)]
    rows = []
    for f in raw:
        area, cen = _face_facts(f)
        key = (surface_kind(f), tuple(round(float(c), 3) for c in cen), round(area, 3))
        rows.append((key, f, area, cen))
    rows.sort(key=lambda r: r[0])
    ordered = [
        FaceInfo(f"F{i + 1:04d}", f, key[0], area, cen)
        for i, (key, f, area, cen) in enumerate(rows)
    ]
    fmap = ShapeMap()
    for fi in ordered:
        fmap.Add(fi.face)
    table = FaceTable(ordered, fmap)
    _fill_adjacency(shape, table)
    return table


def _fill_adjacency(shape: TopoDS_Shape, table: FaceTable) -> None:
    emap = EdgeFaceMap()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, emap)
    for i in range(1, emap.Extent() + 1):
        idx = sorted({table.index_of(as_face(f)) for f in emap.FindFromIndex(i)})
        idx = [j for j in idx if j >= 0]
        for a in idx:
            table.faces[a].neighbors.update(j for j in idx if j != a)


def topology_summary(shape: TopoDS_Shape, table: FaceTable) -> TopologySummary:
    """Counts of solids/shells/faces/edges/vertices and a face-type histogram."""
    types: dict[str, int] = {}
    for fi in table.faces:
        types[fi.kind] = types.get(fi.kind, 0) + 1
    return TopologySummary(
        solids=len(unique_subshapes(shape, TopAbs_SOLID)),
        shells=len(unique_subshapes(shape, TopAbs_SHELL)),
        faces=len(table.faces),
        edges=len(unique_subshapes(shape, TopAbs_EDGE)),
        vertices=len(unique_subshapes(shape, TopAbs_VERTEX)),
        face_types=dict(sorted(types.items())),
    )
