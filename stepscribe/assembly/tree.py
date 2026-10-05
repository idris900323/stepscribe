# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Assembly tree: instances with global transforms, bounding boxes, subassemblies."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from OCP.gp import gp_Trsf
from OCP.TopoDS import TopoDS_Shape

from stepscribe.geometry.occ_utils import Vec, bbox_of, transform_shape
from stepscribe.geometry.properties import make_bbox, vec3
from stepscribe.models.schema import BBox, Instance, Subassembly

if TYPE_CHECKING:
    from stepscribe.analysis import AnalyzedPart


@dataclass
class InstanceData:
    """One placed part: its analysis, transform and (lazily built) global geometry."""

    id: str
    ap: AnalyzedPart
    matrix: np.ndarray  # 4x4 rigid transform local -> global
    path: str
    parent_path: str | None
    subassembly: str | None
    _shape: TopoDS_Shape | None = field(default=None, repr=False)
    _bbox: tuple[Vec, Vec] | None = field(default=None, repr=False)

    @property
    def part_id(self) -> str:
        return self.ap.part.id

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def rotation(self) -> np.ndarray:
        r: np.ndarray = self.matrix[:3, :3]
        return r

    @property
    def translation(self) -> Vec:
        t: Vec = self.matrix[:3, 3]
        return t

    @property
    def shape(self) -> TopoDS_Shape:
        """Global-frame copy of the part's shape (built on first use)."""
        if self._shape is None:
            self._shape = (
                self.ap.geom.shape
                if np.allclose(self.matrix, np.eye(4))
                else transform_shape(self.ap.geom.shape, to_trsf(self.matrix))
            )
        return self._shape

    @property
    def bounds(self) -> tuple[Vec, Vec]:
        """Global axis-aligned bounds (min, max)."""
        if self._bbox is None:
            self._bbox = _global_bounds(self)
        return self._bbox

    def to_global_point(self, p: Vec) -> Vec:
        """Local point -> global point."""
        out: Vec = self.rotation @ p + self.translation
        return out

    def to_global_dir(self, d: Vec) -> Vec:
        """Local direction -> global direction."""
        out: Vec = self.rotation @ d
        return out


def to_trsf(m: np.ndarray) -> gp_Trsf:
    """4x4 numpy matrix -> gp_Trsf."""
    t = gp_Trsf()
    t.SetValues(*(float(m[r, c]) for r in range(3) for c in range(4)))
    return t


def _is_axis_aligned(rot: np.ndarray) -> bool:
    return bool(np.all(np.isclose(np.abs(rot).max(axis=1), 1.0, atol=1e-9)))


def _global_bounds(inst: InstanceData) -> tuple[Vec, Vec]:
    """Exact bounds: corner transform when the rotation is a 90-degree permutation, else the shape."""
    lo_l, hi_l = inst.ap.geom.bounds
    if _is_axis_aligned(inst.rotation):
        corners = np.array(
            [
                [x, y, z]
                for x in (lo_l[0], hi_l[0])
                for y in (lo_l[1], hi_l[1])
                for z in (lo_l[2], hi_l[2])
            ]
        )
        g = corners @ inst.rotation.T + inst.translation
        return g.min(axis=0), g.max(axis=0)
    return bbox_of(inst.shape)


def rotation_axis_angle(rot: np.ndarray) -> tuple[Vec, float]:
    """Axis (unit) and angle in degrees of a rotation matrix; identity gives (+Z, 0)."""
    angle = math.acos(max(-1.0, min(1.0, (float(np.trace(rot)) - 1.0) / 2.0)))
    if angle < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if abs(angle - math.pi) < 1e-6:
        w, v = np.linalg.eigh((rot + np.eye(3)) / 2.0)
        axis = v[:, int(np.argmax(w))]
    else:
        axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
        axis = axis / (2.0 * math.sin(angle))
    axis = axis / np.linalg.norm(axis)
    return axis, math.degrees(angle)


def to_schema_instance(inst: InstanceData, parent_id: str | None) -> Instance:
    """Schema ``Instance`` for an :class:`InstanceData`."""
    axis, angle = rotation_axis_angle(inst.rotation)
    lo, hi = inst.bounds
    return Instance(
        id=inst.id,
        part_id=inst.part_id,
        parent_instance_id=parent_id,
        path=inst.path,
        position=vec3(inst.translation),
        rotation_matrix=[[float(v) for v in row] for row in inst.rotation],
        rotation_axis_angle=(vec3(axis), angle),
        global_bbox=make_bbox(lo, hi),
        subassembly=inst.subassembly,
    )


def union_bounds(instances: list[InstanceData]) -> BBox:
    """Union of global bounds."""
    lo = np.min([i.bounds[0] for i in instances], axis=0)
    hi = np.max([i.bounds[1] for i in instances], axis=0)
    return make_bbox(lo, hi)


def build_subassemblies(instances: list[InstanceData]) -> list[Subassembly]:
    """First-level subassemblies with bbox, mass and a one-line summary."""
    groups: dict[str, list[InstanceData]] = {}
    for inst in instances:
        if inst.subassembly:
            groups.setdefault(inst.subassembly, []).append(inst)
    out: list[Subassembly] = []
    for name in sorted(groups):
        members = groups[name]
        masses = [m.ap.part.mass.mass_g for m in members]
        mass = (
            sum(x for x in masses if x is not None) if all(x is not None for x in masses) else None
        )
        counts: dict[str, int] = {}
        for m in members:
            counts[m.ap.part.name] = counts.get(m.ap.part.name, 0) + 1
        parts = ", ".join(f"{n} × {k}" if n > 1 else k for k, n in sorted(counts.items()))
        out.append(
            Subassembly(
                name=name,
                instance_ids=[m.id for m in members],
                bbox=union_bounds(members),
                mass_g=mass,
                summary=f"{len(members)} instance{'s' if len(members) != 1 else ''}: {parts}.",
            )
        )
    return out
