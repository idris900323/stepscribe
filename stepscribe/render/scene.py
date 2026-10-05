# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Scene description: items, standard cameras, exploded view."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stepscribe.assembly.spatial import frame
from stepscribe.geometry.occ_utils import Vec
from stepscribe.render.tessellate import Mesh

PALETTE = [  # fixed qualitative palette (Okabe-Ito based), indexed by part number
    (0.00, 0.45, 0.70),
    (0.90, 0.62, 0.00),
    (0.00, 0.62, 0.45),
    (0.80, 0.47, 0.65),
    (0.84, 0.37, 0.00),
    (0.34, 0.71, 0.91),
    (0.94, 0.89, 0.26),
    (0.55, 0.55, 0.55),
]
FIT_MARGIN = 1.32  # leaves room around the model for dimension lines and labels
EXPLODE_FACTOR = 0.6
GROUP_EXPLODE = 0.5
VIEWS = ("iso", "front", "top", "right", "exploded")


@dataclass
class SceneItem:
    """One drawable body: a local-frame mesh placed by a rigid transform."""

    key: str  # instance ID (assembly) or part ID (single file)
    part_id: str
    name: str
    mesh: Mesh
    matrix: np.ndarray  # 4x4
    color: tuple[float, float, float]
    group: str | None = None  # subassembly name, for hierarchical explode
    offset: Vec = field(default_factory=lambda: np.zeros(3))

    def world_vertices(self) -> np.ndarray:
        """Vertices in the assembly frame (including any explode offset)."""
        out: np.ndarray = (
            self.mesh.vertices @ self.matrix[:3, :3].T + self.matrix[:3, 3] + self.offset
        )
        return out

    def bounds(self) -> tuple[Vec, Vec]:
        v = self.world_vertices()
        return v.min(axis=0), v.max(axis=0)

    def to_world(self, p: Vec) -> Vec:
        """Local point -> world point (with offset)."""
        out: Vec = self.matrix[:3, :3] @ p + self.matrix[:3, 3] + self.offset
        return out


def part_color(part_id: str, step_color: list[float] | None) -> tuple[float, float, float]:
    """STEP color if present, else a fixed palette entry by part number."""
    if step_color:
        return (step_color[0], step_color[1], step_color[2])
    n = int(part_id[3:]) if part_id[3:].isdigit() else 0
    return PALETTE[n % len(PALETTE)]


def scene_bounds(items: list[SceneItem]) -> tuple[Vec, Vec]:
    """Union of item bounds."""
    lo = np.min([i.bounds()[0] for i in items], axis=0)
    hi = np.max([i.bounds()[1] for i in items], axis=0)
    return lo, hi


@dataclass
class CameraSpec:
    """Orthographic camera."""

    position: Vec
    focal: Vec
    view_up: Vec
    scale: float  # half of the visible height (VTK parallel scale)


def standard_camera(
    view: str, items: list[SceneItem], up: str = "+Z", front: str = "-Y", aspect: float = 4 / 3
) -> CameraSpec:
    """iso / front / top / right camera fitted to the scene bounds."""
    lo, hi = scene_bounds(items)
    centre = 0.5 * (lo + hi)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    u, right, f = frame(up, front)
    if view == "front":
        direction, view_up = f, u
    elif view == "right":
        direction, view_up = right, u
    elif view == "top":
        direction, view_up = u, -f
    else:  # iso: from front, right and above
        direction, view_up = f + right + u, u
    direction = direction / np.linalg.norm(direction)
    return fit_camera(centre, direction, view_up, (lo, hi), diag, aspect)


def fit_camera(
    centre: Vec, direction: Vec, view_up: Vec, bounds: tuple[Vec, Vec], diag: float, aspect: float
) -> CameraSpec:
    """Camera looking along ``-direction`` toward *centre*, scaled to contain the bounds."""
    lo, hi = bounds
    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )
    cam_right = np.cross(view_up, direction)
    cam_right = cam_right / np.linalg.norm(cam_right)
    cam_up = np.cross(direction, cam_right)
    half_w = float(np.max(np.abs((corners - centre) @ cam_right)))
    half_h = float(np.max(np.abs((corners - centre) @ cam_up)))
    scale = max(half_h, half_w / aspect) * FIT_MARGIN or 1.0
    return CameraSpec(centre + direction * diag * 2.0, centre, cam_up, scale)


def apply_explode(items: list[SceneItem]) -> None:
    """Hierarchical exploded view: groups move away from the assembly centre, items within a
    group move away from their group's centre. Offsets are written into each item."""
    for it in items:
        it.offset = np.zeros(3)
    lo, hi = scene_bounds(items)
    centre = 0.5 * (lo + hi)
    diag = float(np.linalg.norm(hi - lo))
    groups: dict[str, list[SceneItem]] = {}
    for it in items:
        groups.setdefault(it.group or f"#{it.key}", []).append(it)
    gcentres = {g: _centre(members) for g, members in groups.items()}
    dist = {g: float(np.linalg.norm(c - centre)) for g, c in gcentres.items()}
    dmax = max(dist.values()) or 1.0
    for g, members in groups.items():
        vec = gcentres[g] - centre
        direction = (
            vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 1e-9 else np.array([0.0, 0.0, 1.0])
        )
        goff = direction * EXPLODE_FACTOR * 0.5 * diag * (dist[g] / dmax)
        glo = np.min([m.bounds()[0] for m in members], axis=0)
        ghi = np.max([m.bounds()[1] for m in members], axis=0)
        gdiag = float(np.linalg.norm(ghi - glo))
        local = {m.key: float(np.linalg.norm(_centre([m]) - gcentres[g])) for m in members}
        lmax = max(local.values()) or 1.0
        for m in members:
            vec_i = _centre([m]) - gcentres[g]
            n = float(np.linalg.norm(vec_i))
            step = (
                vec_i / n * GROUP_EXPLODE * 0.3 * gdiag * (local[m.key] / lmax)
                if n > 1e-9
                else np.zeros(3)
            )
            m.offset = goff + step


def _centre(members: list[SceneItem]) -> Vec:
    lo = np.min([m.bounds()[0] for m in members], axis=0)
    hi = np.max([m.bounds()[1] for m in members], axis=0)
    return 0.5 * (lo + hi)
