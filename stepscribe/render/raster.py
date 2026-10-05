# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""VTK offscreen shaded rendering: white background, black edges, fixed lights."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from functools import cache

import numpy as np
import vtk
from PIL import Image
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

from stepscribe.geometry.occ_utils import Vec
from stepscribe.render.scene import CameraSpec, SceneItem

DEFAULT_SIZE = (1600, 1200)
EDGE_WIDTH = 1.5
DEPTH_TOL = 2e-3
DIM_COLOR = (0.82, 0.82, 0.82)
HIGHLIGHT_COLOR = (0.95, 0.25, 0.15)


@dataclass
class Rendered:
    """A rendered image plus what is needed to project and test visibility of 3D points."""

    image: Image.Image
    zbuffer: np.ndarray  # (h, w) depth in [0, 1], row 0 = top
    matrix: np.ndarray  # 4x4 world -> clip
    view_size: tuple[int, int] | None = None  # size the matrix was rendered for

    def __post_init__(self) -> None:
        if self.view_size is None:
            self.view_size = self.image.size

    @property
    def size(self) -> tuple[int, int]:
        """Pixel size of the rendered view (margins added later do not change it)."""
        assert self.view_size is not None
        return self.view_size

    def project(self, p: Vec) -> tuple[float, float, float]:
        """World point -> (x pixel, y pixel from the top, depth in [0, 1])."""
        w, h = self.size
        clip = self.matrix @ np.array([p[0], p[1], p[2], 1.0])
        ndc = clip[:3] / clip[3]
        return (ndc[0] + 1) / 2 * w, (1 - (ndc[1] + 1) / 2) * h, (ndc[2] + 1) / 2

    def project_many(self, pts: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`project`; returns an (N, 3) array."""
        w, h = self.size
        homog = np.c_[pts, np.ones(len(pts))] @ self.matrix.T
        ndc = homog[:, :3] / homog[:, 3:4]
        result: np.ndarray = np.c_[
            (ndc[:, 0] + 1) / 2 * w, (1 - (ndc[:, 1] + 1) / 2) * h, (ndc[:, 2] + 1) / 2
        ]
        return result

    def visible(self, x: float, y: float, depth: float) -> bool:
        """True if nothing nearer than *depth* covers pixel (x, y) (3x3 neighbourhood)."""
        w, h = self.size
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            return False
        patch = self.zbuffer[max(0, yi - 1) : yi + 2, max(0, xi - 1) : xi + 2]
        return bool(depth <= float(patch.min()) + DEPTH_TOL)

    def visible_mask(self, proj: np.ndarray) -> np.ndarray:
        """Boolean visibility for an (N, 3) array from :meth:`project_many`."""
        w, h = self.size
        xi = np.clip(np.round(proj[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(proj[:, 1]).astype(int), 0, h - 1)
        inside = (proj[:, 0] >= 0) & (proj[:, 0] < w) & (proj[:, 1] >= 0) & (proj[:, 1] < h)
        mask: np.ndarray = inside & (proj[:, 2] <= self.zbuffer[yi, xi] + DEPTH_TOL)
        return mask


def _polydata(points: np.ndarray, cells: np.ndarray, n_cells: int, lines: bool) -> vtk.vtkPolyData:
    pd = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True))
    pd.SetPoints(pts)
    ca = vtk.vtkCellArray()
    ca.ImportLegacyFormat(numpy_to_vtkIdTypeArray(cells.astype(np.int64), deep=True))
    pd.SetLines(ca) if lines else pd.SetPolys(ca)
    return pd


def _surface_actor(item: SceneItem, color: tuple[float, float, float]) -> vtk.vtkActor | None:
    tris = item.mesh.triangles
    if len(tris) == 0:
        return None
    cells = np.hstack([np.full((len(tris), 1), 3), tris]).ravel()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(_polydata(item.world_vertices(), cells, len(tris), False))
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetAmbient(0.35)
    prop.SetDiffuse(0.7)
    prop.SetSpecular(0.0)
    prop.SetInterpolationToFlat()
    return actor


def _edge_actor(item: SceneItem) -> vtk.vtkActor | None:
    if not item.mesh.edges:
        return None
    pts, cells, start = [], [], 0
    rot, trans = item.matrix[:3, :3], item.matrix[:3, 3] + item.offset
    for line in item.mesh.edges:
        pts.append(line @ rot.T + trans)
        cells.append(np.r_[len(line), np.arange(start, start + len(line))])
        start += len(line)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(_polydata(np.vstack(pts), np.concatenate(cells), len(cells), True))
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(0, 0, 0)
    prop.SetLineWidth(EDGE_WIDTH)
    prop.LightingOff()
    return actor


class RenderUnavailable(RuntimeError):
    """This machine cannot render images (no usable OpenGL, for example a bare virtual machine)."""


def _smoke() -> None:
    """Render one empty frame; run in a child process because a broken driver crashes the process."""
    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.SetSize(8, 8)
    win.AddRenderer(vtk.vtkRenderer())
    win.Render()
    win.Finalize()


@cache
def rendering_available() -> bool:
    """True when VTK can draw off-screen here; checked once per process in a child process."""
    try:
        done = subprocess.run(
            [sys.executable, "-c", "from stepscribe.render.raster import _smoke; _smoke()"],
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def render_scene(
    items: list[SceneItem],
    camera: CameraSpec,
    size: tuple[int, int] = DEFAULT_SIZE,
    highlight: set[str] | None = None,
) -> Rendered:
    """Render *items* with an orthographic camera; deterministic (fixed lights, no multisampling)."""
    if not rendering_available():
        raise RenderUnavailable("off-screen rendering does not work on this machine")
    w, h = size
    vtk.vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
    ren = vtk.vtkRenderer()
    ren.SetBackground(1.0, 1.0, 1.0)
    ren.AutomaticLightCreationOff()
    ren.RemoveAllLights()
    key = vtk.vtkLight()
    key.SetLightTypeToHeadlight()
    key.SetIntensity(0.75)
    fill = vtk.vtkLight()
    fill.SetLightTypeToCameraLight()
    fill.SetPosition(-1.0, 1.0, 1.0)
    fill.SetFocalPoint(0, 0, 0)
    fill.SetIntensity(0.45)
    ren.AddLight(key)
    ren.AddLight(fill)
    for item in items:
        color = item.color
        if highlight is not None:
            color = (
                HIGHLIGHT_COLOR if item.key in highlight or item.part_id in highlight else DIM_COLOR
            )
        for actor in (_surface_actor(item, color), _edge_actor(item)):
            if actor is not None:
                ren.AddActor(actor)
    cam = ren.GetActiveCamera()
    cam.ParallelProjectionOn()
    cam.SetPosition(*camera.position)
    cam.SetFocalPoint(*camera.focal)
    cam.SetViewUp(*camera.view_up)
    cam.SetParallelScale(camera.scale)
    ren.ResetCameraClippingRange()
    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.SetMultiSamples(0)
    win.SetAlphaBitPlanes(0)
    win.SetSize(w, h)
    win.AddRenderer(ren)
    win.Render()
    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(win)
    grab.SetInputBufferTypeToRGB()
    grab.ReadFrontBufferOff()
    grab.Update()
    rgb = vtk_to_numpy(grab.GetOutput().GetPointData().GetScalars()).reshape(h, w, 3)[::-1]
    z = vtk.vtkFloatArray()
    win.GetZbufferData(0, 0, w - 1, h - 1, z)
    zbuf = vtk_to_numpy(z).reshape(h, w)[::-1]
    m = cam.GetCompositeProjectionTransformMatrix(w / h, -1, 1)
    matrix = np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])
    win.Finalize()
    return Rendered(Image.fromarray(np.ascontiguousarray(rgb)), np.array(zbuf, copy=True), matrix)
