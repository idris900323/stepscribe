# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Mass properties, bounding boxes and oriented bounding boxes."""

from __future__ import annotations

import numpy as np
from OCP.Bnd import Bnd_OBB
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GProp import GProp_GProps
from OCP.TopoDS import TopoDS_Shape

from stepscribe import config
from stepscribe.config import MATERIAL_KEYWORDS
from stepscribe.geometry.occ_utils import bbox_of, to_np
from stepscribe.models.schema import BBox, MassProperties, OrientedBBox, Vec3


def vec3(v: np.ndarray) -> Vec3:
    """numpy vector -> schema Vec3."""
    return Vec3(x=float(v[0]), y=float(v[1]), z=float(v[2]))


def make_bbox(lo: np.ndarray, hi: np.ndarray) -> BBox:
    """Schema BBox from min/max corners."""
    return BBox(min=vec3(lo), max=vec3(hi), size=vec3(hi - lo))


def aabb(shape: TopoDS_Shape) -> BBox:
    """Exact axis-aligned bounding box (a 10 mm cube reads exactly 10.0)."""
    lo, hi = bbox_of(shape)
    return make_bbox(lo, hi)


def obb(shape: TopoDS_Shape) -> OrientedBBox:
    """Oriented bounding box via OCCT; ``size_sorted`` is descending (a >= b >= c).

    OCCT only gives a tight OBB from a triangulation (without one a rotated plate reads 72 mm
    instead of 60 mm), so the shape is meshed finely first.
    """
    lo, hi = bbox_of(shape)
    deflection = max(
        float(np.linalg.norm(hi - lo)) * config.OBB_MESH_DEFLECTION_REL,
        config.OBB_MESH_DEFLECTION_MIN,
    )
    BRepMesh_IncrementalMesh(shape, deflection, False, config.MESH_ANGULAR_DEFLECTION, False)
    box = Bnd_OBB()
    BRepBndLib.AddOBB_s(shape, box, True, True, False)
    half = np.array([box.XHSize(), box.YHSize(), box.ZHSize()], dtype=float)
    axes = [to_np(box.XDirection()), to_np(box.YDirection()), to_np(box.ZDirection())]
    return OrientedBBox(
        center=vec3(to_np(box.Center())),
        axes=[vec3(a) for a in axes],
        half_sizes=vec3(half),
        size_sorted=sorted((float(2 * h) for h in half), reverse=True),
    )


def guess_density(name: str) -> tuple[float, str] | None:
    """Density (g/cm3) guess from a part name keyword, or None."""
    low = name.lower()
    for key in sorted(MATERIAL_KEYWORDS):
        if key in low:
            return MATERIAL_KEYWORDS[key], key
    return None


def mass_properties(
    shape: TopoDS_Shape,
    name: str = "",
    density: float | None = None,
    material: str | None = None,
) -> MassProperties:
    """Volume, area, centroid; mass/inertia only if a density is known."""
    vol = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, vol)
    area = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, area)
    centroid = to_np(vol.CentreOfMass())
    source = "none"
    if density is not None:
        source = "user"
    else:
        guess = guess_density(name)
        if guess:
            density, material, source = guess[0], guess[1], "name_guess"
    volume = float(vol.Mass())
    mass = None
    inertia = None
    if density is not None:
        mass = volume * density / 1000.0  # mm3 * g/cm3 -> g
        m = vol.MatrixOfInertia()
        scale = density / 1000.0
        inertia = [[float(m.Value(r, c)) * scale for c in (1, 2, 3)] for r in (1, 2, 3)]
    return MassProperties(
        volume_mm3=volume,
        surface_area_mm2=float(area.Mass()),
        centroid=vec3(centroid),
        density_g_cm3=density,
        mass_g=mass,
        material_assumed=material,
        material_source=source,
        inertia_tensor_g_mm2=inertia,
    )
