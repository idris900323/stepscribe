# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Per-part analysis: ties readers, geometry and feature detectors into ``Part`` models."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopoDS import TopoDS_Shape

from stepscribe.features.bosses import detect_bosses
from stepscribe.features.cutouts import detect_cutouts
from stepscribe.features.fillets_chamfers import detect_chamfers, detect_fillets
from stepscribe.features.holes import detect_holes
from stepscribe.features.patterns import detect_patterns
from stepscribe.features.shape_class import classify_shape
from stepscribe.features.slots_pockets import detect_pockets, detect_slots
from stepscribe.geometry.occ_utils import unique_subshapes
from stepscribe.geometry.part_geom import PartGeom
from stepscribe.geometry.properties import aabb, guess_density, mass_properties, obb
from stepscribe.geometry.thickness import sample_wall_thickness
from stepscribe.geometry.topology import topology_summary
from stepscribe.geometry.validity import is_valid_solid
from stepscribe.models.schema import (
    Boss,
    Chamfer,
    Cutout,
    CutoutPattern,
    Fillet,
    Hole,
    HolePattern,
    Part,
    Pocket,
    SemanticTag,
    ShapeClass,
    Slot,
)

T = TypeVar("T")


@dataclass
class AnalyzeOptions:
    """User-facing options shared by the API, CLI and MCP server."""

    density: float | None = None
    material: str | None = None
    up: str = "+Z"
    front: str = "-Y"
    no_timestamp: bool = False
    check_interference: bool = False
    skip_wall_thickness: bool = False
    understanding: bool = True
    check_motion: bool = False
    motion_budget_s: int = 120
    debug_graphs: bool = False
    precision: int = 4
    extra: dict[str, Any] = field(default_factory=dict)


def split_bodies(name: str, shape: TopoDS_Shape) -> list[tuple[str, TopoDS_Shape]]:
    """One part per solid; several solids become ``<name>_body<n>`` (stable order)."""
    solids = unique_subshapes(shape, TopAbs_SOLID)
    if len(solids) <= 1:
        return [(name, solids[0] if solids else shape)]
    keyed = []
    for s in solids:
        bb = aabb(s)
        keyed.append(
            ((round(bb.min.x, 3), round(bb.min.y, 3), round(bb.min.z, 3), round(bb.size.x, 3)), s)
        )
    keyed.sort(key=lambda kv: kv[0])
    return [(f"{name}_body{i}", s) for i, (_k, s) in enumerate(keyed, 1)]


def _guarded(label: str, default: T, warnings: list[str], fn: Callable[[], T]) -> T:
    """Rule 11: per-entity errors become warnings, never tracebacks."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"{label} failed: {type(exc).__name__}: {exc}")
        return default


def content_hash(part: Part) -> str:
    """Hash of rounded volume, area, OBB sizes, face-type counts and sorted hole diameters."""
    parts = [
        f"{part.mass.volume_mm3:.2f}",
        f"{part.mass.surface_area_mm2:.2f}",
        *(f"{s:.2f}" for s in part.obb.size_sorted),
        *(f"{k}={v}" for k, v in sorted(part.topology.face_types.items())),
        *(f"{d:.3f}" for d in sorted(h.diameter_mm for h in part.holes)),
    ]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]  # noqa: S324


@dataclass
class AnalyzedPart:
    """A ``Part`` plus the live geometry that produced it (needed for assembly analysis)."""

    part: Part
    geom: PartGeom


def analyze_geom(
    geom: PartGeom,
    source_file: str,
    opts: AnalyzeOptions,
    color: list[float] | None = None,
    semantics: Callable[[PartGeom, Part], list[SemanticTag]] | None = None,
) -> Part:
    """Run every analysis step on one solid. Failures become warnings on the part."""
    shape = geom.shape
    w: list[str] = []
    solid = is_valid_solid(shape)
    if not solid:
        w.append("shape is not a valid closed solid; results may be unreliable")
    material = opts.extra.get("part_materials", {}).get(geom.name) or opts.material
    density = opts.density
    if density is None and material:
        guess = guess_density(material)
        if guess is not None:
            density = guess[0]
        else:
            w.append(f"material '{material}' is not a known keyword; mass left unknown")
    mass = mass_properties(shape, geom.name, density, material)
    topo = topology_summary(shape, geom.table)
    bb, ob = aabb(shape), obb(shape)
    holes: list[Hole] = _guarded("hole detection", [], w, lambda: detect_holes(geom))
    patterns: list[HolePattern] = _guarded(
        "pattern detection", [], w, lambda: detect_patterns(holes)
    )
    no_slots: tuple[list[Slot], set[str]] = ([], set())
    slots_used = _guarded("slot detection", no_slots, w, lambda: detect_slots(geom))
    slots, slot_faces = slots_used
    fillets: list[Fillet] = _guarded(
        "fillet detection", [], w, lambda: detect_fillets(geom, slot_faces)
    )
    chamfers: list[Chamfer] = _guarded("chamfer detection", [], w, lambda: detect_chamfers(geom))
    bosses: list[Boss] = _guarded("boss detection", [], w, lambda: detect_bosses(geom, holes))
    pockets: list[Pocket] = _guarded("pocket detection", [], w, lambda: detect_pockets(geom, holes))
    wall = None
    if not opts.skip_wall_thickness:
        wall = _guarded("wall thickness", None, w, lambda: sample_wall_thickness(geom))
    shape_class: ShapeClass = _guarded(
        "shape classification",
        ShapeClass(confidence=0.0, evidence="classification failed", label="other"),
        w,
        lambda: classify_shape(geom, ob, mass.volume_mm3, mass.surface_area_mm2, holes, wall),
    )
    part = Part(
        id="",
        name=geom.name,
        source_file=source_file,
        content_hash="",
        is_valid_solid=solid,
        bbox=bb,
        obb=ob,
        mass=mass,
        topology=topo,
        shape_class=shape_class,
        holes=holes,
        hole_patterns=patterns,
        fillets=fillets,
        chamfers=chamfers,
        bosses=bosses,
        slots=slots,
        pockets=pockets,
        color_rgb=color,
        min_wall_thickness_mm=wall[0] if wall else None,
        warnings=[*geom.warnings, *w],
    )
    no_cutouts: tuple[list[Cutout], list[CutoutPattern]] = ([], [])
    part.cutouts, part.cutout_patterns = _guarded(
        "cutout detection", no_cutouts, part.warnings, lambda: detect_cutouts(geom, part)
    )
    if semantics is not None:
        part.semantic_tags = _guarded(
            "semantic matching", [], part.warnings, lambda: semantics(geom, part)
        )
    if opts.understanding:
        from stepscribe.understanding import part_understanding

        override = opts.extra.get("part_processes", {}).get(geom.name) or opts.extra.get(
            "default_process"
        )
        part.understanding = _guarded(
            "understanding", None, part.warnings, lambda: part_understanding(geom, part, override)
        )
    part.content_hash = content_hash(part)
    return part


def analyze_brep_file(
    brep_path: str,
    name: str,
    source_file: str,
    opts: AnalyzeOptions,
    color: list[float] | None,
) -> tuple[Part, float]:
    """Process-pool entry point: load a solid saved as BREP and analyse it (returns Part, seconds)."""
    import time

    from OCP.BRep import BRep_Builder
    from OCP.BRepTools import BRepTools

    from stepscribe.semantics import run_semantics

    started = time.time()
    shape = TopoDS_Shape()
    if not BRepTools.Read_s(shape, brep_path, BRep_Builder()):
        raise RuntimeError(f"could not reload part '{name}' for parallel analysis")
    part = analyze_geom(PartGeom(name, shape), source_file, opts, color, run_semantics)
    return part, time.time() - started


def worker_count(n_parts: int) -> int:
    """Processes to use for part analysis: env STEPSCRIBE_WORKERS, else CPUs - 1 (min 1)."""
    import os

    env = os.environ.get("STEPSCRIBE_WORKERS")
    if env and env.isdigit():
        return max(1, min(int(env), n_parts))
    return max(1, min((os.cpu_count() or 2) - 1, n_parts, 8))


def assign_part_ids(parts: list[Part]) -> None:
    """PRT001... sorted by (name, content hash): Rule 6.4."""
    ordered = sorted(parts, key=lambda p: (p.name, p.content_hash))
    for i, p in enumerate(ordered, 1):
        p.id = f"PRT{i:03d}"


__all__ = [
    "AnalyzeOptions",
    "AnalyzedPart",
    "analyze_brep_file",
    "analyze_geom",
    "assign_part_ids",
    "np",
    "split_bodies",
]
