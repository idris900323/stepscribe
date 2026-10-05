# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Standard shaft diameters and D-flats."""

from __future__ import annotations

import math

import numpy as np

from stepscribe.describe.phrases import fmt
from stepscribe.features.bosses import _is_convex_full_cylinder
from stepscribe.features.hole_standards import load_table
from stepscribe.features.holes import find_bore_faces
from stepscribe.geometry.occ_utils import canonical_dir, uv_bounds
from stepscribe.geometry.part_geom import PartGeom
from stepscribe.geometry.surfaces import cylinder_params, plane_normal_origin
from stepscribe.models.schema import Part, SemanticTag
from stepscribe.semantics.matcher import entries, round_conf

BORE_CLEARANCE_MAX = 0.15  # mm over the standard diameter for a close-fit shaft bore
SHAFT_TOL = 0.05
SCREW_CLEARANCE_TOL = 0.05
MIN_LENGTH_RATIO = 1.0  # shaft stub length / diameter
D_FLAT_MIN_SPAN = 200.0  # degrees of remaining cylinder for a D-shaft


def _screw_clearances() -> list[float]:
    rows = load_table("screws_iso_metric.yaml")["sizes"].values()
    return [float(r[k]) for r in rows for k in ("close", "normal", "loose")]


def match_shaft_bores(part: Part) -> list[SemanticTag]:
    """Through holes at a standard shaft diameter (+0 to +0.15 mm) that are not screw holes."""
    screws = _screw_clearances()
    tags: list[SemanticTag] = []
    for h in part.holes:
        if not h.is_through or h.likely_threaded:
            continue
        if any(abs(h.diameter_mm - s) <= SCREW_CLEARANCE_TOL for s in screws):
            continue
        for key, e in entries("shafts.yaml").items():
            d = float(e["diameter_mm"])
            over = h.diameter_mm - d
            if 0 <= over <= BORE_CLEARANCE_MAX + 1e-9:
                conf = round_conf(0.5 + 0.3 * (1 - over / BORE_CLEARANCE_MAX), 0.8)
                tags.append(
                    SemanticTag(
                        confidence=conf,
                        evidence=f"{e['label']} shaft bore ({conf:.2f}): Ø{fmt(h.diameter_mm, 2)} is {over:+.2f} mm over standard Ø{fmt(d, 3)}",
                        kind="shaft_bore",
                        label=f"{e['label']} shaft bore",
                        knowledge_id=f"shafts.{key}",
                        feature_ids=[h.id],
                    )
                )
                break
    return tags


def match_shafts(geom: PartGeom, part: Part) -> list[SemanticTag]:
    """Convex cylinders at a standard diameter, and D-flats (a plane cut into a cylinder)."""
    tags: list[SemanticTag] = []
    table = geom.table
    boss_faces = {fid for b in part.bosses for fid in b.face_ids}
    concave = {b.idx for b in find_bore_faces(geom)}
    for idx, fi in enumerate(table.faces):
        if fi.kind != "cylinder" or fi.id in boss_faces or idx in concave:
            continue
        cp = cylinder_params(fi.face)
        u0, u1, v0, v1 = uv_bounds(fi.face)
        span = math.degrees(u1 - u0)
        std = _standard_for(2 * cp.radius)
        if std is None:
            continue
        key, e = std
        if span >= 359.0 and _is_convex_full_cylinder(geom, idx):
            if abs(v1 - v0) < MIN_LENGTH_RATIO * 2 * cp.radius:
                continue
            conf = round_conf(0.6)
            tags.append(
                _shaft_tag(
                    fi.id,
                    key,
                    e,
                    conf,
                    f"Ø{fmt(2 * cp.radius, 2)} × {fmt(abs(v1 - v0))} long convex cylinder",
                    "shaft",
                )
            )
        elif D_FLAT_MIN_SPAN <= span < 359.0 and _has_parallel_flat(
            geom, idx, canonical_dir(cp.direction)
        ):
            conf = round_conf(0.7)
            tags.append(
                _shaft_tag(
                    fi.id,
                    key,
                    e,
                    conf,
                    f"Ø{fmt(2 * cp.radius, 2)} cylinder with a flat parallel to its axis",
                    "D-shaft",
                )
            )
    return tags


def _shaft_tag(
    face_id: str, key: str, e: dict[str, object], conf: float, what: str, noun: str
) -> SemanticTag:
    return SemanticTag(
        confidence=conf,
        evidence=f"{e['label']} {noun} ({conf:.2f}): {what}",
        kind="other",
        label=f"{e['label']} {noun}",
        knowledge_id=f"shafts.{key}",
        feature_ids=[face_id],
    )


def _standard_for(diameter: float) -> tuple[str, dict[str, object]] | None:
    for key, e in entries("shafts.yaml").items():
        if abs(diameter - float(e["diameter_mm"])) <= SHAFT_TOL:
            return key, e
    return None


def _has_parallel_flat(geom: PartGeom, idx: int, axis: np.ndarray) -> bool:
    for j in geom.table.faces[idx].neighbors:
        nb = geom.table.faces[j]
        if nb.kind == "plane":
            n, _o = plane_normal_origin(nb.face)
            if abs(float(np.dot(n, axis))) < 1e-3:
                return True
    return False
