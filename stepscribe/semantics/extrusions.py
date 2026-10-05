# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Aluminium extrusion profiles."""

from __future__ import annotations

import numpy as np

from stepscribe.describe.phrases import fmt
from stepscribe.geometry.occ_utils import outward_normal, uv_bounds
from stepscribe.geometry.part_geom import PartGeom
from stepscribe.geometry.surfaces import plane_normal_origin
from stepscribe.models.schema import Part, SemanticTag
from stepscribe.semantics.matcher import entries, linear_score, round_conf

OUTER_TOL = 0.3  # mm
SLOT_TOL = 0.08
BORE_TOL = 0.2
PAIR_ALIGN_TOL = 1.0  # mm: facing slot walls are centred on each other
BASE_CONF, BORE_BONUS, SLOT_BONUS = 0.5, 0.2, 0.25


def _slot_widths(geom: PartGeom, axis: np.ndarray) -> list[float]:
    """Distances between pairs of planar faces that face each other across a void."""
    planes = []
    for fi in geom.table.faces:
        if fi.kind != "plane":
            continue
        n, _o = plane_normal_origin(fi.face)
        if abs(float(np.dot(n, axis))) > 1e-3:
            continue
        u0, u1, v0, v1 = uv_bounds(fi.face)
        out = outward_normal(fi.face, 0.5 * (u0 + u1), 0.5 * (v0 + v1))
        if out is not None:
            planes.append((out, fi.centroid))
    widths = []
    for i, (ni, ci) in enumerate(planes):
        for nj, cj in planes[i + 1 :]:
            if float(np.dot(ni, nj)) < -1 + 1e-3:
                off = cj - ci
                d = float(np.dot(off, ni))
                perp = float(np.linalg.norm(off - ni * d))
                if d > 0.5 and perp <= PAIR_ALIGN_TOL:
                    widths.append(round(d, 2))
    return sorted(widths)


def match_extrusions(geom: PartGeom, part: Part) -> list[SemanticTag]:
    """Match the cross-section (outer size, centre bore, slot width) with catalogue profiles."""
    if part.shape_class.label not in ("extrusion_profile", "bar"):
        return []
    a, b, c = part.obb.size_sorted
    axis = _long_axis(part)
    widths = _slot_widths(geom, axis) if axis is not None else []
    best: SemanticTag | None = None
    for key, e in entries("extrusions.yaml").items():
        o = sorted((float(v) for v in e["outer_mm"]), reverse=True)
        if linear_score(b - o[0], OUTER_TOL) == 0 or linear_score(c - o[1], OUTER_TOL) == 0:
            continue
        conf = BASE_CONF
        notes = [f"outer {fmt(b)} × {fmt(c)} matches {fmt(o[0])} × {fmt(o[1])}"]
        if "bore_mm" in e:
            bore = [h for h in part.holes if abs(h.diameter_mm - float(e["bore_mm"])) <= BORE_TOL]
            if bore:
                conf += BORE_BONUS
                notes.append(
                    f"centre bore Ø{fmt(bore[0].diameter_mm)} matches Ø{fmt(float(e['bore_mm']))}"
                )
        if "slot_width_mm" in e:
            hits = [w for w in widths if abs(w - float(e["slot_width_mm"])) <= SLOT_TOL]
            if hits:
                conf += SLOT_BONUS
                notes.append(f"{len(hits)} slot opening(s) of {fmt(float(e['slot_width_mm']))} mm")
        conf = round_conf(conf)
        tag = SemanticTag(
            confidence=conf,
            evidence=f"{e['label']} ({conf:.2f}): " + "; ".join(notes) + f"; length {fmt(a)} mm",
            kind="extrusion_profile",
            label=f"{e['label']}, {fmt(a)} mm long",
            knowledge_id=f"extrusions.{key}",
            feature_ids=[h.id for h in part.holes][:1],
        )
        if best is None or tag.confidence > best.confidence:
            best = tag
    return [best] if best else []


def _long_axis(part: Part) -> np.ndarray | None:
    order = np.argsort([-part.obb.half_sizes.x, -part.obb.half_sizes.y, -part.obb.half_sizes.z])
    ax = part.obb.axes[int(order[0])]
    return np.array([ax.x, ax.y, ax.z])
