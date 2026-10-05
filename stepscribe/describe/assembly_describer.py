# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Whole design -> Level 0 overview and the assembly core text."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from stepscribe.describe.budget import rank_parts
from stepscribe.describe.part_describer import shape_label
from stepscribe.describe.phrases import fmt, join_and, plural
from stepscribe.models.schema import BBox, Part, Report

_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def size_up_front(bbox: BBox, up: str, front: str) -> tuple[float, float, float]:
    """(width, depth, height) from a global bbox using the up/front axis labels."""
    s = [bbox.size.x, bbox.size.y, bbox.size.z]
    ui, fi = _AXIS_INDEX[up[1]], _AXIS_INDEX[front[1]]
    wi = ({0, 1, 2} - {ui, fi}).pop() if ui != fi else 0
    return s[wi], s[fi], s[ui]


def union_bbox(parts: list[Part]) -> BBox:
    """Union of part bounding boxes (local frames; used when there is no assembly)."""
    from stepscribe.models.schema import Vec3

    lo = [min(getattr(p.bbox.min, a) for p in parts) for a in "xyz"]
    hi = [max(getattr(p.bbox.max, a) for p in parts) for a in "xyz"]
    return BBox(
        min=Vec3(x=lo[0], y=lo[1], z=lo[2]),
        max=Vec3(x=hi[0], y=hi[1], z=hi[2]),
        size=Vec3(x=hi[0] - lo[0], y=hi[1] - lo[1], z=hi[2] - lo[2]),
    )


def quantities(report: Report) -> dict[str, int]:
    """Part ID -> instance count (1 each when there is no assembly)."""
    if report.assembly:
        q = {line.part_id: line.quantity for line in report.assembly.bom}
        return {p.id: q.get(p.id, 1) for p in report.parts}
    return {p.id: 1 for p in report.parts}


HEADLINE_MAX = 8


def headline_semantics(report: Report) -> list[str]:
    """'2 × NEMA 17 motor mount' style counts across the design (the HEADLINE_MAX most common)."""
    qty = quantities(report)
    counts: Counter[str] = Counter()
    for p in report.parts:
        for t in p.semantic_tags:
            counts[t.label] += qty.get(p.id, 1)
    rows = [f"{n} × {label}" for label, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    if len(rows) > HEADLINE_MAX:
        rows = [*rows[:HEADLINE_MAX], f"{len(rows) - HEADLINE_MAX} more kinds (see the part files)"]
    return rows


def total_mass(report: Report) -> float | None:
    """Total mass in grams if every part has a mass, else None."""
    if report.assembly and report.assembly.total_mass_g is not None:
        return report.assembly.total_mass_g
    qty = quantities(report)
    if report.parts and all(p.mass.mass_g is not None for p in report.parts):
        return sum((p.mass.mass_g or 0.0) * qty[p.id] for p in report.parts)
    return None


def level0_context(
    report: Report,
    root_name: str,
    up: str = "+Z",
    front: str = "-Y",
    images: list[str] | None = None,
) -> dict[str, Any]:
    """Data for the overview template (target <= 400 words)."""
    parts = report.parts
    u = report.understanding
    qty = quantities(report)
    bbox = report.assembly.global_bbox if report.assembly else union_bbox(parts)
    w, d, h = size_up_front(bbox, up, front)
    hardware = sum(1 for p in parts if p.likely_purchased_hardware)
    mass = total_mass(report)
    top = rank_parts(parts)[:8]
    shopping = report.assembly.fastener_shopping_list if report.assembly else []
    return {
        "root_name": root_name,
        "n_parts": len(parts),
        "n_instances": sum(qty.values()),
        "n_fabricated": len(parts) - hardware,
        "n_hardware": hardware,
        "subassemblies": [s.name for s in report.assembly.subassemblies] if report.assembly else [],
        "size": f"{fmt(w)} (width) × {fmt(d)} (depth) × {fmt(h)} (height) mm",
        "mass": f"{fmt(mass, 1)} g" if mass is not None else None,
        "mass_estimated": any(p.mass.material_source == "name_guess" for p in parts),
        "com": (
            f"({fmt(report.assembly.center_of_mass.x)}, {fmt(report.assembly.center_of_mass.y)}, "
            f"{fmt(report.assembly.center_of_mass.z)})"
            if report.assembly and report.assembly.center_of_mass
            else None
        ),
        "top_parts": [f"{p.id} {p.name} ({shape_label(p.shape_class)}, ×{qty[p.id]})" for p in top],
        "headline": headline_semantics(report),
        "summary": u.summary if u else "",
        "top_spots": [
            f"{w.id} ({w.severity}): {w.message}" for w in (u.weak_spots[:3] if u else [])
        ],
        "fasteners": [f"{row['qty']} × {row['item']}" for row in shopping],
        "up": up,
        "front": front,
        "images": images or [],
    }


def assembly_core_context(report: Report) -> dict[str, Any]:
    """Relations, joints and shopping list that always appear in the single-file pack."""
    a = report.assembly
    if a is None:
        return {"has_assembly": False}
    return {
        "has_assembly": True,
        "relations": [f"{r.id}: {r.sentence}" for r in a.relations],
        "joints": [
            f"{j.id}: Ø{fmt(j.common_diameter_mm)} through {len(j.instance_ids)} parts, grip {fmt(j.stack_thickness_mm)} mm, "
            f"{j.suggested_fastener or 'fastener unknown'} (Likely, {j.confidence:.2f})"
            for j in a.fastener_joints
        ],
        "shopping": [f"{row['qty']} × {row['item']}" for row in a.fastener_shopping_list],
        "subassemblies": [f"{s.name}: {s.summary}" for s in a.subassemblies],
        "bom": [
            f"{line.part_id} {line.name} × {line.quantity} ({line.category.replace('_', ' ')})"
            for line in a.bom
        ],
        "spatial": [s.sentence for s in a.spatial_facts],
        "placements": [
            f"{i.id} = {i.part_id}: origin of the part's own frame at ({fmt(i.position.x, 2)}, {fmt(i.position.y, 2)}, {fmt(i.position.z, 2)}), "
            f"rotated {fmt(math.degrees(i.rotation_axis_angle[1]), 1)} deg about "
            f"({fmt(i.rotation_axis_angle[0].x, 3)}, {fmt(i.rotation_axis_angle[0].y, 3)}, {fmt(i.rotation_axis_angle[0].z, 3)}); "
            f"world bounds min ({fmt(i.global_bbox.min.x, 2)}, {fmt(i.global_bbox.min.y, 2)}, {fmt(i.global_bbox.min.z, 2)}) "
            f"max ({fmt(i.global_bbox.max.x, 2)}, {fmt(i.global_bbox.max.y, 2)}, {fmt(i.global_bbox.max.z, 2)})"
            for i in a.instances
        ],
        "interferences": [str(i.get("sentence", i)) for i in a.interferences],
        "n_contacts": len(a.contacts),
        "plural": plural,
        "join_and": join_and,
    }
