# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""One part -> Level 1 paragraph and Level 2 detail."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from stepscribe.describe.phrases import fmt, hole_callout, join_and, plural
from stepscribe.models.schema import Cutout, Hole, Part, ShapeClass


def size_text(part: Part) -> str:
    """Stock size from the oriented bounding box, longest first."""
    a, b, c = part.obb.size_sorted
    return f"{fmt(a)} × {fmt(b)} × {fmt(c)} mm"


def inference(label: str, conf: float, evidence: str | None = None) -> str:
    """'Likely: label (0.93)' with optional evidence."""
    base = f"Likely: {label} ({conf:.2f})"
    return f"{base}: {evidence}" if evidence else base


def shape_label(sc: ShapeClass) -> str:
    """Readable class name."""
    return sc.label.replace("_", " ")


def material_text(part: Part) -> str:
    """Material sentence fragment."""
    m = part.mass
    if m.material_assumed and m.material_source == "user":
        return f"material {m.material_assumed} (given)"
    if m.material_assumed:
        return f"material guess {m.material_assumed} from the name (assumed, not measured)"
    return "material unknown"


def mass_text(part: Part) -> str:
    """Mass fragment."""
    if part.mass.mass_g is not None:
        return f"{fmt(part.mass.mass_g, 1)} g ({material_text(part)})"
    return "unknown (no material given)"


def group_by_size(items: list[Any], key: str) -> list[tuple[float, int]]:
    """[(size, count)] sorted by size."""
    counts: dict[float, int] = defaultdict(int)
    for it in items:
        counts[round(getattr(it, key), 2)] += 1
    return sorted(counts.items())


def feature_phrases(part: Part) -> list[str]:
    """Key features as short callouts (patterns first, then loose holes, slots, pockets, bosses)."""
    out: list[str] = []
    in_pattern: set[str] = set()
    for pat in part.hole_patterns:
        out.append(f"{pat.description} ({pat.id})")
        in_pattern.update(pat.hole_ids)
    for h in part.holes:
        if h.id not in in_pattern:
            out.append(f"{hole_callout(h)} ({h.id})")
    in_cutout_pattern: set[str] = set()
    for cp in part.cutout_patterns:
        out.append(f"{cp.description} ({cp.id}; {', '.join(cp.cutout_ids)})")
        in_cutout_pattern.update(cp.cutout_ids)
    for c in part.cutouts:
        if c.id not in in_cutout_pattern:
            out.append(f"{c.description} ({c.id})")
    for s in part.slots:
        if s.cutout_id:
            continue  # reported once, as its cut-out
        out.append(f"slot {fmt(s.width_mm)} × {fmt(s.length_mm)} ({s.id})")
    for p in part.pockets:
        if p.cutout_id:
            continue
        out.append(
            f"pocket {fmt(p.outline_size[0])} × {fmt(p.outline_size[1])} ↧ {fmt(p.depth_mm)} ({p.id})"
        )
    for b in part.bosses:
        out.append(f"boss Ø{fmt(b.diameter_mm)} × {fmt(b.height_mm)} high ({b.id})")
    for r, n in group_by_size(part.fillets, "radius_mm"):
        out.append(f"{n} × R{fmt(r)} fillet")
    for d, n in group_by_size(part.chamfers, "distance_mm"):
        out.append(f"{n} × {fmt(d)} chamfer")
    return out


def level1(part: Part, connections: list[str] | None = None, max_words: int = 120) -> str:
    """One paragraph per part, trimmed to *max_words*."""
    sc = part.shape_class
    thickness = f", t = {fmt(sc.thickness_mm)} mm" if sc.thickness_mm else ""
    sentences = [
        f"{part.id} {part.name}: {inference(shape_label(sc), sc.confidence)}, {size_text(part)}{thickness}."
    ]
    sentences.append(material_text(part).capitalize() + ".")
    feats = feature_phrases(part)
    if feats:
        sentences.append(
            "Features: "
            + "; ".join(feats[:4])
            + (f"; +{len(feats) - 4} more." if len(feats) > 4 else ".")
        )
    for tag in part.semantic_tags[:2]:
        sentences.append(inference(tag.label, tag.confidence) + ".")
    from stepscribe.understanding.describe import understanding_sentences

    sentences += understanding_sentences(part)
    if connections:
        sentences.append("Connects to: " + join_and(connections[:3]) + ".")
    text = " ".join(sentences)
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(";,") + " …"
    return text


def level1_short(part: Part) -> str:
    """One-line form used when the budget is tight."""
    return f"{part.id} {part.name}: {shape_label(part.shape_class)}, {size_text(part)}, {plural(len(part.holes), 'hole')}."


def hole_row(h: Hole) -> dict[str, str]:
    """A formatted hole-table row."""
    o, d = h.axis.origin, h.axis.direction
    std = h.standard_matches[0] if h.standard_matches else None
    return {
        "id": h.id,
        "dia": fmt(h.diameter_mm, 2),
        "depth": "THRU"
        if h.is_through
        else (fmt(h.depth_mm, 2) if h.depth_mm is not None else "?"),
        "entry": h.entry_type.replace("_", " "),
        "bottom": h.bottom_type.replace("_", " "),
        "position": f"({fmt(o.x, 2)}, {fmt(o.y, 2)}, {fmt(o.z, 2)})",
        "direction": f"({fmt(d.x, 2)}, {fmt(d.y, 2)}, {fmt(d.z, 2)})",
        "edge": fmt(h.edge_distance_mm, 2) if h.edge_distance_mm is not None else "-",
        "standard": (
            f"Likely: {std.designation} {std.fit.replace('_', ' ')} ({std.confidence:.2f})"
            if std
            else "-"
        ),
        "threaded": "Likely threaded" if h.likely_threaded else "",
    }


def cutout_row(c: Cutout, part: Part | None = None) -> dict[str, str]:
    """A formatted cut-out-table row; levels are joined with ' ; ' in one cell."""
    lv = []
    for i, x in enumerate(c.levels, 1):
        depth = "THRU" if x.depth_mm is None else f"↧{fmt(x.depth_mm, 2)}"
        size = (
            f"Ø{fmt(x.width_mm, 2)}"
            if x.shape == "circle"
            else f"{fmt(x.width_mm, 2)} × {fmt(x.length_mm, 2)}"
        )
        extra = f", R{fmt(x.corner_radius_mm, 2)}" if x.corner_radius_mm else ""
        rot = (
            f", rot {fmt(x.rotation_deg, 1)}°"
            if abs(x.rotation_deg) > 0.05 and x.shape != "circle"
            else ""
        )
        sides = f" ({x.sides} sides)" if x.sides else ""
        lv.append(
            f"L{i} {x.shape.replace('_', ' ')}{sides} {size}{extra}{rot}, {depth}, "
            f"centre (u {fmt(x.center_uv[0], 2)}, v {fmt(x.center_uv[1], 2)})"
        )
    also = ", ".join(
        [s.id for s in (part.slots if part else []) if s.cutout_id == c.id]
        + [p.id for p in (part.pockets if part else []) if p.cutout_id == c.id]
    )
    return {
        "id": c.id,
        "kind": c.kind,
        "levels": " ; ".join(lv),
        "depth": "THRU" if c.total_depth_mm is None else fmt(c.total_depth_mm, 2),
        "edge": fmt(c.edge_distance_mm, 2) if c.edge_distance_mm is not None else "-",
        "faces": f"{c.host_face_id}; {', '.join(c.face_ids[:8])}"
        + (" …" if len(c.face_ids) > 8 else ""),
        "also": also or "-",
        "note": "; ".join(c.warnings),
    }


def level2_context(
    part: Part, relations: list[str] | None = None, recon: list[str] | None = None
) -> dict[str, Any]:
    """All the data the part template renders."""
    sc = part.shape_class
    return {
        "part": part,
        "size": size_text(part),
        "mass": mass_text(part),
        "shape_label": shape_label(sc),
        "shape_inference": inference(shape_label(sc), sc.confidence, sc.evidence),
        "thickness": fmt(sc.thickness_mm) if sc.thickness_mm else None,
        "holes": [hole_row(h) for h in part.holes],
        "patterns": [
            f"{p.description} ({p.id}; holes {', '.join(p.hole_ids)})" for p in part.hole_patterns
        ],
        "cutouts": [cutout_row(c, part) for c in part.cutouts],
        "cutout_patterns": [
            f"{p.description} ({p.id}; cut-outs {', '.join(p.cutout_ids)})"
            for p in part.cutout_patterns
        ],
        "slots": [
            f"{s.id}: width {fmt(s.width_mm)}, length {fmt(s.length_mm)}, "
            + ("through" if s.is_through else f"depth {fmt(s.depth_mm or 0.0)}")
            + f" at ({fmt(s.center.x)}, {fmt(s.center.y)}, {fmt(s.center.z)})"
            for s in part.slots
            if not s.cutout_id
        ],
        "pockets": [
            f"{p.id}: {fmt(p.outline_size[0])} × {fmt(p.outline_size[1])}, depth {fmt(p.depth_mm)}"
            + (f", corner radius R{fmt(p.corner_radius_mm)}" if p.corner_radius_mm else "")
            for p in part.pockets
            if not p.cutout_id
        ],
        "fillets": [f"{n} × R{fmt(r)}" for r, n in group_by_size(part.fillets, "radius_mm")],
        "chamfers": [f"{n} × {fmt(d)}" for d, n in group_by_size(part.chamfers, "distance_mm")],
        "bosses": [
            f"{b.id}: Ø{fmt(b.diameter_mm)} × {fmt(b.height_mm)} high"
            + (f", carries hole {b.has_hole_id}" if b.has_hole_id else "")
            for b in part.bosses
        ],
        "tags": [inference(t.label, t.confidence, t.evidence) for t in part.semantic_tags],
        "wall": fmt(part.min_wall_thickness_mm, 2)
        if part.min_wall_thickness_mm is not None
        else None,
        "relations": relations or [],
        "recon": recon or [],
        "images": part.images,
        "warnings": part.warnings,
    }
