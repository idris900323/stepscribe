# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Rule-based weak spots.

Each rule is a small function; thresholds come from ``knowledge/weak_spot_rules.yaml`` and the
process guess (or the designer's answer). Every finding has numbers, IDs, and a generic
suggestion; findings that rest on an assumption say which one.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from stepscribe.describe.phrases import fmt
from stepscribe.features.hole_standards import load_table
from stepscribe.models.schema import (
    KinematicModel,
    Part,
    StabilityInfo,
    WeakSpot,
)
from stepscribe.understanding.aag import build_aag
from stepscribe.understanding.kinematics import KinContext, hardware_kind
from stepscribe.understanding.load_path import LoadInfo
from stepscribe.understanding.process import min_wall_for

if TYPE_CHECKING:
    from stepscribe.api import Analysis

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}
BRACKETS = ("l_bracket", "u_bracket", "z_bracket", "angle")
PLASTIC = ("fdm_3d_printed", "sla_or_sls_printed")


def rules() -> dict:  # type: ignore[type-arg]
    return load_table("weak_spot_rules.yaml")


def top_process(part: Part) -> tuple[str | None, str]:
    """(process label, assumption text or '') for a part from its understanding."""
    u = part.understanding
    if u is None or not u.process.ranked:
        return None, ""
    h = u.process.ranked[0]
    if h.label == "unknown":
        return None, ""
    if u.process.source == "user":
        return h.label, ""
    return h.label, f"assumes {h.label.replace('_', ' ')} (inferred, confidence {h.confidence:.2f})"


def _spot(
    severity: str,
    category: str,
    refs: list[str],
    message: str,
    suggestion: str | None = None,
    value: float | None = None,
    threshold: float | None = None,
    unit: str | None = None,
    assumption: str | None = None,
) -> WeakSpot:
    return WeakSpot(
        id="",
        severity=severity,
        category=category,
        refs=refs,
        value=None if value is None else round(value, 3),
        threshold=None if threshold is None else round(threshold, 3),
        unit=unit,
        message=message,
        suggestion=suggestion,
        depends_on_assumption=assumption or None,
    )


def _sev(ratio: float) -> str:
    """Severity from value / threshold (lower is worse)."""
    return "high" if ratio < 0.5 else "medium" if ratio < 0.8 else "low"


def _edge_assumption(proc: str | None, why: str) -> str | None:
    if proc in PLASTIC:
        return why or None
    if proc is None:
        return "no process known: metal factor used"
    return f"{why}; metal factor used" if why else None


def edge_distance(part: Part, label: str) -> list[WeakSpot]:
    cfg = rules()["edge_distance"]
    if part.likely_purchased_hardware:
        return []  # a bought part (servo, bearing...) is not ours to redesign
    proc, why = top_process(part)
    factor = float(cfg["fdm_factor"] if proc in PLASTIC else cfg["metal_factor"])
    groups: dict[float, list] = {}  # type: ignore[type-arg]
    for h in part.holes:
        if h.edge_distance_mm is None or h.diameter_mm > float(cfg["max_hole_diameter_mm"]):
            continue
        if h.edge_distance_mm < float(cfg["open_edge_mm"]):
            continue  # the hole breaks through the edge: a slot or notch, not a wall left too thin
        if h.edge_distance_mm < factor * h.diameter_mm:
            groups.setdefault(round(h.diameter_mm, 2), []).append(h)
    out = []
    for dia, holes in groups.items():
        # one row per part and hole size: the worst hole gives the numbers, the rest are listed
        worst = min(holes, key=lambda x: x.edge_distance_mm)
        need = factor * dia
        ids = ", ".join(h.id for h in holes)
        many = f"{len(holes)} holes of dia {fmt(dia)} mm ({ids}) on {label}; worst is {worst.id}"
        one = f"Hole {worst.id} (dia {fmt(dia)} mm) on {label}"
        out.append(
            _spot(
                _sev(worst.edge_distance_mm / need),
                "edge_distance",
                [part.id, *[h.id for h in holes]],
                f"{many if len(holes) > 1 else one} is {fmt(worst.edge_distance_mm, 2)} mm from "
                f"the nearest edge; the guideline is {fmt(factor, 1)} x diameter = {fmt(need, 1)} mm",
                f"increase the edge distance to at least {fmt(need, 1)} mm "
                f"(move the hole {fmt(need - worst.edge_distance_mm, 1)} mm inward)",
                worst.edge_distance_mm,
                need,
                "mm",
                _edge_assumption(proc, why),
            )
        )
    return out


def cutout_web(part: Part, label: str) -> list[WeakSpot]:
    """Thin bridges between a cut-out and the outline or another opening."""
    cfg = rules()["cutout_web"]
    if part.likely_purchased_hardware or not part.cutouts:
        return []
    proc, why = top_process(part)
    floor = min_wall_for(proc)
    thick = part.shape_class.thickness_mm
    need = max(
        floor[0] if floor else 0.0,
        float(cfg["thickness_factor"]) * thick if thick else 0.0,
    )
    if need <= 0:
        return []
    thin = [
        c
        for c in part.cutouts
        if c.edge_distance_mm is not None
        and float(cfg["open_edge_mm"]) <= c.edge_distance_mm < need
    ]
    if not thin:
        return []
    worst = min(thin, key=lambda c: c.edge_distance_mm or 0.0)
    dist = worst.edge_distance_mm or 0.0
    ids = ", ".join(c.id for c in thin)
    many = f"{len(thin)} cut-outs ({ids}) on {label}; worst is {worst.id}"
    one = f"Cut-out {worst.id} on {label}"
    return [
        _spot(
            _sev(dist / need),
            "cutout_web",
            [part.id, *[c.id for c in thin]],
            f"{many if len(thin) > 1 else one} leaves a {fmt(dist, 2)} mm web to the nearest edge or "
            f"opening; the guideline is {fmt(need, 1)} mm",
            f"widen the web to at least {fmt(need, 1)} mm (move the cut-out {fmt(need - dist, 1)} mm "
            f"or shrink it)",
            dist,
            need,
            "mm",
            _edge_assumption(proc, why),
        )
    ]


def thin_wall(part: Part, label: str) -> list[WeakSpot]:
    if part.min_wall_thickness_mm is None:
        return []
    proc, why = top_process(part)
    got = min_wall_for(proc)
    if got is None:
        return []
    need, source = got
    if part.min_wall_thickness_mm >= need:
        return []
    return [
        _spot(
            _sev(part.min_wall_thickness_mm / need),
            "thin_wall",
            [part.id],
            f"{label} has a thinnest wall of about {fmt(part.min_wall_thickness_mm, 2)} mm, below the "
            f"{fmt(need, 1)} mm guideline for {str(proc).replace('_', ' ')} ({source})",
            f"thicken the thinnest wall to at least {fmt(need, 1)} mm",
            part.min_wall_thickness_mm,
            need,
            "mm",
            why or None,
        )
    ]


def sharp_corners(part: Part, geom, label: str, load_bearing: bool) -> list[WeakSpot]:  # type: ignore[no-untyped-def]
    """Sharp concave edges between flats on a bracket or ribbed part that carries load."""
    u = part.understanding
    has_ribs = u is not None and any(f.kind in ("rib", "gusset") for f in u.structural_features)
    if not load_bearing or not (part.shape_class.label in BRACKETS or has_ribs):
        return []
    aag = build_aag(geom)
    edges = [
        e
        for e in aag.edges
        if e.kind == "concave"
        and aag.attrs[e.a].kind == "plane"
        and aag.attrs[e.b].kind == "plane"
        and e.length >= 3.0
    ]
    if not edges:
        return []
    proc, why = top_process(part)
    bend = part.shape_class.label in BRACKETS
    longest = max(e.length for e in edges)
    ids = sorted({aag.attrs[e.a].id for e in edges} | {aag.attrs[e.b].id for e in edges})
    plastic = proc in PLASTIC
    cat = "unfilleted_bend" if (bend and plastic) else "sharp_internal_corner"
    where = "at the bend" if bend else "at the base of a rib"
    sev = "medium" if plastic else "low"
    return [
        _spot(
            sev,
            cat,
            [part.id, *ids[:4]],
            f"{label} has {len(edges)} sharp internal corner(s) {where} (longest {fmt(longest, 1)} mm, faces "
            f"{', '.join(ids[:4])}); stress concentrates there and printed layers split along them",
            "add a fillet of about 2 mm (or a gusset) at the corner",
            float(len(edges)),
            0.0,
            "corners",
            why or None,
        )
    ]


def floating_parts(model: KinematicModel, ctx: KinContext) -> list[WeakSpot]:
    out = []
    for link in model.links:
        if link.id not in model.floating_groups:
            continue
        names = ", ".join(f"{i} ({ctx.by_id[i].name})" for i in link.instance_ids[:4])
        out.append(
            _spot(
                "high",
                "floating_part",
                list(link.instance_ids),
                f"{link.id} ({names}) is not connected to the ground link: fasteners or mates may be missing in the CAD",
                "check that this part is bolted, mated or contacts something that is",
            )
        )
    return out


def interferences(asm) -> list[WeakSpot]:  # type: ignore[no-untyped-def]
    cfg = rules()["interference"]
    out = []
    for item in asm.interferences:
        sentence = str(item.get("sentence", item))
        refs = [str(item[k]) for k in ("instance_a", "instance_b") if k in item]
        vol = float(item.get("volume_mm3", 0.0))
        small = vol < float(cfg["small_mm3"])
        out.append(
            _spot(
                "low" if small else "medium",
                "interference",
                refs,
                sentence,
                "separate the overlapping parts or add clearance"
                + (
                    " (a small overlap can be a modelling tolerance or a screw modelled inside its hole)"
                    if small
                    else ""
                ),
                vol,
                float(cfg["small_mm3"]),
                "mm3",
            )
        )
    return out


def tipping(stab: StabilityInfo | None) -> list[WeakSpot]:
    if stab is None or stab.tipping_angle_deg is None:
        return []
    cfg = rules()["stability"]
    out = []
    for label, angle in (
        ("now", stab.tipping_angle_deg),
        ("with the arm extended", stab.extended_tipping_angle_deg),
    ):
        if angle is None or angle >= float(cfg["tipping_warn_deg"]):
            continue
        sev = "high" if angle < float(cfg["tipping_high_deg"]) else "medium"
        out.append(
            _spot(
                sev,
                "tipping",
                [],
                f"The design tips over at {fmt(angle, 1)} deg {label} (threshold {fmt(float(cfg['tipping_warn_deg']), 0)} deg); "
                f"{stab.description}",
                "widen the support polygon, lower the centre of mass, or add a counterweight on the opposite side",
                angle,
                float(cfg["tipping_warn_deg"]),
                "deg",
                "uses the masses from the given material or density",
            )
        )
    return out


def single_fasteners(load: LoadInfo, ctx: KinContext) -> list[WeakSpot]:
    out = []
    masses = {i.id: i.ap.part.mass.mass_g for i in ctx.insts}
    for ref, e in sorted(load.edges.items()):
        if e.kind != "fastened" or e.fasteners != 1:
            continue
        carried = load.through.get(ref, [])
        motors = [s for s in carried if hardware_kind(ctx.by_id[s]) == "motor"]
        carried_mass = sum(masses[s] or 0.0 for s in carried)
        if not carried:
            continue
        names = ", ".join(f"{s} ({ctx.by_id[s].name})" for s in carried[:3])
        out.append(
            _spot(
                "high" if motors else "medium",
                "single_fastener",
                [e.a, e.b, ref, *carried[:3]],
                f"Connection {ref} between {e.a} ({ctx.by_id[e.a].name}) and {e.b} ({ctx.by_id[e.b].name}) uses a "
                f"single dia {fmt(e.diameter, 1)} mm fastener but carries {names}"
                + (f" ({fmt(carried_mass, 0)} g)" if carried_mass else ""),
                "use at least two fasteners (or a pin plus a screw) so the part cannot rotate or loosen",
                1.0,
                2.0,
                "fasteners",
            )
        )
    return out


def cantilevers(load: LoadInfo, ctx: KinContext, asm) -> list[WeakSpot]:  # type: ignore[no-untyped-def]
    cfg = rules()["cantilever"]
    by_joint = {j.id: j for j in asm.fastener_joints}
    out = []
    seen: set[str] = set()
    for ref, e in sorted(load.edges.items()):
        if e.kind != "fastened" or e.fasteners < 2:
            continue
        pts = np.array(
            [
                [by_joint[j].axis.origin.x, by_joint[j].axis.origin.y, by_joint[j].axis.origin.z]
                for j in e.joint_ids
                if j in by_joint
            ]
        )
        if len(pts) < 2:
            continue
        axis = by_joint[e.joint_ids[0]].axis.direction
        d = np.array([axis.x, axis.y, axis.z])
        d = d / np.linalg.norm(d)
        spread = max(float(np.linalg.norm(a - b)) for a in pts for b in pts)
        if spread < 1e-6:
            continue
        centroid = pts.mean(axis=0)
        # the loaded side is the instance farther from the support centroid
        for inst_id in (e.a, e.b):
            if inst_id in seen or inst_id in load.ground:
                continue
            inst = ctx.by_id[inst_id]
            lo, hi = inst.bounds
            corners = np.array(
                [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
            )
            rel = corners - centroid
            rel = rel - np.outer(rel @ d, d)
            reach = float(np.linalg.norm(rel, axis=1).max())
            ratio = reach / spread
            if ratio <= float(cfg["min_ratio"]):
                continue
            carried = [s for s in load.through.get(ref, []) if s != inst_id]
            if not carried and inst_id not in load.sources:
                continue
            seen.add(inst_id)
            out.append(
                _spot(
                    "high" if ratio > 1.5 * float(cfg["min_ratio"]) else "medium",
                    "cantilever",
                    [inst_id, ref, *carried[:2]],
                    f"{inst_id} ({inst.name}) is held only near one end: its fasteners ({ref}) span {fmt(spread, 1)} mm "
                    f"but it reaches {fmt(reach, 1)} mm from them ({fmt(ratio, 1)} x; limit {fmt(float(cfg['min_ratio']), 0)} x)",
                    "add support near the free end, a gusset, or move the fasteners apart (a wider footprint resists bending)",
                    ratio,
                    float(cfg["min_ratio"]),
                    "x support width",
                )
            )
    return out


def _screw_size(inst, hole_d: float, sizes: dict) -> str | None:  # type: ignore[no-untyped-def,type-arg]
    """'M3' from the name or hardware guess, else the size whose clearance hole matches the joint."""
    text = f"{inst.ap.part.hardware_guess or ''} {inst.name} {inst.ap.part.name}"
    m = re.search(r"M(\d+(?:\.\d+)?)\s*[x\u00d7]", text) or re.search(r"\bM(\d+(?:\.\d+)?)\b", text)
    if m and f"M{m.group(1)}" in sizes:
        return f"M{m.group(1)}"
    best: tuple[float, str] | None = None
    for name, row in sizes.items():
        dev = min(abs(hole_d - float(row[k])) for k in ("close", "normal", "loose") if k in row)
        if dev <= 0.12 and (best is None or dev < best[0]):
            best = (dev, name)
    return best[1] if best else None


def short_screws(asm, ctx: KinContext) -> list[WeakSpot]:  # type: ignore[no-untyped-def]
    cfg = rules()["short_screw"]
    sizes = load_table("screws_iso_metric.yaml")["sizes"]
    out = []
    for j in asm.fastener_joints:
        if not j.existing_fastener_instance_id:
            continue
        inst = ctx.by_id.get(j.existing_fastener_instance_id)
        if inst is None or hardware_kind(inst) != "fastener":
            continue
        size = _screw_size(inst, j.common_diameter_mm, sizes)
        if size is None:
            continue
        row = sizes[size]
        d = float(row["nominal"])
        total = float(inst.ap.part.obb.size_sorted[0])
        shank = total - float(row.get("head_k", d))
        if j.has_threaded_end:
            need = j.stack_thickness_mm + float(cfg["engagement_diameters"]) * d
            kind = f"engagement of {fmt(float(cfg['engagement_diameters']), 1)} x diameter in the threaded hole"
        else:
            need = j.stack_thickness_mm + float(row.get("nut_m", 0.8 * d))
            kind = "a nut (ISO 4032 height)"
        if shank + 0.05 >= need:
            continue
        pick = next((L for L in cfg["standard_lengths_mm"] if L >= need - 1e-6), None)
        out.append(
            _spot(
                "high" if shank < j.stack_thickness_mm else "medium",
                "short_screw",
                [inst.id, j.id],
                f"{size} screw {inst.id} is about {fmt(shank, 1)} mm long under the head but the joint {j.id} needs "
                f"{fmt(need, 1)} mm (grip {fmt(j.stack_thickness_mm, 1)} mm plus {kind})",
                f"use {size} x {pick} instead"
                if pick
                else f"use a longer {size} screw (at least {fmt(need, 1)} mm)",
                shank,
                need,
                "mm",
            )
        )
    return out


def unsupported_span(
    part: Part, label: str, supports: np.ndarray, plate_bounds: tuple[np.ndarray, np.ndarray] | None
) -> list[WeakSpot]:
    """Plate span between supports over thickness above the limit."""
    t = part.shape_class.thickness_mm
    if part.shape_class.label != "plate" or not t or plate_bounds is None or len(supports) == 0:
        return []
    lo, hi = plate_bounds
    ext = hi - lo
    flat_axes = [i for i in range(3) if ext[i] > 3 * t]
    if len(flat_axes) < 2:
        return []
    gaps = []
    gx = np.linspace(lo[flat_axes[0]], hi[flat_axes[0]], 21)
    gy = np.linspace(lo[flat_axes[1]], hi[flat_axes[1]], 21)
    for x in gx:
        for y in gy:
            q = np.array([x, y])
            gaps.append(min(float(np.linalg.norm(q - s[flat_axes])) for s in supports))
    span = 2 * max(gaps)
    limit = float(rules()["unsupported_span"]["max_span_over_thickness"])
    if span / t <= limit:
        return []
    return [
        _spot(
            "medium",
            "unsupported_span",
            [part.id],
            f"{label}: unsupported span about {fmt(span, 0)} mm over a {fmt(t, 1)} mm thick plate "
            f"({fmt(span / t, 0)} x thickness, limit {fmt(limit, 0)} x)",
            "add fasteners, ribs or a stiffer section so no region is farther than "
            f"{fmt(limit * t / 2, 0)} mm from a support",
            span / t,
            limit,
            "x thickness",
        )
    ]


def joint_play(model: KinematicModel, ctx: KinContext) -> list[WeakSpot]:
    cfg = float(rules()["joint_play"]["arm_over_bearing_width"])
    out = []
    for j in model.joints:
        if j.kind != "revolute":
            continue
        bearings = [e for e in j.evidence if e.code == "bearing:inner"]
        if len(bearings) != 1:
            continue
        b_inst = ctx.by_id[bearings[0].refs[-1]] if bearings[0].refs[-1] in ctx.by_id else None
        # the refs are [a, b, contacts...]; the bearing is the instance with a bearing hardware guess
        cand = [
            ctx.by_id[r]
            for r in bearings[0].refs
            if r in ctx.by_id and hardware_kind(ctx.by_id[r]) == "bearing"
        ]
        b_inst = cand[0] if cand else b_inst
        if b_inst is None:
            continue
        width = float(min(b_inst.ap.part.obb.size_sorted))
        child = next(g for g in model.links if g.id == j.child_link)
        members = [ctx.by_id[i] for i in child.instance_ids]
        ms = np.array([i.ap.part.mass.mass_g or i.ap.part.mass.volume_mm3 * 1e-3 for i in members])
        cents = np.array(
            [
                i.to_global_point(
                    np.array(
                        [
                            i.ap.part.mass.centroid.x,
                            i.ap.part.mass.centroid.y,
                            i.ap.part.mass.centroid.z,
                        ]
                    )
                )
                for i in members
            ]
        )
        com = (ms[:, None] * cents).sum(axis=0) / ms.sum()
        o = np.array([j.axis.origin.x, j.axis.origin.y, j.axis.origin.z])
        d = np.array([j.axis.direction.x, j.axis.direction.y, j.axis.direction.z])
        d = d / np.linalg.norm(d)
        # moment arm: distance from the bearing along the axis (tilting) and across it
        off = com - o
        along = abs(float(np.dot(off, d)))
        across = float(np.linalg.norm(off - d * np.dot(off, d)))
        arm = max(along, across)
        if arm > cfg * width:
            out.append(
                _spot(
                    "medium",
                    "joint_play",
                    [j.id, b_inst.id, *child.instance_ids[:2]],
                    f"{j.id} turns on a single bearing ({b_inst.id}, {fmt(width, 1)} mm wide) while the centre of mass of "
                    f"{j.child_link} is {fmt(arm, 1)} mm away ({fmt(arm / width, 1)} x the bearing width; limit {fmt(cfg, 0)} x): "
                    "the joint will rock",
                    "add a second bearing spaced along the axis (at least 2 x the bearing width apart)",
                    arm / width,
                    cfg,
                    "x bearing width",
                )
            )
    return out


Rule = Callable[..., list[WeakSpot]]


def motion_limits(model: KinematicModel) -> list[WeakSpot]:
    """Info findings for joints the motion check found blocked."""
    out = []
    for j in model.joints:
        if j.blocked_by and j.range_deg_or_mm is not None:
            lo, hi = j.range_deg_or_mm
            unit = "mm" if j.kind == "prismatic" else "deg"
            out.append(
                _spot(
                    "info",
                    "interference",
                    [j.id, *j.blocked_by],
                    f"{j.id} moves freely from {fmt(lo, 0)} to {fmt(hi, 0)} {unit}, then hits "
                    f"{', '.join(j.blocked_by)}",
                    "check that this limit is the intended end stop",
                    hi,
                    None,
                    unit,
                )
            )
    return out


def number(spots: list[WeakSpot]) -> list[WeakSpot]:
    """Sort by severity, category, refs and assign W001..."""
    spots.sort(key=lambda s: (SEVERITY_ORDER[s.severity], s.category, s.refs, s.message))
    for i, s in enumerate(spots, 1):
        s.id = f"W{i:03d}"
    return spots


def find_weak_spots(
    analysis: Analysis,
    model: KinematicModel | None,
    ctx: KinContext | None,
    load: LoadInfo | None,
    stab: StabilityInfo | None,
) -> list[WeakSpot]:
    """All weak spots for a single part or an assembly, numbered W001..."""
    report = analysis.report
    spots: list[WeakSpot] = []
    asm = report.assembly
    on_path: set[str] = set()
    if ctx is not None and load is not None:
        for p in load.paths:
            on_path.update(x for x in p.path if x.startswith("INS"))
        part_of = {i.id: i.part_id for i in ctx.insts}
        on_path_parts = {part_of[i] for i in on_path if i in part_of}
    for ap in analysis.parts:
        part = ap.part
        label = f"{part.id} ({part.name})"
        spots += edge_distance(part, label)
        spots += cutout_web(part, label)
        spots += thin_wall(part, label)
        carrying = asm is None or part.id in on_path_parts
        spots += sharp_corners(part, ap.geom, label, carrying)
    if asm is not None and model is not None and ctx is not None and load is not None:
        spots += floating_parts(model, ctx)
        spots += interferences(asm)
        spots += single_fasteners(load, ctx)
        spots += cantilevers(load, ctx, asm)
        spots += short_screws(asm, ctx)
        spots += joint_play(model, ctx)
        spots += motion_limits(model)
        pos: dict[str, list[np.ndarray]] = {}
        for j in asm.fastener_joints:
            o = np.array([j.axis.origin.x, j.axis.origin.y, j.axis.origin.z])
            for iid in j.instance_ids:
                pos.setdefault(iid, []).append(o)
        for inst in ctx.insts:
            if inst.id in pos:
                lo, hi = inst.bounds
                spots += unsupported_span(
                    inst.ap.part, f"{inst.id} ({inst.name})", np.array(pos[inst.id]), (lo, hi)
                )
    spots += tipping(stab)
    return number(spots)
