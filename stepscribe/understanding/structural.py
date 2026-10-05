# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Structural features from the attributed adjacency graph.

Base body, ribs, gussets, steps, bends, flanges, lightening cutouts and standoff walls.
Every rule is conservative: a miss is better than a false hit, and each feature carries the
face IDs it was built from so a reader can trace it to measured facts.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.BRepGProp import BRepGProp
from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_WIRE

from stepscribe.describe.phrases import fmt
from stepscribe.geometry.occ_utils import as_face, as_wire, unique_subshapes
from stepscribe.geometry.part_geom import PartGeom
from stepscribe.models.schema import Part, StructuralFeature
from stepscribe.understanding.aag import (
    AAG,
    AAGEdge,
    FaceAttr,
    angle_deg,
    face_vertices,
    parallel,
    perpendicular,
    plane_key,
)

RIB_MAX_T_RATIO = 0.35  # thickness < this x min(height, length)
GUSSET_MIN_DEG, GUSSET_MAX_DEG = 60.0, 120.0
STEP_MIN_MM = 0.2
COPLANAR_TOL = 0.05  # mm, same plane
BEND_THICKNESS_TOL = 0.12  # relative error allowed on R_out - R_in = thickness
STANDOFF_RATIO = 2.0
LIGHTENING_MIN_AREA_FRACTION = 0.01  # opening must remove >= 1% of the plate area
MAX_FEATURES = 60


@dataclass
class Plane:
    """A group of coplanar faces (same outward normal and offset)."""

    key: tuple[float, ...]
    normal: np.ndarray
    offset: float
    faces: list[int]
    area: float


def planes_of(aag: AAG) -> list[Plane]:
    groups: dict[tuple[float, ...], list[FaceAttr]] = defaultdict(list)
    for a in aag.planes():
        groups[plane_key(a, 1)].append(a)
    out = []
    for key, fs in groups.items():
        n = fs[0].normal
        assert n is not None
        off = float(np.mean([f.offset for f in fs if f.offset is not None]))
        out.append(Plane(key, n, off, [f.index for f in fs], sum(f.area for f in fs)))
    out.sort(key=lambda p: (-p.area, p.key))
    return out


def _same_plane(a: FaceAttr, b: FaceAttr) -> bool:
    return (
        a.normal is not None
        and b.normal is not None
        and float(np.dot(a.normal, b.normal)) > 0.999
        and abs((a.offset or 0.0) - (b.offset or 0.0)) < COPLANAR_TOL
    )


def base_body(part: Part, aag: AAG) -> str | None:
    """One-line description of the dominant body ("plate 60.0 x 60.0 x 4.0 mm")."""
    label = part.shape_class.label
    size = " x ".join(fmt(v) for v in sorted(part.obb.size_sorted, reverse=True))
    if label in ("plate", "block", "bar", "disc", "ring", "tube", "shaft", "housing"):
        return f"{label} {size} mm"
    if label in ("l_bracket", "u_bracket", "z_bracket", "angle"):
        t = part.shape_class.thickness_mm
        extra = f", thickness {fmt(t)} mm" if t else ""
        return f"{label.replace('_', '-')} {size} mm{extra}"
    planes = planes_of(aag)
    if planes:
        return f"body {size} mm dominated by a flat face of {fmt(planes[0].area, 0)} mm²"
    return f"body {size} mm"


def _vertex_extent(geom: PartGeom, faces: list[int], direction: np.ndarray) -> float:
    pts = np.vstack([face_vertices(geom, i) for i in faces if len(face_vertices(geom, i))])
    if not len(pts):
        return 0.0
    proj = pts @ direction
    return float(proj.max() - proj.min())


def detect_ribs_and_gussets(aag: AAG, ids: dict[str, int]) -> list[StructuralFeature]:
    """Thin walls standing on a larger face (ribs) or bridging two faces at 60-120 deg (gussets)."""
    geom = aag.geom
    planes_attr = aag.planes()
    found: list[StructuralFeature] = []
    used: set[int] = set()
    for a in planes_attr:
        for b in planes_attr:
            if b.index <= a.index or a.normal is None or b.normal is None:
                continue
            if float(np.dot(a.normal, b.normal)) > -0.999:
                continue
            t = float((a.offset or 0.0) + (b.offset or 0.0))  # normals are anti-parallel
            # a faces away from b only if a's plane lies further along a.normal than b's
            if t <= 0.05:
                continue
            if a.index in used or b.index in used:
                continue
            feat = _rib_candidate(aag, a, b, t)
            if feat is None:
                continue
            used.update({a.index, b.index})
            found.append(feat)
    for f in found:
        ids["S"] += 1
        f.id = f"S{ids['S']:03d}"
    del geom
    return found


def _concave_planar_neighbours(aag: AAG, a: FaceAttr) -> list[tuple[FaceAttr, AAGEdge]]:
    out = []
    for e in aag.edges_of_type(a.index, "concave"):
        n = aag.attrs[e.other(a.index)]
        if n.kind == "plane" and n.normal is not None and perpendicular(n.normal, a.normal):  # type: ignore[arg-type]
            out.append((n, e))
    return out


def _rib_candidate(aag: AAG, a: FaceAttr, b: FaceAttr, t: float) -> StructuralFeature | None:
    assert a.normal is not None and b.normal is not None
    # the two faces must be adjacent to a common convex "top" face (or be bridged by one)
    ca = _concave_planar_neighbours(aag, a)
    cb = _concave_planar_neighbours(aag, b)
    if not ca or not cb:
        return None
    # supports: planes that both faces meet concavely
    common: list[tuple[FaceAttr, FaceAttr, AAGEdge, AAGEdge]] = []
    for sa, ea in ca:
        for sb, eb in cb:
            if _same_plane(sa, sb) or sa.index == sb.index:
                common.append((sa, sb, ea, eb))
    if not common:
        return None
    # distinct support planes
    reps: list[tuple[FaceAttr, AAGEdge]] = []
    for sa, _sb, ea, _eb in common:
        if not any(_same_plane(sa, r[0]) for r in reps):
            reps.append((sa, ea))
    top = _common_top(aag, a, b)
    verts_a = face_vertices(aag.geom, a.index)
    if not len(verts_a):
        return None
    first = reps[0][0]
    assert first.normal is not None
    height = float(np.max((verts_a @ first.normal) - (first.offset or 0.0)))
    length = float(max(e.length for _s, e in reps))
    if height <= 0 or length <= 0:
        return None
    if t >= RIB_MAX_T_RATIO * min(height, length):
        return None
    faces = [a.id, b.id] + ([aag.attrs[top].id] if top is not None else [])
    supports = [r[0].id for r in reps]
    if len(reps) >= 2:
        s1, s2 = reps[0][0], reps[1][0]
        assert s1.normal is not None and s2.normal is not None
        between = 180.0 - angle_deg(s1.normal, s2.normal)
        if GUSSET_MIN_DEG <= between <= GUSSET_MAX_DEG:
            leg1, leg2 = reps[0][1].length, reps[1][1].length
            return StructuralFeature(
                id="",
                kind="gusset",
                face_ids=faces,
                thickness_mm=t,
                height_mm=height,
                length_mm=max(leg1, leg2),
                angle_deg=between,
                supports=supports,
                description=(
                    f"gusset {fmt(t, 2)} mm thick bridging faces {s1.id} and {s2.id} "
                    f"(which meet at {fmt(between, 0)} deg), legs {fmt(leg1)} and {fmt(leg2)} mm"
                ),
                confidence=0.85,
            )
    return StructuralFeature(
        id="",
        kind="rib",
        face_ids=faces,
        thickness_mm=t,
        height_mm=height,
        length_mm=length,
        supports=supports,
        description=(
            f"rib {fmt(t, 2)} mm thick, {fmt(height)} mm high, {fmt(length)} mm long, "
            f"standing on face {supports[0]}"
        ),
        confidence=0.85,
    )


def _common_top(aag: AAG, a: FaceAttr, b: FaceAttr) -> int | None:
    na = {e.other(a.index) for e in aag.edges_of_type(a.index, "convex")}
    nb = {e.other(b.index) for e in aag.edges_of_type(b.index, "convex")}
    both = sorted(na & nb)
    return both[0] if both else None


def detect_steps(aag: AAG, exclude: set[str], ids: dict[str, int]) -> list[StructuralFeature]:
    """Two parallel, same-facing planes joined by a wall with one convex and one concave edge."""
    found: list[StructuralFeature] = []
    seen: set[tuple[int, int]] = set()
    refs = _reference_normals(aag)
    for w in aag.planes():
        if w.id in exclude or w.feature:
            continue
        if not any(perpendicular(w.normal, r) for r in refs):  # type: ignore[arg-type]
            continue
        convex = aag.edges_of_type(w.index, "convex")
        concave = aag.edges_of_type(w.index, "concave")
        for ec in concave:
            low = aag.attrs[ec.other(w.index)]
            if low.kind != "plane" or low.normal is None or w.normal is None:
                continue
            if not perpendicular(low.normal, w.normal):
                continue
            for ex in convex:
                high = aag.attrs[ex.other(w.index)]
                if high.kind != "plane" or high.normal is None or high.index == low.index:
                    continue
                if float(np.dot(high.normal, low.normal)) < 0.999:
                    continue
                if not any(parallel(low.normal, r) for r in refs):
                    continue
                h = float((high.offset or 0.0) - (low.offset or 0.0))
                if h < STEP_MIN_MM:
                    continue
                key = (high.index, low.index)
                if key in seen:
                    continue
                if _face_width(aag, high) < RIB_MAX_T_RATIO * h:
                    continue  # a thin wall standing up (flange, rib), not a tread
                if _is_enclosed(aag, w):
                    continue  # pocket walls are reported as pockets
                seen.add(key)
                ids["S"] += 1
                found.append(
                    StructuralFeature(
                        id=f"S{ids['S']:03d}",
                        kind="step",
                        face_ids=[high.id, w.id, low.id],
                        height_mm=h,
                        length_mm=max(ec.length, ex.length),
                        supports=[low.id],
                        description=(
                            f"step {fmt(h, 2)} mm high between faces {high.id} (upper) and "
                            f"{low.id} (lower), wall {w.id}"
                        ),
                        confidence=0.8,
                    )
                )
    return found


def _face_width(aag: AAG, a: FaceAttr) -> float:
    """Rough width of a planar face: area over its longest edge."""
    longest = max((e.length for e in aag.adj.get(a.index, [])), default=0.0)
    return a.area / longest if longest > 0 else 0.0


def _reference_normals(aag: AAG) -> list[np.ndarray]:
    """Normals of the dominant flat faces: treads of a step are parallel to one of them."""
    planes = planes_of(aag)
    if not planes:
        return []
    top = planes[0].area
    refs: list[np.ndarray] = []
    for p in planes:
        if p.area >= 0.5 * top and not any(parallel(p.normal, r) for r in refs):
            refs.append(p.normal)
    return refs


def _is_enclosed(aag: AAG, w: FaceAttr) -> bool:
    """A wall whose every neighbour edge is concave or tangent-closed is a pocket wall."""
    kinds = [e.kind for e in aag.adj.get(w.index, [])]
    convex = sum(k == "convex" for k in kinds)
    concave = sum(k == "concave" for k in kinds)
    return concave >= 3 and convex <= 1


def detect_bends(aag: AAG, ids: dict[str, int], thickness: float | None) -> list[StructuralFeature]:
    """Coaxial convex + concave cylinders whose radii differ by the sheet thickness."""
    cyls = [a for a in aag.attrs if a.kind == "cylinder" and a.axis_dir is not None]
    found: list[StructuralFeature] = []
    used: set[int] = set()
    for inner in cyls:
        if inner.convex is not False or inner.index in used:
            continue
        for outer in cyls:
            if outer.convex is not True or outer.index in used:
                continue
            assert inner.axis_dir is not None and outer.axis_dir is not None
            assert inner.axis_point is not None and outer.axis_point is not None
            if not parallel(inner.axis_dir, outer.axis_dir):
                continue
            delta = outer.axis_point - inner.axis_point
            off = delta - inner.axis_dir * float(np.dot(delta, inner.axis_dir))
            if np.linalg.norm(off) > 0.05:
                continue
            assert inner.radius is not None and outer.radius is not None
            t = outer.radius - inner.radius
            if t <= 0.05:
                continue
            if thickness is not None and abs(t - thickness) > BEND_THICKNESS_TOL * thickness:
                continue
            if not _smooth_to_planes(aag, inner) or not _smooth_to_planes(aag, outer):
                continue
            used.update({inner.index, outer.index})
            angle = _arc_angle(aag, inner)
            ids["S"] += 1
            found.append(
                StructuralFeature(
                    id=f"S{ids['S']:03d}",
                    kind="bend",
                    face_ids=[inner.id, outer.id],
                    thickness_mm=t,
                    angle_deg=angle,
                    radius_mm=inner.radius,
                    supports=sorted(
                        {aag.attrs[j].id for f in (inner, outer) for j in aag.neighbors(f.index)}
                    ),
                    description=(
                        f"bend of {fmt(angle, 0)} deg, inner radius {fmt(inner.radius, 2)} mm, "
                        f"sheet {fmt(t, 2)} mm (faces {inner.id} inner, {outer.id} outer)"
                    ),
                    confidence=0.9 if thickness else 0.75,
                )
            )
            break
    return found


def _smooth_to_planes(aag: AAG, a: FaceAttr) -> bool:
    n = [aag.attrs[e.other(a.index)] for e in aag.edges_of_type(a.index, "smooth")]
    return sum(1 for f in n if f.kind == "plane") >= 2


def _arc_angle(aag: AAG, cyl: FaceAttr) -> float:
    """Bend angle: angle between the two planar faces tangent to the cylinder."""
    flats = [
        aag.attrs[e.other(cyl.index)]
        for e in aag.edges_of_type(cyl.index, "smooth")
        if aag.attrs[e.other(cyl.index)].kind == "plane"
    ]
    if len(flats) >= 2 and flats[0].normal is not None and flats[1].normal is not None:
        return angle_deg(flats[0].normal, flats[1].normal)
    return 90.0


def detect_flanges(
    aag: AAG, bends: list[StructuralFeature], ids: dict[str, int]
) -> list[StructuralFeature]:
    """Planar legs joined through bends: one flange per connected flat region."""
    if not bends:
        return []
    bend_faces = {fid for b in bends for fid in b.face_ids}
    idx_of = {a.id: a.index for a in aag.attrs}
    flat_regions: dict[int, list[int]] = {}
    seeds = []
    for b in bends:
        for fid in b.face_ids:
            for j in aag.neighbors(idx_of[fid]):
                if aag.attrs[j].kind == "plane":
                    seeds.append(j)
    comps = aag.connected_components(
        lambda a: a.kind == "plane" and a.id not in bend_faces,
        lambda e: e.kind in ("convex", "concave", "smooth") and False,
    )
    del flat_regions, comps
    found: list[StructuralFeature] = []
    done: set[int] = set()
    for j in sorted(set(seeds)):
        if j in done:
            continue
        a = aag.attrs[j]
        # partner face: the anti-parallel plane at the sheet thickness reached through a bend
        partner = None
        for b in bends:
            t = b.thickness_mm or 0.0
            for k in aag.planes():
                if k.index == j or k.normal is None or a.normal is None:
                    continue
                if float(np.dot(k.normal, a.normal)) < -0.999 and abs(
                    abs((a.offset or 0.0) + (k.offset or 0.0)) - t
                ) < 0.05 * max(t, 1.0):
                    partner = k
                    break
            if partner:
                break
        if partner is None:
            continue
        done.update({j, partner.index})
        ids["S"] += 1
        sheet_t = bends[0].thickness_mm
        found.append(
            StructuralFeature(
                id=f"S{ids['S']:03d}",
                kind="flange",
                face_ids=[a.id, partner.id],
                thickness_mm=sheet_t,
                supports=[b.id for b in bends],
                description=f"flat sheet leg (faces {a.id}/{partner.id}), {fmt(sheet_t or 0.0, 2)} mm thick, joined through bend(s)",
                confidence=0.75,
            )
        )
    return found


def _loop_area(wire) -> float:  # type: ignore[no-untyped-def]
    mk = BRepBuilderAPI_MakeFace(wire, True)
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(mk.Face(), props)
    return float(props.Mass())


def detect_lightening_cutouts(
    aag: AAG, part: Part, exclude_faces: set[str], ids: dict[str, int]
) -> list[StructuralFeature]:
    """Through openings in a plate-like part that are not screw holes or slots."""
    if part.shape_class.label not in (
        "plate",
        "disc",
        "block",
        "l_bracket",
        "u_bracket",
        "other",
        "bar",
    ):
        return []
    thickness = part.shape_class.thickness_mm or min(part.obb.size_sorted)
    mains = [a for a in aag.planes() if a.area > 0.25 * part.mass.surface_area_mm2 / 2.5]
    if not mains:
        return []
    main = max(mains, key=lambda a: a.area)
    face = as_face(aag.geom.table.faces[main.index].face)
    outer = BRepTools.OuterWire_s(face)
    wires = [as_wire(w) for w in unique_subshapes(face, TopAbs_WIRE)]
    clearance = [
        h.diameter_mm
        for h in part.holes
        if any(m.fit.startswith("clearance") for m in h.standard_matches)
    ]
    limit = 2.0 * max(clearance) if clearance else 12.0
    inner = []
    total_area = main.area
    for w in wires:
        if w.IsSame(outer):
            continue
        area = _loop_area(w)
        total_area += area
        inner.append((w, area))
    midpoints = _loop_midpoints(face, [w for w, _a in inner])
    found: list[StructuralFeature] = []
    for (_w, area), mids in zip(inner, midpoints, strict=True):
        wall_ids = _walls_for(aag, main.index, mids)
        if not wall_ids or any(aag.attrs[j].id in exclude_faces for j in wall_ids):
            continue
        circular = all(aag.attrs[j].kind == "cylinder" for j in wall_ids) and len(wall_ids) <= 2
        if circular:
            diam = 2.0 * math.sqrt(area / math.pi)
            if diam <= limit:
                continue
        if area < LIGHTENING_MIN_AREA_FRACTION * total_area:
            continue
        depth = _vertex_extent(aag.geom, wall_ids, main.normal) if main.normal is not None else 0.0
        if thickness and depth < 0.9 * thickness:
            continue  # blind: a pocket, not a through opening
        ids["S"] += 1
        found.append(
            StructuralFeature(
                id=f"S{ids['S']:03d}",
                kind="lightening_cutout",
                face_ids=[aag.attrs[j].id for j in wall_ids],
                thickness_mm=thickness,
                length_mm=None,
                supports=[main.id],
                description=(
                    f"through opening of {fmt(area, 0)} mm² in face {main.id} "
                    f"({fmt(100 * area / total_area, 1)}% of the plate area removed)"
                ),
                confidence=0.8,
            )
        )
    return found


def _loop_midpoints(face, wires) -> list[list[np.ndarray]]:  # type: ignore[no-untyped-def]
    from OCP.BRepAdaptor import BRepAdaptor_Curve

    from stepscribe.geometry.occ_utils import to_np

    out = []
    for w in wires:
        pts = []
        ex = BRepTools_WireExplorer(w, face)
        while ex.More():
            c = BRepAdaptor_Curve(ex.Current())
            pts.append(to_np(c.Value(0.5 * (c.FirstParameter() + c.LastParameter()))))
            ex.Next()
        out.append(pts)
    return out


def _walls_for(aag: AAG, main_idx: int, mids: list[np.ndarray]) -> list[int]:
    walls: set[int] = set()
    for e in aag.adj.get(main_idx, []):
        if any(float(np.linalg.norm(e.midpoint - m)) < 1e-5 for m in mids):
            walls.add(e.other(main_idx))
    return sorted(walls)


def detect_standoff_walls(part: Part, ids: dict[str, int]) -> list[StructuralFeature]:
    out = []
    for b in part.bosses:
        if b.diameter_mm > 0 and b.height_mm > STANDOFF_RATIO * b.diameter_mm:
            ids["S"] += 1
            out.append(
                StructuralFeature(
                    id=f"S{ids['S']:03d}",
                    kind="standoff_wall",
                    face_ids=list(b.face_ids),
                    height_mm=b.height_mm,
                    thickness_mm=b.diameter_mm,
                    supports=[],
                    description=(
                        f"standoff: boss {b.id} dia {fmt(b.diameter_mm)} mm, "
                        f"{fmt(b.height_mm)} mm tall (more than {STANDOFF_RATIO:g}x its diameter)"
                    ),
                    confidence=0.85,
                )
            )
    return out


def structural_features(aag: AAG, part: Part) -> list[StructuralFeature]:
    """All structural features of *part*, in a deterministic order, IDs S001...."""
    ids = {"S": 0}
    hole_faces = {fid for h in part.holes for s in h.segments for fid in s.face_ids}
    slot_faces = {fid for s in part.slots for fid in s.face_ids}
    pocket_faces = {p.floor_face_id for p in part.pockets}
    taken = hole_faces | slot_faces | pocket_faces
    thickness = part.shape_class.thickness_mm
    feats: list[StructuralFeature] = []
    bends = detect_bends(aag, ids, thickness or part.min_wall_thickness_mm)
    feats += bends
    feats += detect_flanges(aag, bends, ids)
    bend_faces = {fid for b in bends for fid in b.face_ids}
    ribs = detect_ribs_and_gussets(aag, ids)
    feats += ribs
    idx_of = {a.id: a.index for a in aag.attrs}
    rib_walls = {fid for r in ribs for fid in r.face_ids}
    for r in ribs:  # end faces and the sloped edge of a rib are part of it, not steps
        for fid in r.face_ids[:2]:
            for e in aag.edges_of_type(idx_of[fid], "convex"):
                rib_walls.add(aag.attrs[e.other(idx_of[fid])].id)
    feats += detect_steps(aag, taken | bend_faces | rib_walls, ids)
    feats += detect_lightening_cutouts(aag, part, taken, ids)
    feats += detect_standoff_walls(part, ids)
    # renumber in kind order for stable output
    order = {"bend": 0, "flange": 1, "rib": 2, "gusset": 3, "step": 4, "lightening_cutout": 5}
    feats.sort(key=lambda f: (order.get(f.kind, 9), f.face_ids))
    for i, f in enumerate(feats[:MAX_FEATURES], 1):
        f.id = f"S{i:03d}"
    return feats[:MAX_FEATURES]
