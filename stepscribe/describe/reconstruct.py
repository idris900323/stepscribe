# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Build-recipe text: enough measured geometry for a reader to redraw a part.

Everything here is a measured fact in the part's own coordinate system (no inference):
the box frame, the outline of the main flat face as lines and arcs, the thickness, the
position of every feature in that frame, and for turned parts the stepped diameter profile.
"""

from __future__ import annotations

import math

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Line
from OCP.TopAbs import TopAbs_REVERSED, TopAbs_WIRE

from stepscribe.describe.phrases import fmt
from stepscribe.geometry.occ_utils import (
    Vec,
    as_face,
    as_wire,
    bbox_of,
    interior_uv,
    outward_normal,
    surface_point,
    to_np,
    unique_subshapes,
)
from stepscribe.geometry.part_geom import PartGeom
from stepscribe.geometry.surfaces import cone_params, cylinder_params, plane_normal_origin
from stepscribe.models.schema import Part

MAX_LOOP_SEGMENTS = 40
Seg = tuple[str, Vec, Vec, tuple[Vec, float, float] | None]
TURNED = {"shaft", "tube", "ring", "disc"}


def _vec(v: Vec, nd: int = 3) -> str:
    return f"({', '.join(fmt(float(c), nd) for c in v)})"


def _pt(uv: tuple[float, float]) -> str:
    return f"({fmt(uv[0], 2)}, {fmt(uv[1], 2)})"


def _frame(part: Part, n: Vec) -> tuple[Vec, Vec]:
    """Unit in-plane axes (u, v): u follows the longest box axis that lies in the face plane."""
    axes = [np.array([a.x, a.y, a.z]) for a in part.obb.axes]
    hs = [part.obb.half_sizes.x, part.obb.half_sizes.y, part.obb.half_sizes.z]
    order = sorted(range(3), key=lambda i: -hs[i])
    u = None
    for i in order:
        cand = axes[i] - n * float(np.dot(axes[i], n))
        if np.linalg.norm(cand) > 0.5:
            u = cand / np.linalg.norm(cand)
            break
    if u is None:
        ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = ref - n * float(np.dot(ref, n))
        u = u / np.linalg.norm(u)
    first = next((c for c in u if abs(c) > 1e-6), 1.0)
    if first < 0:
        u = -u
    v = np.cross(n, u)
    return u, v


def _edge_points(edge, face) -> tuple[BRepAdaptor_Curve, Vec, Vec, bool]:  # type: ignore[no-untyped-def]
    c = BRepAdaptor_Curve(edge)
    p0, p1 = to_np(c.Value(c.FirstParameter())), to_np(c.Value(c.LastParameter()))
    if edge.Orientation() == TopAbs_REVERSED:
        p0, p1 = p1, p0
    return c, p0, p1, edge.Orientation() == TopAbs_REVERSED


def outline(part: Part, geom: PartGeom) -> list[str] | None:
    """Markdown lines describing the largest flat face as outline + openings + thickness."""
    flats = [f for f in geom.table.faces if f.kind == "plane"]
    if not flats:
        return None
    main = max(flats, key=lambda f: (round(f.area, 3), -f.centroid[0]))
    if main.area < 0.15 * part.mass.surface_area_mm2:
        return None
    n, o = plane_normal_origin(main.face)
    u, v = _frame(part, n)
    lo, hi = geom.bounds
    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )
    along = corners @ n - float(np.dot(o, n))
    sign = 1.0 if abs(along.max()) >= abs(along.min()) else -1.0
    thickness = float(abs(along.max() - along.min()))
    wn = n * sign

    wires = [as_wire(w) for w in unique_subshapes(main.face, TopAbs_WIRE)]
    outer = BRepTools.OuterWire_s(as_face(main.face))
    loops: list[tuple[bool, list[Seg]]] = []
    extent_pts: list[Vec] = []  # points along the outer wire, so arcs that bulge count
    for w in wires:
        segs: list[Seg] = []
        ex = BRepTools_WireExplorer(w, as_face(main.face))
        while ex.More():
            edge = ex.Current()
            c, p0, p1, _rev = _edge_points(edge, main.face)
            kind = c.GetType()
            if w.IsSame(outer):
                f0, f1 = c.FirstParameter(), c.LastParameter()
                extent_pts += [to_np(c.Value(f0 + (f1 - f0) * k / 24)) for k in range(25)]
            if kind == GeomAbs_Line:
                segs.append(("line", p0, p1, None))
            elif kind == GeomAbs_Circle:
                circ = c.Circle()
                sweep = abs(c.LastParameter() - c.FirstParameter())
                segs.append(("arc", p0, p1, (to_np(circ.Location()), float(circ.Radius()), sweep)))
            else:
                mid = to_np(c.Value(0.5 * (c.FirstParameter() + c.LastParameter())))
                segs.append(("curve", p0, p1, (mid, 0.0, 0.0)))
            ex.Next()
        loops.append((w.IsSame(outer), segs))

    def uv(p: Vec) -> tuple[float, float]:
        return float(np.dot(p - o, u)), float(np.dot(p - o, v))

    outer_pts = [uv(p) for p in extent_pts] or [uv(s[1]) for lp in loops for s in lp[1]]
    u0 = min(p[0] for p in outer_pts)
    v0 = min(p[1] for p in outer_pts)

    def rel(p: Vec) -> tuple[float, float]:
        a, b = uv(p)
        return a - u0, b - v0

    lines = [
        f"Main face: the largest flat face ({fmt(main.area, 1)} mm², normal {_vec(n)}).",
        f"Frame: u = {_vec(u)}, v = {_vec(v)}, w = {_vec(wn)} (w points from the main face into the part).",
        f"The part extends {fmt(thickness, 2)} mm along w. Origin of (u, v) = lowest-u, lowest-v corner of the outline; w = 0 on the main face.",
    ]
    circles = 0
    by_loop = _cutout_labels(part, main.id, loops, rel)
    for is_outer, segs in sorted(loops, key=lambda lp: (not lp[0],)):
        if (
            not is_outer
            and len(segs) == 1
            and segs[0][0] == "arc"
            and segs[0][3]
            and segs[0][3][2] > 6.2
        ):
            circles += 1
            continue
        title = "Outer outline" if is_outer else "Cut-out outline (opening through the main face)"
        if not is_outer and id(segs) in by_loop:
            title = f"{by_loop[id(segs)]} outline (opening in the main face)"
        rows = []
        for kind, p0, p1, extra in segs[:MAX_LOOP_SEGMENTS]:
            a, b = rel(p0), rel(p1)
            if kind == "line":
                length = float(np.linalg.norm(p1 - p0))
                rows.append(f"line {_pt(a)} -> {_pt(b)} (length {fmt(length, 2)})")
            elif kind == "arc" and extra:
                centre, r, sweep = extra
                rows.append(
                    f"arc {_pt(a)} -> {_pt(b)}, centre {_pt(rel(centre))}, R{fmt(r, 2)}, sweep {fmt(math.degrees(sweep), 1)} deg"
                )
            elif extra:
                rows.append(f"curve {_pt(a)} -> {_pt(b)} via {_pt(rel(extra[0]))}")
        more = (
            f" (+{len(segs) - MAX_LOOP_SEGMENTS} more segments)"
            if len(segs) > MAX_LOOP_SEGMENTS
            else ""
        )
        lines.append(f"{title}, {len(segs)} segment(s), listed in order, (u, v) in mm{more}:")
        lines.extend(f"  {i}. {r}" for i, r in enumerate(rows, 1))
    if circles:
        lines.append(
            f"{circles} circular opening(s) in the main face: see the hole table (positions below)."
        )

    def frame_pos(p: Vec) -> str:
        a, b = rel(p)
        c = float(np.dot(p - o, wn))
        return f"(u {fmt(a, 2)}, v {fmt(b, 2)}, w {fmt(c, 2)})"

    feats = []
    for h in part.holes:
        feats.append(
            f"{h.id} hole dia {fmt(h.diameter_mm, 2)} at {frame_pos(np.array([h.axis.origin.x, h.axis.origin.y, h.axis.origin.z]))}"
        )
    for s in part.slots:
        feats.append(
            f"{s.id} slot {fmt(s.width_mm, 2)} x {fmt(s.length_mm, 2)} at {frame_pos(np.array([s.center.x, s.center.y, s.center.z]))}"
        )
    for p in part.pockets:
        try:
            feats.append(
                f"{p.id} pocket {fmt(p.outline_size[0], 2)} x {fmt(p.outline_size[1], 2)}, depth {fmt(p.depth_mm, 2)}, floor centre {frame_pos(geom.table.by_id(p.floor_face_id).centroid)}"
            )
        except (IndexError, ValueError):
            continue
    for c in part.cutouts:
        if c.host_face_id != main.id:
            continue
        lv = c.levels[0]
        feats.append(
            f"{c.id} {c.description}; first level centre (u {fmt(lv.center_uv[0], 2)}, v {fmt(lv.center_uv[1], 2)})"
        )
    for bo in part.bosses:
        try:
            feats.append(
                f"{bo.id} boss dia {fmt(bo.diameter_mm, 2)} height {fmt(bo.height_mm, 2)} at {frame_pos(geom.table.by_id(bo.face_ids[0]).centroid)}"
            )
        except (IndexError, ValueError):
            continue
    if feats:
        lines.append(
            "Feature positions in this frame (u, v, w in mm; hole positions are where the hole enters the part):"
        )
        lines.extend(f"  - {f}" for f in feats)
    return lines


def _cutout_labels(part: Part, main_id: str, loops, rel) -> dict[int, str]:  # type: ignore[no-untyped-def]
    """Map an inner loop (by id of its segment list) to the cut-out whose first level it bounds."""
    cuts = [c for c in part.cutouts if c.host_face_id == main_id and c.kind != "notch"]
    out: dict[int, str] = {}
    taken: set[str] = set()
    for is_outer, segs in loops:
        if is_outer or not segs:
            continue
        pts = np.array([rel(s[1]) for s in segs])
        lo, hi = pts.min(axis=0) - 0.05, pts.max(axis=0) + 0.05
        for c in cuts:
            cu, cv = c.levels[0].center_uv
            if c.id not in taken and lo[0] <= cu <= hi[0] and lo[1] <= cv <= hi[1]:
                out[id(segs)] = f"{c.id} ({c.kind})"
                taken.add(c.id)
                break
    return out


def turned_profile(part: Part, geom: PartGeom) -> list[str] | None:
    """Stepped diameter profile along the main axis for shafts, tubes, discs and rings."""
    axes = [np.array([a.x, a.y, a.z]) for a in part.obb.axes]
    cyl_area = sum(f.area for f in geom.table.faces if f.kind in ("cylinder", "cone"))
    if part.shape_class.label not in TURNED and cyl_area < 0.6 * part.mass.surface_area_mm2:
        return None
    # the axis: the box axis along which most cylinders run
    best = None
    for a in axes:
        n = 0.0
        for f in geom.table.faces:
            if f.kind == "cylinder":
                c = cylinder_params(f.face)
                if abs(float(np.dot(c.direction, a))) > 0.999:
                    n += f.area
        if best is None or n > best[0]:
            best = (n, a)
    if best is None or best[0] <= 0:
        return None
    axis = best[1]
    first = next((c for c in axis if abs(c) > 1e-6), 1.0)
    axis = axis if first > 0 else -axis
    lo, hi = geom.bounds
    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )
    z_min = float((corners @ axis).min())
    rows: list[tuple[float, float, str]] = []
    for f in geom.table.faces:
        if f.kind not in ("cylinder", "cone"):
            continue
        fl, fh = bbox_of(f.face)
        fc = (
            np.array(
                [[x, y, z] for x in (fl[0], fh[0]) for y in (fl[1], fh[1]) for z in (fl[2], fh[2])]
            )
            @ axis
        )
        z0, z1 = float(fc.min()) - z_min, float(fc.max()) - z_min
        u_, v_ = interior_uv(f.face)
        nrm = outward_normal(f.face, u_, v_)
        pt = surface_point(f.face, u_, v_)
        if f.kind == "cylinder":
            c = cylinder_params(f.face)
            if abs(float(np.dot(c.direction, axis))) < 0.999:
                continue
            radial = pt - c.origin - c.direction * float(np.dot(pt - c.origin, c.direction))
            outer = nrm is None or float(np.dot(nrm, radial)) > 0
            rows.append((z0, z1, f"{'outer' if outer else 'bore'} dia {fmt(2 * c.radius, 2)}"))
        else:
            cp = cone_params(f.face)
            if abs(float(np.dot(cp.direction, axis))) < 0.999:
                continue
            ang = fmt(abs(math.degrees(cp.semi_angle)), 1)
            rows.append(
                (
                    z0,
                    z1,
                    f"cone (taper), half-angle {ang} deg, radius {fmt(cp.ref_radius, 2)} at its reference plane",
                )
            )
    if not rows:
        return None
    rows.sort(key=lambda r: (round(r[0], 2), round(r[1], 2), r[2]))
    merged: list[tuple[float, float, str]] = []
    for r in rows:
        if merged and merged[-1][2] == r[2] and abs(merged[-1][1] - r[0]) < 0.05:
            merged[-1] = (merged[-1][0], r[1], r[2])
        elif (
            merged
            and merged[-1][2] == r[2]
            and abs(merged[-1][0] - r[0]) < 0.05
            and abs(merged[-1][1] - r[1]) < 0.05
        ):
            continue
        else:
            merged.append(r)
    length = float((corners @ axis).max() - z_min)
    out = [
        f"Turned part: axis direction {_vec(axis)}, overall length {fmt(length, 2)} mm along it (z = 0 at the lowest end).",
        "Diameter steps along z (mm), outer surfaces and bores:",
    ]
    out.extend(f"  - z {fmt(a, 2)} to {fmt(b, 2)}: {t}" for a, b, t in merged[:60])
    return out


MAX_FACES_LISTED = 28


def face_inventory(part: Part, geom: PartGeom) -> list[str]:
    """The largest faces with positions in the oriented-box frame (origin = box corner)."""
    ob = part.obb
    axes = [np.array([a.x, a.y, a.z]) for a in ob.axes]
    half = [ob.half_sizes.x, ob.half_sizes.y, ob.half_sizes.z]
    centre = np.array([ob.center.x, ob.center.y, ob.center.z])
    corner = centre - sum(h * a for h, a in zip(half, axes, strict=True))

    def box(p: Vec) -> str:
        d = p - corner
        return f"({', '.join(fmt(float(np.dot(d, a)), 2) for a in axes)})"

    faces = sorted(geom.table.faces, key=lambda f: (-round(f.area, 3), f.id))[:MAX_FACES_LISTED]
    rows = []
    for f in faces:
        if f.kind == "plane":
            n, _o = plane_normal_origin(f.face)
            rows.append(
                f"{f.id} plane, area {fmt(f.area, 1)}, normal {_vec(n, 2)}, centre {box(f.centroid)}"
            )
        elif f.kind == "cylinder":
            c = cylinder_params(f.face)
            rows.append(
                f"{f.id} cylinder R{fmt(c.radius, 2)}, area {fmt(f.area, 1)}, axis {_vec(c.direction, 2)}, centre {box(f.centroid)}"
            )
        elif f.kind == "cone":
            rows.append(f"{f.id} cone, area {fmt(f.area, 1)}, centre {box(f.centroid)}")
        else:
            rows.append(f"{f.id} {f.kind} surface, area {fmt(f.area, 1)}, centre {box(f.centroid)}")
    head = (
        f"Largest {len(rows)} of {len(geom.table.faces)} faces, positions (a, b, c) in mm along the "
        "oriented-box axes listed above, measured from the box corner that has the lowest a, b and c:"
    )
    return [head, *[f"  - {r}" for r in rows]]


def reconstruction(part: Part, geom: PartGeom) -> list[str]:
    """All build-recipe lines for one part (empty if nothing could be derived)."""
    lines: list[str] = []
    ob = part.obb
    lines.append(
        f"Oriented box: centre {_vec(np.array([ob.center.x, ob.center.y, ob.center.z]), 2)}, "
        f"half sizes {_vec(np.array([ob.half_sizes.x, ob.half_sizes.y, ob.half_sizes.z]), 2)} along axes "
        + "; ".join(_vec(np.array([a.x, a.y, a.z])) for a in ob.axes)
        + "."
    )
    bb = part.bbox
    lines.append(
        f"Axis-aligned bounds: min {_vec(np.array([bb.min.x, bb.min.y, bb.min.z]), 2)}, max {_vec(np.array([bb.max.x, bb.max.y, bb.max.z]), 2)}."
    )
    for fn in (turned_profile, outline):
        try:
            got = fn(part, geom)
        except Exception as exc:  # noqa: BLE001 - never lose the pack over a recipe detail
            got = [f"({fn.__name__} unavailable: {type(exc).__name__})"]
        if got:
            lines.extend(got)
            if fn is turned_profile:
                break
    try:
        lines.extend(face_inventory(part, geom))
    except Exception as exc:  # noqa: BLE001
        lines.append(f"(face inventory unavailable: {type(exc).__name__})")
    return lines
