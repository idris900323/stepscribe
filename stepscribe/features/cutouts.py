# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""General recognizer for openings, recesses, stepped cut-outs and edge notches.

Method, based on face loops instead of special cases:

1. *Host faces* are the large planar faces. Every inner wire of a host is the mouth of an
   opening into it; runs of the outer wire that dip inside the outline's convex hull are
   edge notches. Loops that belong to a detected hole are not cut-outs.
2. From the faces next to the mouth, the *cavity* is explored through wall faces (planes and
   cylinders parallel to the entry normal) and floors (planes parallel to the host below it)
   until it ends on the opposite face (through) or on its deepest floor (blind).
3. The cavity is cut by planes parallel to the host between consecutive floor depths. Each
   cut yields the outline of one level, which is measured in the host plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Line
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
from OCP.OCP.collections import (
    IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher as EdgeFaceMap,
)
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_WIRE
from OCP.TopExp import TopExp

from stepscribe import config
from stepscribe.features import cutout_geometry as cg
from stepscribe.features.cutout_geometry import Seg
from stepscribe.features.patterns import _grid, _linear
from stepscribe.geometry.occ_utils import (
    Vec,
    as_edge,
    as_face,
    as_wire,
    bbox_of,
    interior_uv,
    normal_at_point,
    outward_normal,
    to_np,
    unique_subshapes,
)
from stepscribe.geometry.part_geom import PartGeom
from stepscribe.geometry.properties import vec3
from stepscribe.geometry.surfaces import cylinder_params, plane_normal_origin
from stepscribe.models.schema import Cutout, CutoutLevel, CutoutPattern, Part

PROBE_MM = 0.02  # step used to test which side of a face is material


Found = tuple[Cutout, np.ndarray, "_Host"]
Piece = tuple[np.ndarray, int, Seg]  # one wire edge: polyline in (u, v), neighbour face, segment


@dataclass
class _Face:
    kind: str
    normal: Vec | None = None  # planes: solid-outward unit normal
    origin: Vec | None = None  # planes: a point on the plane
    axis: Vec | None = None  # cylinders: unit axis
    area: float = 0.0


@dataclass
class _Cavity:
    walls: set[int] = field(default_factory=set)
    floors: dict[int, float] = field(default_factory=dict)  # face index -> depth
    opposite: dict[int, float] = field(default_factory=dict)
    others: set[int] = field(default_factory=set)
    protrusion: bool = False  # explored faces rose above the host: not an opening
    draft_deg: float = 0.0


@dataclass
class _Host:
    idx: int
    n: Vec  # outward normal (points out of the material, toward the viewer of the mouth)
    o: Vec  # a point on the host plane
    u: Vec
    v: Vec
    outer: np.ndarray  # outer wire polyline in absolute (u, v)
    inner: list[tuple[np.ndarray, list[int], list[Piece]]]  # polyline, seed faces, wire pieces
    origin_uv: np.ndarray  # lowest-u, lowest-v corner of the outer outline
    outer_edges: list[Piece] = field(default_factory=list)


class _Detector:
    """Holds the shared lookups for one part."""

    def __init__(self, geom: PartGeom, part: Part) -> None:
        self.geom, self.part = geom, part
        self.table = geom.table
        emap = EdgeFaceMap()
        TopExp.MapShapesAndAncestors_s(geom.shape, TopAbs_EDGE, TopAbs_FACE, emap)
        self.emap = emap
        self.hole_faces = {
            int(fid[1:]) - 1 for h in part.holes for s in h.segments for fid in s.face_ids
        }
        self._faces: dict[int, _Face] = {}
        self.warnings: list[str] = []

    # ---- face lookup ------------------------------------------------------------------
    def face(self, i: int) -> _Face:
        if i in self._faces:
            return self._faces[i]
        fi = self.table.faces[i]
        info = _Face(fi.kind, area=fi.area)
        try:
            if fi.kind == "plane":
                u, v = interior_uv(fi.face)
                n = outward_normal(fi.face, u, v)
                gn, go = plane_normal_origin(fi.face)
                info.normal = (n if n is not None else gn) / np.linalg.norm(
                    n if n is not None else gn
                )
                info.origin = go
            elif fi.kind == "cylinder":
                cp = cylinder_params(fi.face)
                info.axis = cp.direction / np.linalg.norm(cp.direction)
        except Exception:  # noqa: BLE001 - degenerate faces stay unclassified
            info = _Face("other", area=fi.area)
        self._faces[i] = info
        return info

    def other_faces(self, edge, own: int) -> list[int]:  # type: ignore[no-untyped-def]
        """Indices of the faces other than *own* that share *edge*."""
        k = self.emap.FindIndex(edge)
        if k <= 0:
            return []
        out = []
        for f in self.emap.FindFromIndex(k):
            j = self.table.index_of(as_face(f))
            if j >= 0 and j != own and j not in out:
                out.append(j)
        return out

    # ---- hosts ------------------------------------------------------------------------
    def hosts(self) -> list[int]:
        planes = [(i, self.face(i)) for i, f in enumerate(self.table.faces) if f.kind == "plane"]
        if not planes:
            return []
        biggest = max(p.area for _i, p in planes)
        keep = [
            (i, p)
            for i, p in planes
            if p.area >= config.CUTOUT_HOST_AREA_REL * biggest and p.normal is not None
        ]
        keep.sort(key=lambda ip: (-round(ip[1].area, 3), _canon_rank(ip[1].normal), ip[0]))
        return [i for i, _p in keep]

    def build_host(self, idx: int) -> _Host | None:
        fi = self.table.faces[idx]
        info = self.face(idx)
        assert info.normal is not None and info.origin is not None
        n = info.normal
        u, v = _frame(self.part, n)
        o = info.origin
        outer_w = BRepTools.OuterWire_s(as_face(fi.face))
        wires = [as_wire(w) for w in unique_subshapes(fi.face, TopAbs_WIRE)]
        outer_pts = np.zeros((0, 2))
        outer_edges: list[Piece] = []
        inner: list[tuple[np.ndarray, list[int], list[Piece]]] = []
        for w in wires:
            pieces = self._wire_pieces(w, fi.face, idx, o, u, v)
            if not pieces:
                continue
            pts = np.vstack([p for p, _j, _s in pieces])
            if w.IsSame(outer_w):
                outer_pts = pts
                outer_edges = pieces
            else:
                seeds = sorted({j for _p, j, _s in pieces if j >= 0})
                inner.append((pts, seeds, pieces))
        if len(outer_pts) == 0:
            return None
        origin_uv = outer_pts.min(axis=0)
        return _Host(idx, n, o, u, v, outer_pts, inner, origin_uv, outer_edges)

    def _wire_pieces(self, wire, face, own: int, o: Vec, u: Vec, v: Vec):  # type: ignore[no-untyped-def]
        """Per edge: (polyline in (u, v), neighbour face index or -1, segment) in wire order."""
        pieces = []
        ex = BRepTools_WireExplorer(wire, face)
        while ex.More():
            edge = as_edge(ex.Current())
            c = BRepAdaptor_Curve(edge)
            rev = edge.Orientation() == TopAbs_REVERSED
            kind = c.GetType()
            f, last = c.FirstParameter(), c.LastParameter()
            if kind == GeomAbs_Line:
                params = [f, last]
            elif kind == GeomAbs_Circle:
                n = max(
                    2, int(math.ceil(math.degrees(abs(last - f)) / config.CUTOUT_ARC_STEP_DEG)) + 1
                )
                params = list(np.linspace(f, last, n))
            else:
                params = list(np.linspace(f, last, 48))
            pts3 = [to_np(c.Value(t)) for t in params]
            if rev:
                pts3 = pts3[::-1]
            uv = np.array([[float(np.dot(p - o, u)), float(np.dot(p - o, v))] for p in pts3])
            others = self.other_faces(edge, own)
            pieces.append((uv, others[0] if others else -1, self._seg_of(c, kind, rev, o, u, v)))
            ex.Next()
        return pieces

    def _seg_of(self, c, kind, rev: bool, o: Vec, u: Vec, v: Vec) -> Seg:  # type: ignore[no-untyped-def]
        f, last = c.FirstParameter(), c.LastParameter()
        p0, p1 = to_np(c.Value(f)), to_np(c.Value(last))
        if rev:
            p0, p1 = p1, p0

        def uv(p: Vec) -> np.ndarray:
            return np.array([float(np.dot(p - o, u)), float(np.dot(p - o, v))])

        if kind == GeomAbs_Line:
            return Seg("line", uv(p0), uv(p1))
        if kind == GeomAbs_Circle:
            circ = c.Circle()
            centre, r = uv(to_np(circ.Location())), float(circ.Radius())
            pm = uv(to_np(c.Value(0.5 * (f + last))))
            a, b = uv(p0), uv(p1)
            return Seg("arc", a, b, centre, r, cg.arc_sweep(a, pm, b, centre))
        return Seg("line", uv(p0), uv(p1))  # curve: chord (only used for classification)

    # ---- cavity exploration -----------------------------------------------------------
    def explore(self, host: _Host, seeds: set[int], excluded: set[int]) -> _Cavity:
        cav = _Cavity()
        d_host = float(np.dot(host.n, host.o))
        stack = sorted(seeds)
        seen = {host.idx} | set(stack)
        while stack:
            cur = stack.pop()
            cls = self._classify(cur, host, d_host, cav)
            if cls == "wall":
                cav.walls.add(cur)
            elif cls == "floor":
                pass
            else:
                continue
            for nb in sorted(self.table.faces[cur].neighbors):
                if nb in seen or nb in excluded or nb in self.hole_faces:
                    continue
                seen.add(nb)
                nb_cls = self._classify(nb, host, d_host, cav)
                if nb_cls in ("wall", "floor"):
                    if cls == "wall" or nb_cls == "wall":
                        stack.append(nb)
        return cav

    def _classify(self, i: int, host: _Host, d_host: float, cav: _Cavity) -> str:
        """'wall', 'floor' (recorded), 'opposite' (recorded), 'above' or 'other' (recorded)."""
        info = self.face(i)
        if info.kind == "plane" and info.normal is not None and info.origin is not None:
            dot = float(np.dot(info.normal, host.n))
            depth = d_host - float(np.dot(host.n, info.origin))
            if abs(dot) < config.CUTOUT_WALL_DOT:
                cav.draft_deg = max(cav.draft_deg, math.degrees(math.asin(min(1.0, abs(dot)))))
                return "wall"
            if dot > config.CUTOUT_FLAT_DOT:
                if depth > config.CUTOUT_MIN_STEP:
                    cav.floors[i] = depth
                    return "floor"
                if depth < -config.CUTOUT_MIN_STEP:
                    cav.protrusion = True
                return "above"
            if dot < -config.CUTOUT_FLAT_DOT:
                cav.opposite[i] = depth
                return "opposite"
            cav.others.add(i)
            return "other"
        if info.kind == "cylinder" and info.axis is not None:
            if abs(float(np.dot(info.axis, host.n))) > config.CUTOUT_AXIS_DOT:
                return "wall"
        cav.others.add(i)
        return "other"

    # ---- measuring --------------------------------------------------------------------
    def section_loops(
        self, host: _Host, walls: set[int], depth: float
    ) -> tuple[list[list[Seg]], list[list[Seg]]]:
        """Closed loops and open chains cut from the wall faces by a plane *depth* below the host."""
        d_host_pt = host.o - host.n * depth
        pln = gp_Pln(gp_Pnt(*map(float, d_host_pt)), gp_Dir(*map(float, host.n)))
        segs: list[Seg] = []
        for w in sorted(walls):
            sec = BRepAlgoAPI_Section(self.table.faces[w].face, pln, True)
            if not sec.IsDone():
                continue
            for e in unique_subshapes(sec.Shape(), TopAbs_EDGE):
                seg = self._section_seg(as_edge(e), host)
                if seg is not None:
                    segs.append(seg)
        closed, opened = cg.chain_loops(segs)
        return [cg.merge_collinear(c) for c in closed], opened

    def _section_seg(self, edge, host: _Host) -> Seg | None:  # type: ignore[no-untyped-def]
        c = BRepAdaptor_Curve(edge)
        kind = c.GetType()
        if kind not in (GeomAbs_Line, GeomAbs_Circle):
            return None
        return self._seg_of(c, kind, False, host.o, host.u, host.v)

    def bottom_extent(self, host: _Host, faces: set[int]) -> float:
        """Greatest depth below the host plane reached by *faces* (used for odd floors)."""
        d_host = float(np.dot(host.n, host.o))
        deepest = 0.0
        for i in faces:
            lo, hi = bbox_of(self.table.faces[i].face)
            corners = np.array(
                [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
            )
            deepest = max(deepest, d_host - float((corners @ host.n).min()))
        return deepest


def _canon_rank(n: Vec | None) -> int:
    """Tie-break between caps of equal area: faces looking along +axis come first."""
    if n is None:
        return 2
    k = int(np.argmax(np.abs(n)))
    return 0 if n[k] > 0 else 1


def _frame(part: Part, n: Vec) -> tuple[Vec, Vec]:
    from stepscribe.describe.reconstruct import _frame as frame

    return frame(part, n)


# ---------------------------------------------------------------------------------------
def detect_cutouts(geom: PartGeom, part: Part) -> tuple[list[Cutout], list[CutoutPattern]]:
    """All cut-outs of *part* with IDs C001.., plus patterns of identical ones (CP001..)."""
    det = _Detector(geom, part)
    found: list[Found] = []
    claimed: set[int] = set()
    notches = 0
    for hi in det.hosts():
        host = det.build_host(hi)
        if host is None:
            continue
        outer_x = {j for _p, j, _s in host.outer_edges if j >= 0}
        for k, (_pts, seeds, _pieces) in enumerate(host.inner):
            cut = _mouth_cutout(det, host, k, seeds, outer_x, claimed)
            if cut is not None:
                found.append(cut)
        for run in _notch_runs(host):
            if notches >= config.CUTOUT_MAX_NOTCHES:
                break
            cut = _notch_cutout(det, host, run, outer_x, claimed)
            if cut is not None:
                notches += 1
                found.append(cut)
    cutouts = _finalise(_drop_inner_duplicates(found), part)
    patterns = _patterns(cutouts)
    return cutouts, patterns


def _mouth_cutout(
    det: _Detector, host: _Host, k: int, seeds: list[int], outer_x: set[int], claimed: set[int]
) -> Found | None:
    seed_set = {s for s in seeds if s >= 0}
    if not seed_set or seed_set <= claimed or seed_set <= det.hole_faces:
        return None
    if not _opens_into_void(det, host, host.inner[k][2]):
        return None
    cav = det.explore(host, seed_set, outer_x - seed_set)
    return _build(det, host, cav, seed_set, claimed, notch=None, mouth=host.inner[k][0])


def _opens_into_void(det: _Detector, host: _Host, pieces: list[Piece]) -> bool:
    """True if the wall at the mouth faces the opening (a cut-out), not away (a boss or rib).

    Probe a point just below the host plane on the side the wall's outward normal points to:
    inside a cut-out that is empty space, next to a boss it is material.
    """
    for poly, j, _seg in pieces:
        if j < 0:
            continue
        mid = poly[len(poly) // 2]
        p3 = host.o + host.u * mid[0] + host.v * mid[1]
        nrm = normal_at_point(det.table.faces[j].face, p3)
        if nrm is None:
            continue
        probe = p3 + PROBE_MM * nrm - PROBE_MM * host.n
        return str(det.geom.classifier.state(probe)) != "in"
    return True


def _notch_runs(host: _Host) -> list[list[int]]:
    """Runs of consecutive outer-wire edges that dip inside the outline's convex hull."""
    if not host.outer_edges:
        return []
    hull = cg.convex_hull(host.outer)
    if len(hull) < 3:
        return []
    flags = []
    for pts, _j, _s in host.outer_edges:
        dist = max(cg._pts_to_segs(np.array([q]), hull) for q in pts)
        flags.append(dist >= config.CUTOUT_NOTCH_MIN_DEPTH)
    n = len(flags)
    if not any(flags) or all(flags):
        return []
    start = next(i for i in range(n) if flags[i] and not flags[i - 1])
    runs: list[list[int]] = []
    cur: list[int] = []
    for off in range(n):
        i = (start + off) % n
        if flags[i]:
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    box = host.outer.max(axis=0) - host.outer.min(axis=0)
    limit = config.CUTOUT_NOTCH_MAX_REL * float(box.min())
    out = []
    for r in runs:
        pts = np.vstack([host.outer_edges[i][0] for i in r])
        chord = float(np.linalg.norm(pts[-1] - pts[0]))
        depth = _depth_from_chord(pts)
        if len(r) <= config.CUTOUT_NOTCH_MAX_EDGES and chord <= limit and depth <= limit * 1.5:
            out.append(r)
    return out


def _depth_from_chord(pts: np.ndarray) -> float:
    a, b = pts[0], pts[-1]
    d = b - a
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return float(np.linalg.norm(pts - a, axis=1).max())
    return float(np.abs((pts - a) @ np.array([-d[1], d[0]]) / n).max())


def _notch_cutout(
    det: _Detector, host: _Host, run: list[int], outer_x: set[int], claimed: set[int]
) -> Found | None:
    seeds = {host.outer_edges[i][1] for i in run if host.outer_edges[i][1] >= 0}
    if not seeds or seeds <= claimed or seeds <= det.hole_faces:
        return None
    if not _opens_into_void(det, host, [host.outer_edges[i] for i in run]):
        return None
    cav = det.explore(host, seeds, outer_x - seeds)
    pts = np.vstack([host.outer_edges[i][0] for i in run])
    return _build(det, host, cav, seeds, claimed, notch=pts, mouth=pts)


# ---------------------------------------------------------------------------------------
def _breakpoints(det: _Detector, host: _Host, cav: _Cavity) -> tuple[list[float], bool, str]:
    """(depth breakpoints from 0 to the bottom, is_through, floor_type)."""
    if cav.opposite:
        bottom, through, ftype = max(cav.opposite.values()), True, "through"
    elif cav.floors and not cav.others:
        bottom, through, ftype = max(cav.floors.values()), False, "flat"
    else:
        used = set(cav.walls) | set(cav.floors) | cav.others
        bottom, through, ftype = det.bottom_extent(host, used), False, "other"
    inner = sorted(
        {round(d, 4) for d in cav.floors.values() if d < bottom - config.CUTOUT_MIN_STEP}
    )
    return [0.0, *inner, float(bottom)], through, ftype


def _build(
    det: _Detector,
    host: _Host,
    cav: _Cavity,
    seeds: set[int],
    claimed: set[int],
    notch: np.ndarray | None,
    mouth: np.ndarray,
) -> Found | None:
    if cav.protrusion or not cav.walls:
        return None
    bps, through, ftype = _breakpoints(det, host, cav)
    if bps[-1] <= config.CUTOUT_MIN_STEP:
        return None
    warnings: list[str] = []
    levels: list[CutoutLevel] = []
    for i in range(len(bps) - 1):
        mid = 0.5 * (bps[i] + bps[i + 1])
        last = i == len(bps) - 2
        lv = _level(
            det,
            host,
            cav,
            bps[i],
            mid,
            bps[i + 1] - bps[i] if not (last and through) else None,
            notch if i == 0 else notch,
            warnings,
        )
        if lv is None:
            warnings.append(f"level {i + 1}: no closed outline found")
            continue
        levels.append(lv)
    if not levels:
        return None
    claimed |= set(cav.walls)
    if cav.draft_deg > 0.05:
        for lv in levels:
            lv.draft_deg = round(cav.draft_deg, 2)
    if ftype == "other":
        warnings.append(
            "floor is curved or not parallel to the entry face; depth is the deepest reach"
        )
    face_ids = sorted(
        det.table.faces[i].id
        for i in (cav.walls | set(cav.floors) | set(cav.opposite) - {host.idx})
        if i != host.idx
    )
    kind = "notch" if notch is not None else _kind(levels, through)
    cut = Cutout(
        id="",
        kind=kind,
        host_face_id=det.table.faces[host.idx].id,
        entry_normal=vec3(host.n),
        levels=levels,
        total_depth_mm=None if through else bps[-1],
        face_ids=face_ids,
        floor_type=ftype,
        description="",
        warnings=warnings,
    )
    if notch is not None:
        cut.edge_side = _edge_side(host, notch)
    cut.description = describe_cutout(cut)
    return cut, mouth, host


def _kind(levels: list[CutoutLevel], through: bool) -> str:
    if len(levels) >= 2:
        return "stepped"
    return "through" if through else "recess"


def _edge_side(host: _Host, pts: np.ndarray) -> str:
    lo, hi = host.outer.min(axis=0), host.outer.max(axis=0)
    mid = 0.5 * (pts[0] + pts[-1])
    cand = {
        "-u": abs(mid[0] - lo[0]),
        "+u": abs(hi[0] - mid[0]),
        "-v": abs(mid[1] - lo[1]),
        "+v": abs(hi[1] - mid[1]),
    }
    return min(sorted(cand), key=lambda k: cand[k])


def _level(
    det: _Detector,
    host: _Host,
    cav: _Cavity,
    entry_depth: float,
    mid: float,
    depth: float | None,
    notch: np.ndarray | None,
    warnings: list[str],
) -> CutoutLevel | None:
    loops, opened = det.section_loops(host, cav.walls, mid)
    loop: list[Seg] | None = None
    if notch is not None and opened:
        chain = max(opened, key=lambda c: sum(s.length() for s in c))
        loop = cg.merge_collinear([*chain, Seg("line", chain[-1].p1, chain[0].p0)])
    elif loops:
        loop = max(loops, key=lambda c: abs(cg.signed_area(c)))
        if len(loops) > 1:
            warnings.append(f"{len(loops)} loops at depth {mid:.2f}; the largest is reported")
    if loop is None:
        return None
    return _measure(host, loop, entry_depth, depth, notch is not None)


def _measure(
    host: _Host, loop: list[Seg], entry_depth: float, depth: float | None, notch: bool
) -> CutoutLevel:
    shape, radius, sides = cg.classify(loop)
    pts = cg.sample_loop(loop)
    width, length, rot, _c = cg.min_area_rect(pts)
    area = abs(cg.signed_area(loop))
    perim = sum(s.length() for s in loop)
    cen = cg.polygon_centroid(pts)
    if notch:
        chord = loop[-1]
        width = float(np.linalg.norm(chord.p1 - chord.p0))
        length = _depth_from_chord(pts)
        d = chord.p1 - chord.p0
        rot = math.degrees(math.atan2(d[1], d[0])) % 180.0
        rot = rot - 180.0 if rot > 90.0 else rot
    if shape == "circle":
        width = length = 2 * loop[0].radius
        rot = 0.0
    origin = host.origin_uv
    p3 = host.o + host.u * cen[0] + host.v * cen[1] - host.n * entry_depth
    lo, hi = pts.min(axis=0) - origin, pts.max(axis=0) - origin
    verts = (
        [(float(s.p0[0] - origin[0]), float(s.p0[1] - origin[1])) for s in loop]
        if shape in ("rectangle", "polygon") and len(loop) <= config.CUTOUT_MAX_VERTICES
        else []
    )
    return CutoutLevel(
        shape=shape,
        width_mm=width,
        length_mm=length,
        rotation_deg=rot,
        corner_radius_mm=radius if radius is not None else cg.corner_radius(loop),
        sides=sides,
        center=vec3(p3),
        center_uv=(float(cen[0] - origin[0]), float(cen[1] - origin[1])),
        u_range=(float(lo[0]), float(hi[0])),
        v_range=(float(lo[1]), float(hi[1])),
        vertices_uv=verts,
        depth_mm=depth,
        area_mm2=area,
        perimeter_mm=perim,
        outline=cg.outline_dicts(loop, origin),
    )


# ---------------------------------------------------------------------------------------
def level_text(lv: CutoutLevel) -> str:
    """'Ø6.0', '20.0×10.0', '20.0×10.0 R2.0' for one level."""
    if lv.shape == "circle":
        return f"Ø{lv.width_mm:.1f}"
    s = f"{lv.width_mm:.1f}×{lv.length_mm:.1f}"
    if lv.shape == "polygon" and lv.sides:
        s += f" {lv.sides}-sided"
    elif lv.corner_radius_mm and lv.shape in ("rounded_rectangle",):
        s += f" R{lv.corner_radius_mm:.1f}"
    return s


def describe_cutout(c: Cutout) -> str:
    """The one-line description, e.g. 'Stepped cutout: 10.0×18.0 recess ↧2.0, then 10.0×6.0 THRU'."""
    lv = c.levels
    if c.kind == "notch":
        side = f", on the {c.edge_side} edge" if c.edge_side else ""
        return f"Edge notch: {lv[0].width_mm:.1f} wide × {lv[0].length_mm:.1f} deep{side}"
    r = abs(lv[0].rotation_deg)
    skew = lv[0].shape != "circle" and min(r, abs(r - 90.0)) > 0.5
    rot = f" (rotated {lv[0].rotation_deg:.1f}°)" if skew else ""
    if c.kind == "through":
        return f"Through cutout: {level_text(lv[0])} THRU{rot}"
    if c.kind == "recess":
        return f"Recess: {level_text(lv[0])} ↧{lv[0].depth_mm:.1f}{rot}"
    parts = []
    for i, x in enumerate(lv):
        tail = "THRU" if x.depth_mm is None else f"↧{x.depth_mm:.1f}"
        parts.append(
            f"{level_text(x)} {'recess ' if i == 0 and x.depth_mm is not None else ''}{tail}"
        )
    return "Stepped cutout: " + ", then ".join(parts) + rot


def _drop_inner_duplicates(found: list[Found]) -> list[Found]:
    """Drop an opening whose faces are all part of a bigger one (seen again from the far side)."""
    keep: list[Found] = []
    for item in sorted(found, key=lambda t: (-len(t[0].face_ids), t[0].host_face_id)):
        ids = set(item[0].face_ids)
        if not any(ids <= set(k[0].face_ids) for k in keep):
            keep.append(item)
    return keep


def ascii_text(c: Cutout, short: bool = False) -> str:
    """Plain-ASCII label for images (the bundled font has no arrow glyphs): 'C001 10.0x18.0 d2.0 / 6.0x10.0 THRU'."""
    if c.kind == "notch":
        lv = c.levels[0]
        return f"{c.id} notch {lv.width_mm:.1f}x{lv.length_mm:.1f}"
    if short and len(c.levels) > 1:
        return f"{c.id} {level_text(c.levels[0]).replace('×', 'x')} stepped"
    parts = []
    for x in c.levels:
        tail = "THRU" if x.depth_mm is None else f"d{x.depth_mm:.1f}"
        parts.append(f"{level_text(x).replace('×', 'x')} {tail}")
    return f"{c.id} " + " / ".join(parts)


def _finalise(found: list[tuple[Cutout, np.ndarray, _Host]], part: Part) -> list[Cutout]:
    """Stable IDs, edge distances and links to slots / pockets."""
    found.sort(
        key=lambda t: (
            t[0].host_face_id,
            *(round(c, 3) for c in t[0].levels[0].center_uv),
            t[0].kind,
        )
    )
    for i, (c, mouth, host) in enumerate(found, 1):
        c.id = f"C{i:03d}"
        _distances(c, mouth, host)
    cutouts = [c for c, _m, _h in found]
    for s in part.slots:
        s.cutout_id = next((c.id for c in cutouts if set(s.face_ids) & set(c.face_ids)), None)
    for p in part.pockets:
        p.cutout_id = next((c.id for c in cutouts if p.floor_face_id in c.face_ids), None)
    return cutouts


def _distances(c: Cutout, mouth: np.ndarray, host: _Host) -> None:
    web = math.inf
    own = c.kind == "notch"
    for pts, _s, _sg in host.inner:
        if pts is mouth or (len(pts) == len(mouth) and np.allclose(pts, mouth)):
            continue
        web = min(web, cg.min_distance(mouth, pts))
    edge = math.inf if own else cg.min_distance(mouth, host.outer)
    if web < math.inf:
        c.web_to_neighbor_mm = web
    best = min(edge, web)
    c.edge_distance_mm = best if best < math.inf else None


def _patterns(cutouts: list[Cutout]) -> list[CutoutPattern]:
    groups: dict[tuple[object, ...], list[Cutout]] = {}
    for c in cutouts:
        sig = (
            c.kind,
            c.host_face_id,
            tuple(
                (
                    lv.shape,
                    round(lv.width_mm, 2),
                    round(lv.length_mm, 2),
                    None if lv.depth_mm is None else round(lv.depth_mm, 2),
                    round(lv.rotation_deg, 0),
                )
                for lv in c.levels
            ),
        )
        groups.setdefault(sig, []).append(c)
    out: list[CutoutPattern] = []
    for _sig, grp in sorted(groups.items(), key=lambda kv: kv[1][0].id):
        if len(grp) < 2:
            continue
        pts = np.array([c.levels[0].center_uv for c in grp])
        kind, pitch = "irregular_group", None
        grid = _grid(pts) if len(grp) == 4 else None
        lin = _linear(pts)
        if grid is not None:
            kind = "rectangular_grid"
        elif lin is not None:
            kind, pitch = "linear", abs(lin)
        word = {
            "through": "through cutout",
            "recess": "recess",
            "stepped": "stepped cutout",
            "notch": "notch",
        }[grp[0].kind]
        size = " / ".join(level_text(lv) for lv in grp[0].levels)
        extra = {
            "linear": f"linear, pitch {pitch:.1f}" if pitch is not None else "linear",
            "rectangular_grid": "rectangular grid",
            "irregular_group": "irregular group",
        }[kind]
        pat = CutoutPattern(
            id=f"CP{len(out) + 1:03d}",
            kind=kind,
            cutout_ids=[c.id for c in grp],
            count=len(grp),
            pitch_mm=pitch,
            description=f"{len(grp)} × {size} {word}, {extra}",
        )
        for c in grp:
            c.pattern_id = pat.id
        out.append(pat)
    return out


__all__ = ["ascii_text", "describe_cutout", "detect_cutouts", "level_text"]
