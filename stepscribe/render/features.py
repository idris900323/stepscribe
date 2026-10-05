# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Feature dimensions for part images: sized callouts, a feature table panel, ordinate dimensions."""

from __future__ import annotations

import math
import textwrap
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from stepscribe import config
from stepscribe.describe.phrases import fmt
from stepscribe.features.cutouts import ascii_text
from stepscribe.geometry.part_geom import PartGeom
from stepscribe.models.schema import Cutout, CutoutLevel, Hole, Part
from stepscribe.render.dimensions import INK
from stepscribe.render.overlay import FONT_SIZE, Label, _font, draw_text, pad_canvas, text_size
from stepscribe.render.raster import Rendered

PANEL_WIDTH = 640
PANEL_CHARS = 46  # wrap width for the 20 px default font
MAX_PANEL_LINES = 42
MAX_ORDINATES = 18
LEVEL_COLORS = ((214, 39, 40), (31, 119, 180), (44, 160, 44), (148, 103, 189))
CROWDED_LABELS = 1  # with more cut-outs than this the label is short; the panel has the full text


@dataclass
class Mark:
    """A feature anchored at a point in the part's own frame."""

    label: str
    point: np.ndarray
    always: bool = False  # draw even if the anchor is not on a visible surface (slot centres)


def _v(p) -> np.ndarray:  # type: ignore[no-untyped-def]
    return np.array([p.x, p.y, p.z])


def hole_text(h: Hole, with_id: bool = True) -> str:
    """'H1 dia 6.5 thru', 'H2 dia 5 x 12 deep'; counterbore / countersink appended."""
    s = f"dia {fmt(h.diameter_mm)}"
    s += " thru" if h.is_through else (f" x {fmt(h.depth_mm)} deep" if h.depth_mm else "")
    if h.counterbore_diameter_mm:
        s += f", cbore dia {fmt(h.counterbore_diameter_mm)}"
        if h.counterbore_depth_mm:
            s += f" x {fmt(h.counterbore_depth_mm)}"
    if h.countersink_diameter_mm:
        s += f", csink dia {fmt(h.countersink_diameter_mm)}"
        if h.countersink_angle_deg:
            s += f" {fmt(h.countersink_angle_deg, 0)} deg"
    return f"{h.id} {s}" if with_id else s


def short_hole_text(h: Hole) -> str:
    """The label drawn next to a hole: ID and diameter, plus depth when blind."""
    s = f"{h.id} dia {fmt(h.diameter_mm)}"
    if not h.is_through and h.depth_mm:
        s += f" x {fmt(h.depth_mm)}"
    return s


def feature_marks(part: Part, geom: PartGeom, limit: int = 40) -> list[Mark]:
    """Anchored callouts: holes, slots, pockets and bosses, each with its sizes."""
    marks = [Mark(short_hole_text(h), _v(h.axis.origin)) for h in part.holes[:limit]]
    short = len(part.cutouts) > CROWDED_LABELS
    for c in part.cutouts:
        marks.append(Mark(ascii_text(c, short), _v(c.levels[0].center), True))
    for s in part.slots:
        if s.cutout_id:
            continue  # shown once, as its cut-out
        marks.append(Mark(f"{s.id} {fmt(s.width_mm)} x {fmt(s.length_mm)}", _v(s.center), True))
    table = geom.table
    for pk in part.pockets:
        if pk.cutout_id:
            continue
        try:
            floor_c = table.by_id(pk.floor_face_id).centroid
        except (IndexError, ValueError):
            continue
        a, b = pk.outline_size
        marks.append(Mark(f"{pk.id} {fmt(a)} x {fmt(b)}, {fmt(pk.depth_mm)} deep", floor_c))
    for bo in part.bosses:
        try:
            boss_c = table.by_id(bo.face_ids[0]).centroid
        except (IndexError, ValueError):
            continue
        marks.append(Mark(f"{bo.id} dia {fmt(bo.diameter_mm)} h {fmt(bo.height_mm)}", boss_c))
    return marks


def visible_labels(rendered: Rendered, marks: list[Mark], to_world=lambda p: p) -> list[Label]:  # type: ignore[no-untyped-def]
    out = []
    for m in marks:
        x, y, d = rendered.project(to_world(m.point))
        if m.always or rendered.visible(x, y, d):
            out.append(Label(m.label, x, y))
    return out


def _ids(ids: list[str]) -> str:
    return ", ".join(ids) if len(ids) <= 6 else ", ".join(ids[:5]) + f", ... ({len(ids)} holes)"


def cutout_panel_lines(part: Part) -> list[str]:
    """The 'Cutouts' group of the panel: patterns on one line each, then the single ones."""
    lines = ["#Cutouts (mm)"]
    in_pattern: set[str] = set()
    for p in part.cutout_patterns:
        in_pattern.update(p.cutout_ids)
        first = next(c for c in part.cutouts if c.id == p.cutout_ids[0])
        pitch = f", pitch {fmt(p.pitch_mm)}" if p.pitch_mm is not None else ""
        lines.append(
            f"{p.cutout_ids[0]}..{p.cutout_ids[-1]} x{p.count} ({p.kind.replace('_', ' ')}{pitch}): "
            + ascii_text(first).split(" ", 1)[1]
        )
    for c in part.cutouts:
        if c.id in in_pattern:
            continue
        lines.append(ascii_text(c))
        lv = c.levels[0]
        extra = f"  centre u {fmt(lv.center_uv[0])}, v {fmt(lv.center_uv[1])}"
        if c.edge_distance_mm is not None:
            extra += f", edge {fmt(c.edge_distance_mm)}"
        lines.append(extra)
    return lines


def panel_lines(part: Part) -> list[str]:
    """Plain-text rows for the feature panel; '#' prefixes a heading."""
    m = part.mass
    sx, sy, sz = part.obb.size_sorted[::-1] if part.obb.size_sorted else (0, 0, 0)
    lines = [
        f"#{part.id}  {part.name}",
        f"Box (oriented) {fmt(sx)} x {fmt(sy)} x {fmt(sz)} mm",
        f"Volume {fmt(m.volume_mm3, 0)} mm3",
        f"Surface area {fmt(m.surface_area_mm2, 0)} mm2",
    ]
    if m.mass_g is not None:
        lines.append(f"Mass {fmt(m.mass_g)} g ({m.material_assumed or 'material given'})")
    if part.min_wall_thickness_mm is not None:
        lines.append(f"Thinnest wall about {fmt(part.min_wall_thickness_mm)} mm")
    if part.holes:
        lines.append("#Holes (mm)")
        groups: dict[str, list[Hole]] = defaultdict(list)
        for h in part.holes:
            groups[hole_text(h, with_id=False)].append(h)
        for text, hs in groups.items():
            n = f" x{len(hs)}" if len(hs) > 1 else ""
            lines.append(f"{_ids([h.id for h in hs])}{n}: {text}")
        edge = [h.edge_distance_mm for h in part.holes if h.edge_distance_mm is not None]
        if edge:
            lines.append(f"Nearest hole to an edge: {fmt(min(edge))} mm")
    if part.cutouts:
        lines.extend(cutout_panel_lines(part))
    slots = [s for s in part.slots if not s.cutout_id]
    if slots:
        lines.append("#Slots (mm)")
        for s in slots:
            depth = "through" if s.is_through else f"{fmt(s.depth_mm)} deep" if s.depth_mm else ""
            lines.append(f"{s.id}: width {fmt(s.width_mm)}, length {fmt(s.length_mm)}, {depth}")
    pockets = [p for p in part.pockets if not p.cutout_id]
    if pockets:
        lines.append("#Pockets (mm)")
        for p in pockets:
            pa, pb = p.outline_size
            corner = f", corner R{fmt(p.corner_radius_mm)}" if p.corner_radius_mm else ""
            lines.append(f"{p.id}: {fmt(pa)} x {fmt(pb)}, {fmt(p.depth_mm)} deep{corner}")
    if part.bosses:
        lines.append("#Bosses (mm)")
        for b in part.bosses:
            lines.append(f"{b.id}: dia {fmt(b.diameter_mm)}, height {fmt(b.height_mm)}")
    if part.fillets:
        by_r: dict[tuple[float, bool], int] = defaultdict(int)
        for f in part.fillets:
            by_r[(round(f.radius_mm, 2), f.convex)] += 1
        lines.append("#Fillets (mm)")
        lines.extend(
            f"R{fmt(r, 2)} {'convex' if cv else 'concave'} x{n}"
            for (r, cv), n in sorted(by_r.items())
        )
    if part.chamfers:
        by_c: dict[tuple[float, float], int] = defaultdict(int)
        for c in part.chamfers:
            by_c[(round(c.distance_mm, 2), round(c.angle_deg))] += 1
        lines.append("#Chamfers (mm)")
        lines.extend(f"{fmt(d, 2)} at {a} deg x{n}" for (d, a), n in sorted(by_c.items()))
    return lines


def _wrap_to_width(text: str, font, max_px: float) -> list[str]:  # type: ignore[no-untyped-def]
    """Wrap *text* so that every piece is narrower than *max_px* (starting from PANEL_CHARS)."""
    width = PANEL_CHARS
    while True:
        pieces = textwrap.wrap(text, width, subsequent_indent="   ") or [""]
        if width <= 8 or all(font.getlength(p) <= max_px for p in pieces):
            return pieces
        width -= 2


def with_panel(image: Image.Image, lines: list[str]) -> Image.Image:
    """Return *image* with a white feature-table panel appended on the right.

    The canvas grows downward when the table is taller than the picture, so no row is cut off.
    """
    font = _font(FONT_SIZE)
    rows: list[tuple[str, bool]] = []
    for ln in lines:
        head = ln.startswith("#")
        text = ln[1:] if head else ln
        for piece in _wrap_to_width(text, font, PANEL_WIDTH - 36):
            rows.append((piece, head))
    if len(rows) > MAX_PANEL_LINES:
        extra = len(rows) - MAX_PANEL_LINES + 1
        rows = rows[: MAX_PANEL_LINES - 1] + [
            (f"... {extra} more lines in the context pack", False)
        ]
    need = 14 + sum((FONT_SIZE + 18) if head else (FONT_SIZE + 5) for _t, head in rows) + 14
    w, h = image.size
    out = Image.new("RGB", (w + PANEL_WIDTH, max(h, need)), (255, 255, 255))
    out.paste(image.convert("RGB"), (0, 0))
    d = ImageDraw.Draw(out)
    d.line([(w, 0), (w, out.height)], fill=(0, 0, 0), width=2)
    y = 14
    for text, head in rows:
        if head:
            y += 8
            draw_text(d, (w + 18, y), text, font, INK)
            d.line(
                [(w + 18, y + FONT_SIZE + 3), (w + PANEL_WIDTH - 18, y + FONT_SIZE + 3)], fill=INK
            )
            y += FONT_SIZE + 10
        else:
            draw_text(d, (w + 18, y), text, font)
            y += FONT_SIZE + 5
    return out


def level_polyline(part: Part, c: Cutout, lv: CutoutLevel) -> np.ndarray:
    """Sampled outline of one cut-out level as an (N, 3) array in the part frame."""
    from stepscribe.describe.reconstruct import _frame

    n = np.array([c.entry_normal.x, c.entry_normal.y, c.entry_normal.z])
    u, v = _frame(part, n)
    cen = np.array([lv.center.x, lv.center.y, lv.center.z])
    cu, cv = lv.center_uv
    pts2: list[list[float]] = []
    for seg in lv.outline:
        a, b = seg["from"], seg["to"]
        if seg["type"] == "arc":
            ctr, sweep = seg["centre"], math.radians(float(seg["sweep_deg"]))
            a0 = math.atan2(a[1] - ctr[1], a[0] - ctr[0])
            k = max(2, int(math.ceil(abs(math.degrees(sweep)) / config.CUTOUT_ARC_STEP_DEG)) + 1)
            r = float(seg["radius"])
            for i in range(k):
                ang = a0 + sweep * i / (k - 1)
                pts2.append([ctr[0] + r * math.cos(ang), ctr[1] + r * math.sin(ang)])
        else:
            pts2 += [a, b]
    arr = np.array(pts2)
    return np.asarray(cen + np.outer(arr[:, 0] - cu, u) + np.outer(arr[:, 1] - cv, v))


def draw_cutout_outlines(
    rendered: Rendered,
    part: Part,
    to_world: Callable[[np.ndarray], np.ndarray] = lambda p: p,
) -> dict[str, np.ndarray]:
    """Outline every cut-out level in its own colour.

    Returns the first level's outline per cut-out (world coordinates) for the ordinate and size
    dimensions."""
    img = rendered.image.convert("RGB")
    draw = ImageDraw.Draw(img)
    first: dict[str, np.ndarray] = {}
    for c in part.cutouts:
        for k, lv in enumerate(c.levels):
            world = np.array([to_world(p) for p in level_polyline(part, c, lv)])
            if k == 0:
                first[c.id] = world
            proj = rendered.project_many(world)
            vis = rendered.visible_mask(proj)
            color = LEVEL_COLORS[k % len(LEVEL_COLORS)]
            for i in range(len(proj) - 1):
                if vis[i] and vis[i + 1]:
                    seg = [(proj[i, 0], proj[i, 1]), (proj[i + 1, 0], proj[i + 1, 1])]
                    draw.line(seg, fill=color, width=4)
    rendered.image = img
    return first


def cutout_size_dimensions(
    rendered: Rendered, outlines: dict[str, np.ndarray], part: Part, mm_per_px: float
) -> None:
    """Width and length dimension lines on one cut-out of each pattern (and every single one).

    The values are measured from the drawn outline (orthographic view: pixels x scale)."""
    if mm_per_px <= 0:
        return
    pattern_of = {cid: p.id for p in part.cutout_patterns for cid in p.cutout_ids}
    done: set[str] = set()
    img = rendered.image.convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font(16)
    for c in part.cutouts:
        key = pattern_of.get(c.id, c.id)
        if key in done or c.id not in outlines:
            continue
        done.add(key)
        proj = rendered.project_many(outlines[c.id])
        x0, y0 = float(proj[:, 0].min()), float(proj[:, 1].min())
        x1, y1 = float(proj[:, 0].max()), float(proj[:, 1].max())
        horizontal = (x0, y0 - 14, x1, y0 - 14, (x1 - x0) * mm_per_px, True)
        vertical = (x1 + 14, y0, x1 + 14, y1, (y1 - y0) * mm_per_px, False)
        for xa, ya, xb, yb, mm, horiz in (horizontal, vertical):
            if mm < 0.5:
                continue
            draw.line([(xa, ya), (xb, yb)], fill=INK, width=2)
            tick = (0, 6) if horiz else (6, 0)
            for tx, ty in ((xa, ya), (xb, yb)):
                draw.line(
                    [(tx - tick[0], ty - tick[1]), (tx + tick[0], ty + tick[1])], fill=INK, width=2
                )
            t = fmt(mm)
            tw, th = text_size(draw, t, font)
            cx, cy = (xa + xb) / 2, (ya + yb) / 2
            bx = cx - tw / 2 if horiz else xb + 8
            by = cy - th - 8 if horiz else cy - th / 2
            bx = min(max(bx, 4), img.width - tw - 4)
            by = min(max(by, 4), img.height - th - 4)
            draw.rectangle([bx - 2, by - 2, bx + tw + 2, by + th + 2], fill=(255, 255, 255))
            draw_text(draw, (bx, by), t, font, INK)
    rendered.image = img


def ordinate_dimensions(
    rendered: Rendered,
    corners: np.ndarray,
    w_mm: float,
    h_mm: float,
    points: list[np.ndarray],
    outlines: dict[str, np.ndarray] | None = None,
) -> float:
    """Draw feature positions measured from the model's lower-left corner, along the top
    (distance from the left edge) and the left (distance from the bottom edge).

    Holes give their centre; each cut-out in *outlines* gives its two outer edges in each
    direction. *corners* are the 8 world corners of the model box; the view is orthographic so
    pixel offsets scale linearly with millimetres. The canvas grows on the left and top so no
    value is cut off. Returns the millimetres per pixel of the view (0 if nothing was drawn).
    """
    font = _font(16)
    px = rendered.project_many(corners)
    x0, x1 = float(px[:, 0].min()), float(px[:, 0].max())
    y0, y1 = float(px[:, 1].min()), float(px[:, 1].max())
    if x1 - x0 < 60 or y1 - y0 < 60 or w_mm <= 0 or h_mm <= 0:
        return 0.0
    xs = [rendered.project(p)[0] for p in points]
    ys = [rendered.project(p)[1] for p in points]
    for pts in (outlines or {}).values():
        proj = rendered.project_many(pts)
        xs += [float(proj[:, 0].min()), float(proj[:, 0].max())]
        ys += [float(proj[:, 1].min()), float(proj[:, 1].max())]
    cols = _ordinates(xs, lambda x: (x - x0) / (x1 - x0) * w_mm)
    rows = _ordinates(ys, lambda y: (y1 - y) / (y1 - y0) * h_mm)
    probe = ImageDraw.Draw(rendered.image.convert("RGB"))
    widest = max([text_size(probe, fmt(v), font)[0] for _p, v in rows] + [0.0])
    stagger = 56 if len(rows) > 1 else 0
    pad_left = max(0, math.ceil(8 + widest + 10 + stagger - x0)) if rows else 0
    pad_top = max(0, math.ceil(8 + 26 + (20 if len(cols) > 1 else 0) - y0)) if cols else 0
    img = pad_canvas(rendered.image.convert("RGB"), left=pad_left, top=pad_top)
    draw = ImageDraw.Draw(img)
    dx, dy = float(pad_left), float(pad_top)
    light = (225, 150, 150)
    white = (255, 255, 255)
    # top: distance from the left edge, alternating two text rows so neighbours do not collide
    for k, (x, v) in enumerate(cols):
        ty = y0 + dy - 26 - (k % 2) * 20
        draw.line([(x + dx, y0 + dy - 6), (x + dx, y1 + dy)], fill=light, width=1)
        t = fmt(v)
        tw, th = text_size(draw, t, font)
        draw.rectangle([x + dx - tw / 2 - 2, ty - 2, x + dx + tw / 2 + 2, ty + th + 2], fill=white)
        draw_text(draw, (x + dx - tw / 2, ty), t, font, INK)
    # left: distance from the bottom edge
    for k, (y, v) in enumerate(rows):
        tx = x0 + dx - 10 - (k % 2) * 56
        t = fmt(v)
        tw, th = text_size(draw, t, font)
        draw.line([(x0 + dx - 6, y + dy), (x1 + dx, y + dy)], fill=light, width=1)
        draw.rectangle([tx - tw - 2, y + dy - th / 2 - 2, tx + 2, y + dy + th / 2 + 2], fill=white)
        draw_text(draw, (tx - tw, y + dy - th / 2), t, font, INK)
    rendered.image = img
    return w_mm / (x1 - x0)


def _ordinates(pixels: list[float], to_mm) -> list[tuple[float, float]]:  # type: ignore[no-untyped-def]
    """(pixel, mm) pairs, one per distinct 0.1 mm value, sorted by value, at most MAX_ORDINATES."""
    seen: set[float] = set()
    out: list[tuple[float, float]] = []
    for p in pixels:
        v = round(float(to_mm(p)), 1)
        if v not in seen and len(seen) < MAX_ORDINATES:
            seen.add(v)
            out.append((p, v))
    return sorted(out, key=lambda t: t[1])
