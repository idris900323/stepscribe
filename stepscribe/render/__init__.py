# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Labelled renders: assembly views, exploded view and per-part views."""

from __future__ import annotations

import copy
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from stepscribe.assembly import build_instances
from stepscribe.assembly.spatial import frame
from stepscribe.describe.budget import rank_parts
from stepscribe.geometry.occ_utils import Vec, canonical_dir
from stepscribe.progress import Tracker
from stepscribe.render.dimensions import annotate, size_caption, view_spans
from stepscribe.render.features import (
    cutout_size_dimensions,
    draw_cutout_outlines,
    feature_marks,
    ordinate_dimensions,
    panel_lines,
    visible_labels,
    with_panel,
)
from stepscribe.render.kinematics_view import (
    draw_joint_arrows,
    link_labels,
    link_legend,
    recolor_by_link,
)
from stepscribe.render.overlay import Label, draw_labels, save_png
from stepscribe.render.raster import Rendered, render_scene, rendering_available
from stepscribe.render.scene import (
    CameraSpec,
    SceneItem,
    apply_explode,
    fit_camera,
    part_color,
    scene_bounds,
    standard_camera,
)
from stepscribe.render.tessellate import Mesh, tessellate

if TYPE_CHECKING:
    from stepscribe.api import Analysis

MAX_PART_IMAGES = 20
MAX_HOLE_LABELS = 40
ASSEMBLY_VIEWS = ("iso", "front", "top")


_clock = {"t": time.time()}


def _save(tracker: Tracker | None, image, path: str) -> None:  # type: ignore[no-untyped-def]
    """Save a PNG and report how long the image took to produce."""
    save_png(image, path)
    now = time.time()
    if tracker is not None:
        tracker.image_done(Path(path).name, now - _clock["t"])
    _clock["t"] = now


def build_scene(analysis: Analysis) -> list[SceneItem]:
    """One :class:`SceneItem` per placed body (meshes are shared between instances of a part)."""
    meshes: dict[str, Mesh] = {}

    def mesh_of(ap_id: str, geom) -> Mesh:  # type: ignore[no-untyped-def]
        if ap_id not in meshes:
            meshes[ap_id] = tessellate(geom)
        return meshes[ap_id]

    items: list[SceneItem] = []
    if analysis.report.assembly is not None and analysis.model is not None:
        for inst in build_instances(analysis):
            part = inst.ap.part
            items.append(
                SceneItem(
                    inst.id,
                    part.id,
                    inst.name,
                    mesh_of(part.id, inst.ap.geom),
                    inst.matrix,
                    part_color(part.id, part.color_rgb),
                    inst.subassembly,
                )
            )
    else:
        for ap in analysis.parts:
            items.append(
                SceneItem(
                    ap.part.id,
                    ap.part.id,
                    ap.part.name,
                    mesh_of(ap.part.id, ap.geom),
                    np.eye(4),
                    part_color(ap.part.id, ap.part.color_rgb),
                )
            )
    return [i for i in items if len(i.mesh.triangles)]


def item_labels(items: list[SceneItem]) -> dict[str, str]:
    """Label text per item: the part ID, or ``PRT001#2`` when a part is placed several times."""
    counts = Counter(i.part_id for i in items)
    seen: Counter[str] = Counter()
    out: dict[str, str] = {}
    for it in sorted(items, key=lambda i: i.key):
        seen[it.part_id] += 1
        out[it.key] = it.part_id if counts[it.part_id] == 1 else f"{it.part_id}#{seen[it.part_id]}"
    return out


def part_anchor(rendered: Rendered, item: SceneItem) -> tuple[float, float] | None:
    """A visible surface point of *item* closest (in the image) to its projected centre."""
    verts = item.world_vertices()
    proj = rendered.project_many(verts)
    mask = rendered.visible_mask(proj)
    if not mask.any():
        return None
    lo, hi = item.bounds()
    cx, cy, _d = rendered.project(0.5 * (lo + hi))
    vis = proj[mask]
    k = int(np.argmin((vis[:, 0] - cx) ** 2 + (vis[:, 1] - cy) ** 2))
    return float(vis[k, 0]), float(vis[k, 1])


def render_labelled(
    items: list[SceneItem],
    camera: CameraSpec,
    labels: dict[str, str],
    only: set[str] | None = None,
    highlight: set[str] | None = None,
    legend: list[str] | None = None,
) -> Rendered:
    """Render and draw part labels (hidden parts are omitted); the image is replaced in place."""
    rendered = render_scene(items, camera, highlight=highlight)
    marks = []
    for it in items:
        if only is not None and it.key not in only and it.part_id not in only:
            continue
        anchor = part_anchor(rendered, it)
        if anchor is not None:
            marks.append(Label(labels[it.key], *anchor))
    rendered.image = draw_labels(rendered.image, marks, legend)
    return rendered


def _legend(items: list[SceneItem], labels: dict[str, str]) -> list[str]:
    return [f"{labels[i.key]}  {i.name}" for i in sorted(items, key=lambda i: labels[i.key])][:30]


def dominant_hole_axis(part) -> tuple[Vec, Vec] | None:  # type: ignore[no-untyped-def]
    """(entry point, direction into the material) of the largest group of parallel holes, or, when
    cut-outs outnumber the holes (or there are none), of the cut-outs sharing one entry face."""
    cut = _dominant_cutout_axis(part)
    if cut is not None and len(cut[2]) > len(part.holes):
        return cut[0], cut[1]
    if not part.holes:
        return None
    groups: dict[tuple[float, ...], list] = {}  # type: ignore[type-arg]
    for h in part.holes:
        d = canonical_dir(np.array([h.axis.direction.x, h.axis.direction.y, h.axis.direction.z]))
        groups.setdefault(tuple(np.round(d, 3)), []).append(h)
    best = max(groups.values(), key=lambda g: (len(g), max(h.diameter_mm for h in g)))
    h = best[0]
    return (
        np.array([h.axis.origin.x, h.axis.origin.y, h.axis.origin.z]),
        np.array([h.axis.direction.x, h.axis.direction.y, h.axis.direction.z]),
    )


def _dominant_cutout_axis(part) -> tuple[Vec, Vec, list[Any]] | None:  # type: ignore[no-untyped-def]
    """(a point, direction into the material, cut-outs) of the largest group sharing an entry axis."""
    groups: dict[tuple[float, ...], list] = {}  # type: ignore[type-arg]
    for c in part.cutouts:
        n = np.array([c.entry_normal.x, c.entry_normal.y, c.entry_normal.z])
        groups.setdefault(tuple(np.round(canonical_dir(n), 3)), []).append(c)
    if not groups:
        return None
    best = max(groups.values(), key=lambda g: (len(g), sum(c.levels[0].area_mm2 for c in g)))
    c = best[0]
    return (
        np.array([c.levels[0].center.x, c.levels[0].center.y, c.levels[0].center.z]),
        -np.array([c.entry_normal.x, c.entry_normal.y, c.entry_normal.z]),
        best,
    )


def _cutout_normal(part, cid: str) -> np.ndarray:  # type: ignore[no-untyped-def]
    c = next(c for c in part.cutouts if c.id == cid)
    return np.array([c.entry_normal.x, c.entry_normal.y, c.entry_normal.z])


def _unit(h) -> np.ndarray:  # type: ignore[no-untyped-def]
    v = np.array([h.axis.direction.x, h.axis.direction.y, h.axis.direction.z])
    return np.asarray(v / np.linalg.norm(v))


def render_part_images(
    analysis: Analysis, folder: Path, up: str, front: str, tracker: Tracker | None = None
) -> None:
    """``images/parts/PRTxxx_iso.png`` and ``PRTxxx_axis.png`` (looking down the dominant hole axis)."""
    out_dir = folder / "images" / "parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {ap.part.id: ap for ap in analysis.parts}
    for part in rank_parts([ap.part for ap in analysis.parts])[:MAX_PART_IMAGES]:
        ap = by_id[part.id]
        item = SceneItem(
            part.id,
            part.id,
            part.name,
            tessellate(ap.geom),
            np.eye(4),
            part_color(part.id, part.color_rgb),
        )
        cam = standard_camera("iso", [item], up, front)
        rendered = render_scene([item], cam)
        marks = feature_marks(part, ap.geom, MAX_HOLE_LABELS)
        draw_cutout_outlines(rendered, part)
        rendered.image = draw_labels(rendered.image, visible_labels(rendered, marks))
        part_size = " x ".join(f"{v:.1f}" for v in part.obb.size_sorted)
        annotate(
            rendered,
            cam,
            scene_bounds([item]),
            f"{part.id}: {part_size} mm (oriented box)",
            lines=False,
        )
        _save(
            tracker,
            with_panel(rendered.image, panel_lines(part)),
            str(out_dir / f"{part.id}_iso.png"),
        )
        part.images.append(f"images/parts/{part.id}_iso.png")
        axis = dominant_hole_axis(part)
        if axis is not None:
            _origin, direction = axis
            d = direction / np.linalg.norm(direction)
            lo, hi = scene_bounds([item])
            u, _r, _f = frame(up, front)
            view_up = u - d * float(np.dot(u, d))
            if np.linalg.norm(view_up) < 1e-6:
                view_up = np.array([0.0, 1.0, 0.0]) - d * d[1]
            cam = fit_camera(
                0.5 * (lo + hi),
                -d,
                view_up / np.linalg.norm(view_up),
                (lo, hi),
                float(np.linalg.norm(hi - lo)) or 1.0,
                4 / 3,
            )
            rendered = render_scene([item], cam)
            outlines = draw_cutout_outlines(rendered, part)
            rendered.image = draw_labels(rendered.image, visible_labels(rendered, marks))
            annotate(
                rendered,
                cam,
                (lo, hi),
                f"{part.id}: looking down the main hole axis; dimensions in mm",
                lines=True,
            )
            along = [
                np.array([h.axis.origin.x, h.axis.origin.y, h.axis.origin.z])
                for h in part.holes
                if abs(float(np.dot(_unit(h), d))) > 0.99
            ]
            outlines = {
                cid: pts
                for cid, pts in outlines.items()
                if abs(float(np.dot(_cutout_normal(part, cid), d))) > 0.99
            }
            if along or outlines:
                w_mm, h_mm, corners = view_spans(cam, (lo, hi))
                px = rendered.project_many(corners)
                span = float(px[:, 0].max() - px[:, 0].min())
                if span > 0:
                    cutout_size_dimensions(rendered, outlines, part, w_mm / span)
                ordinate_dimensions(rendered, corners, w_mm, h_mm, along, outlines)
            _save(
                tracker,
                with_panel(rendered.image, panel_lines(part)),
                str(out_dir / f"{part.id}_axis.png"),
            )
            part.images.append(f"images/parts/{part.id}_axis.png")


def render_pack_images(
    analysis: Analysis, folder: Path, tracker: Tracker | None = None
) -> list[str]:
    """Write the assembly image set and part images into *folder*; returns assembly image paths."""
    report = analysis.report
    if not rendering_available():
        return []  # no usable OpenGL: the pack is written without images
    _clock["t"] = time.time()
    if tracker is not None:
        tracker.stage("render", "rendering labelled images")
    up = report.assembly.up_axis if report.assembly else str(report.meta.options.get("up", "+Z"))
    front = (
        report.assembly.front_axis
        if report.assembly
        else str(report.meta.options.get("front", "-Y"))
    )
    items = build_scene(analysis)
    written: list[str] = []
    if items:
        (folder / "images").mkdir(parents=True, exist_ok=True)
        labels = item_labels(items)
        legend = _legend(items, labels) if len(items) > 1 else None
        assembled = scene_bounds(items)
        caption = size_caption(assembled, up, front)
        kin = report.understanding.kinematics if report.understanding else None
        diag = float(np.linalg.norm(assembled[1] - assembled[0]))
        for view in ASSEMBLY_VIEWS:
            cam = standard_camera(view, items, up, front)
            r = render_labelled(items, cam, labels, legend=legend)
            annotate(r, cam, assembled, caption, lines=view != "iso")
            if view == "iso" and kin is not None and kin.joints:
                draw_joint_arrows(r, kin, diag)
            _save(tracker, r.image, str(folder / "images" / f"assembly_{view}.png"))
            written.append(f"images/assembly_{view}.png")
        exploded = copy.deepcopy([_shallow(i) for i in items])
        apply_explode(exploded)
        cam = standard_camera("iso", exploded, up, front)
        r = render_labelled(exploded, cam, labels, legend=legend)
        annotate(
            r,
            cam,
            scene_bounds(exploded),
            "Exploded view (parts moved apart). Assembled: " + caption.removeprefix("Overall "),
            lines=False,
        )
        _save(tracker, r.image, str(folder / "images" / "assembly_exploded.png"))
        written.append("images/assembly_exploded.png")
        if kin is not None and kin.joints:
            kitems = recolor_by_link(items, kin)
            cam = standard_camera("iso", kitems, up, front)
            r = render_labelled(kitems, cam, link_labels(kin), legend=link_legend(kin))
            annotate(
                r,
                cam,
                assembled,
                "Links coloured by rigid group (L0 = ground, grey); arrows are joint axes",
                lines=False,
            )
            draw_joint_arrows(r, kin, diag)
            _save(tracker, r.image, str(folder / "images" / "kinematics_iso.png"))
            written.append("images/kinematics_iso.png")
    render_part_images(analysis, folder, up, front, tracker)
    return written


def _shallow(item: SceneItem) -> SceneItem:
    """Copy an item sharing its mesh (the explode step only changes the offset)."""
    return SceneItem(
        item.key, item.part_id, item.name, item.mesh, item.matrix, item.color, item.group
    )


def render_view_file(
    path: str | Path, view: str, part: str | None, highlight: list[str] | None, out: str | Path
) -> Path:
    """CLI helper: analyse *path* and render one view (optionally one part / highlighted IDs)."""
    from stepscribe.analysis import AnalyzeOptions
    from stepscribe.api import analyze_full

    analysis = analyze_full(path, AnalyzeOptions(skip_wall_thickness=True, no_timestamp=True))
    if not analysis.parts:
        raise SystemExit(f"nothing to render: {analysis.report.errors}")
    report = analysis.report
    up = report.assembly.up_axis if report.assembly else "+Z"
    front = report.assembly.front_axis if report.assembly else "-Y"
    items = build_scene(analysis)
    if part:
        items = [i for i in items if i.part_id == part or i.key == part]
        if not items:
            raise SystemExit(f"no part or instance named {part}")
    labels = item_labels(items)
    if view == "exploded":
        items = [_shallow(i) for i in items]
        apply_explode(items)
        camera = standard_camera("iso", items, up, front)
    elif view in ("iso", "front", "top", "right"):
        camera = standard_camera(view, items, up, front)
    else:
        raise SystemExit(f"unknown view '{view}' (iso|front|top|right|exploded)")
    hl = set(highlight) if highlight else None
    r = render_labelled(
        items,
        camera,
        labels,
        only=hl,
        highlight=hl,
        legend=_legend(items, labels) if len(items) > 1 else None,
    )
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    save_png(r.image, str(target))
    return target


def highlight_keys(analysis: Analysis, refs: list[str]) -> tuple[set[str], list[str]]:
    """(instance/part keys to highlight, joint IDs to draw) for question references.

    Parts (PRT), instances (INS), links (L) and joints (KJ, whose child link is highlighted) are
    understood; other references (holes, patterns) fall back to their part when it is named.
    """
    keys: set[str] = set()
    joints: list[str] = []
    u = analysis.report.understanding
    for r in refs:
        if r.startswith(("PRT", "INS")):
            keys.add(r)
        elif u is not None and r.startswith("L"):
            link = next((g for g in u.kinematics.links if g.id == r), None)
            keys.update(link.instance_ids if link else [])
        elif u is not None and r.startswith("KJ"):
            joint = next((j for j in u.kinematics.joints if j.id == r), None)
            if joint is not None:
                joints.append(r)
                link = next(g for g in u.kinematics.links if g.id == joint.child_link)
                keys.update(link.instance_ids)
    return keys, joints


def render_highlight(analysis: Analysis, refs: list[str], folder: Path, view: str = "iso") -> Path:
    """Render the assembly with the referenced parts in a strong colour and the rest dimmed.

    Joint questions also get the joint axis as an arrow. The image is cached by its inputs.
    """
    import hashlib

    from stepscribe.render.scene import scene_bounds

    out_dir = folder / "images" / "q"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha1((view + "|" + "|".join(sorted(refs))).encode()).hexdigest()[:12]  # noqa: S324
    target = out_dir / f"{name}.png"
    if target.is_file():
        return target
    report = analysis.report
    up = report.assembly.up_axis if report.assembly else str(report.meta.options.get("up", "+Z"))
    front = (
        report.assembly.front_axis
        if report.assembly
        else str(report.meta.options.get("front", "-Y"))
    )
    items = build_scene(analysis)
    keys, joints = highlight_keys(analysis, refs)
    cam = standard_camera(view, items, up, front)
    labels = item_labels(items)
    r = render_labelled(items, cam, labels, only=keys or None, highlight=keys or None)
    u = report.understanding
    if joints and u is not None:
        lo, hi = scene_bounds(items)
        shown = u.kinematics.model_copy(
            update={"joints": [j for j in u.kinematics.joints if j.id in joints]}
        )
        draw_joint_arrows(r, shown, float(np.linalg.norm(hi - lo)))
    save_png(r.image, str(target))
    return target
