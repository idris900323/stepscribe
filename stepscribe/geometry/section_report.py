# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""``stepscribe section``: a planar cut as a PNG plus a text summary (Phase 8 core, no HLR)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepGProp import BRepGProp
from OCP.GCPnts import GCPnts_QuasiUniformDeflection
from OCP.GProp import GProp_GProps
from PIL import Image, ImageDraw

from stepscribe.analysis import AnalyzeOptions
from stepscribe.api import analyze_full
from stepscribe.assembly import build_instances
from stepscribe.describe.phrases import fmt
from stepscribe.geometry.occ_utils import as_edge, edges_of, to_np
from stepscribe.geometry.sections import section_shape

IMAGE_SIZE = (1200, 900)
MARGIN = 60
AXES = {"x": 0, "y": 1, "z": 2}
PLANE_RE = re.compile(r"^\s*([xyzXYZ])\s*=\s*(-?\d+(?:\.\d+)?)\s*$")


def parse_plane(text: str) -> tuple[np.ndarray, np.ndarray]:
    """'z=12.5' -> (origin, unit normal)."""
    m = PLANE_RE.match(text)
    if not m:
        raise ValueError(f"plane must look like 'z=12.5', got '{text}'")
    idx = AXES[m.group(1).lower()]
    normal = np.zeros(3)
    normal[idx] = 1.0
    return normal * float(m.group(2)), normal


def _polylines(shape) -> list[np.ndarray]:  # type: ignore[no-untyped-def]
    out = []
    for e in edges_of(shape):
        curve = BRepAdaptor_Curve(as_edge(e))
        sampler = GCPnts_QuasiUniformDeflection(curve, 0.05)
        if sampler.IsDone() and sampler.NbPoints() >= 2:
            out.append(
                np.array([to_np(sampler.Value(i)) for i in range(1, sampler.NbPoints() + 1)])
            )
    return out


def section_file(path: str | Path, plane: str, part: str | None, out: str | Path) -> str:
    """Cut the model at *plane*, write a PNG, and return a text summary."""
    origin, normal = parse_plane(plane)
    analysis = analyze_full(path, AnalyzeOptions(skip_wall_thickness=True, no_timestamp=True))
    if not analysis.parts:
        raise SystemExit(f"nothing to cut: {analysis.report.errors}")
    targets = (
        [(i.name, i.part_id, i.shape) for i in build_instances(analysis)]
        if analysis.report.assembly
        else [(ap.part.name, ap.part.id, ap.geom.shape) for ap in analysis.parts]
    )
    if part:
        targets = [t for t in targets if t[1] == part or t[0] == part]
        if not targets:
            raise SystemExit(f"no part or instance named {part}")
    extent = 10 * max(
        float(np.linalg.norm(ap.geom.bounds[1] - ap.geom.bounds[0])) for ap in analysis.parts
    )
    axis = int(np.argmax(np.abs(normal)))
    keep = [i for i in range(3) if i != axis]
    lines: list[tuple[str, np.ndarray]] = []
    summary = [
        f"Section at {plane} (plane normal {tuple(int(v) for v in normal)}); axes shown: "
        + ", ".join("xyz"[k] for k in keep)
        + "."
    ]
    for name, pid, shape in targets:
        sec = section_shape(shape, origin, normal, extent)
        if sec is None or sec.IsNull():
            continue
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(sec, props)
        area = float(props.Mass())
        if area <= 1e-6:
            continue
        polys = _polylines(sec)
        pts = np.vstack(polys)
        lo, hi = pts.min(axis=0)[keep], pts.max(axis=0)[keep]
        summary.append(
            f"{pid} {name}: cut area {fmt(area, 1)} mm²; extent {fmt(float(hi[0] - lo[0]))} × {fmt(float(hi[1] - lo[1]))} mm in the plane."
        )
        lines.extend((pid, p) for p in polys)
    if not lines:
        summary.append("The plane does not cut any solid.")
        return "\n".join(summary)
    allpts = np.vstack([p for _n, p in lines])[:, keep]
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    scale = min(
        (IMAGE_SIZE[0] - 2 * MARGIN) / max(hi[0] - lo[0], 1e-6),
        (IMAGE_SIZE[1] - 2 * MARGIN) / max(hi[1] - lo[1], 1e-6),
    )
    img = Image.new("RGB", IMAGE_SIZE, "white")
    draw = ImageDraw.Draw(img)
    for _pid, poly in lines:
        xy = [
            (
                MARGIN + (p[keep[0]] - lo[0]) * scale,
                IMAGE_SIZE[1] - MARGIN - (p[keep[1]] - lo[1]) * scale,
            )
            for p in poly
        ]
        draw.line(xy, fill=(0, 0, 0), width=2)
    draw.text((10, 10), f"section {plane}   scale {scale:.2f} px/mm", fill=(0, 0, 0))
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    img.save(target)
    summary.append(f"Image: {target}")
    return "\n".join(summary)
