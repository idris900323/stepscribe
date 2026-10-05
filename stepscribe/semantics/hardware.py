# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Purchased hardware: screws, nuts, washers, standoffs, bearings, motors."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from stepscribe.describe.phrases import fmt
from stepscribe.features.hole_standards import load_table
from stepscribe.geometry.part_geom import PartGeom
from stepscribe.geometry.surfaces import cylinder_params, plane_normal_origin
from stepscribe.models.schema import Hole, Part, SemanticTag
from stepscribe.semantics.matcher import point, round_conf

NAME_CONF = 0.85
SHAPE_CONF = 0.6
NOT_HARDWARE = r"(?!\s*(block|seat|mount|mounting|holder|housing|plate|cover|cap|retainer|bracket|shim|fix|clip|bore|circle|hole|holes|pattern|pitch|pocket|trap|slot|recess|cutout|clearance))"

NAME_RULES: list[tuple[str, str]] = [
    (r"\bM\d+(\.\d+)?\s*[x×]\s*\d+", "screw"),
    (r"\bISO\s?4762\b", "socket head cap screw"),
    (r"\bDIN\s?912\b", "socket head cap screw"),
    (r"\bscrew" + NOT_HARDWARE, "screw"),
    (r"\bbolt" + NOT_HARDWARE, "bolt"),
    (r"\bnut" + NOT_HARDWARE, "nut"),
    (r"\bwasher", "washer"),
    (r"\b\d{3,4}\s?(zz|2rs)\b", "bearing"),
    (r"\bbearing" + NOT_HARDWARE, "bearing"),
    (r"\bLM\d+UU\b", "linear bearing"),
    (r"\b(standoff|spacer)" + NOT_HARDWARE, "standoff"),
    (r"\bservo" + NOT_HARDWARE, "servo"),
    (r"\bmotor" + NOT_HARDWARE, "motor"),
    # servo model numbers: Feetech STS3215 / SCS0009, TowerPro SG90 / MG996, Hitec HS-485, Dynamixel AX-12
    (r"\b(STS|SCS|SMS|HLS)\s?\d{3,4}\b", "servo"),
    (r"\b(SG|MG|DS|HS)[\s-]?\d{2,4}\b", "servo"),
    (r"\bAX[\s-]?\d{2}[A-Z]?\b|\bXL\d{3}\b|\bXM\d{3}\b", "servo"),
]


@dataclass
class HardwareGuess:
    """What a part probably is, and why."""

    kind: str
    description: str
    confidence: float
    evidence: str


def _name_text(name: str) -> str:
    return re.sub(r"[_\-.]+", " ", name)


def guess_from_name(name: str) -> HardwareGuess | None:
    """Name regexes (case-insensitive)."""
    text = _name_text(name)
    for pattern, kind in NAME_RULES:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return HardwareGuess(
                kind,
                f"{kind} ('{m.group(0).strip()}' in the name)",
                NAME_CONF,
                f"name '{name}' matches '{kind}'",
            )
    return None


def _metric_sizes() -> dict[str, dict[str, float]]:
    return {k: v for k, v in load_table("screws_iso_metric.yaml")["sizes"].items()}


def _size_for(diameter: float, column: str, tol: float = 0.12) -> str | None:
    best: tuple[float, str] | None = None
    for size, row in _metric_sizes().items():
        dev = abs(diameter - float(row[column]))
        if dev <= tol and (best is None or dev < best[0]):
            best = (dev, size)
    return best[1] if best else None


def _axis_planes(geom: PartGeom, axis: np.ndarray) -> int:
    count = 0
    for fi in geom.table.faces:
        if fi.kind == "plane":
            n, _o = plane_normal_origin(fi.face)
            if abs(float(np.dot(n, axis))) < 1e-3:
                count += 1
    return count


def _all_coaxial(cyls, origin: np.ndarray, axis: np.ndarray) -> bool:  # type: ignore[no-untyped-def]
    """True if every cylinder shares one axis line (a turned part, unlike a block with holes)."""
    for f in cyls:
        c = cylinder_params(f.face)
        d = c.direction / np.linalg.norm(c.direction)
        off = (c.origin - origin) - axis * float(np.dot(c.origin - origin, axis))
        if abs(abs(float(np.dot(d, axis))) - 1.0) > 1e-3 or float(np.linalg.norm(off)) > 0.05:
            return False
    return True


def _bearing_by_size(part: Part) -> HardwareGuess | None:
    """A ring whose bore, outside diameter and width match one catalogued bearing."""
    from stepscribe.semantics.matcher import entries

    a, b, c = part.obb.size_sorted
    bore = part.holes[0].diameter_mm
    best: tuple[float, str] | None = None
    for name, e in entries("bearings.yaml").items():
        od, wd, bd = float(e["od_mm"]), float(e["width_mm"]), float(e["id_mm"])
        dev = max(abs(bore - bd), abs(a - od), abs(b - od), abs(c - wd))
        if dev <= 0.15 and (best is None or dev < best[0]):
            best = (dev, name)
    if best is None:
        return None
    name = best[1]
    kind = "linear bearing" if name.startswith("LM") else "bearing"
    return HardwareGuess(
        kind,
        f"{name} {kind}",
        SHAPE_CONF,
        f"ring with bore Ø{fmt(bore, 2)}, OD Ø{fmt(a, 2)} and width {fmt(c, 2)} matching {name}",
    )


def guess_from_shape(geom: PartGeom, part: Part) -> HardwareGuess | None:
    """Conservative shape rules: nut, washer, socket head cap screw, standoff."""
    cyls = [f for f in geom.table.faces if f.kind == "cylinder"]
    if not cyls:
        return None
    big = max(cyls, key=lambda f: f.area)
    cp = cylinder_params(big.face)
    axis = cp.direction / np.linalg.norm(cp.direction)
    planes = _axis_planes(geom, axis)
    a, b, c = part.obb.size_sorted
    label = part.shape_class.label
    if (
        len(part.holes) == 1
        and part.holes[0].is_through
        and c <= 0.25 * b
        and label in ("ring", "fastener")
    ):
        bore = min((h.diameter_mm for h in part.holes), default=None)
        size = _size_for(bore, "close") if bore else None
        if size:
            return HardwareGuess(
                "washer",
                f"{size} washer",
                SHAPE_CONF,
                f"thin ring, bore Ø{fmt(bore or 0, 2)} = ISO 273 close clearance of {size}",
            )
    if len(part.holes) == 1 and part.holes[0].is_through and label in ("ring", "tube", "other"):
        found = _bearing_by_size(part)
        if found:
            return found
    if planes == 6 and part.holes and len(part.holes) == 1 and part.holes[0].is_through:
        size = _size_for(part.holes[0].diameter_mm, "tap_drill", 0.15) or _size_for(
            part.holes[0].diameter_mm, "nominal", 0.1
        )
        if size:
            return HardwareGuess(
                "nut",
                f"{size} hex nut",
                SHAPE_CONF,
                f"hex prism (6 flats) with a Ø{fmt(part.holes[0].diameter_mm, 2)} through bore matching {size}",
            )
    radii = sorted({round(cylinder_params(f.face).radius, 2) for f in cyls})
    if (
        planes >= 6
        and len(radii) == 2
        and not part.holes
        and _all_coaxial(cyls, cp.origin, axis)
        and label in ("shaft", "other", "block", "fastener")
    ):
        shank = _size_for(2 * radii[0], "nominal", 0.1)
        if shank:
            return HardwareGuess(
                "screw",
                f"{shank} socket head cap screw",
                SHAPE_CONF,
                f"two coaxial diameters (Ø{fmt(2 * radii[0], 2)} shank, Ø{fmt(2 * radii[1], 2)} head) and a {planes}-flat socket",
            )
    blind = [h for h in part.holes if not h.is_through and h.likely_threaded]
    if _threaded_at_both_ends(blind, a) and label in ("tube", "shaft", "bar", "block") and a >= b:
        return HardwareGuess(
            "standoff",
            "standoff",
            SHAPE_CONF,
            f"long prism with {len(blind)} threaded blind holes at its ends",
        )
    return None


def _threaded_at_both_ends(blind: list[Hole], length: float) -> bool:
    """Two threaded blind holes on one axis, entering from opposite ends of the part."""
    for i, h1 in enumerate(blind):
        for h2 in blind[i + 1 :]:
            d1 = np.array([h1.axis.direction.x, h1.axis.direction.y, h1.axis.direction.z])
            d2 = np.array([h2.axis.direction.x, h2.axis.direction.y, h2.axis.direction.z])
            if float(np.dot(d1, d2)) > -1 + 1e-3:
                continue  # must point at each other
            off = np.array(point(h2)) - np.array(point(h1))
            radial = float(np.linalg.norm(off - d1 * np.dot(off, d1)))
            if radial < 0.1 and abs(float(np.dot(off, d1))) >= 0.5 * length:
                return True
    return False


def apply_hardware(geom: PartGeom, part: Part) -> list[SemanticTag]:
    """Set the hardware flags on *part* and return a standoff/fastener tag if any."""
    guess = guess_from_name(part.name) or guess_from_shape(geom, part)
    if guess is None:
        return []
    part.likely_purchased_hardware = True
    part.hardware_guess = guess.description
    if guess.kind in ("screw", "bolt", "socket head cap screw", "nut", "washer"):
        part.shape_class = part.shape_class.model_copy(
            update={
                "label": "fastener",
                "confidence": round_conf(guess.confidence),
                "evidence": guess.evidence,
            }
        )
    kind = "standoff" if guess.kind == "standoff" else "other"
    return [
        SemanticTag(
            confidence=round_conf(guess.confidence),
            evidence=f"{guess.description} ({guess.confidence:.2f}): {guess.evidence}",
            kind=kind,
            label=guess.description,
            knowledge_id=None,
            feature_ids=[],
        )
    ]
