# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Generic helpers for matching part features against sourced knowledge entries."""

from __future__ import annotations

from typing import Any

from stepscribe.features.hole_standards import load_table
from stepscribe.models.schema import Hole

MAX_CONFIDENCE = 0.95


def entries(filename: str) -> dict[str, dict[str, Any]]:
    """Entries of a knowledge YAML file (each carries a ``source``)."""
    table = load_table(filename)
    result: dict[str, dict[str, Any]] = table["entries"]
    return result


def linear_score(deviation: float, tolerance: float) -> float:
    """1.0 at zero deviation, falling linearly to 0.0 at *tolerance*."""
    if tolerance <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(deviation) / tolerance)


def screw_hole_range(screw: str) -> tuple[float, float]:
    """(nominal, loose-clearance) diameters of an ISO metric screw: the plausible hole range."""
    row = load_table("screws_iso_metric.yaml")["sizes"][screw]
    return float(row["nominal"]), float(row["loose"])


def round_conf(x: float, cap: float = MAX_CONFIDENCE) -> float:
    """Clamp an inference confidence to [0, cap] and round to two decimals."""
    return round(max(0.0, min(cap, x)), 2)


def point(h: Hole) -> tuple[float, float, float]:
    """Entry point of a hole as a tuple."""
    o = h.axis.origin
    return o.x, o.y, o.z
