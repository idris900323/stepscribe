# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Deterministic sentence builders: short declarative text, units, IDs."""

from __future__ import annotations

from stepscribe.models.schema import Hole


def fmt(x: float, nd: int = 1) -> str:
    """Number with fixed decimals, never '-0.0'."""
    s = f"{x:.{nd}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def hole_callout(h: Hole) -> str:
    """Drawing-style callout: ``Ø5.5 THRU ⌴ Ø9.5 ↧ 5.4``."""
    parts = [f"Ø{fmt(h.diameter_mm)}"]
    if h.is_through:
        parts.append("THRU")
    elif h.depth_mm is not None:
        parts.append(f"↧ {fmt(h.depth_mm)}")
    if h.counterbore_diameter_mm is not None:
        cb = f"⌴ Ø{fmt(h.counterbore_diameter_mm)}"
        if h.counterbore_depth_mm is not None:
            cb += f" ↧ {fmt(h.counterbore_depth_mm)}"
        parts.append(cb)
    if h.countersink_diameter_mm is not None:
        ang = h.countersink_angle_deg
        parts.append(f"⌵ Ø{fmt(h.countersink_diameter_mm)} × {fmt(ang or 0.0, 0)}°")
    return " ".join(parts)


def plural(n: int, word: str) -> str:
    """'1 plate', '2 plates'."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def join_and(items: list[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]
