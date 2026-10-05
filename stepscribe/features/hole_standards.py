# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Match a hole diameter against screw clearance / tap-drill / nominal tables."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import yaml

from stepscribe import config
from stepscribe.models.schema import StandardMatch

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"

_METRIC_FITS = (
    ("close", "clearance_close", "ISO 273 close"),
    ("normal", "clearance_normal", "ISO 273 medium"),
    ("loose", "clearance_loose", "ISO 273 coarse"),
)


@cache
def load_table(filename: str) -> dict[str, Any]:
    """Load a YAML knowledge table from ``stepscribe/knowledge``."""
    with (KNOWLEDGE_DIR / filename).open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return data


def _candidates() -> list[tuple[str, str, str, float, str, float | None]]:
    """(designation, fit, standard label, value_mm, size key, size nominal) for every table cell."""
    rows: list[tuple[str, str, str, float, str, float | None]] = []
    metric = load_table("screws_iso_metric.yaml")["sizes"]
    for size, row in metric.items():
        for col, fit, label in _METRIC_FITS:
            rows.append((size, fit, label, float(row[col]), size, float(row["nominal"])))
        rows.append(
            (
                size,
                "tap_drill",
                "ISO 2306 tap drill",
                float(row["tap_drill"]),
                size,
                float(row["nominal"]),
            )
        )
        rows.append(
            (
                size,
                "nominal",
                "ISO 261 major diameter",
                float(row["nominal"]),
                size,
                float(row["nominal"]),
            )
        )
    unified = load_table("screws_unified.yaml")["sizes"]
    for size, row in unified.items():
        rows.append(
            (
                size,
                "tap_drill",
                "UNC tap drill",
                float(row["tap_drill"]),
                size,
                float(row["nominal"]),
            )
        )
        rows.append(
            (
                size,
                "nominal",
                "ASME B1.1 major diameter",
                float(row["nominal"]),
                size,
                float(row["nominal"]),
            )
        )
    return rows


def match_standards(diameter_mm: float, counterbore_mm: float | None = None) -> list[StandardMatch]:
    """Top matches for a hole diameter; confidence = max(0, 1 - |dev| / 0.15 mm).

    A counterbore equal to the same size's ISO 4762 counterbore boosts confidence by 0.2.
    """
    metric = load_table("screws_iso_metric.yaml")["sizes"]
    found: list[StandardMatch] = []
    for designation, fit, label, value, size, _nom in _candidates():
        dev = diameter_mm - value
        conf = max(0.0, 1.0 - abs(dev) / config.STANDARD_MATCH_SCALE)
        if conf <= config.STANDARD_MATCH_MIN_CONF:
            continue
        evidence = (
            f"Ø{diameter_mm:.2f} vs {label} Ø{value:.2f} for {designation} (dev {dev:+.2f} mm)"
        )
        cb_ref = metric.get(size, {}).get("cbore_d")
        if (
            counterbore_mm is not None
            and cb_ref is not None
            and fit.startswith("clearance")
            and abs(counterbore_mm - float(cb_ref)) <= config.CBORE_MATCH_TOL
        ):
            conf = min(1.0, conf + config.CBORE_MATCH_BOOST)
            evidence += f"; counterbore Ø{counterbore_mm:.2f} matches ISO 4762 Ø{float(cb_ref):.1f}"
        found.append(
            StandardMatch(
                confidence=round(conf, 4),
                evidence=evidence,
                standard=label,
                designation=designation,
                fit=fit,
                nominal_diameter_mm=value,
                deviation_mm=round(dev, 4),
            )
        )
    found.sort(key=lambda m: (-m.confidence, abs(m.deviation_mm), m.designation, m.fit))
    return found[: config.STANDARD_MATCH_TOP_N]


def is_likely_threaded(matches: list[StandardMatch], diameter_mm: float) -> bool:
    """Threaded if the best match is a tap drill >= 0.7, or the diameter equals a nominal size."""
    if (
        matches
        and matches[0].fit == "tap_drill"
        and matches[0].confidence >= config.THREAD_TAP_CONF
    ):
        return True
    for _d, fit, _l, value, _s, _n in _candidates():
        if fit == "nominal" and abs(diameter_mm - value) <= config.THREAD_EXACT_TOL:
            return True
    return False
