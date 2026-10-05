# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Bearing seats: a concave bore sized for a bearing OD and its width."""

from __future__ import annotations

from stepscribe.describe.phrases import fmt
from stepscribe.models.schema import Hole, Part, SemanticTag
from stepscribe.semantics.matcher import entries, round_conf

OD_FIT_MIN = -0.05  # mm: seat diameter minus OD (slight interference allowed)
OD_FIT_GOOD = 0.10
OD_FIT_MAX = 0.25
WIDTH_TOL = 0.3  # seat depth vs bearing width
WIDTH_TOL_LOOSE = 0.8
WEIGHT_OD, WEIGHT_WIDTH = 0.6, 0.4
SHOULDER_BONUS = 0.05
BEARING_CAP = 0.9


def _seat_candidates(h: Hole) -> list[tuple[float, float, bool]]:
    """(diameter, depth, has_shoulder) readings of a hole that could be a bearing seat."""
    out: list[tuple[float, float, bool]] = []
    if h.counterbore_diameter_mm is not None and h.counterbore_depth_mm is not None:
        out.append((h.counterbore_diameter_mm, h.counterbore_depth_mm, True))
    if h.depth_mm is not None and (not h.is_through or h.entry_type == "plain"):
        out.append((h.diameter_mm, h.depth_mm, not h.is_through))
    return out


def _score(od_dev: float, depth_dev: float) -> float | None:
    if od_dev < OD_FIT_MIN or od_dev > OD_FIT_MAX:
        return None
    s_od = 1.0 if od_dev <= OD_FIT_GOOD else 0.7
    if abs(depth_dev) <= WIDTH_TOL:
        s_w = 1.0
    elif abs(depth_dev) <= WIDTH_TOL_LOOSE:
        s_w = 0.6
    else:
        return None
    return WEIGHT_OD * s_od + WEIGHT_WIDTH * s_w


def match_bearing_seats(part: Part) -> list[SemanticTag]:
    """One tag per hole that fits a catalogued bearing's OD and width.

    Bearings with identical OD and width (e.g. 608 and 627) cannot be told apart from the seat
    alone, so all of them are named in the label.
    """
    tags: list[SemanticTag] = []
    for h in part.holes:
        scored: list[tuple[float, str, float, float, bool]] = []
        for dia, depth, shoulder in _seat_candidates(h):
            for name, e in entries("bearings.yaml").items():
                s = _score(dia - float(e["od_mm"]), depth - float(e["width_mm"]))
                if s is not None:
                    scored.append(
                        (s + (SHOULDER_BONUS if shoulder else 0.0), name, dia, depth, shoulder)
                    )
        if not scored:
            continue
        top = max(sc[0] for sc in scored)
        near = sorted((sc for sc in scored if top - sc[0] < 1e-6), key=lambda sc: sc[1])
        names = sorted({sc[1] for sc in near}, key=_name_key)
        _s, _n, dia, depth, shoulder = near[0]
        first = entries("bearings.yaml")[names[0]]
        conf = round_conf(BEARING_CAP * top, BEARING_CAP)
        joined = " / ".join(names)
        evidence = (
            f"{joined} bearing seat ({conf:.2f}): Ø{fmt(dia, 2)} × {fmt(depth, 2)} deep "
            f"{'with a shoulder' if shoulder else 'without a shoulder'} vs OD {fmt(float(first['od_mm']))} × "
            f"width {fmt(float(first['width_mm']))}"
        )
        if len(names) > 1:
            evidence += "; these bearings share OD and width, so the seat cannot tell them apart"
        tags.append(
            SemanticTag(
                confidence=conf,
                evidence=evidence,
                kind="bearing_seat",
                label=f"{joined} bearing seat",
                knowledge_id=f"bearings.{names[0]}",
                feature_ids=[h.id],
            )
        )
    return tags


def _name_key(name: str) -> tuple[int, str]:
    return (
        0 if name.isdigit() and len(name) == 3 and name[0] == "6" and name[1] == "0" else 1,
        name,
    )
