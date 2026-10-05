# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Motor and board mounting patterns."""

from __future__ import annotations

import numpy as np

from stepscribe.describe.phrases import fmt, hole_callout
from stepscribe.models.schema import Hole, HolePattern, Part, SemanticTag
from stepscribe.semantics.matcher import (
    entries,
    linear_score,
    point,
    round_conf,
    screw_hole_range,
)

SPACING_TOL = 0.3  # mm: both spacings must be within this of the catalogue value
PILOT_SLACK = 0.5  # mm over the pilot diameter that still scores 1.0
PILOT_MAX_OVERSIZE = 6.0  # mm: a bigger bore than this is not a pilot clearance
MOTOR_WEIGHTS = (0.5, 0.2, 0.3)  # spacing, hole diameter, pilot bore
BOARD_WEIGHTS = (0.75, 0.25)
NO_PILOT_CAP = 0.7
BOARD_CAP = 0.8


def _dia_score(pattern: HolePattern, screw: str) -> float:
    lo, hi = screw_hole_range(screw)
    d = pattern.diameter_mm
    if lo - 0.05 <= d <= hi + 0.1:
        return 1.0
    return linear_score(min(abs(d - lo), abs(d - hi)), 0.5)


def _central_bore(part: Part, pattern: HolePattern, pilot: float) -> tuple[Hole | None, float]:
    """A bore near the pattern centre that can take the motor's pilot boss."""
    if pattern.center is None:
        return None, 0.0
    c = np.array([pattern.center.x, pattern.center.y, pattern.center.z])
    n = np.array([pattern.normal.x, pattern.normal.y, pattern.normal.z])
    best: tuple[Hole | None, float] = (None, 0.0)
    for h in part.holes:
        if h.id in pattern.hole_ids:
            continue
        off = np.array(point(h)) - c
        radial = float(np.linalg.norm(off - n * np.dot(off, n)))
        d = np.array([h.axis.direction.x, h.axis.direction.y, h.axis.direction.z])
        if radial > 0.5 or abs(abs(float(np.dot(d, n))) - 1.0) > 1e-3:
            continue
        over = h.diameter_mm - pilot
        if -0.1 <= over <= PILOT_SLACK:
            score = 1.0
        elif PILOT_SLACK < over <= PILOT_MAX_OVERSIZE:
            score = 0.7
        else:
            continue
        if score > best[1]:
            best = (h, score)
    return best


def _grid_dims(p: HolePattern) -> tuple[float, float] | None:
    if p.kind != "rectangular_grid" or p.col_pitch_mm is None or p.row_pitch_mm is None:
        return None
    a, b = sorted((p.col_pitch_mm, p.row_pitch_mm), reverse=True)
    return a, b


def match_motor_mounts(part: Part) -> list[SemanticTag]:
    """4 holes on a catalogue square, optionally plus a pilot bore, make a motor mount."""
    tags: list[SemanticTag] = []
    for pat in part.hole_patterns:
        dims = _grid_dims(pat)
        if dims is None or pat.count != 4:
            continue
        for key, e in entries("motors.yaml").items():
            sp = float(e["spacing_mm"])
            s_space = (
                linear_score(dims[0] - sp, SPACING_TOL) + linear_score(dims[1] - sp, SPACING_TOL)
            ) / 2
            if (
                min(
                    linear_score(dims[0] - sp, SPACING_TOL), linear_score(dims[1] - sp, SPACING_TOL)
                )
                == 0
            ):
                continue
            s_dia = _dia_score(pat, e["screw"])
            bore, s_pilot = _central_bore(part, pat, float(e["pilot_diameter_mm"]))
            w = MOTOR_WEIGHTS
            conf = round_conf(0.95 * (w[0] * s_space + w[1] * s_dia + w[2] * s_pilot))
            if bore is None:
                conf = round_conf(NO_PILOT_CAP * (0.7 * s_space + 0.3 * s_dia), NO_PILOT_CAP)
            evidence = f"{hole_callout_pattern(pat)} on {fmt(dims[0])} × {fmt(dims[1])} square"
            evidence += (
                f" + Ø{fmt(bore.diameter_mm)} central bore (pilot Ø{fmt(float(e['pilot_diameter_mm']))})"
                if bore
                else "; no central pilot bore found"
            )
            ids = [pat.id, *pat.hole_ids, *([bore.id] if bore else [])]
            tags.append(
                SemanticTag(
                    confidence=conf,
                    evidence=f"{e['label']} motor mount ({conf:.2f}): {evidence}",
                    kind="motor_mount",
                    label=f"{e['label']} motor mount",
                    knowledge_id=f"motors.{key}",
                    feature_ids=ids,
                )
            )
    return _best_per_pattern(tags)


def hole_callout_pattern(pat: HolePattern) -> str:
    """'4 × Ø3.4' style text for a pattern's holes."""
    return f"{pat.count} × Ø{fmt(pat.diameter_mm)}"


def match_board_mounts(part: Part) -> list[SemanticTag]:
    """Rectangular hole patterns on a catalogue board footprint."""
    tags: list[SemanticTag] = []
    for pat in part.hole_patterns:
        dims = _grid_dims(pat)
        if dims is None or pat.count != 4:
            continue
        for key, e in entries("boards.yaml").items():
            a, b = (float(v) for v in e["spacing_mm"])
            sa, sb = linear_score(dims[0] - a, SPACING_TOL), linear_score(dims[1] - b, SPACING_TOL)
            if sa == 0 or sb == 0:
                continue
            w = BOARD_WEIGHTS
            conf = round_conf(w[0] * (sa + sb) / 2 + w[1] * _dia_score(pat, e["screw"]), BOARD_CAP)
            tags.append(
                SemanticTag(
                    confidence=conf,
                    evidence=(
                        f"{e['label']} board mount ({conf:.2f}): {pat.count} × Ø{fmt(pat.diameter_mm)} on "
                        f"{fmt(dims[0])} × {fmt(dims[1])} rectangle"
                    ),
                    kind="board_mount",
                    label=f"{e['label']} board mount",
                    knowledge_id=f"boards.{key}",
                    feature_ids=[pat.id, *pat.hole_ids],
                )
            )
    return _best_per_pattern(tags)


def _best_per_pattern(tags: list[SemanticTag]) -> list[SemanticTag]:
    """Keep the highest-confidence tag for each pattern (ties: knowledge id order)."""
    best: dict[str, SemanticTag] = {}
    for t in sorted(tags, key=lambda t: (-t.confidence, t.knowledge_id or "")):
        best.setdefault(t.feature_ids[0], t)
    return sorted(best.values(), key=lambda t: t.feature_ids[0])


def fastener_patterns(part: Part) -> list[SemanticTag]:
    """Plain 'N × M3 clearance holes' tags for patterns with a strong screw match."""
    tags: list[SemanticTag] = []
    for pat in part.hole_patterns:
        hole = next(h for h in part.holes if h.id == pat.hole_ids[0])
        if not hole.standard_matches or hole.standard_matches[0].confidence < 0.7:
            continue
        m = hole.standard_matches[0]
        what = m.fit.replace("_", " ")
        conf = round_conf(m.confidence * 0.9)
        tags.append(
            SemanticTag(
                confidence=conf,
                evidence=f"{pat.description}; {m.evidence}",
                kind="fastener_hole_pattern",
                label=f"{pat.count} × {m.designation} {what} holes ({hole_callout(hole)})",
                knowledge_id=None,
                feature_ids=[pat.id, *pat.hole_ids],
            )
        )
    return tags
