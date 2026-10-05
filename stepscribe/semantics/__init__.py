# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Semantic interpretation: motor mounts, bearings, extrusions, shafts, hardware."""

from __future__ import annotations

from stepscribe.geometry.part_geom import PartGeom
from stepscribe.models.schema import Part, SemanticTag
from stepscribe.semantics.bearings import match_bearing_seats
from stepscribe.semantics.extrusions import match_extrusions
from stepscribe.semantics.hardware import apply_hardware
from stepscribe.semantics.mounts import fastener_patterns, match_board_mounts, match_motor_mounts
from stepscribe.semantics.shafts import match_shaft_bores, match_shafts


def _motor_bodies(part: Part, tags: list[SemanticTag]) -> list[SemanticTag]:
    """A hole square with the catalogue's centring boss (not a bore) is the motor itself."""
    from stepscribe.semantics.matcher import entries

    out: list[SemanticTag] = []
    for t in tags:
        key = (t.knowledge_id or "").removeprefix("motors.")
        row = entries("motors.yaml").get(key)
        if row and any(
            abs(b.diameter_mm - float(row["pilot_diameter_mm"])) <= 0.6 for b in part.bosses
        ):
            part.likely_purchased_hardware = True
            part.hardware_guess = f"{row['label']} stepper motor"
            t = t.model_copy(
                update={
                    "label": f"{row['label']} stepper motor (body)",
                    "evidence": t.evidence
                    + f"; Ø{row['pilot_diameter_mm']} centring boss present, so this is the motor, not its mount",
                }
            )
        out.append(t)
    return out


def run_semantics(geom: PartGeom, part: Part) -> list[SemanticTag]:
    """All semantic tags for one part; also sets the purchased-hardware flags on *part*.

    Order is stable so that output is deterministic: hardware, mounts, bearings, extrusions,
    shafts, then plain fastener hole patterns.
    """
    tags: list[SemanticTag] = []
    tags += apply_hardware(geom, part)
    tags += _motor_bodies(part, match_motor_mounts(part))
    tags += match_board_mounts(part)
    tags += match_bearing_seats(part)
    tags += match_extrusions(geom, part)
    if not part.likely_purchased_hardware:
        tags += match_shaft_bores(part)
        tags += match_shafts(geom, part)
        tags += fastener_patterns(part)
    return tags
