# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Bill of materials: instances grouped by part ID and content hash."""

from __future__ import annotations

from stepscribe.assembly.tree import InstanceData
from stepscribe.models.schema import BOMLine


def build_bom(instances: list[InstanceData]) -> list[BOMLine]:
    """One line per unique part (grouped by ``part_id`` + ``content_hash``), sorted by part ID."""
    groups: dict[tuple[str, str], list[InstanceData]] = {}
    for inst in instances:
        groups.setdefault((inst.part_id, inst.ap.part.content_hash), []).append(inst)
    lines: list[BOMLine] = []
    for (pid, _hash), members in sorted(groups.items()):
        part = members[0].ap.part
        lines.append(
            BOMLine(
                part_id=pid,
                name=part.name,
                quantity=len(members),
                mass_each_g=part.mass.mass_g,
                category="purchased_hardware" if part.likely_purchased_hardware else "fabricated",
            )
        )
    return lines
