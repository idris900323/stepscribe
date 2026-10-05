# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Token estimation and context-pack budgeting."""

from __future__ import annotations

import math
from dataclasses import dataclass

from stepscribe import config
from stepscribe.models.schema import Part


def estimate_tokens(text: str) -> int:
    """Rough token count: ``ceil(chars / 4)``."""
    return math.ceil(len(text) / config.CHARS_PER_TOKEN)


def feature_count(part: Part) -> int:
    """Number of recognised features on a part."""
    return (
        len(part.holes)
        + len(part.cutouts)
        + sum(1 for s in part.slots if not s.cutout_id)
        + sum(1 for p in part.pockets if not p.cutout_id)
        + len(part.bosses)
        + len(part.fillets)
        + len(part.chamfers)
        + len(part.semantic_tags)
    )


def importance(part: Part) -> float:
    """Size x feature count; purchased hardware is de-emphasised."""
    size = max(part.obb.size_sorted) if part.obb.size_sorted else 0.0
    score = size * (1 + feature_count(part))
    return score * (0.2 if part.likely_purchased_hardware else 1.0)


def rank_parts(parts: list[Part]) -> list[Part]:
    """Parts by descending importance; ties by ID for determinism."""
    return sorted(parts, key=lambda p: (-importance(p), p.id))


@dataclass
class PackSections:
    """Pieces composed into ``context_pack.md``."""

    level0: str
    assembly_core: str
    level1: dict[str, str]  # part id -> paragraph
    level1_short: dict[str, str]  # part id -> one line
    level2: dict[str, str]  # part id -> full markdown
    ranked_ids: list[str]
    understanding: str = ""  # compact understanding block, placed right after the header


def compose_budgeted(
    sections: PackSections, budget_tokens: int, header: str = ""
) -> tuple[str, list[str], list[str]]:
    """Build the single-file pack under a token budget.

    Returns ``(text, parts_with_full_detail, parts_one_line_only)``.
    Level 0 and the assembly core are always present; Level 1 for every part (dropping the least
    important to one line if over budget); Level 2 in importance order while budget remains.
    """
    fixed = "\n\n".join(
        x for x in (sections.understanding, sections.level0, sections.assembly_core) if x
    )
    ids = list(sections.ranked_ids)
    short_ids: list[str] = []
    l1 = dict(sections.level1)

    def l1_tokens() -> int:
        return estimate_tokens("\n\n".join(l1[i] for i in ids))

    used = estimate_tokens(header) + estimate_tokens(fixed)
    while ids and used + l1_tokens() > budget_tokens and len(short_ids) < len(ids):
        victim = ids[len(ids) - 1 - len(short_ids)]
        l1[victim] = sections.level1_short[victim]
        short_ids.append(victim)
    used += l1_tokens()
    detailed: list[str] = []
    bodies: list[str] = []
    for pid in ids:
        if pid in short_ids:
            continue
        cost = estimate_tokens(sections.level2[pid])
        if used + cost > budget_tokens:
            continue
        used += cost
        detailed.append(pid)
        bodies.append(sections.level2[pid])
    parts_text = "\n\n".join(l1[i] for i in ids)
    text = "\n\n".join(
        x
        for x in (header, fixed, "## Part summaries\n\n" + parts_text if ids else "", *bodies)
        if x
    )
    return text.rstrip() + "\n", detailed, short_ids
