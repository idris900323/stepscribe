# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""``stepscribe eval``: factual questions with answers derived from report.json.

Measures how well a given model understands the context pack. The questions and expected answers
are computed from the report, never from a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from stepscribe.models.schema import Report

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
SUFFIX = "Answer with only the value (a single number or a single word), no explanation."


@dataclass(frozen=True)
class Question:
    """One factual question."""

    text: str
    expected: str
    numeric: bool
    tolerance: float = 0.0


def build_questions(report: Report, limit: int = 12) -> list[Question]:
    """Deterministic question list for a report (counts, sizes, hole facts, relations)."""
    qs: list[Question] = [
        Question(
            f"How many unique parts does the design have? {SUFFIX}", str(len(report.parts)), True
        )
    ]
    if report.assembly:
        a = report.assembly
        qs.append(
            Question(
                f"How many part instances are in the assembly? {SUFFIX}",
                str(len(a.instances)),
                True,
            )
        )
        qs.append(
            Question(
                f"How many fastener joints were found? {SUFFIX}", str(len(a.fastener_joints)), True
            )
        )
        multi = [b for b in a.bom if b.quantity > 1]
        if multi:
            b = sorted(multi, key=lambda x: (-x.quantity, x.part_id))[0]
            qs.append(
                Question(
                    f"What is the BOM quantity of part {b.part_id}? {SUFFIX}", str(b.quantity), True
                )
            )
    for p in sorted(report.parts, key=lambda x: -x.mass.volume_mm3)[:3]:
        qs.append(
            Question(f"How many holes does part {p.id} have? {SUFFIX}", str(len(p.holes)), True)
        )
        qs.append(
            Question(
                f"What is the shape class of part {p.id}? {SUFFIX}",
                p.shape_class.label.replace("_", " "),
                False,
            )
        )
        qs.append(
            Question(
                f"What is the largest overall dimension of part {p.id} in mm? {SUFFIX}",
                f"{p.obb.size_sorted[0]:.1f}",
                True,
                0.15,
            )
        )
        if p.holes:
            h = p.holes[0]
            qs.append(
                Question(
                    f"What is the diameter in mm of hole {h.id} of part {p.id}? {SUFFIX}",
                    f"{h.diameter_mm:.1f}",
                    True,
                    0.15,
                )
            )
        if p.shape_class.thickness_mm:
            qs.append(
                Question(
                    f"What is the thickness in mm of part {p.id}? {SUFFIX}",
                    f"{p.shape_class.thickness_mm:.1f}",
                    True,
                    0.15,
                )
            )
    return qs[:limit]


def score(answer: str, q: Question) -> bool:
    """Numeric: the first number must match within tolerance. Text: expected words appear."""
    if q.numeric:
        found = NUMBER.findall(answer)
        if not found:
            return False
        return abs(float(found[0]) - float(q.expected)) <= max(q.tolerance, 1e-9)
    return q.expected.lower() in answer.lower()


@dataclass
class EvalResult:
    """Outcome of one question."""

    question: Question
    answer: str
    correct: bool


def accuracy(results: list[EvalResult]) -> float:
    """Fraction correct (0 for an empty list)."""
    return sum(r.correct for r in results) / len(results) if results else 0.0
