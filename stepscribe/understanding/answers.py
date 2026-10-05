# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Designer answers and their deterministic effects.

An answer never needs the STEP file again: effects are applied to the cached analysis and the
cheap stages are re-run. User answers always win over inferences; when an answer contradicts
strong geometric evidence it is accepted and a visible conflict note is added.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from stepscribe import config
from stepscribe.models.schema import (
    Evidence,
    Hypothesis,
    Part,
    ProcessGuess,
    Question,
    WeakSpot,
)
from stepscribe.understanding.process import PROCESSES

CONFLICT_GEOMETRY_CONF = 0.5  # the geometry's own favourite must be at least this sure
CONFLICT_ANSWER_CONF = 0.1  # ...and the designer's choice at most this likely
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class Effects:
    """What the answers change, grouped by stage."""

    material: str | None = None
    payload_g: float | None = None
    process: dict[str, str] = field(default_factory=dict)  # part id -> label
    roles: dict[str, str] = field(default_factory=dict)  # part id -> label
    drives: dict[str, str] = field(default_factory=dict)  # joint id -> "designer:motor"
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    suppress_floating: set[str] = field(default_factory=set)  # instance ids
    fasteners_missing: list[tuple[str, str]] = field(default_factory=list)
    text: dict[str, str] = field(default_factory=dict)  # purpose / load / environment
    unconfirmed: list[str] = field(default_factory=list)  # question ids skipped
    unsure: list[str] = field(default_factory=list)


def answers_from_context(raw: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """The ``answers:`` section of design_context.yaml, ignoring empty stubs."""
    out: dict[str, dict[str, Any]] = {}
    for qid, row in (raw or {}).items():
        if isinstance(row, dict) and row.get("status") in ("answered", "skipped", "not_sure"):
            out[str(qid)] = dict(row)
    return out


def collect_effects(answers: dict[str, dict[str, Any]], questions: dict[str, Question]) -> Effects:
    """Translate answers into effects; answers about questions that no longer exist are ignored."""
    fx = Effects()
    for qid, a in sorted(answers.items()):
        q = questions.get(qid)
        status = a.get("status")
        if q is None:
            continue
        if status == "skipped":
            fx.unconfirmed.append(qid)
            continue
        if status == "not_sure":
            fx.unsure.append(qid)
            continue
        if status != "answered":
            continue
        v = a.get("value")
        if q.kind == "material" and isinstance(v, str) and v in config.MATERIAL_KEYWORDS:
            fx.material = v
        elif q.kind == "payload":
            nums = NUMBER.findall(str(v))
            if nums:
                fx.payload_g = float(nums[0])
        elif q.kind == "process" and isinstance(v, str) and v in PROCESSES:
            for pid in q.refs:
                fx.process[pid] = v
        elif q.kind in ("confirm_role", "choose_role") and isinstance(v, str) and q.refs:
            fx.roles[q.refs[0]] = v
        elif q.kind == "joint_drive":
            chosen = v if isinstance(v, list) else [v]
            if len(q.refs) > 1 or q.answer_type == "multi_choice":
                for jid in q.refs:
                    if jid in chosen and jid != "none":
                        fx.drives[jid] = "designer:motor"
                    else:
                        fx.drives.setdefault(jid, "designer:passive")
            elif q.refs and isinstance(v, str):
                fx.drives[q.refs[0]] = f"designer:{v}"
        elif q.kind == "joint_range" and q.refs:
            nums = NUMBER.findall(str(v))
            if len(nums) >= 2:
                lo, hi = sorted(float(n) for n in nums[:2])
                fx.ranges[q.refs[0]] = (lo, hi)
        elif q.kind == "missing_part" and v in ("missing", "purchased", "loose"):
            fx.suppress_floating.update(q.refs)
        elif q.kind == "connection_type" and v == "fasteners_missing" and len(q.refs) == 2:
            fx.fasteners_missing.append((q.refs[0], q.refs[1]))
        elif q.kind in ("purpose", "load", "environment") and v:
            fx.text[q.kind] = str(v)
    return fx


def apply_material(analysis: Any, material: str) -> None:
    """Set density, mass and the assembly's totals from a material keyword (the designer's)."""
    density = config.MATERIAL_KEYWORDS[material]
    for ap in analysis.parts:
        m = ap.part.mass
        m.density_g_cm3 = density
        m.mass_g = m.volume_mm3 * 1e-3 * density
        m.material_assumed = material
        m.material_source = "user"
        _plastic_process(ap.part, material)
    asm = analysis.report.assembly
    if asm is None:
        return
    by_part = {ap.part.id: ap.part for ap in analysis.parts}
    for line in asm.bom:
        line.mass_each_g = by_part[line.part_id].mass.mass_g
    total = 0.0
    for inst in asm.instances:
        total += by_part[inst.part_id].mass.mass_g or 0.0
    asm.total_mass_g = total
    import numpy as np

    from stepscribe.geometry.properties import vec3

    wsum = np.zeros(3)
    for inst in asm.instances:
        p = by_part[inst.part_id]
        c = np.array([p.mass.centroid.x, p.mass.centroid.y, p.mass.centroid.z])
        rot = np.array(inst.rotation_matrix)
        g = rot @ c + np.array([inst.position.x, inst.position.y, inst.position.z])
        wsum += (p.mass.mass_g or 0.0) * g
    if total > 0:
        asm.center_of_mass = vec3(wsum / total)
    for s in asm.subassemblies:
        ms = [by_part[i.part_id].mass.mass_g for i in asm.instances if i.subassembly == s.name]
        s.mass_g = (
            sum(x for x in ms if x is not None) if ms and all(x is not None for x in ms) else None
        )


PLASTICS = {"pla", "petg", "abs"}


def _plastic_process(part: Part, material: str) -> None:
    """A printable plastic makes 3D printing the likely process (unless the designer said otherwise)."""
    u = part.understanding
    if (
        material not in PLASTICS
        or u is None
        or u.process.source == "user"
        or part.likely_purchased_hardware
    ):
        return
    ranked = u.process.ranked
    if ranked and ranked[0].label == "fdm_3d_printed" and ranked[0].confidence >= 0.8:
        return
    ev = Evidence(
        code=f"material:{material}",
        refs=[part.id],
        weight=1.0,
        text=f"the designer gave {material.upper()}, a filament material",
    )
    first = Hypothesis(label="fdm_3d_printed", confidence=0.8, evidence=[ev])
    u.process = ProcessGuess(
        ranked=[first] + [h for h in ranked if h.label != "fdm_3d_printed"], source="inferred"
    )


def apply_process(parts: list[Part], overrides: dict[str, str]) -> list[str]:
    """Replace the process guess of the named parts; returns conflict notes."""
    conflicts: list[str] = []
    for p in parts:
        label = overrides.get(p.id)
        if label is None or p.understanding is None:
            continue
        before = p.understanding.process.ranked
        if p.understanding.process.source != "user":
            top = before[0] if before else None
            mine = next((h for h in before if h.label == label), None)
            if (
                top is not None
                and top.label != label
                and top.confidence >= CONFLICT_GEOMETRY_CONF
                and (mine is None or mine.confidence <= CONFLICT_ANSWER_CONF)
            ):
                conflicts.append(
                    f"Designer says {p.id} ({p.name}) is made by {label.replace('_', ' ')}, but the geometry "
                    f"points to {top.label.replace('_', ' ')} (confidence {top.confidence:.2f}) and shows "
                    f"little sign of {label.replace('_', ' ')}"
                )
        p.understanding.process = ProcessGuess(
            source="user",
            ranked=[
                Hypothesis(
                    label=label,
                    confidence=1.0,
                    evidence=[
                        Evidence(code="user", refs=[p.id], weight=1.0, text="given by the designer")
                    ],
                )
            ],
        )
    return conflicts


def apply_roles(parts: list[Part], overrides: dict[str, str]) -> list[str]:
    """Put the designer's role first (confidence 1.0); returns conflict notes."""
    conflicts: list[str] = []
    for p in parts:
        label = overrides.get(p.id)
        if label is None or p.understanding is None:
            continue
        roles = p.understanding.roles
        top = roles[0] if roles else None
        mine = next((h for h in roles if h.label == label), None)
        if (
            top is not None
            and top.label != label
            and top.confidence >= CONFLICT_GEOMETRY_CONF
            and (mine is None or mine.confidence <= CONFLICT_ANSWER_CONF)
        ):
            conflicts.append(
                f"Designer says {p.id} ({p.name}) is a {label.replace('_', ' ')}, but the geometry and "
                f"connections point to {top.label.replace('_', ' ')} ({top.confidence:.2f})"
            )
        ev = Evidence(code="user", refs=[p.id], weight=1.0, text="confirmed by the designer")
        first = Hypothesis(label=label, confidence=1.0, evidence=[ev])
        p.understanding.roles = [first] + [h for h in roles if h.label != label][:2]
    return conflicts


def info_spot(message: str, refs: list[str]) -> WeakSpot:
    return WeakSpot(
        id="",
        severity="info",
        category="other",
        refs=refs,
        message=message,
        suggestion=None,
        depends_on_assumption=None,
    )
