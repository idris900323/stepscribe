# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Questions for the designer: what geometry cannot tell.

Each question carries a stable ID (hash of its kind and references), the reason it matters, an
answer type, the tool's current guess and the analysis stages an answer would re-run. Questions
are ranked by expected value: ``impact x uncertainty x cheapness``; global questions (purpose,
material, process, payload) always come before per-part ones because they cascade.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from stepscribe import config
from stepscribe.models.schema import AnswerOption, Question, Report

MAX_QUESTIONS = 10
GLOBAL_KINDS = ("purpose", "material", "process", "payload")
CHEAPNESS = {
    "yes_no": 1.0,
    "single_choice": 1.0,
    "multi_choice": 0.8,
    "number": 0.7,
    "number_with_unit": 0.7,
    "text": 0.4,
}


def role_label(s: str) -> str:
    return s.replace("_", " ")


PROCESS_OPTIONS = [
    ("fdm_3d_printed", "3D printed (FDM)"),
    ("sla_or_sls_printed", "3D printed (resin or powder)"),
    ("cnc_milled", "CNC milled"),
    ("cnc_turned", "CNC turned (lathe)"),
    ("sheet_metal", "sheet metal (bent)"),
    ("laser_or_waterjet", "laser or waterjet cut"),
    ("injection_molded", "injection molded"),
]


def question_id(kind: str, refs: list[str]) -> str:
    """Stable across runs: the same kind about the same things always has the same ID."""
    raw = kind + "|" + "|".join(sorted(refs))
    return "Q" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:6].upper()  # noqa: S324


def _q(
    kind: str,
    text: str,
    why: str,
    answer_type: str,
    *,
    refs: list[str] | None = None,
    options: list[tuple[str, str]] | None = None,
    unit: str | None = None,
    guess: str | None = None,
    applies: list[str] | None = None,
) -> Question:
    refs = refs or []
    return Question(
        id=question_id(kind, refs),
        kind=kind,
        priority=0,
        text=text,
        refs=refs,
        why=why,
        answer_type=answer_type,
        options=[AnswerOption(value=v, label=lab) for v, lab in options] if options else None,
        unit=unit,
        default_guess=guess,
        applies_to=applies or [],
    )


def _score(q: Question, impact: float, uncertainty: float) -> float:
    return impact * max(uncertainty, 0.05) * CHEAPNESS[q.answer_type]


def generate_questions(
    report: Report,
    answers: dict[str, dict[str, Any]] | None = None,
    context: dict[str, Any] | None = None,
    wheel_joints: set[str] | None = None,
    limit: int | None = MAX_QUESTIONS,
) -> list[Question]:
    """Up to ten questions, most valuable first; answered or skipped ones are not asked again."""
    answers = answers or {}
    u = report.understanding
    if u is None:
        return []
    ctx = context or {}
    scored: list[tuple[float, Question]] = []
    parts = report.parts
    fab = [p for p in parts if not p.likely_purchased_hardware and p.understanding]
    asm = report.assembly

    def add(q: Question, impact: float, uncertainty: float) -> None:
        scored.append((_score(q, impact, uncertainty), q))

    # ---- global questions
    if not ctx.get("purpose"):
        add(
            _q(
                "purpose",
                "What is this design for? (one sentence)",
                "The purpose decides the role of every part and what counts as a risk.",
                "text",
                applies=["roles", "summary"],
            ),
            6.0,
            1.0,
        )
    unknown_mass = [p for p in parts if p.mass.mass_g is None]
    assumption_spots = [w for w in u.weak_spots if w.depends_on_assumption]
    if unknown_mass or any("mass" in (w.depends_on_assumption or "") for w in assumption_spots):
        impact = (
            len(unknown_mass) + len(assumption_spots) + (3 if u.stability is None and asm else 0)
        )
        add(
            _q(
                "material",
                "What material are the parts made of?",
                "Mass, centre of mass, tipping and the wall-thickness guidelines all depend on it.",
                "single_choice",
                options=[(k, k) for k in config.MATERIAL_KEYWORDS],
                applies=["mass", "stability", "weak_spots", "summary"],
            ),
            float(max(impact, 4)),
            1.0,
        )
    weak = [
        p
        for p in fab
        if p.understanding
        and p.understanding.process.source != "user"
        and p.understanding.process.ranked[0].confidence < 0.7
    ]
    if weak:
        procs = Counter(p.understanding.process.ranked[0].label for p in weak if p.understanding)
        guess = procs.most_common(1)[0][0]
        conf = min(p.understanding.process.ranked[0].confidence for p in weak if p.understanding)
        add(
            _q(
                "process",
                f"How are the fabricated parts made? {len(weak)} of {len(fab)} look {role_label(guess)} but I am not sure.",
                "The process sets the minimum wall thickness, hole edge distances and corner advice.",
                "single_choice",
                refs=[p.id for p in weak][:20],
                options=PROCESS_OPTIONS,
                guess=guess,
                applies=["process", "weak_spots", "summary"],
            ),
            float(len(weak) + len(assumption_spots)),
            1.0 - conf,
        )
    arm_end = [j for j in u.kinematics.joints if j.kind == "revolute"]
    if len(arm_end) >= 2:
        add(
            _q(
                "payload",
                "What payload (mass at the end of the arm) should it carry?",
                "A payload adds a point mass at the end effector for load paths and tipping.",
                "number_with_unit",
                unit="g",
                applies=["stability", "load_paths", "weak_spots", "summary"],
            ),
            3.0,
            1.0,
        )
    # ---- per-part and per-joint questions
    gap = 0.1
    for p in fab:
        roles = p.understanding.roles if p.understanding else []
        if (
            len(roles) >= 2
            and roles[0].confidence - roles[1].confidence < gap
            and roles[0].label != "unknown"
        ):
            add(
                _q(
                    "choose_role",
                    f"Is {p.id} ({p.name}) a {role_label(roles[0].label)} or a {role_label(roles[1].label)}?",
                    "Its role is ambiguous; the answer sharpens the roles of the parts connected to it.",
                    "single_choice",
                    refs=[p.id],
                    options=[(r.label, role_label(r.label)) for r in roles[:3]],
                    guess=roles[0].label,
                    applies=["roles", "summary"],
                ),
                2.0,
                1.0 - roles[0].confidence,
            )
    if asm is not None:
        for rel in asm.relations:
            if rel.predicate == "rests_on" and not _fastened(report, rel.subject, rel.object):
                add(
                    _q(
                        "connection_type",
                        f"{rel.subject} rests on {rel.object} with no fasteners. Is it glued, clamped, or are fasteners missing from the CAD?",
                        "A rest without fasteners is treated as an uncertain fixed connection; the answer fixes the load path.",
                        "single_choice",
                        refs=[rel.subject, rel.object],
                        options=[
                            ("fixed", "fixed (press fit or bonded)"),
                            ("glued", "glued"),
                            ("clamped", "clamped"),
                            ("fasteners_missing", "fasteners are missing from the CAD"),
                        ],
                        guess="fasteners_missing",
                        applies=["load_paths", "weak_spots", "summary"],
                    ),
                    2.0,
                    0.7,
                )
        for link_id in u.kinematics.floating_groups:
            link = next(g for g in u.kinematics.links if g.id == link_id)
            add(
                _q(
                    "missing_part",
                    f"{link_id} ({link.description}) is not connected to anything. Is it loose, or is a part or fastener missing from the CAD?",
                    "A floating group usually means missing mates; the answer removes or confirms the warning.",
                    "single_choice",
                    refs=list(link.instance_ids),
                    options=[
                        ("missing", "something is missing from the CAD"),
                        ("loose", "it is a loose part"),
                        ("purchased", "it is a purchased part held by something not modelled"),
                    ],
                    guess="missing",
                    applies=["weak_spots", "summary"],
                ),
                3.0,
                0.8,
            )
        wheels = sorted(wheel_joints or set())
        undriven = [
            w for w in wheels if not next(j for j in u.kinematics.joints if j.id == w).driven_by
        ]
        if undriven:
            add(
                _q(
                    "joint_drive",
                    f"Which wheels are driven? ({', '.join(undriven)} are wheel joints and none is "
                    "connected to a motor in the CAD)",
                    "The drivetrain decides the load on the wheels and the mobile-base summary.",
                    "multi_choice",
                    refs=undriven,
                    options=[(w, w) for w in undriven] + [("none", "none: all passive")],
                    applies=["summary"],
                ),
                2.0,
                1.0,
            )
        for j in u.kinematics.joints:
            if j.id in wheels:
                continue
            if (
                j.kind in ("revolute", "prismatic", "cylindrical")
                and not j.driven_by
                and not _is_input(u, j.id)
            ):
                add(
                    _q(
                        "joint_drive",
                        f"What drives {j.id} ({j.kind} about its axis)?",
                        "The drive decides the load on the joint and the mechanism summary.",
                        "single_choice",
                        refs=[j.id],
                        options=[
                            ("motor", "a motor (direct)"),
                            ("geared", "a motor through gears or a belt"),
                            ("manual", "by hand"),
                            ("passive", "nothing: it moves freely"),
                        ],
                        applies=["summary"],
                    ),
                    1.5,
                    1.0,
                )
            if j.range_deg_or_mm is None and j.kind in ("revolute", "prismatic"):
                add(
                    _q(
                        "joint_range",
                        f"What is the allowed range of {j.id}? (degrees or mm, from the lower to the upper limit)",
                        "The range tells which poses to check for collisions and tipping (or run --check-motion).",
                        "text",
                        refs=[j.id],
                        applies=["weak_spots"],
                    ),
                    1.0,
                    1.0,
                )
    add(
        _q(
            "load",
            "What loads should it carry? (forces, torques, payload, speeds)",
            "Loads decide whether the weak spots matter.",
            "text",
            applies=["weak_spots"],
        ),
        1.0,
        1.0,
    )
    add(
        _q(
            "environment",
            "Where will it be used? (indoor, outdoor, dust, water, temperature)",
            "The environment changes the material and the process.",
            "text",
            applies=["summary"],
        ),
        0.8,
        1.0,
    )
    return _rank(scored, answers, limit)


def _fastened(report: Report, a: str, b: str) -> bool:
    asm = report.assembly
    if asm is None:
        return False
    return any(a in j.instance_ids and b in j.instance_ids for j in asm.fastener_joints)


def _is_input(u, joint_id: str) -> bool:  # type: ignore[no-untyped-def]
    return any(m.parameters.get("input_joint") == joint_id for m in u.kinematics.mechanisms)


def _rank(
    scored: list[tuple[float, Question]], answers: dict[str, dict[str, Any]], limit: int | None
) -> list[Question]:
    """Rank by value; with ``limit=None`` every question is returned, answered ones included."""
    out: list[tuple[float, Question]] = []
    seen: set[str] = set()
    for s, q in scored:
        if q.id in seen:
            continue
        seen.add(q.id)
        a = answers.get(q.id)
        if limit is not None and a and a.get("status") in ("answered", "skipped"):
            continue
        if a and a.get("status") == "not_sure":
            s *= 0.1  # a lower priority, permanently
        unsure = bool(a and a.get("status") == "not_sure")
        boost = 1000.0 if q.kind in GLOBAL_KINDS and not unsure else 0.0
        out.append((s + boost, q))
    out.sort(key=lambda t: (-t[0], t[1].id))
    ranked = [q for _s, q in (out if limit is None else out[:limit])]
    for i, q in enumerate(ranked, 1):
        q.priority = i
    return ranked
