# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Text for the understanding layer: ``05_understanding.md``, the compact block for the single-file
pack, and the sentences added to part paragraphs."""

from __future__ import annotations

from collections import Counter

from stepscribe.describe.phrases import fmt
from stepscribe.models.schema import Part, Report, Understanding, WeakSpot

MAX_CORE_SPOTS = 8
MAX_CORE_QUESTIONS = 5


def _cell(text: object) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def _role_text(label: str) -> str:
    return label.replace("_", " ")


def understanding_sentences(part: Part) -> list[str]:
    """Role, process, structure, symmetry and mirror-pair sentences for a part paragraph."""
    u = part.understanding
    if u is None:
        return []
    out: list[str] = []
    if u.roles and u.roles[0].label != "unknown" and u.roles[0].confidence >= 0.3:
        r = u.roles[0]
        out.append(f"Likely: {_role_text(r.label)} ({r.confidence:.2f}).")
    if u.process.ranked and u.process.ranked[0].label != "unknown":
        p = u.process.ranked[0]
        src = "given by the designer" if u.process.source == "user" else f"{p.confidence:.2f}"
        if u.process.source == "user" or p.confidence >= 0.4:
            out.append(f"Likely made by {_role_text(p.label)} ({src}).")
    if u.structural_features:
        counts = Counter(f.kind for f in u.structural_features)
        out.append(
            "Structure: "
            + ", ".join(
                f"{n} {_role_text(k)}{'s' if n > 1 else ''}" for k, n in sorted(counts.items())
            )
            + "."
        )
    if u.symmetry.mirror_planes or u.symmetry.rotational:
        out.append(u.symmetry.description.capitalize() + ".")
    if u.mirror_of_part_id:
        out.append(f"Mirror image of {u.mirror_of_part_id}.")
    return out


def _spot_row(w: WeakSpot) -> str:
    return (
        f"| {w.id} | {w.severity} | {_cell(w.message)} | {_cell(w.suggestion or '')} | "
        f"{_cell(w.depends_on_assumption or '')} |"
    )


def spots_table(spots: list[WeakSpot], limit: int | None = None) -> list[str]:
    rows = [
        "| ID | Severity | Finding | Suggestion | Depends on |",
        "|---|---|---|---|---|",
    ]
    rows += [_spot_row(w) for w in (spots[:limit] if limit else spots)]
    if limit and len(spots) > limit:
        rows.append(f"| ... | | {len(spots) - limit} more in 05_understanding.md | | |")
    return rows


def answers_section(u: Understanding, level: int = 2) -> list[str]:
    """Designer answers (confirmed, unsure) and conflicts with the geometry."""
    h = "#" * level
    lines: list[str] = []
    confirmed = [a for a in u.answers if a.status == "answered"]
    unsure = [a for a in u.answers if a.status == "not_sure"]
    qtext = {a.question_id: a.question or a.question_id for a in u.answers}
    qtext.update({q.id: q.text for q in u.questions})
    if confirmed:
        lines.append(f"{h} Confirmed by designer")
        for a in confirmed:
            note = f" ({a.note})" if a.note else ""
            lines.append(
                f"- Confirmed by designer: {qtext.get(a.question_id, a.question_id)} -> {a.value}{note}"
            )
        lines.append("")
    if unsure:
        lines.append(f"{h} Designer unsure")
        for a in unsure:
            lines.append(f"- Designer unsure: {qtext.get(a.question_id, a.question_id)}")
        lines.append("")
    if u.conflicts:
        lines.append(f"{h} Designer vs. geometry conflicts")
        lines += [f"- {c}" for c in u.conflicts]
        lines.append("")
    return lines


def understanding_core(report: Report) -> str:
    """Compact block that leads the single-file pack: summary, kinematics, weak spots, questions."""
    u = report.understanding
    if u is None:
        return ""
    lines = [
        "## Design understanding (rule-based hypotheses; each has evidence in 05_understanding.md)",
        "",
    ]
    lines += ["### Summary", u.summary, "", "### Kinematics", u.kinematics.description]
    for j in u.kinematics.joints:
        lines.append(f"- {j.description} (confidence {j.confidence:.2f})")
    for m in u.kinematics.mechanisms:
        lines.append(f"- {m.id} {m.kind.replace('_', ' ')}: {m.description}")
    if u.weak_spots:
        lines += ["", "### Weak spots", *spots_table(u.weak_spots, MAX_CORE_SPOTS)]
    if u.questions:
        lines += ["", "### Open questions for the designer"]
        lines += [f"{q.priority}. {q.text}" for q in u.questions[:MAX_CORE_QUESTIONS]]
    lines += [""] + answers_section(u, level=3)
    return "\n".join(lines).rstrip() + "\n"


def understanding_markdown(report: Report) -> str:
    """The full ``05_understanding.md``."""
    u = report.understanding
    if u is None:
        return "# Understanding\n\nThe understanding layer was not run.\n"
    k = u.kinematics
    names = (
        {i.id: i.path.rsplit("/", 1)[-1] for i in report.assembly.instances}
        if report.assembly
        else {}
    )
    lines = ["# Understanding: how the design works", ""]
    lines += [
        "Everything here is a rule-based hypothesis (no model, no network). Statements start with "
        '"Likely:" where they are inferences; each carries a confidence from 0 to 1 and evidence that '
        "refers to part (PRT), placed part (INS), contact (C), joint (J, KJ), link (L) and mechanism (M) IDs.",
        "",
        "## Design summary",
        u.summary,
        "",
        "## Kinematics",
        k.description,
        "",
        "```mermaid",
        k.mermaid,
        "```",
        "",
    ]
    if k.joints:
        lines += [
            "### Joints",
            "",
            "| ID | Kind | Parent | Child | Confidence | Driven by | Evidence |",
            "|---|---|---|---|---|---|---|",
        ]
        for j in k.joints:
            ev = "; ".join(e.text for e in j.evidence[:2])
            lines.append(
                f"| {j.id} | {j.kind} | {j.parent_link} | {j.child_link} | {j.confidence:.2f} | "
                f"{j.driven_by or ''} | {_cell(ev)} |"
            )
        lines.append("")
    if k.mechanisms:
        lines += [
            "### Mechanisms",
            "",
            "| ID | Kind | Ratio | Input | Output joint | Confidence | Description |",
            "|---|---|---|---|---|---|---|",
        ]
        for m in k.mechanisms:
            ratio = f"{fmt(m.ratio, 2)}:1" if m.ratio else ""
            lines.append(
                f"| {m.id} | {m.kind.replace('_', ' ')} | {ratio} | {m.input_instance_id or ''} | "
                f"{m.output_joint_id or ''} | {m.confidence:.2f} | {_cell(m.description)} |"
            )
        lines.append("")
    lines += ["### Links (rigid groups)", ""]
    for g in k.links:
        members = ", ".join(f"{i} ({names.get(i, i)})" for i in g.instance_ids)
        tag = " (ground)" if g.is_ground else ""
        lines.append(
            f"- {g.id}{tag}: {members}" + (f"; mass {fmt(g.mass_g, 0)} g" if g.mass_g else "")
        )
        for e in g.ground_evidence:
            lines.append(f"  - ground evidence: {e.text}")
    if k.floating_groups:
        lines.append(f"- Floating (not connected to ground): {', '.join(k.floating_groups)}")
    lines.append("")
    if u.subassembly_roles:
        lines += ["### Structures", ""]
        for s in u.subassembly_roles:
            for h in s.roles:
                lines.append(
                    f"- {s.subassembly}: likely {_role_text(h.label)} ({h.confidence:.2f}); {h.evidence[0].text if h.evidence else ''}"
                )
        lines.append("")
    lines += [
        "## Part roles and process",
        "",
        "| Part | Name | Likely role | Confidence | Key evidence | Likely process | Structure | Symmetry |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in report.parts:
        pu = p.understanding
        if pu is None:
            continue
        r = pu.roles[0] if pu.roles else None
        ev = r.evidence[0].text if r and r.evidence else ""
        proc = pu.process.ranked[0] if pu.process.ranked else None
        counts = Counter(f.kind for f in pu.structural_features)
        struct = ", ".join(f"{n} {kk}" for kk, n in sorted(counts.items()))
        lines.append(
            f"| {p.id} | {_cell(p.name)} | {_role_text(r.label) if r else ''} | "
            f"{f'{r.confidence:.2f}' if r else ''} | {_cell(ev)} | "
            f"{_role_text(proc.label) + f' ({proc.confidence:.2f})' if proc else ''} | {struct} | "
            f"{_cell(pu.symmetry.description)} |"
        )
    lines.append("")
    if u.load_paths:
        lines += ["## Load paths", ""]
        lines += [f"- {lp.id}: {lp.description}" for lp in u.load_paths]
        lines.append("")
    lines += ["## Weak spots", ""]
    lines += spots_table(u.weak_spots) if u.weak_spots else ["No rule fired."]
    lines.append("")
    if u.stability:
        lines += ["## Stability", u.stability.description, ""]
    if u.questions:
        lines += ["## Questions for the designer", ""]
        for q in u.questions:
            opts = f" Options: {', '.join(o.label for o in q.options)}." if q.options else ""
            lines.append(f"{q.priority}. [{q.id}] {q.text} (why: {q.why}){opts}")
        lines.append("")
    lines += answers_section(u)
    return "\n".join(lines).rstrip() + "\n"
