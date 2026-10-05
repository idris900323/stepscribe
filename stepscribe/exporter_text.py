# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Text of the portable export: ``<name>_FULL.md`` and ``<name>_SMALL.md``.

Both files are plain GitHub-flavoured Markdown, UTF-8, LF, no raw HTML, and deterministic.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stepscribe import __version__
from stepscribe.describe.assembly_describer import assembly_core_context
from stepscribe.describe.budget import PackSections, estimate_tokens
from stepscribe.describe.part_describer import level2_context
from stepscribe.models.schema import Part, Report
from stepscribe.pack.writer import PackBuild
from stepscribe.understanding.describe import answers_section, understanding_markdown

NOT_ANALYSED = "Not analysed (layer not enabled)."
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}
_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^(#{1,6}) +(.*?)\s*$")


# ------------------------------------------------------------------ markdown helpers
def slug(heading: str) -> str:
    """GitHub-style anchor for a heading."""
    text = re.sub(r"[`*_]", "", heading.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return text.replace(" ", "-")


def retitle(md: str, top_level: int, prefix: str | None = None) -> str:
    """Move the headings of *md* so its first heading sits at *top_level*.

    Sub-headings get ``prefix`` in front of their text so that headings stay unique when many
    documents (one per part) are joined into one file.
    """
    lines = md.splitlines()
    levels = []
    fenced = False
    for line in lines:
        if _FENCE.match(line):
            fenced = not fenced
        m = None if fenced else _HEADING.match(line)
        if m:
            levels.append(len(m.group(1)))
    if not levels:
        return md
    shift = top_level - min(levels)
    out: list[str] = []
    fenced = False
    first = True
    for line in lines:
        if _FENCE.match(line):
            fenced = not fenced
        m = None if fenced else _HEADING.match(line)
        if not m:
            out.append(line)
            continue
        level = min(6, len(m.group(1)) + shift)
        text = m.group(2)
        if prefix and not first:
            text = f"{prefix}: {text}"
        first = False
        out.append(f"{'#' * level} {text}")
    return "\n".join(out)


def uniquify_headings(md: str) -> str:
    """Rename repeated headings ("Name (2)") so every anchor is unique."""
    seen: dict[str, int] = {}
    out: list[str] = []
    fenced = False
    for line in md.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
        m = None if fenced else _HEADING.match(line)
        if m:
            key = slug(m.group(2))
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > 1:
                line = f"{m.group(1)} {m.group(2)} ({seen[key]})"
        out.append(line)
    return "\n".join(out)


def table_of_contents(md: str, max_level: int = 3) -> str:
    """Bullet list of links to every heading down to *max_level*."""
    rows: list[str] = []
    fenced = False
    for line in md.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
        m = None if fenced else _HEADING.match(line)
        if m and 2 <= len(m.group(1)) <= max_level:
            indent = "  " * (len(m.group(1)) - 2)
            rows.append(f"{indent}- [{m.group(2)}](#{slug(m.group(2))})")
    return "\n".join(rows)


def split_sections(md: str) -> dict[str, str]:
    """Split a Markdown document at its ``## `` headings (text before the first one is ``""``)."""
    out: dict[str, list[str]] = {"": []}
    key = ""
    fenced = False
    for line in md.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
        if not fenced and line.startswith("## "):
            key = line[3:].strip()
            out[key] = []
        out[key].append(line)
    return {k: "\n".join(v).strip("\n") for k, v in out.items()}


def clean(text: str) -> str:
    """LF only, no trailing spaces, one final newline."""
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip("\n") + "\n"


# ------------------------------------------------------------------ images
_VIEW_TEXT = {
    "iso": "Isometric view of the whole assembly.",
    "front": "Front orthographic view of the whole assembly with overall dimension lines.",
    "top": "Top orthographic view of the whole assembly with overall dimension lines.",
    "right": "Right orthographic view of the whole assembly with overall dimension lines.",
    "exploded": "Exploded view: the parts are moved apart so each one is visible. The caption "
    "states the assembled size.",
}


def panel_text(p: Part) -> str:
    ctx = level2_context(p)
    bits = [f"overall size {ctx['size']}", str(ctx["mass"])]
    if ctx["wall"]:
        bits.append(f"thinnest wall {ctx['wall']} mm")
    counts = [
        (len(p.holes), "holes"),
        (len(p.cutouts), "cutouts"),
        (len(p.bosses), "bosses"),
        (len(p.fillets), "fillets"),
        (len(p.chamfers), "chamfers"),
    ]
    bits += [f"{n} {word}" for n, word in counts if n]
    return "; ".join(bits)


def describe_image(rel: str, report: Report) -> tuple[str, str]:
    """(alt text, description) for an image path such as ``images/assembly_iso.png``."""
    name = Path(rel).name
    stem = name.rsplit(".", 1)[0]
    parts = {p.id: p for p in report.parts}
    if stem.startswith("assembly_"):
        view = stem[len("assembly_") :]
        listed = ", ".join(f"{p.id} {p.name}" for p in report.parts[:30])
        more = " and more" if len(report.parts) > 30 else ""
        text = (
            f"{_VIEW_TEXT.get(view, f'{view} view of the whole assembly.')} Labels are part IDs "
            f"with names. Parts: {listed}{more}."
        )
        return f"{view} view of the assembly", text
    if stem == "kinematics_iso":
        u = report.understanding
        joints = (
            " ".join(
                f"{j.id} {j.kind} {j.parent_link}->{j.child_link}." for j in u.kinematics.joints
            )
            if u
            else ""
        )
        text = (
            "Isometric view with each rigid link in its own colour (L0 is the ground link, grey). "
            f"Arrows are joint axes labelled with the joint ID and kind. {joints}"
        )
        return "links and joint axes", text.strip()
    m = re.match(r"(PRT\d+)_(\w+)", stem)
    if m and m.group(1) in parts:
        p = parts[m.group(1)]
        how = (
            "along its hole axis; hole labels (H###) and ordinate dimensions are measured from "
            "the lower-left corner"
            if m.group(2) == "axis"
            else f"in a {m.group(2)} view"
        )
        text = (
            f"Part {p.id} {p.name}, viewed {how}. The white panel on the right of the image repeats: "
            f"{panel_text(p)}."
        )
        return f"{p.id} {p.name}, {m.group(2)} view", text
    return stem.replace("_", " "), "Rendered view; labels use the same IDs as the text."


# ------------------------------------------------------------------ FULL
def _title_block(build: PackBuild, report: Report, name: str, no_timestamp: bool) -> str:
    meta = report.meta
    lines = [
        f"# {name}: stepscribe full export",
        "",
        f"- Source file: {', '.join(meta.source_files)}",
        f"- Tool: stepscribe {__version__}; report schema {meta.schema_version}",
    ]
    if meta.generated_at and not no_timestamp:
        lines.append(f"- Generated: {meta.generated_at}")
    lines += [
        f"- Units: millimetres, degrees, grams (the file used {meta.original_length_unit})",
        f"- Up axis {build.up}, front axis {build.front}",
        "- Generated by stepscribe (AGPL-3.0). The analysis output belongs to you.",
    ]
    return "\n".join(lines)


def _how_to_read() -> str:
    return "\n".join(
        [
            "## How to read this document",
            "",
            "- **IDs:** PRT part, INS placed part, H hole, P hole pattern, C contact (or CP cut-out pattern), "
            "J fastener joint, R relation, S slot, PK pocket, B boss, FL fillet, CH chamfer, "
            "L link, KJ kinematic joint, M mechanism, W weak spot, Q question. Face IDs look like F0001.",
            "- **Facts and guesses:** statements without a prefix are measured. Statements starting "
            "with `Likely:` are rule-based inferences with a confidence from 0 to 1 and evidence.",
            "- **Designer input:** `Confirmed by designer:` is an answer the designer gave; "
            "`Designer unsure:` is a soft answer; skipped questions were not answered.",
            "- **Not known:** thread data, material (unless given), tolerances, loads and the "
            "intended purpose, unless the design context says so.",
            "- **Images:** every image has a text description in the Images section, so a model that "
            "cannot see pictures still knows what each one shows.",
        ]
    )


def _understanding_blocks(report: Report) -> dict[str, str]:
    """Understanding Markdown regrouped into the sections of the full file."""
    u = report.understanding
    keys = ("kinematics", "stability", "questions", "answers")
    if u is None:
        return {k: NOT_ANALYSED for k in keys}
    sec = split_sections(understanding_markdown(report))

    def grab(*titles: str) -> str:
        return "\n\n".join(retitle(sec[t], 3) for t in titles if t in sec)

    return {
        "kinematics": grab("Design summary", "Kinematics", "Part roles and process", "Load paths"),
        "stability": grab("Weak spots", "Stability"),
        "questions": grab("Questions for the designer") or "No open questions.",
        "answers": "\n".join(answers_section(u)),
    }


def compose_full(
    build: PackBuild,
    report: Report,
    name: str,
    images: list[str],
    no_timestamp: bool = False,
) -> str:
    """The paste-anywhere file: every layer, every part in full (including Reconstruction)."""
    sections: PackSections = build.sections
    env = build.env
    u = _understanding_blocks(report)
    core = assembly_core_context(report)
    if core["has_assembly"]:
        assembly = retitle(
            env.get_template("02_assembly.md.j2")
            .render(a=core, level1=sections.level1)
            .replace("## Part summaries", "## Part summaries (short)"),
            2,
        )
    else:
        assembly = "## Assembly\n\nThis file contains a single part, so there are no assembly relationships."
    ctx_md = ""
    if build.ctx or build.qa:
        ctx_md = retitle(
            env.get_template("04_design_context.md.j2").render(ctx=build.ctx, qa=build.qa), 3
        )
    answers = u["answers"] if u["answers"] != NOT_ANALYSED else ""
    context_block = "\n\n".join(x for x in (ctx_md, retitle(answers, 3) if answers else "") if x)
    ranked = [pid for pid in sections.ranked_ids]
    part_blocks = [retitle(sections.level2[pid], 3, prefix=pid) for pid in ranked]
    img_lines: list[str] = []
    for rel in images:
        alt, desc = describe_image(rel, report)
        img_lines += [f"![{alt}]({rel})", "", desc, ""]
    appendix = [
        "## Appendix",
        "",
        "### Tolerances used",
        "",
        *[f"- {k}: {v}" for k, v in sorted(report.meta.tolerances.items())],
        "",
        "### Options",
        "",
        "```json",
        json.dumps(report.meta.options, indent=2, sort_keys=True, default=str),
        "```",
        "",
        "### Warnings",
        "",
        *([f"- {w}" for w in report.warnings] or ["- none"]),
        *[f"- {p.id}: {w}" for p in report.parts for w in p.warnings],
        "",
        "### Errors",
        "",
        *([f"- {json.dumps(e, default=str)}" for e in report.errors] or ["- none"]),
    ]
    body = "\n\n".join(
        [
            _how_to_read(),
            "<<TOC>>",
            retitle(sections.level0, 2),
            "## Design context and designer answers\n\n"
            + (context_block or "No design context or designer answers were supplied."),
            "## Kinematics, mechanisms and roles\n\n" + u["kinematics"],
            assembly,
            "## Weak spots and stability\n\n" + u["stability"],
            "## Open questions for the designer\n\n" + u["questions"],
            "## Parts\n\nEvery part in full detail, most important first.\n\n"
            + "\n\n".join(part_blocks),
            "## Images\n\n"
            + ("\n".join(img_lines).strip() if img_lines else "No images were rendered."),
            "\n".join(appendix),
        ]
    )
    body = uniquify_headings(body)
    body = body.replace("<<TOC>>", "## Contents\n\n" + table_of_contents(body), 1)
    text = _title_block(build, report, name, no_timestamp) + "\n\n" + body
    return clean(text)


# ------------------------------------------------------------------ SMALL
def _fit(builder: Callable[[int], str], start: int, cap: int) -> str:
    """The longest text ``builder(n)`` with n <= start that fits in *cap* tokens ("" if none)."""
    for n in range(start, 0, -1):
        text = builder(n)
        if estimate_tokens(text) <= cap:
            return text
    return ""


def _spot_sort(report: Report) -> list[Any]:
    u = report.understanding
    spots = list(u.weak_spots) if u else []
    return sorted(spots, key=lambda w: (SEVERITY_ORDER.get(w.severity, 9), w.id))


def compose_small(
    report: Report,
    sections: PackSections,
    name: str,
    up: str,
    front: str,
    budget_tokens: int,
) -> str:
    """The small version: summary, kinematics, top weak spots, questions, BOM, part paragraphs."""
    u = report.understanding
    header = (
        f"# {name}: small version\n\n"
        f"Small version (about {budget_tokens} tokens) for free-tier chats and small models. "
        f"`{name}_COMPACT.md` has more detail and `{name}_FULL.md` has everything. Units are mm, "
        f"degrees, grams; up axis {up}, front axis {front}. Statements starting with `Likely:` "
        "are inferences with a confidence; all others are measured."
    )
    blocks: list[str] = [header]
    used = estimate_tokens(header)

    def add(text: str) -> None:
        nonlocal used
        if text:
            blocks.append(text)
            used += estimate_tokens(text) + 1

    def pct(share: float) -> int:
        return int(budget_tokens * share)

    if u is not None:
        add(f"## Summary\n\n{u.summary}")
        k = u.kinematics

        def kin(n: int) -> str:
            rows = [f"## Kinematics\n\n{k.description}"]
            rows += [f"- {j.description} (confidence {j.confidence:.2f})" for j in k.joints[:n]]
            rows += [f"- {m.id}: {m.description}" for m in k.mechanisms[:n]]
            if n >= len(k.joints) and k.mermaid and n > 4:
                rows += ["", "```mermaid", k.mermaid, "```"]
            return "\n".join(rows)

        add(_fit(kin, max(len(k.joints), 5), pct(0.30)))
        spots = _spot_sort(report)

        def spot_block(n: int) -> str:
            rows = [f"## Top weak spots ({min(n, len(spots))} of {len(spots)})", ""]
            rows += [
                f"- {w.id} ({w.severity}): {w.message}"
                + (f" -> {w.suggestion}" if w.suggestion else "")
                for w in spots[:n]
            ]
            return "\n".join(rows)

        if spots:
            add(_fit(spot_block, min(len(spots), 8), pct(0.18)))

        def question_block(n: int) -> str:
            rows = ["## Open questions for the designer", ""]
            rows += [f"- [{q.id}] {q.text}" for q in u.questions[:n]]
            return "\n".join(rows)

        if u.questions:
            add(_fit(question_block, min(len(u.questions), 6), pct(0.10)))
    else:
        add(sections.level0)
    core = assembly_core_context(report)
    if core["has_assembly"]:

        def bom_block(n: int) -> str:
            rows = ["## Bill of materials", "", *[f"- {b}" for b in core["bom"][:n]]]
            rows += ["", "## Fastener shopping list", "", *[f"- {s}" for s in core["shopping"][:n]]]
            if not core["shopping"]:
                rows.append("- none")
            return "\n".join(rows)

        add(_fit(bom_block, max(len(core["bom"]), len(core["shopping"]), 1), pct(0.15)))
    ids = list(sections.ranked_ids)
    left = budget_tokens - used - 8
    full = list(ids)
    short: list[str] = []

    def cost() -> int:
        return estimate_tokens(
            "\n\n".join(
                sections.level1_short[i] if i in short else sections.level1[i] for i in full
            )
        )

    while full and cost() > left and len(short) < len(full):
        short.append(full[len(full) - 1 - len(short)])
    dropped = 0
    while full and cost() > left:
        full.pop()
        short = [s for s in short if s in full]
        dropped += 1
    if full:
        rows = [sections.level1_short[i] if i in short else sections.level1[i] for i in full]
        tail = (
            f"\n\n{dropped} less important part(s) are not listed here; see `{name}_COMPACT.md`."
            if dropped
            else ""
        )
        add("## Parts\n\n" + "\n\n".join(rows) + tail)
    return clean("\n\n".join(blocks))
