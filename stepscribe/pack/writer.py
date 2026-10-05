# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Builds the ``<name>_context_pack/`` folder."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from stepscribe import config
from stepscribe.api import Analysis
from stepscribe.describe.assembly_describer import assembly_core_context, level0_context
from stepscribe.describe.budget import PackSections, compose_budgeted, rank_parts
from stepscribe.describe.part_describer import level1, level1_short, level2_context
from stepscribe.io.report_writer import write_report
from stepscribe.models.schema import Part, Report
from stepscribe.pack.design_context import context_text, load_context
from stepscribe.progress import Tracker

TEMPLATE_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=StrictUndefined,
        autoescape=False,
    )


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "part"


def instance_part_map(report: Report) -> dict[str, str]:
    """Instance ID -> part ID."""
    return {i.id: i.part_id for i in report.assembly.instances} if report.assembly else {}


def instance_names(report: Report) -> dict[str, str]:
    """Instance ID -> readable name (last path element)."""
    if not report.assembly:
        return {}
    return {i.id: i.path.rsplit("/", 1)[-1] for i in report.assembly.instances}


def relations_by_part(report: Report) -> dict[str, list[str]]:
    """Part ID -> relation sentences that involve it."""
    out: dict[str, list[str]] = {}
    if not report.assembly:
        return out
    pmap = instance_part_map(report)
    for r in report.assembly.relations:
        for iid in (r.subject, r.object):
            pid = pmap.get(iid)
            row = f"{r.id}: {r.sentence}"
            if pid and row not in out.setdefault(pid, []):  # both ends may be the same part
                out[pid].append(row)
    return out


def connections_by_part(report: Report) -> dict[str, list[str]]:
    """Part ID -> names of the other parts it is related to (for Level 1 paragraphs)."""
    out: dict[str, list[str]] = {}
    if not report.assembly:
        return out
    pmap, names = instance_part_map(report), instance_names(report)
    for r in report.assembly.relations:
        a, b = pmap.get(r.subject), pmap.get(r.object)
        if a and b and a != b:
            out.setdefault(a, [])
            out.setdefault(b, [])
            if names[r.object] not in out[a]:
                out[a].append(names[r.object])
            if names[r.subject] not in out[b]:
                out[b].append(names[r.subject])
    return out


def build_sections(
    report: Report,
    root_name: str,
    env: Environment,
    up: str,
    front: str,
    images: list[str] | None = None,
    recon: dict[str, list[str]] | None = None,
) -> PackSections:
    """Render Level 0 / assembly core / Level 1 / Level 2 text for every part."""
    conns, rels = connections_by_part(report), relations_by_part(report)
    ranked = rank_parts(report.parts)
    l1 = {p.id: level1(p, conns.get(p.id)) for p in ranked}
    l1s = {p.id: level1_short(p) for p in ranked}
    l2 = {
        p.id: env.get_template("part.md.j2")
        .render(c=level2_context(p, rels.get(p.id), (recon or {}).get(p.id)))
        .rstrip()
        + "\n"
        for p in ranked
    }
    overview = (
        env.get_template("01_overview.md.j2")
        .render(o=level0_context(report, root_name, up, front, images))
        .rstrip()
    )
    core = _assembly_core_text(report)
    from stepscribe.understanding.describe import understanding_core

    return PackSections(
        overview, core, l1, l1s, l2, [p.id for p in ranked], understanding_core(report)
    )


def _assembly_core_text(report: Report) -> str:
    a = assembly_core_context(report)
    if not a["has_assembly"]:
        return ""
    lines = ["## Relations", *([f"- {r}" for r in a["relations"]] or ["- none found"])]
    lines += ["", "## Fastener shopping list", *([f"- {s}" for s in a["shopping"]] or ["- none"])]
    if a["placements"]:
        lines += [
            "",
            "## Placement of every part in the assembly (world coordinates, mm)",
            *[f"- {s}" for s in a["placements"]],
        ]
    return "\n".join(lines)


def _qa_rows(report: Report) -> list[str]:
    """One readable line per answer the designer gave (confirmed, unsure or skipped)."""
    u = report.understanding
    if u is None:
        return []
    word = {
        "answered": "Confirmed by designer",
        "not_sure": "Designer unsure",
        "skipped": "Skipped",
    }
    rows = []
    for a in u.answers:
        value = f" -> {a.value}" if a.status == "answered" else ""
        note = f" (note: {a.note})" if a.note else ""
        rows.append(f"{word[a.status]}: {a.question or a.question_id}{value}{note}")
    return rows


@dataclass
class PackBuild:
    """A written pack folder plus the pieces it was composed from (the export reuses them)."""

    folder: Path
    sections: PackSections
    env: Environment
    root_name: str
    up: str
    front: str
    ctx: dict[str, Any]
    qa: list[str]
    images: list[str]


def write_pack(
    analysis: Analysis,
    out_dir: Path,
    budget_tokens: int = config.DEFAULT_BUDGET_TOKENS,
    context_file: str | Path | None = None,
    images: bool = False,
    tracker: Tracker | None = None,
) -> Path:
    """Write the whole pack folder and return its path."""
    return write_pack_build(analysis, out_dir, budget_tokens, context_file, images, tracker).folder


def write_pack_build(
    analysis: Analysis,
    out_dir: Path,
    budget_tokens: int = config.DEFAULT_BUDGET_TOKENS,
    context_file: str | Path | None = None,
    images: bool = False,
    tracker: Tracker | None = None,
    folder: Path | None = None,
) -> PackBuild:
    """Write the pack (into *folder*, or ``<out_dir>/<name>_context_pack``) and keep its parts."""
    report = analysis.report
    root_name = (
        analysis.model.root_name if analysis.model else Path(report.meta.source_files[0]).stem
    )
    up = report.assembly.up_axis if report.assembly else str(report.meta.options.get("up", "+Z"))
    front = (
        report.assembly.front_axis
        if report.assembly
        else str(report.meta.options.get("front", "-Y"))
    )
    env = _env()
    folder = folder or out_dir / f"{_safe(Path(report.meta.source_files[0]).stem)}_context_pack"
    (folder / "03_parts").mkdir(parents=True, exist_ok=True)
    ctx: dict[str, Any] = load_context(context_file) if context_file else {}
    images_written: list[str] = []
    if images:
        from stepscribe.render import render_pack_images

        images_written = render_pack_images(analysis, folder, tracker)
    else:  # reuse images rendered earlier (the interview rebuilds the text only)
        existing = folder / "images"
        images_written = (
            [f"images/{p.name}" for p in sorted(existing.glob("*.png"))]
            if existing.is_dir()
            else []
        )
    from stepscribe.describe.reconstruct import reconstruction

    recon = {ap.part.id: reconstruction(ap.part, ap.geom) for ap in analysis.parts}
    sections = build_sections(report, root_name, env, up, front, images_written, recon)

    def put(rel: str, text: str) -> None:
        (folder / rel).write_text(text, encoding="utf-8", newline="\n")

    put(
        "00_READ_ME_FIRST.md",
        env.get_template("00_READ_ME_FIRST.md.j2").render(up=up, front=front, has_images=images),
    )
    put("01_overview.md", sections.level0 + "\n")
    if report.understanding is not None:
        from stepscribe.understanding.describe import understanding_markdown

        put("05_understanding.md", understanding_markdown(report))
    if report.meta.options.get("debug_graphs"):
        import json

        from stepscribe.understanding.aag import aag_to_dict, build_aag

        (folder / "data").mkdir(parents=True, exist_ok=True)
        for ap in analysis.parts:
            graph = aag_to_dict(build_aag(ap.geom))
            put(
                f"data/aag_{ap.part.id}.json",
                json.dumps(graph, indent=1, ensure_ascii=False) + "\n",
            )
    put(
        "02_assembly.md",
        env.get_template("02_assembly.md.j2").render(
            a=assembly_core_context(report), level1=sections.level1
        ),
    )
    by_id: dict[str, Part] = {p.id: p for p in report.parts}
    for pid in sorted(by_id):
        p = by_id[pid]
        put(f"03_parts/{pid}_{_safe(p.name)}.md", sections.level2[pid])
    qa = _qa_rows(report)
    if ctx or qa:
        put(
            "04_design_context.md",
            env.get_template("04_design_context.md.j2").render(ctx=ctx, qa=qa),
        )
    header_args = {
        "name": root_name,
        "up": up,
        "front": front,
        "budget": budget_tokens,
        "n_parts": len(report.parts),
        "ctx_text": context_text(ctx),
    }
    # First pass to learn which parts fit, then render the header with that information.
    _text, detailed, short = compose_budgeted(sections, budget_tokens, "")
    header = (
        env.get_template("context_header.md.j2")
        .render(detailed=detailed, short=short, **header_args)
        .rstrip()
    )
    text, detailed, short = compose_budgeted(sections, budget_tokens, header)
    put("context_pack.md", text)
    write_report(report, folder / "data" / "report.json", config.FLOAT_PRECISION)
    return PackBuild(folder, sections, env, root_name, up, front, ctx, qa, images_written)
