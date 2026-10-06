# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""MCP tool implementations (read-only). Plain functions: the server only wraps them.

Results are cached by file content hash, so follow-up questions about the same robot do not
re-run the analysis.
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from OCP.BRepExtrema import BRepExtrema_DistShapeShape

from stepscribe.analysis import AnalyzeOptions
from stepscribe.api import Analysis, analyze_full
from stepscribe.assembly import build_instances
from stepscribe.assembly.relations import labels
from stepscribe.assembly.tree import InstanceData
from stepscribe.describe.part_describer import level2_context, shape_label
from stepscribe.describe.phrases import fmt, hole_callout
from stepscribe.geometry.occ_utils import to_np
from stepscribe.models.schema import Part

CACHE_SIZE = 4
_cache: OrderedDict[str, Analysis] = OrderedDict()
_sessions: dict[str, Any] = {}  # interview sessions, keyed by analysis identity


def _key(path: Path, options: AnalyzeOptions) -> str:
    digest = hashlib.sha1(path.read_bytes()).hexdigest()  # noqa: S324
    return f"{digest}:{options.material}:{options.density}:{options.up}:{options.front}"


def get_analysis(path: str, material: str | None = None) -> Analysis:
    """Analysis of *path*, cached by file hash (and the options that change results)."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"file not found: {path}")
    options = AnalyzeOptions(material=material, no_timestamp=True)
    key = _key(p, options)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    analysis = analyze_full(p, options)
    if not analysis.parts:
        errors = "; ".join(str(e.get("error")) for e in analysis.report.errors)
        raise ValueError(f"nothing could be analysed: {errors}")
    _cache[key] = analysis
    while len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return analysis


def clear_cache() -> None:
    """Forget all cached analyses."""
    _cache.clear()
    _sessions.clear()


def _find_part(analysis: Analysis, token: str) -> Part:
    parts = analysis.report.parts
    exact = [p for p in parts if token in (p.id, p.name)]
    if len(exact) == 1:
        return exact[0]
    loose = [p for p in parts if token.lower() in p.name.lower()]
    if len(loose) == 1:
        return loose[0]
    options = ", ".join(f"{p.id} {p.name}" for p in parts[:20])
    raise ValueError(f"part '{token}' not found or ambiguous; parts are: {options}")


# ------------------------------------------------------------------ tools
def get_overview(path: str) -> str:
    """Level 0 overview: size, parts, headline features, fasteners."""
    from stepscribe.pack.writer import _env, build_sections

    a = get_analysis(path)
    rep = a.report
    up = rep.assembly.up_axis if rep.assembly else "+Z"
    front = rep.assembly.front_axis if rep.assembly else "-Y"
    root = a.model.root_name if a.model else Path(path).stem
    return build_sections(rep, root, _env(), up, front).level0


def get_part(path: str, part: str) -> str:
    """Level 2 detail for one part (by ID such as PRT003, or by name)."""
    from stepscribe.pack.writer import _env, relations_by_part

    a = get_analysis(path)
    p = _find_part(a, part)
    rel = relations_by_part(a.report).get(p.id)
    return _env().get_template("part.md.j2").render(c=level2_context(p, rel)).rstrip() + "\n"


def list_holes(
    path: str, part: str | None = None, min_d: float | None = None, max_d: float | None = None
) -> str:
    """Hole table, optionally for one part and a diameter range (mm)."""
    a = get_analysis(path)
    parts = [_find_part(a, part)] if part else a.report.parts
    rows = []
    for p in parts:
        for h in p.holes:
            if (min_d is not None and h.diameter_mm < min_d) or (
                max_d is not None and h.diameter_mm > max_d
            ):
                continue
            std = h.standard_matches[0] if h.standard_matches else None
            o = h.axis.origin
            rows.append(
                f"| {p.id} | {h.id} | {hole_callout(h)} | {h.entry_type} | {h.bottom_type} | "
                f"({fmt(o.x, 2)}, {fmt(o.y, 2)}, {fmt(o.z, 2)}) | "
                f"{fmt(h.edge_distance_mm, 2) if h.edge_distance_mm is not None else '-'} | "
                f"{('Likely: ' + std.designation + ' ' + std.fit.replace('_', ' ')) if std else '-'} |"
            )
    if not rows:
        return "No holes match."
    head = "| Part | Hole | Callout | Entry | Bottom | Position (part frame) | Edge distance | Standard |\n|---|---|---|---|---|---|---|---|\n"
    return head + "\n".join(rows)


def list_cutouts(path: str, part: str | None = None) -> str:
    """Cut-out table (ID, kind, each level's shape, size, depth, centre, edge distance)."""
    from stepscribe.describe.part_describer import cutout_row

    a = get_analysis(path)
    parts = [_find_part(a, part)] if part else a.report.parts
    rows = []
    for p in parts:
        for c in p.cutouts:
            r = cutout_row(c, p)
            rows.append(
                f"| {p.id} | {r['id']} | {r['kind']} | {r['levels']} | {r['depth']} | {r['edge']} | {r['also']} |"
            )
    if not rows:
        return "No cut-outs found."
    head = (
        "| Part | Cut-out | Kind | Levels (shape, size, depth, centre u, v from the outline corner) "
        "| Total depth | Edge distance | Also detected as |\n|---|---|---|---|---|---|---|\n"
    )
    return head + "\n".join(rows)


def get_relations(path: str, part: str | None = None) -> str:
    """Relation sentences, optionally only those involving one part."""
    from stepscribe.pack.writer import relations_by_part

    a = get_analysis(path)
    if a.report.assembly is None:
        return "This file is a single part; there are no assembly relations."
    if part:
        lines = relations_by_part(a.report).get(_find_part(a, part).id, [])
    else:
        lines = [f"{r.id}: {r.sentence}" for r in a.report.assembly.relations]
    return "\n".join(f"- {s}" for s in lines) or "No relations found."


def _resolve_instance(a: Analysis, token: str) -> InstanceData:
    instances = build_instances(a)
    names = labels(instances)
    hits = [i for i in instances if token in (i.id, i.name, i.path, names[i.id])]
    if not hits:
        hits = [i for i in instances if i.part_id == token]
    if not hits:
        hits = [i for i in instances if token.lower() in i.path.lower()]
    if len(hits) != 1:
        choices = ", ".join(f"{i.id} {names[i.id]}" for i in instances[:25])
        raise ValueError(f"'{token}' matches {len(hits)} instances; use one of: {choices}")
    return hits[0]


def measure_distance(path: str, a: str, b: str) -> str:
    """Minimum distance between two instances (ID, name, path, or a part ID used once)."""
    an = get_analysis(path)
    if an.report.assembly is None:
        raise ValueError("measure_distance needs an assembly with at least two instances")
    ia, ib = _resolve_instance(an, a), _resolve_instance(an, b)
    dss = BRepExtrema_DistShapeShape(ia.shape, ib.shape)
    dss.Perform()
    if not dss.IsDone():
        raise ValueError("the distance computation failed for these shapes")
    d = float(dss.Value())
    names = labels(build_instances(an))
    if d <= 1e-9:
        return f"{names[ia.id]} and {names[ib.id]} touch or overlap (distance 0.0 mm)."
    p1, p2 = to_np(dss.PointOnShape1(1)), to_np(dss.PointOnShape2(1))

    def fmt3(p: Any) -> str:
        return "(" + ", ".join(fmt(float(v), 2) for v in p) + ")"

    return (
        f"Minimum distance between {names[ia.id]} and {names[ib.id]} is {fmt(d, 3)} mm, "
        f"between {fmt3(p1)} on the first and {fmt3(p2)} on the second (assembly frame, mm)."
    )


def find_parts(path: str, query: str) -> str:
    """Parts whose name, shape class, hardware guess or semantic tags contain *query*."""
    a = get_analysis(path)
    q = query.lower()
    out = []
    for p in a.report.parts:
        haystack = " ".join(
            [
                p.name,
                shape_label(p.shape_class),
                p.hardware_guess or "",
                *[t.label for t in p.semantic_tags],
            ]
        ).lower()
        if q in haystack:
            tags = "; ".join(t.label for t in p.semantic_tags)
            out.append(
                f"- {p.id} {p.name}: {shape_label(p.shape_class)}" + (f" ({tags})" if tags else "")
            )
    return "\n".join(out) or f"No part matches '{query}'."


def get_shopping_list(path: str) -> str:
    """Fastener shopping list derived from the joints."""
    a = get_analysis(path)
    if a.report.assembly is None or not a.report.assembly.fastener_shopping_list:
        return "No fasteners could be derived (single part, or no aligned holes across parts)."
    return "\n".join(
        f"- {r['qty']} × {r['item']}" for r in a.report.assembly.fastener_shopping_list
    )


def section(path: str, plane: str, part: str | None = None) -> tuple[str, bytes]:
    """(text summary, PNG bytes) of a cut such as ``z=12.5``."""
    from stepscribe.geometry.section_report import section_file

    get_analysis(path)  # validates the file early and warms the cache
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "section.png"
        text = section_file(path, plane, part, png)
        return text, png.read_bytes() if png.is_file() else b""


def render_view(
    path: str, view: str = "iso", part: str | None = None, highlight: list[str] | None = None
) -> bytes:
    """PNG bytes of a labelled view (iso, front, top, right, exploded)."""
    from stepscribe.render import render_view_file

    get_analysis(path)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "view.png"
        render_view_file(path, view, part, highlight, out)
        return out.read_bytes()


MAX_IMAGE_PX = 1200


def downscale_png(data: bytes, max_px: int = MAX_IMAGE_PX) -> bytes:
    """PNG bytes with the long side at most *max_px* (unchanged when already small)."""
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(data))
    if max(img.size) <= max_px:
        return data
    scale = max_px / max(img.size)
    out = io.BytesIO()
    img.resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
        Image.Resampling.LANCZOS,
    ).save(out, format="PNG", optimize=True)
    return out.getvalue()


def _save_image(data: bytes, stem: str) -> Path:
    from stepscribe.jobs import cache_dir

    target = cache_dir() / "images"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{stem}.png"
    path.write_bytes(data)
    return path


def _digest(*parts: object) -> str:
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:12]  # noqa: S324


def render_view_result(
    path: str, view: str = "iso", part: str | None = None, highlight: list[str] | None = None
) -> tuple[str, bytes]:
    """(text, PNG <= 1200 px). A part view has no info panel; its contents come as text."""
    png = downscale_png(render_view(path, view, part, highlight))
    lines = [f"{view} view" + (f" of {part}" if part else "") + "; labels use the IDs in the text."]
    if part:
        from stepscribe.exporter_text import panel_text

        p = _find_part(get_analysis(path), part)
        lines.append(f"Info panel of {p.id} {p.name}: {panel_text(p)}.")
    saved = _save_image(png, "view_" + _digest(path, view, part, highlight))
    lines.append(f"Saved at {saved}")
    return "\n".join(lines), png


def section_result(path: str, plane: str, part: str | None = None) -> tuple[str, bytes]:
    """(text, PNG <= 1200 px) of a cut."""
    text, png = section(path, plane, part)
    if not png:
        return text, b""
    png = downscale_png(png)
    saved = _save_image(png, "section_" + _digest(path, plane, part))
    return f"{text}\nSaved at {saved}", png


def question_result(path: str) -> tuple[str, bytes]:
    """(questions text, highlight PNG <= 1200 px of the first question's parts, or b"")."""
    text = get_questions(path)
    s = _session(path)
    refs = [r for r in (s.questions[0].refs if s.questions else []) if r.startswith(("PRT", "INS"))]
    if not refs:
        return text, b""
    try:
        return text, downscale_png(render_view(path, "iso", None, refs))
    except SystemExit:
        return text, b""


def start_analysis(path: str, material: str | None = None) -> str:
    """Begin the analysis in the background; returns job id, whether it is cached, and an ETA."""
    import json

    from stepscribe.exporter import cached_export
    from stepscribe.jobs import MANAGER, initial_eta_s, job_out_dir

    p = Path(path)
    if not p.is_file():
        raise ValueError(f"file not found: {path}")
    options = AnalyzeOptions(material=material, no_timestamp=True)
    hit = cached_export(p, job_out_dir(p, options), options)
    state = MANAGER.start(p, options)
    return json.dumps(
        {
            "job_id": state.id,
            "cached": hit is not None,
            "eta_s": 0 if hit else initial_eta_s(p),
            "note": "poll get_job_status; when it is done, call export_pack",
        }
    )


def get_job_status(job_id: str) -> str:
    """Stage, percent and ETA of a job; a short summary once it has finished."""
    import json

    from stepscribe.jobs import MANAGER

    s = MANAGER.poll(job_id)
    info: dict[str, Any] = {"job_id": s.id, "done": s.done, "error": s.error}
    if not s.done:
        info |= {
            "stage": s.last.get("stage", "starting"),
            "message": s.last.get("message", ""),
            "percent": round(100 * float(s.last.get("fraction", 0.0))),
            "eta_s": s.last.get("eta_s"),
            "elapsed_s": round(time.time() - s.started, 1),
        }
    elif s.summary and not s.error:
        sm = s.summary
        info |= {
            "percent": 100,
            "summary": f"{sm['parts']} parts, {sm['holes']} holes, {sm['joints']} fastener "
            f"joints, {sm['relations']} relations in {sm['seconds']} s",
        }
    return json.dumps(info)


def export_pack(
    path_or_job_id: str,
    small_budget_tokens: int = 12000,
    compact_budget_tokens: int = 40000,
    chat_max_images: int = 4,
    chat_max_mb: float = 10.0,
    material: str | None = None,
    copy_to: str | None = None,
) -> str:
    """Write (or reuse) the portable export; returns paths, token estimates and the SMALL text.

    *copy_to* is a folder that also receives a copy (``<name>_stepscribe/`` and its zip), so the
    result sits next to the STEP file and not only in the cache.
    """
    from stepscribe.exporter import export, load_result
    from stepscribe.jobs import MANAGER, job_out_dir

    options = AnalyzeOptions(material=material, no_timestamp=True)
    if path_or_job_id in MANAGER.jobs:
        state = MANAGER.poll(path_or_job_id)
        if not state.done:
            return get_job_status(path_or_job_id)
        if state.error or not state.summary:
            raise ValueError(state.error or "the job produced no result")
        result = load_result(state.summary["pack"])
        if result is None:
            raise ValueError("the finished job left no export")
    else:
        p = Path(path_or_job_id)
        if not p.is_file():
            raise ValueError(f"neither a job id nor a file: {path_or_job_id}")
        result = export(
            p,
            job_out_dir(p, options),
            options,
            small_budget_tokens=small_budget_tokens,
            compact_budget_tokens=compact_budget_tokens,
            chat_max_images=chat_max_images,
            chat_max_mb=chat_max_mb,
        )
    tok = result.tokens
    lines = [
        f"Export {'reused from the cache' if result.cached else 'written'}: {result.folder}",
        f"- FULL: {result.full} ({tok.get(Path(result.full).name, '?')} tokens)",
        f"- COMPACT: {result.compact} ({tok.get(Path(result.compact).name, '?')} tokens)",
        f"- SMALL: {result.small} ({tok.get(Path(result.small).name, '?')} tokens)",
        f"- Manifest: {result.manifest}",
        f"- Zip: {result.zip or 'not written'}",
        f"- Chat bundle: {', '.join(Path(b).name for b in result.chat_bundle)}",
    ]
    if result.dropped_images:
        lines.append("- Left out of the chat bundle: " + "; ".join(result.dropped_images))
    if copy_to:
        copy = _copy_export(result, Path(copy_to))
        lines.append(f"- Saved a copy in: {copy}")
        if result.zip:
            lines.append(f"- Zip copy: {copy.parent / Path(result.zip).name}")
    return "\n".join(lines) + "\n\n----- SMALL version -----\n\n" + result.small_text()


def _copy_export(result: Any, dest: Path) -> Path:
    """Copy the export folder (and its zip) into *dest*; returns the copied folder."""
    import shutil

    src = Path(result.folder)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / src.name
    shutil.copytree(src, target, dirs_exist_ok=True)
    if result.zip:
        shutil.copy2(result.zip, dest / Path(result.zip).name)
    return target


def diff(path_a: str, path_b: str) -> str:
    """What changed between two versions of a design (parts by name, BOM, counts)."""
    ra, rb = get_analysis(path_a).report, get_analysis(path_b).report
    pa = {p.name: p for p in ra.parts}
    pb = {p.name: p for p in rb.parts}
    lines: list[str] = []
    for name in sorted(pb.keys() - pa.keys()):
        lines.append(f"- Added part {name} ({shape_label(pb[name].shape_class)}).")
    for name in sorted(pa.keys() - pb.keys()):
        lines.append(f"- Removed part {name}.")
    for name in sorted(pa.keys() & pb.keys()):
        lines.extend(_part_changes(pa[name], pb[name]))
    qa = {b.name: b.quantity for b in ra.assembly.bom} if ra.assembly else {}
    qb = {b.name: b.quantity for b in rb.assembly.bom} if rb.assembly else {}
    for name in sorted(qa.keys() & qb.keys()):
        if qa[name] != qb[name]:
            lines.append(f"- Quantity of {name} changed from {qa[name]} to {qb[name]}.")
    ja = len(ra.assembly.fastener_joints) if ra.assembly else 0
    jb = len(rb.assembly.fastener_joints) if rb.assembly else 0
    if ja != jb:
        lines.append(f"- Fastener joints changed from {ja} to {jb}.")
    return "\n".join(lines) or "No differences found in parts, quantities or joints."


def _part_changes(a: Part, b: Part) -> list[str]:
    out: list[str] = []
    if a.content_hash == b.content_hash:
        return out
    if abs(a.mass.volume_mm3 - b.mass.volume_mm3) > 1e-3 * max(a.mass.volume_mm3, 1.0):
        out.append(
            f"- {a.name}: volume {fmt(a.mass.volume_mm3, 1)} → {fmt(b.mass.volume_mm3, 1)} mm³."
        )
    sa, sb = a.obb.size_sorted, b.obb.size_sorted
    if any(abs(x - y) > 0.01 for x, y in zip(sa, sb, strict=True)):
        out.append(
            f"- {a.name}: size {' × '.join(fmt(v) for v in sa)} → {' × '.join(fmt(v) for v in sb)} mm."
        )
    ha = sorted(round(h.diameter_mm, 2) for h in a.holes)
    hb = sorted(round(h.diameter_mm, 2) for h in b.holes)
    if ha != hb:
        out.append(f"- {a.name}: holes {len(ha)} → {len(hb)} (diameters {ha} → {hb}).")
    return out or [f"- {a.name}: geometry changed (content hash differs)."]


TOOLS: dict[str, Any] = {
    "get_overview": get_overview,
    "get_part": get_part,
    "list_holes": list_holes,
    "list_cutouts": list_cutouts,
    "get_relations": get_relations,
    "measure_distance": measure_distance,
    "section": section,
    "render_view": render_view,
    "find_parts": find_parts,
    "get_shopping_list": get_shopping_list,
    "diff": diff,
}


# ------------------------------------------------------------------ interview tools
def _session(path: str) -> Any:
    from stepscribe.understanding.session import Session

    a = get_analysis(path)
    key = str(id(a))
    if key not in _sessions:
        pipe = a.understanding_pipeline
        if pipe is None:
            raise ValueError("the understanding layer is not available for this file")
        _sessions[key] = Session(a, pipe.opts)
    return _sessions[key]


def _format_question(q: Any) -> str:
    lines = [f"[{q.id}] ({q.kind}, priority {q.priority}) {q.text}", f"  why it matters: {q.why}"]
    if q.options:
        lines.append("  options: " + "; ".join(f"{o.value} = {o.label}" for o in q.options))
    if q.unit:
        lines.append(f"  give a number in {q.unit}")
    if q.default_guess:
        lines.append(f"  my current guess: {q.default_guess}")
    lines.append(f"  answer type: {q.answer_type}")
    return "\n".join(lines)


def get_questions(path: str) -> str:
    """The open questions about a design, most valuable first, with the summary so far."""
    s = _session(path)
    u = s.understanding
    if not s.questions:
        return f"{u.summary}\n\nNo open questions."
    return (
        f"{u.summary}\n\nOpen questions (answer the first ones first; global ones change the most):\n\n"
        + "\n\n".join(_format_question(q) for q in s.questions)
    )


def submit_answer(
    path: str,
    question_id: str,
    status: str = "answered",
    value: str | None = None,
    note: str | None = None,
) -> str:
    """Record the designer's answer (answered, skipped or not_sure); returns what changed and the next question."""
    s = _session(path)
    known = {q.id: q for q in s.all_questions()}
    if question_id not in known:
        raise ValueError(f"unknown question {question_id}; ask get_questions for the current ones")
    q = known[question_id]
    parsed: Any = value
    if q.answer_type == "multi_choice" and isinstance(value, str):
        parsed = [v.strip() for v in value.split(",") if v.strip()]
    change = s.answer(question_id, status, parsed if status == "answered" else None, note)
    nxt = s.questions[0] if s.questions else None
    tail = "\n\nNext question:\n" + _format_question(nxt) if nxt else "\n\nNo more open questions."
    return f"{change.text}\n\nUpdated summary: {s.understanding.summary}{tail}"


TOOLS["start_analysis"] = start_analysis
TOOLS["get_job_status"] = get_job_status
TOOLS["export_pack"] = export_pack
TOOLS["get_questions"] = get_questions
TOOLS["submit_answer"] = submit_answer

__all__ = ["TOOLS", "clear_cache", "get_analysis", "np"]
