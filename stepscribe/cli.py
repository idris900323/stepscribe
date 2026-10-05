# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Typer command line: ``stepscribe pack|analyze|render|section|init-context|schema|version``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from stepscribe import __version__, config
from stepscribe.analysis import AnalyzeOptions
from stepscribe.batch import Job, options_dict, run_jobs, write_index
from stepscribe.geometry.occ_utils import occt_version
from stepscribe.io.discovery import find_step_files
from stepscribe.progress import Progress as Progress_
from stepscribe.progress import format_eta

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="Turn STEP files into an LLM Context Pack."
)
console = Console()


def _utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


_utf8()

Path_ = Annotated[Path, typer.Argument(exists=True, help="A .step/.stp file or a folder of them.")]
Out = Annotated[Path, typer.Option("--out", "-o", help="Output directory.")]


def _options(
    material: str | None,
    density: float | None,
    up: str,
    front: str,
    check_interference: bool,
    precision: int,
    no_timestamp: bool,
    fast: bool = False,
    understanding: bool = True,
    check_motion: bool = False,
    motion_budget_s: int = 120,
    debug_graphs: bool = False,
) -> AnalyzeOptions:
    return AnalyzeOptions(
        density=density,
        material=material,
        up=up,
        front=front,
        no_timestamp=no_timestamp,
        check_interference=check_interference,
        precision=precision,
        skip_wall_thickness=fast,
        understanding=understanding,
        check_motion=check_motion,
        motion_budget_s=motion_budget_s,
        debug_graphs=debug_graphs,
    )


def _run(
    files: list[Path], jobs_tpl: dict[str, Any], jobs: int, timeout_s: int
) -> list[dict[str, Any]]:
    todo = [Job(path=str(f), **jobs_tpl) for f in files]
    results: list[dict[str, Any]] = []
    if len(todo) > 1:
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as bar:
            task = bar.add_task("stepscribe", total=len(todo))
            results = run_jobs(todo, jobs, timeout_s, on_done=lambda r: bar.advance(task))
    else:
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as bar:
            task = bar.add_task("starting", total=100)

            def on_progress(p: Progress_) -> None:
                bar.update(
                    task,
                    completed=100 * p.fraction,
                    description=f"{p.message[:48]:<48} ETA {format_eta(p.eta_s)}",
                )

            results = run_jobs(todo, 1, timeout_s, callback=on_progress)
    return results


def _summary(results: list[dict[str, Any]]) -> None:
    table = Table(title="stepscribe summary")
    for col in ("file", "parts", "holes", "joints", "relations", "mass g", "warnings", "status"):
        table.add_column(col)
    for r in results:
        status = r["status"] if r["status"] == "ok" else f"[red]{r['status']}[/red]"
        table.add_row(
            Path(r["file"]).name,
            str(r["parts"]),
            str(r["holes"]),
            str(r["joints"]),
            str(r["relations"]),
            "-" if r["mass_g"] is None else str(r["mass_g"]),
            str(r["warnings"]),
            status,
        )
    console.print(table)
    for r in results:
        for e in r["errors"]:
            console.print(f"[red]{Path(r['file']).name}:[/red] {e}")


@app.command()
def pack(
    path: Path_,
    out: Out = Path("out"),
    context: Annotated[Path | None, typer.Option("--context", help="design_context.yaml")] = None,
    material: Annotated[
        str | None, typer.Option(help="Material keyword, e.g. alu, steel, pla.")
    ] = None,
    density: Annotated[
        float | None, typer.Option(help="Density in g/cm3 (overrides --material).")
    ] = None,
    up: Annotated[str, typer.Option(help="Up axis: +X,-X,+Y,-Y,+Z,-Z")] = "+Z",
    front: Annotated[str, typer.Option(help="Front axis")] = "-Y",
    budget_tokens: Annotated[
        int, typer.Option(help="Token budget for context_pack.md")
    ] = config.DEFAULT_BUDGET_TOKENS,
    no_images: Annotated[
        bool, typer.Option("--no-images", help="Skip rendering (fast mode).")
    ] = False,
    check_interference: Annotated[bool, typer.Option("--check-interference")] = False,
    understanding: Annotated[
        bool,
        typer.Option(
            "--understanding/--no-understanding",
            help="Kinematics, mechanisms, roles, weak spots and questions (05_understanding.md).",
        ),
    ] = True,
    check_motion: Annotated[
        bool, typer.Option("--check-motion", help="Move each joint and test collisions (slow).")
    ] = False,
    motion_budget_s: Annotated[int, typer.Option(help="Time budget for --check-motion.")] = 120,
    debug_graphs: Annotated[
        bool, typer.Option("--debug-graphs", help="Also write data/aag_<part>.json.")
    ] = False,
    jobs: Annotated[int, typer.Option(help="Parallel workers for folders.")] = 1,
    timeout_s: Annotated[
        int, typer.Option(help="Per-file timeout in seconds.")
    ] = config.DEFAULT_TIMEOUT_S,
    precision: Annotated[
        int, typer.Option(help="Decimals in report.json")
    ] = config.FLOAT_PRECISION,
    no_timestamp: Annotated[bool, typer.Option("--no-timestamp")] = False,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Build a context pack for a STEP file or a folder of STEP files."""
    files = find_step_files(path)
    if not files:
        console.print("[red]No .step/.stp files found.[/red]")
        raise typer.Exit(1)
    opts = _options(
        material,
        density,
        up,
        front,
        check_interference,
        precision,
        no_timestamp,
        understanding=understanding,
        check_motion=check_motion,
        motion_budget_s=motion_budget_s,
        debug_graphs=debug_graphs,
    )
    tpl = {
        "out_dir": str(out),
        "mode": "pack",
        "options": options_dict(opts),
        "context_file": str(context) if context else None,
        "budget_tokens": budget_tokens,
        "images": not no_images,
    }
    results = _run(files, tpl, jobs, timeout_s)
    if path.is_dir():
        write_index(results, out)
    _summary(results)
    for r in results:
        if r["pack"] and verbose or (r["pack"] and len(results) == 1):
            console.print(f"Pack written to {r['pack']}")
    if any(r["status"] != "ok" for r in results):
        raise typer.Exit(1)


@app.command()
def export(
    path: Path_,
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory.")] = Path(
        "stepscribe_export"
    ),
    small_budget_tokens: Annotated[
        int, typer.Option(help="Token budget of the SMALL file and the chat bundle.")
    ] = 12000,
    compact_budget_tokens: Annotated[
        int, typer.Option(help="Token budget of the COMPACT file.")
    ] = config.DEFAULT_BUDGET_TOKENS,
    chat_max_images: Annotated[int, typer.Option(help="Images in the chat bundle.")] = 4,
    chat_max_mb: Annotated[float, typer.Option(help="Total size cap of the chat bundle.")] = 10.0,
    embed_images: Annotated[
        bool, typer.Option("--embed-images", help="Also write *_FULL_embedded.md (large).")
    ] = False,
    no_zip: Annotated[bool, typer.Option("--no-zip")] = False,
    no_images: Annotated[bool, typer.Option("--no-images", help="Skip rendering.")] = False,
    context: Annotated[Path | None, typer.Option("--context", help="design_context.yaml")] = None,
    material: Annotated[str | None, typer.Option(help="Material keyword, e.g. alu.")] = None,
    density: Annotated[float | None, typer.Option(help="Density in g/cm3.")] = None,
    up: Annotated[str, typer.Option(help="Up axis: +X,-X,+Y,-Y,+Z,-Z")] = "+Z",
    front: Annotated[str, typer.Option(help="Front axis")] = "-Y",
    check_interference: Annotated[bool, typer.Option("--check-interference")] = False,
    understanding: Annotated[bool, typer.Option("--understanding/--no-understanding")] = True,
    check_motion: Annotated[bool, typer.Option("--check-motion")] = False,
    jobs: Annotated[int, typer.Option(help="Parallel workers for folders.")] = 1,
    timeout_s: Annotated[
        int, typer.Option(help="Per-file timeout in seconds.")
    ] = config.DEFAULT_TIMEOUT_S,
    precision: Annotated[int, typer.Option()] = config.FLOAT_PRECISION,
    no_timestamp: Annotated[bool, typer.Option("--no-timestamp")] = False,
) -> None:
    """Write a portable export: FULL, COMPACT and SMALL Markdown, a chat bundle and a zip."""
    from stepscribe.exporter import load_result
    from stepscribe.exporter import write_index as write_export_index

    files = find_step_files(path)
    if not files:
        console.print("[red]No .step/.stp files found.[/red]")
        raise typer.Exit(1)
    opts = _options(
        material,
        density,
        up,
        front,
        check_interference,
        precision,
        no_timestamp,
        understanding=understanding,
        check_motion=check_motion,
    )
    tpl = {
        "out_dir": str(out),
        "mode": "export",
        "options": options_dict(opts),
        "context_file": str(context) if context else None,
        "budget_tokens": 0,
        "images": not no_images,
        "export": {
            "small_budget_tokens": small_budget_tokens,
            "compact_budget_tokens": compact_budget_tokens,
            "chat_max_images": chat_max_images,
            "chat_max_mb": chat_max_mb,
            "embed_images": embed_images,
            "make_zip": not no_zip,
        },
    }
    results = _run(files, tpl, jobs, timeout_s)
    _summary(results)
    exported = [r for r in (load_result(r["pack"]) for r in results if r["pack"]) if r]
    if path.is_dir() and exported:
        console.print(f"Index written to {write_export_index(exported, out)}")
    for r in exported:
        console.print(f"Export written to {r.folder}" + (" (reused)" if r.cached else ""))
    if any(r["status"] != "ok" for r in results):
        raise typer.Exit(1)


@app.command()
def analyze(
    path: Path_,
    out: Out = Path("out"),
    material: Annotated[str | None, typer.Option()] = None,
    density: Annotated[float | None, typer.Option()] = None,
    jobs: Annotated[int, typer.Option()] = 1,
    timeout_s: Annotated[int, typer.Option()] = config.DEFAULT_TIMEOUT_S,
    precision: Annotated[int, typer.Option()] = config.FLOAT_PRECISION,
    no_timestamp: Annotated[bool, typer.Option("--no-timestamp")] = False,
) -> None:
    """Write report.json only (no Markdown, no images)."""
    files = find_step_files(path)
    if not files:
        console.print("[red]No .step/.stp files found.[/red]")
        raise typer.Exit(1)
    opts = _options(material, density, "+Z", "-Y", False, precision, no_timestamp)
    tpl = {
        "out_dir": str(out),
        "mode": "analyze",
        "options": options_dict(opts),
        "context_file": None,
        "budget_tokens": 0,
        "images": False,
    }
    results = _run(files, tpl, jobs, timeout_s)
    _summary(results)
    if any(r["status"] != "ok" for r in results):
        raise typer.Exit(1)


@app.command("init-context")
def init_context(
    path: Annotated[Path, typer.Argument()] = Path("design_context.yaml"),
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Write a commented design_context.yaml template."""
    from stepscribe.pack.design_context import CONTEXT_TEMPLATE

    if path.exists() and not force:
        console.print(f"[red]{path} exists; use --force to overwrite.[/red]")
        raise typer.Exit(1)
    path.write_text(CONTEXT_TEMPLATE, encoding="utf-8", newline="\n")
    console.print(f"Wrote {path}")


@app.command()
def schema(out: Annotated[Path | None, typer.Option("--out", "-o")] = None) -> None:
    """Print (or write) the JSON Schema of report.json."""
    from stepscribe.models.schema import json_schema

    text = json.dumps(json_schema(), indent=2, ensure_ascii=False) + "\n"
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", newline="\n")
        console.print(f"Wrote {out}")
    else:
        sys.stdout.write(text)


@app.command()
def version() -> None:
    """Print the stepscribe and OpenCASCADE versions."""
    console.print(f"stepscribe {__version__} (OpenCASCADE via cadquery-ocp {occt_version()})")


@app.command()
def render(
    path: Annotated[Path, typer.Argument(exists=True)],
    view: Annotated[str, typer.Option(help="iso|front|top|right|exploded")] = "iso",
    part: Annotated[str | None, typer.Option(help="Part ID, e.g. PRT001")] = None,
    highlight: Annotated[str | None, typer.Option(help="Comma-separated IDs to highlight")] = None,
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("render.png"),
) -> None:
    """Render one labelled view to a PNG."""
    from stepscribe.render import render_view_file

    target = render_view_file(path, view, part, highlight.split(",") if highlight else None, out)
    console.print(f"Wrote {target}")


@app.command()
def section(
    path: Annotated[Path, typer.Argument(exists=True)],
    plane: Annotated[str, typer.Option(help='Plane such as "z=12.5"')] = "z=0",
    part: Annotated[str | None, typer.Option(help="Part ID")] = None,
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("section.png"),
) -> None:
    """Cut a plane through the model: writes a PNG and prints a text summary."""
    from stepscribe.geometry.section_report import section_file

    summary = section_file(path, plane, part, out)
    console.print(summary)


def _understood(
    path: Path,
    context: Path | None,
    material: str | None,
    density: float | None,
    check_motion: bool = False,
    quiet: bool = False,
) -> Any:
    """Analyse one file with the understanding layer (no pack is written)."""
    from stepscribe.api import analyze_full
    from stepscribe.pack.design_context import apply_context, load_context

    opts = _options(material, density, "+Z", "-Y", False, config.FLOAT_PRECISION, True, False)
    opts.check_motion = check_motion
    if context and context.is_file():
        apply_context(opts, load_context(context))
    if quiet:
        analysis = analyze_full(path, opts)
    else:
        with console.status("analysing..."):
            analysis = analyze_full(path, opts)
    if analysis.report.understanding is None:
        console.print(
            "[red]No understanding was produced.[/red] " + "; ".join(analysis.report.warnings)
        )
        raise typer.Exit(1)
    return analysis


@app.command()
def kinematics(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="A STEP file.")],
    check_motion: Annotated[bool, typer.Option("--check-motion")] = False,
    context: Annotated[Path | None, typer.Option("--context")] = None,
) -> None:
    """Print the kinematic description: links, joints, mechanisms and a Mermaid diagram."""
    analysis = _understood(path, context, None, None, check_motion)
    u = analysis.report.understanding
    assert u is not None
    console.print(u.summary, markup=False)
    console.print()
    console.print(u.kinematics.description, markup=False)
    for j in u.kinematics.joints:
        console.print(f"  {j.description} (confidence {j.confidence:.2f})", markup=False)
    for m in u.kinematics.mechanisms:
        console.print(f"  {m.id} {m.kind}: {m.description}", markup=False)
    console.print()
    console.print(u.kinematics.mermaid, markup=False)


@app.command("weak-spots")
def weak_spots_cmd(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="A STEP file.")],
    material: Annotated[
        str | None, typer.Option(help="Material keyword (enables mass and stability).")
    ] = None,
    density: Annotated[float | None, typer.Option(help="Density in g/cm3.")] = None,
    context: Annotated[Path | None, typer.Option("--context")] = None,
) -> None:
    """Print the weak-spot table, most severe first."""
    analysis = _understood(path, context, material, density)
    u = analysis.report.understanding
    assert u is not None
    if not u.weak_spots:
        console.print("No weak spot rule fired.")
        return
    table = Table(title="weak spots")
    for col in ("ID", "severity", "finding", "suggestion", "depends on"):
        table.add_column(col, overflow="fold")
    for w in u.weak_spots:
        table.add_row(
            w.id, w.severity, w.message, w.suggestion or "", w.depends_on_assumption or ""
        )
    console.print(table)


@app.command()
def questions(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="A STEP file.")],
    context: Annotated[
        Path, typer.Option("--context", help="design_context.yaml to read and update.")
    ] = Path("design_context.yaml"),
    material: Annotated[str | None, typer.Option()] = None,
    density: Annotated[float | None, typer.Option()] = None,
    write: Annotated[bool, typer.Option("--write/--no-write", help="Add an answers: stub.")] = True,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the questions as JSON (nothing is written).")
    ] = False,
) -> None:
    """Print the questions the design raises; add an `answers:` stub to the context file."""
    from stepscribe.pack.design_context import write_answer_stubs

    analysis = _understood(path, context, material, density, quiet=as_json)
    u = analysis.report.understanding
    assert u is not None
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "summary": u.summary,
                    "questions": [q.model_dump(mode="json") for q in u.questions],
                    "answers": [a.model_dump(mode="json") for a in u.answers],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if not u.questions:
        console.print("Nothing to ask: every open question has an answer.")
        return
    for q in u.questions:
        console.print(f"{q.priority}. [{q.id}] {q.text}", markup=False)
        console.print(f"     why: {q.why}", markup=False)
        if q.options:
            console.print("     options: " + ", ".join(o.label for o in q.options), markup=False)
    if write:
        added = write_answer_stubs(context, u.questions)
        if added:
            console.print(
                f"\n{added} answer stub(s) added to {context} (fill status/value, then re-run)."
            )
        else:
            console.print(f"\nNo new questions: {context} already has a stub for each.")


@app.command()
def answer(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="A STEP file.")],
    question_id: Annotated[
        str | None, typer.Option("--id", help="Question ID, e.g. QD4A1E6.")
    ] = None,
    value: Annotated[
        str | None, typer.Option("--value", help="The answer (a, b for several).")
    ] = None,
    note: Annotated[
        str | None, typer.Option("--note", help="A remark kept with the answer.")
    ] = None,
    skip: Annotated[bool, typer.Option("--skip", help="Skip the question.")] = False,
    not_sure: Annotated[
        bool, typer.Option("--not-sure", help="Mark the answer as a soft fact.")
    ] = False,
    undo: Annotated[bool, typer.Option("--undo", help="Remove the most recent answer.")] = False,
    context: Annotated[
        Path, typer.Option("--context", help="design_context.yaml to read and update.")
    ] = Path("design_context.yaml"),
    material: Annotated[str | None, typer.Option()] = None,
    density: Annotated[float | None, typer.Option()] = None,
) -> None:
    """Answer one question without a prompt (for scripts and AI assistants), or undo the last.

    The answer is saved in the context file at once; run `stepscribe export` or `pack` with
    `--context` to see the effect.
    """
    from stepscribe.understanding.session import Session, read_answers, save_answers

    if undo:
        rows = read_answers(context)
        if not rows:
            console.print("Nothing to undo: no answers in the context file.")
            raise typer.Exit(1)
        last = max(rows, key=lambda k: (str(rows[k].get("answered_at") or ""), k))
        save_answers(context, {}, only=(), remove=[last])
        console.print(f"Removed the answer to {last}.")
        return
    if question_id is None:
        console.print("[red]Give --id (or --undo).[/red]")
        raise typer.Exit(2)
    flags = sum([skip, not_sure, value is not None])
    if flags != 1:
        console.print("[red]Give exactly one of --value, --skip or --not-sure.[/red]")
        raise typer.Exit(2)
    analysis = _understood(path, context, material, density, quiet=True)
    session = Session(analysis, analysis.understanding_pipeline.opts)
    known = {q.id: q for q in session.all_questions()}
    if question_id not in known:
        console.print(
            f"[red]Unknown question {question_id}.[/red] Known: " + ", ".join(sorted(known))
        )
        raise typer.Exit(1)
    q = known[question_id]
    status = "skipped" if skip else "not_sure" if not_sure else "answered"
    parsed: Any = value
    if status == "answered" and q.answer_type == "multi_choice" and value is not None:
        parsed = [v.strip() for v in value.split(",") if v.strip()]
    change = session.answer(question_id, status, parsed if status == "answered" else None, note)
    session.save(context)
    console.print(f"{question_id}: {status}. {change.text}", markup=False)
    if session.questions:
        console.print(
            f"Next: [{session.questions[0].id}] {session.questions[0].text}", markup=False
        )


@app.command()
def interview(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="A STEP file.")],
    context: Annotated[
        Path, typer.Option("--context", help="design_context.yaml to read and update.")
    ] = Path("design_context.yaml"),
    out: Out = Path("out"),
    max_questions: Annotated[int, typer.Option(help="Questions per session.")] = 15,
    material: Annotated[str | None, typer.Option()] = None,
    density: Annotated[float | None, typer.Option()] = None,
    images: Annotated[
        bool, typer.Option("--images/--no-images", help="Render images in the pack.")
    ] = False,
) -> None:
    """Answer the questions the design raises; the pack is rebuilt with your answers."""
    from stepscribe.pack.writer import write_pack
    from stepscribe.understanding.interview import run_interview
    from stepscribe.understanding.session import Session

    analysis = _understood(path, context, material, density)
    pipe = analysis.understanding_pipeline
    session = Session(analysis, pipe.opts, limit=max_questions)
    console.print(analysis.report.understanding.summary, markup=False)
    n = run_interview(session, console, context_file=context)
    if n:
        folder = write_pack(
            analysis,
            out,
            config.DEFAULT_BUDGET_TOKENS,
            context if context.is_file() else None,
            images,
        )
        console.print(f"Pack rebuilt with your answers: {folder}")
    else:
        console.print("No answers recorded; the pack was not rebuilt.")


@app.command()
def ui(
    path: Annotated[
        Path | None,
        typer.Argument(exists=True, dir_okay=False, help="Optional STEP file to analyse at once."),
    ] = None,
    port: Annotated[int, typer.Option(help="Port to listen on (127.0.0.1 only).")] = 8765,
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Do not open a browser.")
    ] = False,
    samples: Annotated[
        Path | None, typer.Option(help="Folder of STEP files offered as samples.")
    ] = None,
) -> None:
    """Start the local web UI: upload a STEP file, see an ETA, inspect the context pack."""
    from stepscribe.ui.server import serve

    default = Path("robot_steps")
    folder = samples or (default if default.is_dir() else None)
    serve(port, not no_browser, folder, path)


@app.command()
def mcp() -> None:
    """Run the read-only MCP server over stdio (needs: pip install stepscribe[mcp])."""
    from stepscribe.mcp.server import main

    main()


def _llm_pack(path: Path, vision: bool) -> tuple[str, list[Path], Any]:
    """Build a pack in a temp folder; returns (context_pack.md text, images, report)."""
    import tempfile

    from stepscribe.api import build_context_pack
    from stepscribe.models.schema import Report

    tmp = Path(tempfile.mkdtemp(prefix="stepscribe_llm_"))
    folder = build_context_pack(path, tmp, AnalyzeOptions(no_timestamp=True), images=vision)
    text = (folder / "context_pack.md").read_text(encoding="utf-8")
    report = Report.model_validate_json(
        (folder / "data" / "report.json").read_text(encoding="utf-8")
    )
    images = [folder / "images" / n for n in ("assembly_iso.png", "assembly_exploded.png")]
    return text, images, report


def _llm_config() -> Any:
    from stepscribe.llm.adapter import LLMConfig, LLMConfigError

    try:
        return LLMConfig.from_env()
    except LLMConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc


@app.command()
def ask(path: Path_, question: Annotated[str, typer.Argument(help="Your question.")]) -> None:
    """Ask a question about a design: builds the pack and sends it to your configured LLM."""
    from stepscribe.llm.adapter import build_prompt, chat

    cfg = _llm_config()
    text, images, _report = _llm_pack(path, cfg.vision)
    console.print(chat(cfg, build_prompt(text, f"Question: {question}"), images))


@app.command("eval")
def eval_cmd(path: Path_, limit: Annotated[int, typer.Option(help="Max questions.")] = 12) -> None:
    """Score how well your configured LLM answers factual questions about a design."""
    from stepscribe.llm.adapter import build_prompt, chat
    from stepscribe.llm.evaluate import SUFFIX, EvalResult, accuracy, build_questions, score

    cfg = _llm_config()
    text, images, report = _llm_pack(path, cfg.vision)
    results = []
    for q in build_questions(report, limit):
        answer = chat(cfg, build_prompt(text, q.text), images)
        results.append(EvalResult(q, answer, score(answer, q)))
    table = Table(title=f"stepscribe eval: {cfg.model}")
    for col in ("question", "expected", "model answer", "ok"):
        table.add_column(col)
    for r in results:
        ok = "yes" if r.correct else "[red]no[/red]"
        table.add_row(
            r.question.text.replace(" " + SUFFIX, ""), r.question.expected, r.answer[:40], ok
        )
    console.print(table)
    hits = sum(r.correct for r in results)
    console.print(f"Accuracy: {accuracy(results) * 100:.0f}% ({hits}/{len(results)})")


if __name__ == "__main__":
    app()
