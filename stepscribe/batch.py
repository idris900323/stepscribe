# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Folder runs: parallel workers, per-file timeouts, one failure never stops the others."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from stepscribe.analysis import AnalyzeOptions
from stepscribe.procutil import kill_tree
from stepscribe.progress import ProgressFn, Tracker


@dataclass
class Job:
    """What a worker does for one file."""

    path: str
    out_dir: str
    mode: str = "pack"  # pack | analyze | export
    options: dict[str, Any] = field(default_factory=dict)
    context_file: str | None = None
    budget_tokens: int = 12000
    images: bool = False
    export: dict[str, Any] = field(default_factory=dict)  # extra export() parameters


def options_from_dict(d: dict[str, Any]) -> AnalyzeOptions:
    """Rebuild :class:`AnalyzeOptions` inside a worker process."""
    return AnalyzeOptions(**d)


def _summarise(
    path: str, report: Any, status: str, pack_dir: str | None, seconds: float
) -> dict[str, Any]:
    a = report.assembly
    mass = a.total_mass_g if a and a.total_mass_g is not None else None
    if mass is None and report.parts and all(p.mass.mass_g is not None for p in report.parts):
        mass = sum(p.mass.mass_g or 0.0 for p in report.parts)
    return {
        "file": path,
        "status": status,
        "parts": len(report.parts),
        "holes": sum(len(p.holes) for p in report.parts),
        "joints": len(a.fastener_joints) if a else 0,
        "relations": len(a.relations) if a else 0,
        "mass_g": None if mass is None else round(mass, 2),
        "warnings": len(report.warnings) + sum(len(p.warnings) for p in report.parts),
        "errors": [e.get("error", "") for e in report.errors],
        "pack": pack_dir,
        "seconds": round(seconds, 2),
    }


def process_file(job: Job, callback: ProgressFn | None = None) -> dict[str, Any]:
    """Analyse one file and write its output. Never raises."""
    from stepscribe.api import analyze_full
    from stepscribe.io.report_writer import write_report
    from stepscribe.pack.design_context import apply_context, load_context
    from stepscribe.pack.writer import write_pack

    start = time.time()
    try:
        opts = options_from_dict(job.options)
        if job.context_file:
            ctx = load_context(job.context_file)
            apply_context(opts, ctx)
        tracker = Tracker(callback)
        tracker.images_wanted = job.images and job.mode == "pack"
        if job.mode == "export":
            return _export_file(job, opts, tracker, start)
        analysis = analyze_full(job.path, opts, tracker)
        out = Path(job.out_dir)
        pack_dir: str | None = None
        if analysis.report.errors and not analysis.report.parts:
            status = "error"
        elif job.mode == "analyze":
            write_report(
                analysis.report, out / f"{Path(job.path).stem}.report.json", opts.precision
            )
            status = "ok"
        else:
            pack_dir = str(
                write_pack(analysis, out, job.budget_tokens, job.context_file, job.images, tracker)
            )
            status = "ok"
        tracker.finish("finished")
        return _summarise(job.path, analysis.report, status, pack_dir, time.time() - start)
    except Exception as exc:  # noqa: BLE001
        return {
            "file": job.path,
            "status": "error",
            "parts": 0,
            "holes": 0,
            "joints": 0,
            "relations": 0,
            "mass_g": None,
            "warnings": 0,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "pack": None,
            "seconds": round(time.time() - start, 2),
        }


def _export_file(job: Job, opts: AnalyzeOptions, tracker: Tracker, start: float) -> dict[str, Any]:
    """Export mode: reuse the export on disk when nothing changed, else analyse and write it."""
    from stepscribe.exporter import cached_export, export
    from stepscribe.models.schema import Report

    extra = dict(job.export)
    hit = cached_export(job.path, job.out_dir, opts, job.context_file, **extra)
    result = hit or export(
        job.path,
        job.out_dir,
        opts,
        job.context_file,
        images=job.images,
        tracker=tracker,
        **extra,
    )
    report = Report.model_validate_json(
        (Path(result.folder) / "data" / "report.json").read_text(encoding="utf-8")
    )
    tracker.finish("exported")
    return _summarise(job.path, report, "ok", result.folder, time.time() - start)


def _worker(job: Job, queue: Any) -> None:
    queue.put(process_file(job))


def run_jobs(
    jobs: list[Job],
    n_workers: int = 1,
    timeout_s: int = 300,
    on_done: Any = None,
    callback: ProgressFn | None = None,
) -> list[dict[str, Any]]:
    """Run jobs in separate processes (so a hang can be killed); results keep input order."""
    results: dict[int, dict[str, Any]] = {}
    if n_workers <= 1 and len(jobs) == 1:
        results[0] = process_file(jobs[0], callback)
        if on_done:
            on_done(results[0])
        return [results[0]]
    if n_workers > 1:
        os.environ["STEPSCRIBE_WORKERS"] = "1"  # files already run in parallel
    ctx = mp.get_context("spawn")
    pending = list(enumerate(jobs))
    running: dict[int, tuple[Any, Any, float]] = {}
    while pending or running:
        while pending and len(running) < max(1, n_workers):
            idx, job = pending.pop(0)
            q = ctx.Queue()
            proc = ctx.Process(target=_worker, args=(job, q))
            proc.start()
            running[idx] = (proc, q, time.time())
        time.sleep(0.05)
        for idx in list(running):
            proc, q, started = running[idx]
            res: dict[str, Any] | None = None
            if not q.empty():
                res = q.get()
                proc.join()
            elif time.time() - started > timeout_s:
                kill_tree(proc)
                res = _failure(
                    jobs[idx].path, f"timed out after {timeout_s} s", time.time() - started
                )
            elif not proc.is_alive():
                res = (
                    q.get()
                    if not q.empty()
                    else _failure(jobs[idx].path, "worker crashed", time.time() - started)
                )
            if res is not None:
                results[idx] = res
                del running[idx]
                if on_done:
                    on_done(res)
    return [results[i] for i in range(len(jobs))]


def _failure(path: str, message: str, seconds: float = 0.0) -> dict[str, Any]:
    return {
        "file": path,
        "status": "error",
        "parts": 0,
        "holes": 0,
        "joints": 0,
        "relations": 0,
        "mass_g": None,
        "warnings": 0,
        "errors": [message],
        "pack": None,
        "seconds": round(seconds, 2),
    }


def options_dict(opts: AnalyzeOptions) -> dict[str, Any]:
    """Picklable dict form of options."""
    return asdict(opts)


def write_index(results: list[dict[str, Any]], out_dir: Path) -> None:
    """``index.json`` and ``index.md`` for a folder run (stable order, no timestamps)."""
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(results, key=lambda r: r["file"].lower())
    (out_dir / "index.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    lines = [
        "# Index",
        "",
        "| File | Status | Parts | Holes | Joints | Relations | Mass (g) | Warnings | Pack |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        pack = Path(r["pack"]).name if r["pack"] else "-"
        mass = "-" if r["mass_g"] is None else r["mass_g"]
        lines.append(
            f"| {Path(r['file']).name} | {r['status']} | {r['parts']} | {r['holes']} | {r['joints']} | {r['relations']} | {mass} | {r['warnings']} | {pack} |"
        )
        for e in r["errors"]:
            lines.append(f"|  | error: {str(e)[:120]} | | | | | | | |")
    (out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
