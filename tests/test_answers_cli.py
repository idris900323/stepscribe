# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Non-interactive answering (``questions --json``, ``answer``) and safe concurrent writers."""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import yaml
from typer.testing import CliRunner

from stepscribe.cli import app
from stepscribe.understanding.session import read_answers, save_answers

runner = CliRunner()


def _questions(step: Path, ctx: Path) -> list[dict]:  # type: ignore[type-arg]
    res = runner.invoke(app, ["questions", str(step), "--json", "--context", str(ctx)])
    assert res.exit_code == 0, res.output
    return json.loads(res.output)["questions"]  # type: ignore[no-any-return]


def test_questions_json_lists_ids_and_options(step_files) -> None:  # type: ignore[no-untyped-def]
    ctx = Path(step_files["two_link_arm"]).parent / "ctx_json.yaml"
    qs = _questions(step_files["two_link_arm"], ctx)
    assert qs and all({"id", "text", "why", "answer_type"} <= set(q) for q in qs)
    assert not ctx.exists()  # --json never writes


def test_answer_variants_and_undo(step_files, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    step, ctx = step_files["two_link_arm"], tmp_path / "design_context.yaml"
    qs = _questions(step, ctx)
    material = next(q for q in qs if q["kind"] == "material")["id"]

    def run(*args: str) -> str:
        res = runner.invoke(app, ["answer", str(step), "--context", str(ctx), *args])
        assert res.exit_code == 0, res.output
        return res.output

    out = run("--id", material, "--value", "pla", "--note", "best guess")
    assert "answered" in out
    rows = read_answers(ctx)
    assert rows[material]["status"] == "answered" and rows[material]["value"] == "pla"
    assert rows[material]["note"] == "best guess"

    process = next(q for q in _questions(step, ctx) if q["kind"] == "purpose")["id"]
    run("--id", process, "--skip")
    assert read_answers(ctx)[process]["status"] == "skipped"
    run("--id", process, "--not-sure")
    assert read_answers(ctx)[process]["status"] == "not_sure"

    run("--undo")  # the most recent answer (process) goes, the material answer stays
    rows = read_answers(ctx)
    assert process not in rows and material in rows

    bad = runner.invoke(app, ["answer", str(step), "--context", str(ctx), "--id", "NOPE", "--skip"])
    assert bad.exit_code == 1 and "Unknown question" in bad.output
    two = runner.invoke(
        app, ["answer", str(step), "--context", str(ctx), "--id", material, "--skip", "--not-sure"]
    )
    assert two.exit_code == 2
    # the answered material now shows in the next question list
    assert material not in {q["id"] for q in _questions(step, ctx)}


def _writer(path: str, qid: str, rounds: int) -> None:
    for i in range(rounds):
        save_answers(
            path,
            {
                qid: {
                    "status": "answered",
                    "value": f"v{i}",
                    "answered_at": f"2026-01-01T00:00:{i:02d}",
                }
            },
            only=[qid],
        )


def test_two_concurrent_writers_keep_both_answers(tmp_path: Path) -> None:
    ctx = tmp_path / "design_context.yaml"
    ctx.write_text("material: pla\npurpose: test\n", encoding="utf-8")
    ctxp = mp.get_context("spawn")
    procs = [ctxp.Process(target=_writer, args=(str(ctx), q, 12)) for q in ("QAAAAAA", "QBBBBBB")]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    rows = read_answers(ctx)
    assert rows["QAAAAAA"]["value"] == "v11" and rows["QBBBBBB"]["value"] == "v11"
    data = yaml.safe_load(ctx.read_text(encoding="utf-8"))
    assert data["material"] == "pla" and data["purpose"] == "test"
    assert not list(tmp_path.glob("*.lock")) and not list(tmp_path.glob("*.tmp"))


def test_stale_session_does_not_overwrite_a_newer_answer(tmp_path: Path) -> None:
    ctx = tmp_path / "c.yaml"
    save_answers(ctx, {"Q1": {"status": "answered", "value": "old"}})
    save_answers(ctx, {"Q1": {"status": "answered", "value": "new"}}, only=["Q1"])
    # a writer that only changed Q2 carries a stale Q1 but must not write it
    save_answers(
        ctx,
        {"Q1": {"status": "answered", "value": "old"}, "Q2": {"status": "skipped"}},
        only=["Q2"],
    )
    rows = read_answers(ctx)
    assert rows["Q1"]["value"] == "new" and rows["Q2"]["status"] == "skipped"
