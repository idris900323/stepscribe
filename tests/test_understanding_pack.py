"""U7: questions, the understanding pack file, motion check, CLI commands, evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from stepscribe.analysis import AnalyzeOptions
from stepscribe.api import analyze_full, build_context_pack
from stepscribe.cli import app
from stepscribe.pack.design_context import write_answer_stubs
from stepscribe.understanding.evaluate import score_report
from stepscribe.understanding.questions import GLOBAL_KINDS, MAX_QUESTIONS, generate_questions


def _u(analysis):  # type: ignore[no-untyped-def]
    u = analysis.report.understanding
    assert u is not None
    return u


def test_questions_are_ranked_global_first_and_limited(analyses) -> None:  # type: ignore[no-untyped-def]
    qs = _u(analyses("mini_mobile_manipulator")).questions
    assert 1 <= len(qs) <= MAX_QUESTIONS
    assert [q.priority for q in qs] == list(range(1, len(qs) + 1))
    kinds = [q.kind for q in qs]
    first_local = next(i for i, k in enumerate(kinds) if k not in GLOBAL_KINDS)
    assert all(k in GLOBAL_KINDS for k in kinds[:first_local]) and first_local >= 3
    assert {"material", "purpose", "process"} <= set(kinds[:first_local])
    for q in qs:
        assert q.why and q.text and q.answer_type and q.id.startswith("Q")


def test_question_ids_are_stable_across_runs(step_files) -> None:  # type: ignore[no-untyped-def]
    a = analyze_full(step_files["two_link_arm"], AnalyzeOptions(no_timestamp=True))
    b = analyze_full(step_files["two_link_arm"], AnalyzeOptions(no_timestamp=True))
    assert [q.id for q in _u(a).questions] == [q.id for q in _u(b).questions]


def test_answered_questions_are_not_asked_again(analyses) -> None:  # type: ignore[no-untyped-def]
    report = analyses("two_link_arm").report
    first = _u(analyses("two_link_arm")).questions
    answers = {
        first[0].id: {"status": "answered", "value": "alu"},
        first[1].id: {"status": "skipped"},
    }
    again = generate_questions(report, answers)
    assert first[0].id not in {q.id for q in again} and first[1].id not in {q.id for q in again}
    unsure = generate_questions(report, {first[0].id: {"status": "not_sure"}})
    pos = next((i for i, q in enumerate(unsure) if q.id == first[0].id), None)
    assert pos is None or pos > 0  # permanently lower priority


def test_pack_has_the_understanding_file_and_overview_lines(step_files, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    folder = build_context_pack(
        step_files["two_link_arm"], tmp_path, AnalyzeOptions(no_timestamp=True), images=True
    )
    text = (folder / "05_understanding.md").read_text(encoding="utf-8")
    for needle in (
        "## Design summary",
        "```mermaid",
        "### Joints",
        "## Part roles and process",
        "## Weak spots",
        "## Questions for the designer",
    ):
        assert needle in text, needle
    overview = (folder / "01_overview.md").read_text(encoding="utf-8")
    assert "## Design summary" in overview
    pack = (folder / "context_pack.md").read_text(encoding="utf-8")
    assert pack.index("## Design understanding") < pack.index("# Overview")
    assert "Likely: " in pack and "KJ1" in pack
    assert (folder / "images" / "kinematics_iso.png").is_file()
    assert "05_understanding.md" in (folder / "00_READ_ME_FIRST.md").read_text(encoding="utf-8")


def test_part_paragraphs_gain_role_and_process(analyses) -> None:  # type: ignore[no-untyped-def]
    from stepscribe.describe.part_describer import level1

    text = level1(analyses("two_link_arm").report.parts[0])
    assert "Likely: arm link" in text and "Likely made by" in text


def test_debug_graphs_write_a_graph_per_part(step_files, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    folder = build_context_pack(
        step_files["l_bracket"], tmp_path, AnalyzeOptions(no_timestamp=True, debug_graphs=True)
    )
    path = folder / "data" / "aag_PRT001.json"
    assert path.is_file()
    g = json.loads(path.read_text(encoding="utf-8"))
    assert {e["kind"] for e in g["edges"]} >= {"convex", "concave"}


def test_motion_check_finds_the_stop(step_files) -> None:  # type: ignore[no-untyped-def]
    a = analyze_full(
        step_files["limited_rotation_joint"], AnalyzeOptions(check_motion=True, no_timestamp=True)
    )
    u = _u(a)
    j = u.kinematics.joints[0]
    assert j.kind == "revolute" and j.range_deg_or_mm is not None
    lo, hi = j.range_deg_or_mm
    assert abs(hi - 90.0) <= 5.0 and lo <= -175.0  # blocked at about +90, free the other way
    assert j.blocked_by and "free range" in j.description
    spots = [w for w in u.weak_spots if w.category == "interference"]
    assert spots and spots[0].severity == "info"


def test_motion_check_is_off_by_default_and_asks_for_the_range(analyses) -> None:  # type: ignore[no-untyped-def]
    u = _u(analyses("limited_rotation_joint"))
    assert u.kinematics.joints[0].range_deg_or_mm is None
    assert any(q.kind == "joint_range" for q in u.questions)


def test_motion_check_prismatic_travel(step_files) -> None:  # type: ignore[no-untyped-def]
    a = analyze_full(
        step_files["linear_slide"], AnalyzeOptions(check_motion=True, no_timestamp=True)
    )
    j = _u(a).kinematics.joints[0]
    assert j.kind == "prismatic" and j.range_deg_or_mm is not None
    lo, hi = j.range_deg_or_mm
    assert -90 <= lo <= -70 and 70 <= hi <= 90  # the carriage stops at the end plates
    assert len(j.blocked_by) == 2


def test_cli_kinematics_weak_spots_and_questions(step_files, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    runner = CliRunner()
    path = str(step_files["two_link_arm"])
    r = runner.invoke(app, ["kinematics", path])
    assert r.exit_code == 0 and "flowchart LR" in r.output and "KJ1" in r.output
    r = runner.invoke(app, ["weak-spots", path])
    assert r.exit_code == 0
    ctx = tmp_path / "design_context.yaml"
    r = runner.invoke(app, ["questions", path, "--context", str(ctx)])
    assert r.exit_code == 0 and "answer stub" in r.output
    data = yaml.safe_load(ctx.read_text(encoding="utf-8"))
    assert data["answers"] and all("status" in v for v in data["answers"].values())
    r2 = runner.invoke(app, ["questions", path, "--context", str(ctx)])
    assert r2.exit_code == 0 and "answer stub" not in r2.output  # nothing new to add


def test_answer_stubs_keep_existing_answers(tmp_path: Path, analyses) -> None:  # type: ignore[no-untyped-def]
    qs = _u(analyses("two_link_arm")).questions
    ctx = tmp_path / "c.yaml"
    ctx.write_text(
        f"purpose: arm\nanswers:\n  {qs[0].id}:\n    status: answered\n    value: alu\n",
        encoding="utf-8",
    )
    added = write_answer_stubs(ctx, qs)
    assert added == len(qs) - 1
    data = yaml.safe_load(ctx.read_text(encoding="utf-8"))
    assert data["answers"][qs[0].id]["value"] == "alu" and data["purpose"] == "arm"


def test_evaluation_scores_joints_mechanisms_roles(analyses) -> None:  # type: ignore[no-untyped-def]
    labels = {
        "joints": [
            {"type": "revolute", "axis": [1, 0, 0], "point": [0, 0, 30]},
            {"type": "revolute", "axis": [1, 0, 0], "point": [0, 0, 95]},
            {"type": "prismatic", "axis": [0, 1, 0]},
        ],
        "mechanisms": [{"kind": "belt_drive", "ratio": 3.0}],
        "roles": {"ArmLink1": ["arm_link"], "Bearing608": ["bearing"], "Base": ["wheel"]},
        "weak_spots": ["tipping"],
    }
    s = score_report(analyses("two_link_arm").report, labels)
    assert s["joints"].tp == 2 and s["joints"].recall == 2 / 3 and s["joints"].precision == 1.0
    assert s["mechanisms"].recall == 0.0 and s["roles"].tp == 2
    assert any("Base" in m for m in s["roles"].misses)
    belt = score_report(
        analyses("arm_with_belt").report, {"mechanisms": [{"kind": "belt_drive", "ratio": 3.0}]}
    )
    assert belt["mechanisms"].recall == 1.0
