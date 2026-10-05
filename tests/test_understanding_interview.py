"""U8: the interview engine with scripted answers (no UI), the terminal interview, saved answers."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from stepscribe.analysis import AnalyzeOptions
from stepscribe.api import analyze_full
from stepscribe.cli import app
from stepscribe.understanding.session import Session


@pytest.fixture(scope="module")
def make(step_files):  # type: ignore[no-untyped-def]
    """A fresh Session on a shared analysis (every Session starts from the saved base state)."""

    @cache
    def analysis(name: str):  # type: ignore[no-untyped-def]
        return analyze_full(step_files[name], AnalyzeOptions(no_timestamp=True))

    def build(name: str) -> Session:
        a = analysis(name)
        return Session(a, AnalyzeOptions(no_timestamp=True), limit=15)

    return build


def _q(session: Session, kind: str, refs_contain: str | None = None):  # type: ignore[no-untyped-def]
    for q in session.all_questions():
        if q.kind == kind and (refs_contain is None or refs_contain in q.refs):
            return q
    raise AssertionError(f"no {kind} question")


def test_material_changes_mass_stability_and_thresholds(make) -> None:  # type: ignore[no-untyped-def]
    s = make("mini_mobile_manipulator")
    assert s.analysis.report.assembly.total_mass_g is None  # type: ignore[union-attr]
    assert s.understanding.stability is None
    edge_before = {w.threshold for w in s.understanding.weak_spots if w.category == "edge_distance"}
    ch = s.answer(_q(s, "material").id, "answered", "pla")
    mass = s.analysis.report.assembly.total_mass_g  # type: ignore[union-attr]
    assert mass and "Mass updated: unknown ->" in ch.text
    vol = sum(i.ap.part.mass.volume_mm3 for i in s.pipe.ctx.insts)  # type: ignore[union-attr]
    assert mass == pytest.approx(vol * 1e-3 * 1.24, rel=1e-6)
    assert s.understanding.stability is not None
    edge_after = {w.threshold for w in s.understanding.weak_spots if w.category == "edge_distance"}
    assert edge_after != edge_before or not edge_before  # PLA makes printing the likely process
    assert all(
        p.understanding.process.ranked[0].label == "fdm_3d_printed"
        for p in s.analysis.report.parts
        if p.understanding and not p.likely_purchased_hardware
    )


def test_every_answer_applies_in_under_a_second(make) -> None:  # type: ignore[no-untyped-def]
    s = make("mini_mobile_manipulator")
    times = []
    for q in [_q(s, "material"), _q(s, "payload")]:
        value = "alu" if q.kind == "material" else "300 g"
        times.append(s.answer(q.id, "answered", value).seconds)
    times.append(s.undo().seconds)  # type: ignore[union-attr]
    assert max(times) < 1.0, times


def test_confirming_a_role_sets_full_confidence_and_changes_neighbours(make) -> None:  # type: ignore[no-untyped-def]
    s = make("two_link_arm")
    base = next(p for p in s.analysis.report.parts if p.name == "Base")
    s.answer(_q(s, "choose_role", base.id).id, "answered", "bearing_housing")
    top = base.understanding.roles[0]  # type: ignore[union-attr]
    assert top.label == "bearing_housing" and top.confidence == 1.0
    assert top.evidence[0].code == "user"
    # a part confirmed as an arm link makes its link-mates look like joint brackets
    from stepscribe.understanding.function import assign_roles

    arm = next(p for p in s.analysis.report.parts if p.name == "ArmLink1")
    shaft = next(p for p in s.analysis.report.parts if p.name == "ShaftA")
    assign_roles(s.analysis, s.pipe.kin, s.pipe.ctx, confirmed={arm.id: "arm_link"})  # type: ignore[arg-type]
    joint_bracket = next(h for h in shaft.understanding.roles if h.label == "joint_bracket")  # type: ignore[union-attr]
    assert any(e.code == "neighbour:confirmed_arm_link" for e in joint_bracket.evidence)
    s.undo()


def test_skip_keeps_the_guess_and_is_not_asked_again(make) -> None:  # type: ignore[no-untyped-def]
    s = make("two_link_arm")
    q = s.questions[0]
    s.answer(q.id, "skipped")
    assert q.id not in {x.id for x in s.questions}
    assert s.analysis.report.assembly.total_mass_g is None  # type: ignore[union-attr]
    # a later session started from the same answers does not ask it either
    s2 = Session(s.analysis, AnalyzeOptions(no_timestamp=True))
    s2.pipe.finish(s.answers, s.context)
    assert q.id not in {x.id for x in s2.pipe.analysis.report.understanding.questions}  # type: ignore[union-attr]
    s.undo()


def test_not_sure_lowers_priority_permanently(make) -> None:  # type: ignore[no-untyped-def]
    s = make("two_link_arm")
    first = s.questions[0]
    s.answer(first.id, "not_sure")
    assert s.questions[0].id != first.id  # no longer the most urgent
    ids = [q.id for q in s.questions]
    assert first.id not in ids[:3]
    again = Session(s.analysis, AnalyzeOptions(no_timestamp=True))
    rerun = again.pipe.finish(s.answers, s.context).questions
    assert first.id not in [q.id for q in rerun][:3]  # permanently
    s.undo()


def test_contradicting_answers_add_a_conflict_note(make) -> None:  # type: ignore[no-untyped-def]
    s = make("two_link_arm")
    q = _q(s, "process")
    s.answer(q.id, "answered", "cnc_turned")
    u = s.understanding
    assert u.conflicts and "Designer says" in u.conflicts[0] and "cnc turned" in u.conflicts[0]
    parts = [p for p in s.analysis.report.parts if p.id in q.refs]
    assert parts and all(p.understanding.process.source == "user" for p in parts)  # type: ignore[union-attr]
    from stepscribe.understanding.describe import understanding_markdown

    md = understanding_markdown(s.analysis.report)
    assert "## Designer vs. geometry conflicts" in md and "Confirmed by designer" in md


def test_payload_adds_a_point_mass_and_lowers_the_tipping_angle(make) -> None:  # type: ignore[no-untyped-def]
    s = make("mini_mobile_manipulator")
    s.answer(_q(s, "material").id, "answered", "alu")
    ext0 = s.understanding.stability.extended_tipping_angle_deg  # type: ignore[union-attr]
    tip0 = s.understanding.stability.tipping_angle_deg  # type: ignore[union-attr]
    s.answer(_q(s, "payload").id, "answered", "2000 g")
    st = s.understanding.stability  # type: ignore[union-attr]
    assert st is not None and "payload" in s.understanding.summary
    assert (st.tipping_angle_deg or 0) != tip0 or (st.extended_tipping_angle_deg or 0) != ext0


def test_joint_drive_range_and_missing_part_answers(make) -> None:  # type: ignore[no-untyped-def]
    s = make("two_link_arm")
    jd = _q(s, "joint_drive", "KJ1")
    jr = _q(s, "joint_range", "KJ2")
    s.answer(jd.id, "answered", "manual")
    s.answer(jr.id, "answered", "-90 to 120")
    j1, j2 = s.understanding.kinematics.joints
    assert j1.driven_by == "designer:manual" and j2.range_deg_or_mm == (-90.0, 120.0)
    f = make("floating_part_assembly")
    assert any(w.category == "floating_part" for w in f.understanding.weak_spots)
    f.answer(_q(f, "missing_part").id, "answered", "missing")
    assert not any(w.category == "floating_part" for w in f.understanding.weak_spots)


def test_question_ids_survive_a_rerun_and_answers_reapply(step_files, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    a = analyze_full(step_files["two_link_arm"], AnalyzeOptions(no_timestamp=True))
    s = Session(a, AnalyzeOptions(no_timestamp=True))
    mat = _q(s, "material")
    s.answer(mat.id, "answered", "steel")
    ctx = tmp_path / "design_context.yaml"
    s.save(ctx)
    data = yaml.safe_load(ctx.read_text(encoding="utf-8"))
    assert data["answers"][mat.id]["value"] == "steel"
    again = analyze_full(step_files["two_link_arm"], _opts_with(ctx))
    assert again.report.assembly.total_mass_g  # type: ignore[union-attr]
    assert mat.id not in {q.id for q in again.report.understanding.questions}  # type: ignore[union-attr]
    assert any(x.question_id == mat.id for x in again.report.understanding.answers)  # type: ignore[union-attr]


def _opts_with(ctx: Path) -> AnalyzeOptions:
    from stepscribe.pack.design_context import apply_context, load_context

    opts = AnalyzeOptions(no_timestamp=True)
    apply_context(opts, load_context(ctx))
    return opts


def test_terminal_interview_with_scripted_input(step_files, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    ctx = tmp_path / "design_context.yaml"
    out = tmp_path / "pack"
    # 1 = material list, option 4 is pla; then skip one, mark one not sure, then done
    script = "4\ns\n?\nd\n"
    r = CliRunner().invoke(
        app,
        [
            "interview",
            str(step_files["two_link_arm"]),
            "--context",
            str(ctx),
            "--out",
            str(out),
        ],
        input=script,
    )
    assert r.exit_code == 0, r.output
    assert "Mass updated" in r.output and "Answers saved" in r.output
    data = yaml.safe_load(ctx.read_text(encoding="utf-8"))
    statuses = sorted(v["status"] for v in data["answers"].values())
    assert statuses == ["answered", "not_sure", "skipped"]
    md = next(out.glob("*_context_pack")) / "05_understanding.md"
    text = md.read_text(encoding="utf-8")
    assert "Confirmed by designer" in text and "Designer unsure" in text
    qa = (md.parent / "04_design_context.md").read_text(encoding="utf-8")
    assert (
        "## Questions the designer answered" in qa and "Confirmed by designer: What material" in qa
    )


def test_interview_back_key_undoes_the_last_answer(step_files, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    ctx = tmp_path / "c.yaml"
    r = CliRunner().invoke(
        app,
        [
            "interview",
            str(step_files["two_link_arm"]),
            "--context",
            str(ctx),
            "--out",
            str(tmp_path / "o"),
        ],
        input="4\nb\nd\n",
    )
    assert r.exit_code == 0 and "Undone" in r.output
    assert "No answers recorded" in r.output or not ctx.exists()
