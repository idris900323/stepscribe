"""Reading, determinism, JSON and pack output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stepscribe.analysis import AnalyzeOptions
from stepscribe.api import analyze, build_context_pack
from stepscribe.cli import app
from stepscribe.io.discovery import find_step_files
from stepscribe.io.report_writer import report_to_json


def test_json_is_byte_identical_across_runs(step_files) -> None:  # type: ignore[no-untyped-def]
    opts = AnalyzeOptions(no_timestamp=True)
    a = report_to_json(analyze(step_files["plate_4xM3"], opts))
    b = report_to_json(analyze(step_files["plate_4xM3"], opts))
    assert a == b


def test_multi_solid_names(analyses) -> None:  # type: ignore[no-untyped-def]
    names = sorted(p.name for p in analyses("multi_solid").report.parts)
    assert names == ["multi_solid_body1", "multi_solid_body2"]


def test_inch_cube_is_converted_to_mm(analyses) -> None:  # type: ignore[no-untyped-def]
    p = analyses("cube_1in").report.parts[0]
    assert abs(p.bbox.size.x - 25.4) < 1e-3
    assert analyses("cube_1in").report.meta.original_length_unit == "in"


def test_bad_file_becomes_error_not_traceback(tmp_path: Path) -> None:
    bad = tmp_path / "bad.step"
    bad.write_text("this is not a step file")
    report = analyze(bad)
    assert report.errors and not report.parts


def test_folder_discovery_is_recursive_case_insensitive(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    for n in ("b.STEP", "a.stp", "sub/c.Step", "d.txt"):
        (tmp_path / n).write_text("x")
    assert [f.name for f in find_step_files(tmp_path)] == ["a.stp", "b.STEP", "c.Step"]


def test_pack_layout_and_golden(step_files, tmp_path: Path, golden) -> None:  # type: ignore[no-untyped-def]
    folder = build_context_pack(
        step_files["plate_4xM3"], tmp_path, AnalyzeOptions(no_timestamp=True, material="alu")
    )
    for rel in (
        "00_READ_ME_FIRST.md",
        "01_overview.md",
        "02_assembly.md",
        "context_pack.md",
        "data/report.json",
    ):
        assert (folder / rel).is_file(), rel
    assert any((folder / "03_parts").glob("PRT001_*.md"))
    golden("plate_4xM3.report.json", (folder / "data" / "report.json").read_text(encoding="utf-8"))
    golden(
        "plate_4xM3.PRT001.md",
        next((folder / "03_parts").glob("PRT001_*.md")).read_text(encoding="utf-8"),
    )
    text = (folder / "context_pack.md").read_text(encoding="utf-8")
    assert "Likely:" in text and "4 × Ø3.4 THRU" in text


def test_context_pack_respects_token_budget(step_files, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    folder = build_context_pack(
        step_files["bolt_circle"], tmp_path, AnalyzeOptions(no_timestamp=True), budget_tokens=200
    )
    text = (folder / "context_pack.md").read_text(encoding="utf-8")
    assert "Only a one-line summary fits" in text or "none" in text


def test_design_context_sets_material_and_is_copied(step_files, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    ctx = tmp_path / "ctx.yaml"
    ctx.write_text(
        "purpose: Test plate\nmaterial: steel\nquestions:\n  - Is it stiff?\n", encoding="utf-8"
    )
    folder = build_context_pack(
        step_files["plate_4xM3"],
        tmp_path / "o",
        AnalyzeOptions(no_timestamp=True),
        context_file=ctx,
    )
    assert "Is it stiff?" in (folder / "04_design_context.md").read_text(encoding="utf-8")
    report = json.loads((folder / "data" / "report.json").read_text(encoding="utf-8"))
    assert report["parts"][0]["mass"]["density_g_cm3"] == 7.85
    assert report["parts"][0]["mass"]["material_source"] == "user"


def test_cli_pack_and_init_context(step_files, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    r = CliRunner().invoke(
        app,
        ["pack", str(step_files["cube_10"]), "-o", str(tmp_path), "--no-images", "--no-timestamp"],
    )
    assert r.exit_code == 0, r.output
    target = tmp_path / "design_context.yaml"
    r = CliRunner().invoke(app, ["init-context", str(target)])
    assert r.exit_code == 0 and target.read_text(encoding="utf-8").startswith(
        "# design_context.yaml"
    )


def test_assembly_with_external_references_gives_a_clear_error(tmp_path: Path) -> None:
    """A STEP whose parts live in other files has empty prototypes: say so instead of crashing."""
    from assembly_fixtures import Node, write_assembly
    from build123d import Box, Location

    from stepscribe.io.step_reader import StepReadError, read_step

    path = write_assembly(
        tmp_path / "a.step", {"B": Box(5, 5, 5)}, Node("A", children=[Node("x", "B", Location())])
    )
    assert len(read_step(path).protos) == 1  # sanity: a normal file still reads
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound

    import stepscribe.io.step_reader as sr

    model = read_step(path)
    empty = TopoDS_Compound()
    BRep_Builder().MakeCompound(empty)
    for proto in model.protos.values():
        proto.shape = empty
    with pytest.raises(StepReadError, match="external references"):
        sr._drop_empty_prototypes(model, "a.step")


def test_headline_features_are_capped() -> None:
    from types import SimpleNamespace

    from stepscribe.describe.assembly_describer import HEADLINE_MAX, headline_semantics

    tags = [SimpleNamespace(label=f"kind {i:02d}") for i in range(HEADLINE_MAX + 5)]
    part = SimpleNamespace(id="PRT001", semantic_tags=tags)
    report = SimpleNamespace(parts=[part], assembly=None)
    rows = headline_semantics(report)  # type: ignore[arg-type]
    assert len(rows) == HEADLINE_MAX + 1
    assert rows[-1].startswith("5 more kinds")
