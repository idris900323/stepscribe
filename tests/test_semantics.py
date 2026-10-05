"""Phase 5: semantic matching and the sourced-knowledge rule."""

from __future__ import annotations

from pathlib import Path

import yaml

from stepscribe.semantics.hardware import guess_from_name, guess_from_shape

KNOWLEDGE = Path(__file__).parent.parent / "stepscribe" / "knowledge"


def tags(analyses, name):  # type: ignore[no-untyped-def]
    return analyses(name).report.parts[0].semantic_tags


def test_every_knowledge_value_has_a_source() -> None:
    """Rule 8: a test fails if any sourced-data entry lacks `source`."""
    checked = 0
    sourced_tables = ("entries", "sizes", "profiles", "screws", "min_wall_mm")
    for path in sorted(KNOWLEDGE.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        rows: dict = {}  # type: ignore[type-arg]
        for table in sourced_tables:
            rows.update(data.get(table) or {})
        if path.name == "gears.yaml":
            rows["pressure_angle"] = data["pressure_angle_deg"]
            rows["formulas"] = data["formulas"]
        if path.name == "process_rules.yaml":
            rows["min_wall_plastic_cnc"] = data["min_wall_plastic_cnc_mm"]
        if path.name in ("weak_spot_rules.yaml", "roles.yaml"):
            rows = {k: v for k, v in data.items() if isinstance(v, dict)}
        assert rows, f"{path.name} has no sourced entries"
        for key, row in rows.items():
            assert str(row.get("source", "")).strip(), f"{path.name}:{key} has no source"
            checked += 1
    assert checked > 50


def test_nema17_motor_mount(analyses) -> None:  # type: ignore[no-untyped-def]
    mount = [t for t in tags(analyses, "nema17_plate") if t.kind == "motor_mount"]
    assert len(mount) == 1
    assert "NEMA 17" in mount[0].label and mount[0].confidence >= 0.8
    assert "central bore" in mount[0].evidence and mount[0].knowledge_id == "motors.nema17"


def test_forty_by_forty_plate_is_not_a_motor_mount(analyses) -> None:  # type: ignore[no-untyped-def]
    assert not [t for t in tags(analyses, "plate_4xM3") if t.kind in ("motor_mount", "board_mount")]


def test_bearing_seat_names_608(analyses) -> None:  # type: ignore[no-untyped-def]
    seat = [t for t in tags(analyses, "bearing_block") if t.kind == "bearing_seat"]
    assert len(seat) == 1 and "608" in seat[0].label and seat[0].confidence >= 0.8


def test_extrusion_2020(analyses) -> None:  # type: ignore[no-untyped-def]
    ext = [t for t in tags(analyses, "extrusion_2020_like") if t.kind == "extrusion_profile"]
    assert len(ext) == 1 and ext[0].knowledge_id == "extrusions.2020"
    assert "4 slot opening" in ext[0].evidence and "200.0 mm long" in ext[0].label


def test_shaft_stub_is_tagged_not_a_hole(analyses) -> None:  # type: ignore[no-untyped-def]
    assert any("Ø6 shaft" in t.label for t in tags(analyses, "boss_and_shaft"))


def test_names_that_only_look_like_hardware_are_not_hardware() -> None:
    for name in (
        "bolt_circle",
        "bearing_block",
        "Motor_Mount",
        "screw_hole_plate",
        "nut_pocket_plate",
    ):
        assert guess_from_name(name) is None, name
    assert guess_from_name("M3x10 ISO4762")
    assert guess_from_name("Motor_NEMA17").kind == "motor"  # type: ignore[union-attr]
    assert guess_from_name("608ZZ").kind == "bearing"  # type: ignore[union-attr]


def test_hardware_shape_rules_work_without_names(analyses) -> None:  # type: ignore[no-untyped-def]
    for fixture, kind in (("washer_m3", "washer"), ("nut_m3", "nut"), ("screw_m3x10", "screw")):
        ap = analyses(fixture).parts[0]
        guess = guess_from_shape(ap.geom, ap.part)
        assert guess is not None and guess.kind == kind, fixture
        assert "M3" in guess.description


def test_hardware_is_flagged_and_classed_as_fastener(analyses) -> None:  # type: ignore[no-untyped-def]
    for fixture in ("washer_m3", "nut_m3", "screw_m3x10"):
        part = analyses(fixture).report.parts[0]
        assert part.likely_purchased_hardware and part.shape_class.label == "fastener"


def test_two_parallel_blind_holes_are_not_a_standoff(analyses) -> None:  # type: ignore[no-untyped-def]
    assert not analyses("blind_holes").report.parts[0].likely_purchased_hardware


def test_every_inference_has_confidence_and_evidence(analyses, step_files) -> None:  # type: ignore[no-untyped-def]
    for name in step_files:
        for part in analyses(name).report.parts:
            assert 0 <= part.shape_class.confidence <= 1 and part.shape_class.evidence
            for t in part.semantic_tags:
                assert 0 < t.confidence <= 0.95 and t.evidence
            for h in part.holes:
                for m in h.standard_matches:
                    assert 0 < m.confidence <= 1 and m.evidence
