"""Phase 6: assembly tree, BOM, contacts, joints, relations, spatial facts, packs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from assembly_fixtures import Node, write_assembly
from build123d import Box, Location

from stepscribe.analysis import AnalyzeOptions
from stepscribe.api import analyze, build_context_pack


def asm(analyses, name):  # type: ignore[no-untyped-def]
    report = analyses(name).report
    assert report.assembly is not None and not report.errors
    return report, report.assembly


def test_two_plates_bom_names_and_single_analysis(analyses) -> None:  # type: ignore[no-untyped-def]
    report, a = asm(analyses, "two_plates_assembly")
    assert [p.name for p in report.parts] == ["Plate"], "a part used twice is analysed once"
    assert [(b.name, b.quantity) for b in a.bom] == [("Plate", 2)]
    assert sorted(i.path for i in a.instances) == ["TwoPlates/BasePlate", "TwoPlates/TopPlate"]
    assert {i.part_id for i in a.instances} == {"PRT001"}


def test_two_plates_joints_grip_10(analyses) -> None:  # type: ignore[no-untyped-def]
    _r, a = asm(analyses, "two_plates_assembly")
    assert len(a.fastener_joints) == 4
    for j in a.fastener_joints:
        assert j.stack_thickness_mm == pytest.approx(10.0, abs=1e-3)
        assert j.common_diameter_mm == pytest.approx(3.4)
        assert j.suggested_fastener == "M3 × 14 SHCS" and not j.has_threaded_end
        assert len(j.hole_refs) == 2
    assert {"item": "M3 × 14 SHCS", "qty": 4} in a.fastener_shopping_list
    assert {"item": "M3 nut", "qty": 4} in a.fastener_shopping_list


def test_two_plates_relations_and_contact_area(analyses) -> None:  # type: ignore[no-untyped-def]
    _r, a = asm(analyses, "two_plates_assembly")
    preds = {r.predicate: r for r in a.relations}
    assert set(preds) == {"bolted_to", "rests_on"}
    assert preds["rests_on"].sentence.startswith("TopPlate rests on BasePlate")
    assert (
        preds["bolted_to"].sentence
        == "TopPlate is bolted to BasePlate with 4 × M3 screws (J001–J004)."
    )
    (c,) = [c for c in a.contacts if c.kind == "planar_face"]
    expected = 60 * 60 - 4 * np.pi * 1.7**2
    assert c.contact_area_mm2 == pytest.approx(expected, abs=0.1)
    assert c.normal is not None and c.normal.z == pytest.approx(
        1.0, abs=1e-6
    )  # from A (lower) up into B


def test_tapped_joint_has_threaded_end_and_rule_length(analyses) -> None:  # type: ignore[no-untyped-def]
    _r, a = asm(analyses, "tapped_assembly")
    assert len(a.fastener_joints) == 4
    j = a.fastener_joints[0]
    assert (
        j.has_threaded_end and j.suggested_fastener == "M3 × 12 SHCS"
    )  # 13 mm available, largest standard <= 13
    assert j.stack_thickness_mm == pytest.approx(13.0, abs=1e-3)
    assert a.fastener_shopping_list == [{"item": "M3 × 12 SHCS", "qty": 4}]


def test_shaft_in_bearing_cylindrical_fit(analyses) -> None:  # type: ignore[no-untyped-def]
    _r, a = asm(analyses, "shaft_in_bearing_assembly")
    (c,) = a.contacts
    assert c.kind == "cylindrical_fit" and c.fit == "clearance"
    assert c.fit_value_mm == pytest.approx(0.06, abs=1e-6)
    names = {i.id: i.path.rsplit("/", 1)[-1] for i in a.instances}
    assert (names[c.instance_a], names[c.instance_b]) == ("Shaft", "Housing")
    (rel,) = a.relations
    assert rel.predicate == "inserted_into" and "diametral clearance 0.06 mm" in rel.sentence


def test_nested_assembly_paths_transforms_subassemblies(analyses) -> None:  # type: ignore[no-untyped-def]
    _r, a = asm(analyses, "nested_assembly")
    by_path = {i.path: i for i in a.instances}
    assert set(by_path) == {"Robot/Base", "Robot/Arm/Gripper/FingerL", "Robot/Arm/Gripper/FingerR"}
    left = by_path["Robot/Arm/Gripper/FingerL"]
    assert (left.position.x, left.position.y, left.position.z) == pytest.approx(
        (100.0, -8.0, 30.0), abs=1e-6
    )
    axis, angle = left.rotation_axis_angle
    assert angle == pytest.approx(90.0, abs=1e-6) and abs(axis.z) == pytest.approx(1.0)
    assert [s.name for s in a.subassemblies] == ["Arm"]
    assert len(a.subassemblies[0].instance_ids) == 2
    # the 10 x 4 x 20 finger is rotated 90 deg about Z, so its global footprint is 4 x 10 x 20
    size = left.global_bbox.size
    assert (size.x, size.y, size.z) == pytest.approx((4.0, 10.0, 20.0), abs=1e-6)


def test_spatial_facts_use_up_and_front(analyses) -> None:  # type: ignore[no-untyped-def]
    _r, a = asm(analyses, "nested_assembly")
    text = " ".join(s.sentence for s in a.spatial_facts)
    assert "Base is to the left of the assembly centre" in text  # right = +X for front -Y, up +Z
    assert "FingerL is to the right of the assembly centre" in text
    _r, plates = asm(analyses, "two_plates_assembly")
    assert any(
        s.relation == "above" and "TopPlate is above BasePlate" in s.sentence
        for s in plates.spatial_facts
    )


def test_center_of_mass_and_total_mass_with_material(step_files) -> None:  # type: ignore[no-untyped-def]
    report = analyze(
        step_files["two_plates_assembly"], AnalyzeOptions(no_timestamp=True, material="alu")
    )
    a = report.assembly
    assert a is not None and a.total_mass_g is not None and a.center_of_mass is not None
    assert a.total_mass_g == pytest.approx(2 * report.parts[0].mass.mass_g, rel=1e-9)
    assert a.center_of_mass.z == pytest.approx(2.5, abs=1e-6)  # plates centred at z = 0 and z = 5


def test_interference_check_is_opt_in(tmp_path: Path) -> None:
    path = write_assembly(
        tmp_path / "overlap.step",
        {"Cube": Box(10, 10, 10)},
        Node(
            "Overlap",
            children=[Node("A", "Cube", Location()), Node("B", "Cube", Location((5, 0, 0)))],
        ),
    )
    assert analyze(path).assembly.interferences == []  # type: ignore[union-attr]
    found = analyze(path, AnalyzeOptions(check_interference=True)).assembly.interferences  # type: ignore[union-attr]
    assert len(found) == 1 and found[0]["volume_mm3"] == pytest.approx(500.0)


def test_pack_includes_assembly_sections(step_files, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    folder = build_context_pack(
        step_files["two_plates_assembly"], tmp_path, AnalyzeOptions(no_timestamp=True)
    )
    asm_md = (folder / "02_assembly.md").read_text(encoding="utf-8")
    assert "TopPlate is bolted to BasePlate with 4 × M3 screws (J001–J004)." in asm_md
    assert "Plate × 2" in asm_md and "4 × M3 × 14 SHCS" in asm_md
    pack = (folder / "context_pack.md").read_text(encoding="utf-8")
    assert "## Relations" in pack and "TopPlate rests on BasePlate" in pack
    assert "Fastener shopping list" in pack
    overview = (folder / "01_overview.md").read_text(encoding="utf-8")
    assert (
        "has 2 parts (1 different)" in overview
        and "60.0 (width) × 60.0 (depth) × 10.0 (height) mm" in overview
    )
    part_md = next((folder / "03_parts").glob("PRT001_*.md")).read_text(encoding="utf-8")
    assert "## Relations" in part_md


def test_parallel_part_analysis_matches_serial(step_files, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A process pool must give exactly the same report as a single process."""
    from stepscribe import api

    path = step_files["shaft_in_bearing_assembly"]
    monkeypatch.setattr(api, "PARALLEL_MIN_FACES", 0)
    monkeypatch.setattr(api, "PARALLEL_MIN_PARTS", 2)
    monkeypatch.setenv("STEPSCRIBE_WORKERS", "1")
    serial = api.analyze_full(path).report.parts
    monkeypatch.setenv("STEPSCRIBE_WORKERS", "2")
    parallel = api.analyze_full(path).report.parts
    assert [p.model_dump(mode="json") for p in serial] == [
        p.model_dump(mode="json") for p in parallel
    ]


def test_mutual_rests_on_is_said_once() -> None:
    from stepscribe.assembly.relations import _merge_mutual_rests
    from stepscribe.models.schema import Contact, Relation

    def rel(s: str, o: str, via: str) -> tuple[tuple[object, ...], Relation]:
        return (
            (0, s, o),
            Relation(
                id="",
                subject=s,
                predicate="rests_on",
                object=o,
                via=[via],
                sentence="x",
                confidence=0.9,
            ),
        )

    rels = [rel("INS001", "INS002", "C001"), rel("INS002", "INS001", "C002")]
    contacts = [
        Contact(
            id=f"C00{n}",
            instance_a="INS001",
            instance_b="INS002",
            kind="planar_face",
            min_distance_mm=0.0,
            contact_area_mm2=10.0 * n,
        )
        for n in (1, 2)
    ]
    added: list[tuple[str, str, str, list[str], str, float]] = []
    _merge_mutual_rests(
        rels, contacts, {"INS001": "A", "INS002": "B"}, lambda *args: added.append(args)
    )
    assert rels == []
    assert len(added) == 1
    assert added[0][0] == "adjacent_to"
    assert "opposite faces" in added[0][4]
    assert "30" in added[0][4]
