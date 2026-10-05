"""U6: part roles, name-free inference, subassembly roles and the design summary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROLES = Path(__file__).parent / "fixtures" / "expected" / "roles"
NAMES = sorted(p.stem for p in ROLES.glob("*.json"))


def _expected(name: str) -> dict[str, list[str]]:
    return json.loads((ROLES / f"{name}.json").read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_expected_role_is_top_one_for_at_least_ninety_percent(analyses) -> None:  # type: ignore[no-untyped-def]
    hits = total = 0
    misses = []
    for name in NAMES:
        report = analyses(name).report
        assert not [w for w in report.warnings if "understanding" in w], report.warnings
        by_name = {p.name: p for p in report.parts}
        for part_name, accepted in _expected(name).items():
            part = by_name[part_name]
            assert part.understanding is not None
            got = part.understanding.roles[0].label
            total += 1
            if got in accepted:
                hits += 1
            else:
                misses.append((name, part_name, got, accepted))
    assert total >= 60
    assert hits / total >= 0.9, misses


def test_name_only_roles_never_exceed_the_cap(analyses) -> None:  # type: ignore[no-untyped-def]
    for name in NAMES:
        for p in analyses(name).report.parts:
            assert p.understanding is not None
            for h in p.understanding.roles:
                if h.evidence and all(e.code.startswith("name:") for e in h.evidence):
                    assert h.confidence <= 0.6 + 1e-9, (name, p.name, h.label)


def test_every_role_evidence_names_ids_and_weights(analyses) -> None:  # type: ignore[no-untyped-def]
    report = analyses("mini_mobile_manipulator").report
    assert report.assembly is not None
    ids = {i.id for i in report.assembly.instances} | {p.id for p in report.parts}
    for p in report.parts:
        assert p.understanding is not None
        top = p.understanding.roles[0]
        assert 0 < top.confidence < 1 or top.label == "unknown"
        for e in top.evidence:
            assert e.code and e.text and e.weight > 0 and e.refs
            assert all(
                r in ids or r.startswith(("J", "C", "KJ", "M", "L", "S", "P", "H")) for r in e.refs
            )


def test_mini_mobile_manipulator_end_to_end(analyses) -> None:  # type: ignore[no-untyped-def]
    u = analyses("mini_mobile_manipulator").report.understanding
    assert u is not None
    k = u.kinematics
    assert k.topology == "mobile_manipulator" and not k.floating_groups
    assert sum("wheel" in s.subassembly for s in u.subassembly_roles) == 1
    assert {h.label for s in u.subassembly_roles for h in s.roles} >= {
        "arm",
        "drivetrain",
        "chassis",
    }
    belts = [m for m in k.mechanisms if m.kind == "belt_drive"]
    assert len(belts) == 1 and abs((belts[0].ratio or 0) - 3.0) < 0.01
    assert "mobile manipulator" in u.summary and "2-DOF serial arm" in u.summary


def test_summary_golden(analyses, golden) -> None:  # type: ignore[no-untyped-def]
    u = analyses("mini_mobile_manipulator").report.understanding
    assert u is not None
    golden("mini_mobile_manipulator.summary.txt", u.summary + "\n")


def test_summary_is_deterministic_and_short(analyses) -> None:  # type: ignore[no-untyped-def]
    for name in NAMES:
        s = analyses(name).report.understanding.summary  # type: ignore[union-attr]
        assert 1 <= s.count(". ") + 1 <= 6 and s.startswith("Likely")


def test_single_part_roles(analyses) -> None:  # type: ignore[no-untyped-def]
    p = analyses("nema17_plate").report.parts[0]
    assert p.understanding is not None and p.understanding.roles
    assert p.understanding.roles[0].label in ("motor_mount", "base_plate")
    u = analyses("nema17_plate").report.understanding
    assert u is not None and u.summary.startswith("Likely a single")


@pytest.mark.parametrize("name", ["two_link_arm", "mini_mobile_manipulator"])
def test_joint_roles_summary_mentions_drive(analyses, name) -> None:  # type: ignore[no-untyped-def]
    s = analyses(name).report.understanding.summary  # type: ignore[union-attr]
    assert "KJ" in s


def test_name_hits_open_digit_letter_boundaries() -> None:
    from stepscribe.understanding.function import name_hits

    kw = {"spacer_standoff": ["standoff"], "bearing": ["608"], "electronics_plate": ["wave share"]}
    assert name_hits(["M5standoff"], kw) == {"spacer_standoff": "standoff"}
    assert name_hits(["608zz_bearing"], kw)["bearing"] == "608"
    assert name_hits(["WaveShare_Mounting_Plate_01d"], kw) == {"electronics_plate": "wave share"}
    assert name_hits(["Passive_Horn_01"], kw) == {}


def test_a_learned_role_model_adds_evidence_but_is_off_by_default(analyses) -> None:  # type: ignore[no-untyped-def]
    from stepscribe.understanding.function import assign_roles
    from stepscribe.understanding.ml_roles import load_role_models

    assert load_role_models() == {}  # nothing is installed by default
    a = analyses("two_link_arm")
    pipe = a.understanding_pipeline
    assert pipe is not None
    part = a.report.parts[0]

    def fake(p):  # type: ignore[no-untyped-def]
        return {"counterweight": 1.0}

    assign_roles(a, pipe.kin, pipe.ctx, None, {"fake": fake})  # type: ignore[arg-type]
    roles = {r.label: r for r in part.understanding.roles}  # type: ignore[union-attr]
    assert "counterweight" in roles
    assert any(e.code == "ml:fake" for e in roles["counterweight"].evidence)
    assign_roles(a, pipe.kin, pipe.ctx, None, {})  # back to rules only
    assert all(r.label != "counterweight" for r in part.understanding.roles)  # type: ignore[union-attr]
