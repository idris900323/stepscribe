"""U4: direct drive, gear pairs, belts, lead screws (and their evidence)."""

from __future__ import annotations

import math

import pytest

from stepscribe.understanding.mechanisms import belt_length


def _mechs(analyses, name, kind):  # type: ignore[no-untyped-def]
    u = analyses(name).report.understanding
    assert u is not None
    return [m for m in u.kinematics.mechanisms if m.kind == kind], u.kinematics


def test_gear_pair_3_to_1(analyses) -> None:  # type: ignore[no-untyped-def]
    (m,), k = _mechs(analyses, "gear_pair_3to1", "gear_pair")
    assert abs((m.ratio or 0) - 3.0) < 0.01
    assert m.parameters["module_mm"] == 1.5
    assert {m.parameters["teeth_driver"], m.parameters["teeth_driven"]} == {16, 48}
    assert abs(m.parameters["centre_distance_mm"] - 48.0) < 0.05
    codes = {e.code for e in m.evidence}
    assert {"gear:teeth", "gear:module", "gear:centre_distance", "assumed_driver"} <= codes
    out = next(j for j in k.joints if j.id == m.output_joint_id)
    assert out.driven_by == m.id and k.dof == 1  # two gears turning together: one freedom


def test_belt_drive_gt2(analyses) -> None:  # type: ignore[no-untyped-def]
    (m,), _k = _mechs(analyses, "belt_drive_gt2", "belt_drive")
    assert abs((m.ratio or 0) - 3.0) < 0.01 and m.parameters["profile"] == "GT2-2mm"
    expected = belt_length(20 * 2 / math.pi, 60 * 2 / math.pi, 60.0)
    assert abs(m.parameters["belt_length_mm"] - expected) < 1.0
    assert m.parameters["belt_teeth"] == math.ceil(expected / 2.0)


def test_direct_drive_sets_driven_by(analyses) -> None:  # type: ignore[no-untyped-def]
    (m,), k = _mechs(analyses, "direct_drive_joint", "direct_drive")
    assert m.ratio == 1.0 and m.output_joint_id == "KJ1" and m.input_instance_id
    assert k.joints[0].driven_by == m.id and k.joints[0].kind == "revolute" and k.dof == 1
    motor = analyses("direct_drive_joint").report.parts
    assert any("stepper motor" in (p.hardware_guess or "") for p in motor)


def test_lead_screw_drives_prismatic_joint(analyses) -> None:  # type: ignore[no-untyped-def]
    (m,), k = _mechs(analyses, "leadscrew_axis", "lead_screw")
    prism = next(j for j in k.joints if j.kind == "prismatic")
    assert m.output_joint_id == prism.id and prism.driven_by == m.id
    assert m.parameters["lead_mm"] == 8.0 and m.confidence >= 0.8
    assert k.dof == 1 and "closed_loop" not in k.topology  # the nut interface is not a joint


def test_arm_with_belt_sets_driven_by_through_the_belt(analyses) -> None:  # type: ignore[no-untyped-def]
    belts, k = _mechs(analyses, "arm_with_belt", "belt_drive")
    assert len(belts) == 1 and abs((belts[0].ratio or 0) - 3.0) < 0.01
    shoulder = next(j for j in k.joints if j.id == belts[0].output_joint_id)
    assert shoulder.driven_by == belts[0].id and belts[0].input_instance_id
    assert k.dof == 2 and not k.floating_groups
    assert all(j.kind == "revolute" for j in k.joints)


@pytest.mark.parametrize("name", ["two_link_arm", "wheeled_base", "four_bar_linkage"])
def test_no_false_mechanisms_on_plain_assemblies(analyses, name) -> None:  # type: ignore[no-untyped-def]
    u = analyses(name).report.understanding
    assert u is not None
    assert all(m.kind == "linkage_loop" for m in u.kinematics.mechanisms)


def test_mechanism_evidence_refers_to_real_ids(analyses) -> None:  # type: ignore[no-untyped-def]
    for name in ("gear_pair_3to1", "belt_drive_gt2", "leadscrew_axis", "arm_with_belt"):
        r = analyses(name).report
        assert r.understanding is not None and r.assembly is not None
        ids = {i.id for i in r.assembly.instances} | {
            j.id for j in r.understanding.kinematics.joints
        }
        for m in r.understanding.kinematics.mechanisms:
            for e in m.evidence:
                assert e.refs and set(e.refs) <= ids, (name, e)


def test_servo_horn_is_a_revolute_joint(analyses) -> None:  # type: ignore[no-untyped-def]
    a = analyses("servo_horn_arm")
    u = a.report.understanding
    assert u is not None
    servo = next(p for p in a.report.parts if p.name == "STS3215")
    assert servo.likely_purchased_hardware and "servo" in (servo.hardware_guess or "")
    assert len(u.kinematics.joints) == 1
    (j,) = u.kinematics.joints
    assert j.kind == "revolute" and abs(abs(j.axis.direction.z) - 1.0) < 1e-6
    assert any(e.code == "servo:horn" for e in j.evidence)


def test_servo_horn_joint_is_driven_by_its_servo(analyses) -> None:  # type: ignore[no-untyped-def]
    a = analyses("servo_horn_arm")
    u = a.report.understanding
    assert u is not None
    (j,) = u.kinematics.joints
    mech = next(m for m in u.kinematics.mechanisms if m.output_joint_id == j.id)
    assert mech.kind == "direct_drive" and j.driven_by == mech.id
    assert not any(q.kind == "joint_drive" for q in u.questions)
