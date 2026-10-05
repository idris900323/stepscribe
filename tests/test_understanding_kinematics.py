"""U3: connections, rigid links, ground, joints, DOF and topology."""

from __future__ import annotations

import numpy as np
import pytest

from stepscribe.understanding.kinematics import build_kinematics


def _axis(j):  # type: ignore[no-untyped-def]
    a = j.axis
    return (
        np.array([a.origin.x, a.origin.y, a.origin.z]),
        np.array([a.direction.x, a.direction.y, a.direction.z]),
    )


def _dist_to_axis(point, joint) -> float:  # type: ignore[no-untyped-def]
    o, d = _axis(joint)
    d = d / np.linalg.norm(d)
    v = np.asarray(point) - o
    return float(np.linalg.norm(v - d * float(np.dot(v, d))))


def _cos(joint, vec) -> float:  # type: ignore[no-untyped-def]
    _o, d = _axis(joint)
    return abs(float(np.dot(d / np.linalg.norm(d), vec)))


@pytest.mark.parametrize("name", ["two_link_arm", "two_link_arm_generic"])
def test_two_link_arm(analyses, name) -> None:  # type: ignore[no-untyped-def]
    k = build_kinematics(analyses(name))
    assert len(k.links) == 3 and k.dof == 2 and k.topology == "serial_chain"
    assert not k.floating_groups
    assert [j.kind for j in k.joints] == ["revolute", "revolute"]
    j1, j2 = k.joints
    assert _cos(j1, [1, 0, 0]) > 0.9999 and _cos(j2, [1, 0, 0]) > 0.9999
    assert _dist_to_axis((0, 0, 30), j1) < 0.05 and _dist_to_axis((0, 0, 95), j2) < 0.05
    ground = next(g for g in k.links if g.is_ground)
    assert j1.parent_link == ground.id and j2.parent_link == j1.child_link
    assert ground.ground_evidence and all(e.code and e.refs for e in j1.evidence)
    assert "flowchart" in k.mermaid and "KJ1 revolute" in k.mermaid


def test_linear_slide_is_one_prismatic_joint(analyses) -> None:  # type: ignore[no-untyped-def]
    k = build_kinematics(analyses("linear_slide"))
    assert len(k.joints) == 1 and k.joints[0].kind == "prismatic"
    assert _cos(k.joints[0], [1, 0, 0]) > 0.9999 and k.dof == 1
    assert len(k.links) == 2 and k.topology == "serial_chain"


def test_four_bar_is_a_closed_loop(analyses) -> None:  # type: ignore[no-untyped-def]
    k = build_kinematics(analyses("four_bar_linkage"))
    assert k.topology == "closed_loop" and len(k.joints) == 4 and k.dof == 1
    assert all(j.kind == "revolute" and _cos(j, [0, 0, 1]) > 0.9999 for j in k.joints)
    pivots = [(0, 0), (0, 30), (60, 30), (100, 0)]
    for x, y in pivots:
        assert any(_dist_to_axis((x, y, 8), j) < 0.05 for j in k.joints)
    assert [m.kind for m in k.mechanisms] == ["linkage_loop"]


@pytest.mark.parametrize("name", ["wheeled_base", "wheeled_base_generic"])
def test_wheeled_base(analyses, name) -> None:  # type: ignore[no-untyped-def]
    k = build_kinematics(analyses(name))
    assert len(k.joints) == 4 and k.topology == "mobile_base" and k.dof == 4
    assert all(j.kind == "revolute" and _cos(j, [0, 1, 0]) > 0.9999 for j in k.joints)
    xs = sorted(round(_axis(j)[0][0]) for j in k.joints)
    assert xs == [-40, -40, 40, 40]
    assert all(abs(_axis(j)[0][2] - 15) < 0.05 for j in k.joints)


def test_floating_part_and_planar_rest(analyses) -> None:  # type: ignore[no-untyped-def]
    k = build_kinematics(analyses("floating_part_assembly"))
    assert len(k.floating_groups) == 1 and k.topology == "static"
    ground = next(g for g in k.links if g.is_ground)
    assert len(ground.instance_ids) == 2  # the lid rests on the plate: attached, uncertainly
    assert "attached only by a planar rest" in k.description


def test_single_part_has_a_static_model(analyses) -> None:  # type: ignore[no-untyped-def]
    u = analyses("nema17_plate").report.understanding
    assert u is not None and u.kinematics.topology == "static" and len(u.kinematics.links) == 1


def test_understanding_is_attached_to_the_report(analyses) -> None:  # type: ignore[no-untyped-def]
    r = analyses("two_link_arm").report
    assert r.understanding is not None and r.understanding.kinematics.dof == 2
    assert all(p.understanding is not None for p in r.parts)
