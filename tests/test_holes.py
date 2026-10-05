"""Hole recognition: negatives, orientation handling, segment merging, determinism, invariance."""

from __future__ import annotations

import numpy as np
from build123d import Box, Cylinder, Pos, Rot

from stepscribe.features.hole_standards import is_likely_threaded, match_standards
from stepscribe.features.holes import (
    AxisGroup,
    BoreFace,
    build_segments,
    detect_holes,
    stack_segments,
)
from stepscribe.geometry.part_geom import PartGeom


def test_slot_is_not_a_hole(analyses) -> None:  # type: ignore[no-untyped-def]
    assert analyses("slot_and_fillets").report.parts[0].holes == []


def test_internal_fillets_are_not_holes(analyses) -> None:  # type: ignore[no-untyped-def]
    assert analyses("pocket").report.parts[0].holes == []  # R2 pocket corners are concave cylinders


def test_convex_shaft_is_not_a_hole(analyses) -> None:  # type: ignore[no-untyped-def]
    holes = analyses("boss_and_shaft").report.parts[0].holes
    assert len(holes) == 1 and holes[0].diameter_mm == 2.5  # the Ø6 shaft stub is ignored


def test_outer_cylinder_is_not_a_hole(analyses) -> None:  # type: ignore[no-untyped-def]
    assert analyses("bolt_circle").report.parts[0].holes[0].diameter_mm == 4.0
    assert (
        len(analyses("bolt_circle").report.parts[0].holes) == 6
    )  # the Ø80 outer edge is not counted


def test_tube_bore_is_a_hole(analyses) -> None:  # type: ignore[no-untyped-def]
    holes = analyses("tube").report.parts[0].holes
    assert len(holes) == 1 and holes[0].is_through


def test_rotated_plate_matches_unrotated(analyses) -> None:  # type: ignore[no-untyped-def]
    a, b = analyses("plate_4xM3").report.parts[0], analyses("rotated_plate").report.parts[0]
    assert [(h.diameter_mm, h.depth_mm, h.is_through) for h in a.holes] == [
        (h.diameter_mm, h.depth_mm, h.is_through) for h in b.holes
    ]
    assert [round(h.edge_distance_mm or 0, 3) for h in a.holes] == [
        round(h.edge_distance_mm or 0, 3) for h in b.holes
    ]


def test_reversed_face_orientation_is_handled() -> None:
    """Hole from a boolean cut has REVERSED bore faces; a mirrored copy flips orientations again."""
    plate = Box(40, 40, 8) - Pos(5, 5, 0) * Cylinder(3, 10)
    for shape in (plate, plate.mirror()):
        holes = detect_holes(PartGeom("p", shape.wrapped))
        assert len(holes) == 1 and abs(holes[0].diameter_mm - 6.0) < 1e-6


def test_hole_axis_points_into_material_from_entry() -> None:
    block = Box(20, 20, 20) - Pos(0, 0, 10 - 4) * Cylinder(2, 8)  # blind hole from the top (+Z)
    (h,) = detect_holes(PartGeom("b", block.wrapped))
    assert not h.is_through
    assert h.axis.direction.z == -1.0 and abs(h.axis.origin.z - 10.0) < 1e-6


def _bf(idx: int, span: float, t0: float = 0.0, t1: float = 10.0) -> BoreFace:
    return BoreFace(idx, "cylinder", np.zeros(3), np.array([0.0, 0.0, 1.0]), span, t0, t1, 3.0)


def test_two_half_faces_on_one_range_make_a_full_bore() -> None:
    g = AxisGroup(np.zeros(3), np.array([0.0, 0.0, 1.0]), [_bf(0, 180), _bf(1, 180)])
    stacks = stack_segments(build_segments(g))
    assert len(stacks) == 1 and stacks[0][0].span_deg >= 359


def test_half_faces_end_to_end_stay_partial() -> None:
    g = AxisGroup(np.zeros(3), np.array([0.0, 0.0, 1.0]), [_bf(0, 180, 0, 5), _bf(1, 180, 5, 10)])
    assert stack_segments(build_segments(g)) == []


def test_partial_cylinder_is_not_a_bore() -> None:
    g = AxisGroup(np.zeros(3), np.array([0.0, 0.0, 1.0]), [_bf(0, 90)])
    assert stack_segments(build_segments(g)) == []


def test_standards_confidence_and_threading() -> None:
    m = match_standards(3.4)
    assert (m[0].designation, m[0].fit, m[0].confidence) == ("M3", "clearance_normal", 1.0)
    assert not is_likely_threaded(m, 3.4)
    t = match_standards(4.2)
    assert t[0].designation == "M5" and t[0].fit == "tap_drill" and is_likely_threaded(t, 4.2)


def test_unrelated_diameter_has_no_match() -> None:
    assert match_standards(7.7) == [] or match_standards(7.7)[0].confidence <= 0.7


def test_results_invariant_under_rigid_transform() -> None:
    base = Box(50, 30, 6) - Pos(10, 5, 0) * Cylinder(2.1, 8) - Pos(-10, -5, 0) * Cylinder(2.1, 8)
    moved = Pos(13, -7, 22) * Rot(31, 47, 12) * base
    a = detect_holes(PartGeom("a", base.wrapped))
    b = detect_holes(PartGeom("b", moved.wrapped))
    assert [(h.diameter_mm, h.depth_mm, h.is_through) for h in a] == [
        (h.diameter_mm, h.depth_mm, h.is_through) for h in b
    ]
    assert [round(h.edge_distance_mm or 0, 3) for h in a] == [
        round(h.edge_distance_mm or 0, 3) for h in b
    ]
