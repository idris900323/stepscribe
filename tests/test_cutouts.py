# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Cut-out recognizer against fixtures with known ground truth."""

from __future__ import annotations

import pytest

TOL = 1e-3


def _part(analyses, name):  # type: ignore[no-untyped-def]
    report = analyses(name).report
    assert not report.errors
    return report.parts[0]


def _sizes(level) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    return level.width_mm, level.length_mm


def test_rect_through(analyses) -> None:  # type: ignore[no-untyped-def]
    p = _part(analyses, "plate_rect_thru")
    (c,) = p.cutouts
    assert c.kind == "through" and c.total_depth_mm is None
    (lv,) = c.levels
    assert lv.shape == "rectangle" and lv.corner_radius_mm is None
    assert _sizes(lv) == pytest.approx((10.0, 20.0), abs=TOL)
    assert lv.area_mm2 == pytest.approx(200.0, abs=TOL)
    assert lv.depth_mm is None
    assert lv.center_uv == pytest.approx((60.0, 35.0), abs=TOL)
    assert c.edge_distance_mm == pytest.approx(20.0, abs=TOL)


def test_rounded_rect_through(analyses) -> None:  # type: ignore[no-untyped-def]
    (c,) = _part(analyses, "plate_rounded_rect_thru").cutouts
    (lv,) = c.levels
    assert lv.shape == "rounded_rectangle"
    assert lv.corner_radius_mm == pytest.approx(2.0, abs=TOL)
    assert _sizes(lv) == pytest.approx((10.0, 20.0), abs=TOL)
    assert lv.area_mm2 == pytest.approx(200 - (4 - 3.141592653589793) * 4, abs=TOL)


def test_rect_recess(analyses) -> None:  # type: ignore[no-untyped-def]
    p = _part(analyses, "plate_rect_recess")
    (c,) = p.cutouts
    assert c.kind == "recess"
    assert c.total_depth_mm == pytest.approx(2.0, abs=TOL)
    assert c.levels[0].depth_mm == pytest.approx(2.0, abs=TOL)
    assert c.floor_type == "flat"
    (pk,) = p.pockets  # still one feature: the pocket points at its cut-out
    assert pk.cutout_id == c.id


def test_stepped_cutout(analyses) -> None:  # type: ignore[no-untyped-def]
    (c,) = _part(analyses, "plate_stepped_cutout").cutouts
    assert c.kind == "stepped" and len(c.levels) == 2 and c.total_depth_mm is None
    a, b = c.levels
    assert _sizes(a) == pytest.approx((10.0, 18.0), abs=TOL) and a.depth_mm == pytest.approx(2.0)
    assert _sizes(b) == pytest.approx((6.0, 10.0), abs=TOL) and b.depth_mm is None
    assert a.center_uv == pytest.approx(b.center_uv, abs=TOL)
    assert a.center_uv == pytest.approx((50.0, 30.0), abs=TOL)
    assert "recess ↧2.0" in c.description and c.description.endswith("THRU")


def test_stepped_with_ring_floor(analyses) -> None:  # type: ignore[no-untyped-def]
    p = _part(analyses, "plate_stepped_offset")
    (c,) = p.cutouts  # the inner window is not reported again as its own cut-out
    assert [lv.shape for lv in c.levels] == ["rectangle", "rectangle"]
    assert _sizes(c.levels[1]) == pytest.approx((6.0, 8.0), abs=TOL)


def test_three_stepped_make_a_pattern(analyses) -> None:  # type: ignore[no-untyped-def]
    p = _part(analyses, "plate_three_stepped")
    assert len(p.cutouts) == 3 and all(c.kind == "stepped" for c in p.cutouts)
    (pat,) = p.cutout_patterns
    assert pat.kind == "linear" and pat.count == 3
    assert pat.pitch_mm == pytest.approx(30.0, abs=TOL)
    assert {c.pattern_id for c in p.cutouts} == {pat.id}
    assert "linear, pitch 30.0" in pat.description


def test_edge_notch(analyses) -> None:  # type: ignore[no-untyped-def]
    (c,) = _part(analyses, "plate_edge_notch").cutouts
    assert c.kind == "notch" and c.edge_side == "+u"
    (lv,) = c.levels
    assert lv.width_mm == pytest.approx(8.0, abs=TOL)  # along the boundary
    assert lv.length_mm == pytest.approx(5.0, abs=TOL)  # into the part


def test_l_cutout(analyses) -> None:  # type: ignore[no-untyped-def]
    (c,) = _part(analyses, "plate_L_cutout").cutouts
    (lv,) = c.levels
    assert lv.shape == "polygon" and lv.sides == 6
    assert lv.area_mm2 == pytest.approx(300.0, abs=TOL)
    assert len(lv.vertices_uv) == 6


def test_rotated_window(analyses) -> None:  # type: ignore[no-untyped-def]
    (c,) = _part(analyses, "plate_rotated_window").cutouts
    (lv,) = c.levels
    assert lv.shape == "rectangle"
    assert lv.rotation_deg == pytest.approx(30.0, abs=0.01)
    assert _sizes(lv) == pytest.approx((10.0, 20.0), abs=TOL)


def test_block_side_recess(analyses) -> None:  # type: ignore[no-untyped-def]
    p = _part(analyses, "block_side_recess")
    (c,) = p.cutouts
    assert c.kind == "recess" and c.total_depth_mm == pytest.approx(3.0, abs=TOL)
    n = c.entry_normal
    assert (n.x, n.y, n.z) == pytest.approx((1.0, 0.0, 0.0), abs=TOL)  # the +x side face
    assert _sizes(c.levels[0]) == pytest.approx((10.0, 16.0), abs=TOL)


def test_thin_web_between_cutouts(analyses) -> None:  # type: ignore[no-untyped-def]
    p = _part(analyses, "plate_close_cutouts")
    assert len(p.cutouts) == 2
    assert all(c.web_to_neighbor_mm == pytest.approx(1.2, abs=TOL) for c in p.cutouts)
    assert all(c.edge_distance_mm == pytest.approx(1.2, abs=TOL) for c in p.cutouts)


def test_holes_stay_holes(analyses) -> None:  # type: ignore[no-untyped-def]
    p = _part(analyses, "plate_cutout_and_holes")
    assert len(p.holes) == 2 and len(p.cutouts) == 1
    assert p.cutouts[0].web_to_neighbor_mm == pytest.approx(18.5, abs=TOL)


def test_hole_only_parts_have_no_cutouts(analyses) -> None:  # type: ignore[no-untyped-def]
    for name in ("plate_4xM3", "nema17_plate", "cbore", "csk", "bolt_circle", "bearing_block"):
        assert _part(analyses, name).cutouts == [], name


def test_slot_is_one_feature(analyses) -> None:  # type: ignore[no-untyped-def]
    p = _part(analyses, "slot_and_fillets")
    (s,) = p.slots
    (c,) = p.cutouts
    assert s.cutout_id == c.id
    assert c.levels[0].shape == "obround"
    assert _sizes(c.levels[0]) == pytest.approx((6.0, 20.0), abs=TOL)


def test_cutout_ids_are_deterministic(analyses) -> None:  # type: ignore[no-untyped-def]
    p = _part(analyses, "plate_three_stepped")
    assert [c.id for c in p.cutouts] == ["C001", "C002", "C003"]
    xs = [c.levels[0].center_uv[0] for c in p.cutouts]
    assert xs == sorted(xs)


def test_outputs_mention_cutouts(step_files, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from stepscribe.mcp import tools

    path = str(step_files["plate_stepped_cutout"])
    text = tools.get_part(path, "PRT001")
    assert "## Cutouts, recesses and notches" in text and "C001" in text
    assert "Pockets" not in text and "Slots" not in text  # one feature, one record
    table = tools.list_cutouts(path)
    assert "C001" in table and "stepped" in table and "THRU" in table
    assert tools.list_cutouts(str(step_files["plate_4xM3"])) == "No cut-outs found."


def test_thin_web_weak_spot(analyses) -> None:  # type: ignore[no-untyped-def]
    from stepscribe.understanding.weak_spots import cutout_web

    part = analyses("plate_close_cutouts").report.parts[0]
    spots = cutout_web(part, "PRT001")
    assert len(spots) == 1 and spots[0].category == "cutout_web"
    assert spots[0].value == pytest.approx(1.2, abs=0.01)
