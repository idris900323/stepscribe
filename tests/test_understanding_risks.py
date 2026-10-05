"""U5: load paths, weak spots and stability."""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import pytest

from stepscribe.analysis import AnalyzeOptions
from stepscribe.api import analyze_full
from stepscribe.understanding.stability import convex_hull_2d, polygon_margin


@pytest.fixture(scope="module")
def dense(step_files):  # type: ignore[no-untyped-def]
    """Analyses with a known density (2.7 g/cm3) so masses and stability exist."""

    @cache
    def get(name: str):  # type: ignore[no-untyped-def]
        return analyze_full(step_files[name], AnalyzeOptions(density=2.7, no_timestamp=True))

    return get


def _spots(analysis, category):  # type: ignore[no-untyped-def]
    u = analysis.report.understanding
    assert u is not None
    return [w for w in u.weak_spots if w.category == category]


def test_hull_and_margin_helpers() -> None:
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5], [3, 2]], dtype=float)
    hull = convex_hull_2d(pts)
    assert len(hull) == 4
    margin, outward = polygon_margin(hull, np.array([8.0, 5.0]))
    assert abs(margin - 2.0) < 1e-9 and abs(outward[0] - 1.0) < 1e-9
    assert polygon_margin(hull, np.array([12.0, 5.0]))[0] < 0


def test_tippy_mast_has_the_expected_tipping_angle(dense) -> None:  # type: ignore[no-untyped-def]
    a = dense("tippy_mast_robot")
    s = a.report.understanding.stability  # type: ignore[union-attr]
    assert s is not None
    base, mast = 100 * 100 * 10, 20 * 20 * 500
    x = (mast * 40) / (base + mast)
    z = (base * 5 + mast * 260) / (base + mast)
    expected = math.degrees(math.atan2(50 - x, z))
    assert abs(s.tipping_angle_deg - expected) < 0.5, (s.tipping_angle_deg, expected)  # type: ignore[operator]
    assert abs(s.center_of_mass.x - x) < 0.05 and abs(s.center_of_mass.z - z) < 0.05
    assert s.tipping_direction is not None and s.tipping_direction.x > 0.99
    (w,) = _spots(a, "tipping")
    assert w.severity == "high" and w.value is not None and w.value < 8.0
    assert "mass" in (w.depends_on_assumption or "")


def test_stability_needs_masses(analyses) -> None:  # type: ignore[no-untyped-def]
    assert analyses("tippy_mast_robot").report.understanding.stability is None  # type: ignore[union-attr]


def test_cantilever_shelf_is_flagged_with_numbers(dense) -> None:  # type: ignore[no-untyped-def]
    a = dense("cantilever_bracket_assembly")
    spots = _spots(a, "cantilever")
    assert len(spots) == 1
    w = spots[0]
    assert abs((w.value or 0) - math.hypot(95, 20) / 20) < 0.05 and w.threshold == 4.0
    assert "J001" in w.refs and "gusset" in (w.suggestion or "")
    paths = a.report.understanding.load_paths  # type: ignore[union-attr]
    payload = next(p for p in paths if p.path[0] == "INS002")
    assert payload.path[-1] == "INS001" and payload.fastener_count_min == 2
    assert payload.weakest_connection == "C002"


def test_single_screw_motor_mount(dense) -> None:  # type: ignore[no-untyped-def]
    a = dense("single_screw_motor_mount")
    spots = _spots(a, "single_fastener")
    assert len(spots) == 1 and spots[0].severity == "high"
    assert "INS003" in spots[0].refs and "J001" in spots[0].refs
    motor_path = next(p for p in a.report.understanding.load_paths if p.load_source == "INS003")  # type: ignore[union-attr]
    assert motor_path.fastener_count_min == 1


def test_short_screw_numbers(dense) -> None:  # type: ignore[no-untyped-def]
    (w,) = _spots(dense("short_screw_assembly"), "short_screw")
    assert w.value == pytest.approx(10.0, abs=0.05) and w.threshold == pytest.approx(12.4, abs=0.05)
    assert "M3 x 14" in (w.suggestion or "")


def test_thin_printed_bracket(analyses) -> None:  # type: ignore[no-untyped-def]
    a = analyses("thin_printed_bracket")
    (wall,) = _spots(a, "thin_wall")
    assert wall.value == pytest.approx(0.6, abs=0.02) and wall.threshold == 0.8
    assert "assumes fdm" in (wall.depends_on_assumption or "") and "formlabs" in wall.message
    (edge,) = _spots(a, "edge_distance")
    assert edge.value == pytest.approx(1.0, abs=0.05) and edge.threshold == pytest.approx(8.0)
    bend = _spots(a, "unfilleted_bend")
    assert len(bend) == 1 and "fillet" in (bend[0].suggestion or "")
    ids = [w.id for w in a.report.understanding.weak_spots]  # type: ignore[union-attr]
    assert ids == [f"W{i:03d}" for i in range(1, len(ids) + 1)]


def test_user_process_changes_the_threshold(step_files) -> None:  # type: ignore[no-untyped-def]
    opts = AnalyzeOptions(no_timestamp=True)
    opts.extra["default_process"] = "cnc_milled"
    a = analyze_full(step_files["thin_printed_bracket"], opts)
    (edge,) = _spots(a, "edge_distance")
    assert edge.threshold == pytest.approx(6.0) and edge.depends_on_assumption is None
    assert not _spots(a, "unfilleted_bend")  # sharp corners only matter for printed brackets


def test_floating_part_is_a_weak_spot(analyses) -> None:  # type: ignore[no-untyped-def]
    (w,) = _spots(analyses("floating_part_assembly"), "floating_part")
    assert w.severity == "high" and "INS002" in w.refs


def test_plain_assemblies_have_no_false_alarms(analyses) -> None:  # type: ignore[no-untyped-def]
    for name in ("two_link_arm", "linear_slide", "wheeled_base"):
        u = analyses(name).report.understanding
        assert u is not None
        assert not [
            w
            for w in u.weak_spots
            if w.category in ("floating_part", "single_fastener", "cantilever")
        ]
