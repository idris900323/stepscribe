"""Every fixture against its ground truth in expected/fixtures.json (compared at 1e-3 mm)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

EXPECTED = json.loads(
    (Path(__file__).parent / "fixtures" / "expected" / "fixtures.json").read_text(encoding="utf-8")
)
TOL = 1e-3


def approx(x: float) -> object:
    return pytest.approx(x, abs=TOL)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fixture(name: str, analyses) -> None:  # type: ignore[no-untyped-def]
    exp = EXPECTED[name]
    report = analyses(name).report
    assert not report.errors
    if "parts" in exp:
        assert len(report.parts) == exp["parts"]
        return
    part = report.parts[0]
    if "volume" in exp:
        assert part.mass.volume_mm3 == approx(exp["volume"])
        assert part.mass.surface_area_mm2 == approx(exp["area"])
    if "bbox" in exp:
        assert sorted((part.bbox.size.x, part.bbox.size.y, part.bbox.size.z)) == [
            approx(v) for v in sorted(exp["bbox"])
        ]
    if "bbox_obb" in exp:
        assert part.obb.size_sorted == [approx(v) for v in sorted(exp["bbox_obb"], reverse=True)]
    if "unit" in exp:
        assert report.meta.original_length_unit == exp["unit"]
    if "shape" in exp:
        assert part.shape_class.label == exp["shape"]
    if "thickness" in exp:
        assert part.shape_class.thickness_mm == approx(exp["thickness"])
    if "holes" in exp:
        assert len(part.holes) == exp["holes"]
    _check_hole(part, exp)
    _check_pattern(part, exp)
    _check_features(part, exp)


def _check_hole(part, exp) -> None:  # type: ignore[no-untyped-def]
    if "hole" in exp:
        h = part.holes[0]
        e = exp["hole"]
        assert h.diameter_mm == approx(e["d"])
        assert h.is_through is e["through"]
        assert h.edge_distance_mm == approx(e["edge"])
        top = h.standard_matches[0]
        assert (top.designation, top.fit) == (e["standard"], e["fit"])
    if "d" in exp and part.holes:
        assert min(h.diameter_mm for h in part.holes) == approx(exp["d"])
    if "depths" in exp:
        assert sorted(h.depth_mm for h in part.holes) == [approx(d) for d in exp["depths"]]
        assert sorted(h.bottom_type for h in part.holes) == sorted(exp["bottoms"])
        assert all(h.likely_threaded for h in part.holes) is exp["threaded"]
    if "cbore_d" in exp:
        h = part.holes[0]
        assert h.entry_type == "counterbore"
        assert h.counterbore_diameter_mm == approx(exp["cbore_d"])
        if "cbore_depth" in exp:
            assert h.counterbore_depth_mm == approx(exp["cbore_depth"])
    if "standard" in exp and "hole" not in exp:
        assert part.holes[0].standard_matches[0].designation == exp["standard"]
        assert part.holes[0].standard_matches[0].confidence > 0.99  # includes the counterbore boost
    if "csk_d" in exp:
        h = part.holes[0]
        assert h.entry_type == "countersink"
        assert h.countersink_diameter_mm == approx(exp["csk_d"])
        assert h.countersink_angle_deg == approx(exp["csk_angle"])


def _check_pattern(part, exp) -> None:  # type: ignore[no-untyped-def]
    if "pattern" not in exp:
        return
    e = exp["pattern"]
    pat = next(p for p in part.hole_patterns if p.count == e["count"])
    assert pat.kind == e["kind"]
    if "pitch" in e and isinstance(e["pitch"], list):
        assert sorted((pat.col_pitch_mm, pat.row_pitch_mm)) == [
            approx(v) for v in sorted(e["pitch"])
        ]
    elif "pitch" in e:
        assert pat.pitch_mm == approx(e["pitch"])
    if "pcd" in e:
        assert pat.pitch_circle_diameter_mm == approx(e["pcd"])
        assert pat.angular_pitch_deg == pytest.approx(e["pitch_deg"], abs=0.1)


def _check_features(part, exp) -> None:  # type: ignore[no-untyped-def]
    if "bosses" in exp:
        assert len(part.bosses) == exp["bosses"]
        assert part.bosses[0].diameter_mm == approx(exp["boss_d"])
        assert part.bosses[0].has_hole_id == part.holes[0].id
    if "slots" in exp:
        assert len(part.slots) == exp["slots"]
        assert part.slots[0].width_mm == approx(exp["slot_width"])
        assert part.slots[0].length_mm == approx(exp["slot_length"])
        assert len(part.fillets) == exp["fillets"]
        assert all(f.radius_mm == approx(exp["fillet_r"]) for f in part.fillets)
    if "pockets" in exp:
        assert len(part.pockets) == exp["pockets"]
        assert part.pockets[0].depth_mm == approx(exp["pocket_depth"])
        assert part.pockets[0].corner_radius_mm == approx(exp["pocket_corner"])
    if "chamfers" in exp:
        assert len(part.chamfers) == exp["chamfers"]
        assert all(c.distance_mm == approx(exp["chamfer_distance"]) for c in part.chamfers)
    if "min_wall" in exp:
        assert part.min_wall_thickness_mm == pytest.approx(exp["min_wall"], abs=0.05)
