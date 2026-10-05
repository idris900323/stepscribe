"""U2: part symmetry, mirror-pair parts and manufacturing process inference."""

from __future__ import annotations

import numpy as np
import pytest
from build123d import Plane, Rot
from part_fixtures import chiral_part

from stepscribe.geometry.part_geom import PartGeom
from stepscribe.understanding.aag import build_aag
from stepscribe.understanding.process import infer_process, name_hits, rules
from stepscribe.understanding.structural import structural_features
from stepscribe.understanding.symmetry import find_mirror_pairs, part_symmetry, relation


def _sym(analyses, name):  # type: ignore[no-untyped-def]
    ap = analyses(name).parts[0]
    axes = [np.array([a.x, a.y, a.z]) for a in ap.part.obb.axes]
    return part_symmetry(ap.geom, axes)


def test_one_mirror_plane_and_none_for_chiral(analyses) -> None:  # type: ignore[no-untyped-def]
    s = _sym(analyses, "symmetric_bracket")
    assert len(s.mirror_planes) == 1 and not s.rotational
    assert abs(abs(s.mirror_planes[0]["normal"]["x"]) - 1.0) < 1e-6
    c = _sym(analyses, "chiral_part")
    assert not c.mirror_planes and not c.rotational and "no mirror" in c.description


def test_hex_flange_has_six_fold(analyses) -> None:  # type: ignore[no-untyped-def]
    s = _sym(analyses, "hex_flange")
    assert s.rotational and s.rotational[0]["fold"] == 6
    assert s.rotational[0]["axis"]["direction"]["z"] ** 2 > 0.99
    assert s.rotational[0]["max_dev_mm"] < 0.2


def test_square_plate_is_four_fold(analyses) -> None:  # type: ignore[no-untyped-def]
    assert _sym(analyses, "nema17_plate").rotational[0]["fold"] == 4


def test_mirror_pair_same_part_and_unrelated(analyses) -> None:  # type: ignore[no-untyped-def]
    geom = analyses("chiral_part").parts[0].geom
    left = geom
    shape = chiral_part()
    right = PartGeom("right", shape.mirror(Plane.YZ).wrapped)
    moved = PartGeom("moved", (Rot(0, 0, 90) * shape).wrapped)
    other = analyses("symmetric_bracket").parts[0].geom
    assert relation(left, right) == "mirror"
    assert relation(left, moved) == "same"
    assert relation(left, other) is None
    pairs = find_mirror_pairs(
        {"PRT001": left, "PRT002": right, "PRT003": other},
        {"PRT001": "a", "PRT002": "b", "PRT003": "c"},
    )
    assert pairs["PRT001"] == ("PRT002", "mirror") and "PRT003" not in pairs


def _process(analyses, name, rename=None):  # type: ignore[no-untyped-def]
    ap = analyses(name).parts[0]
    part = ap.part.model_copy(update={"name": rename}) if rename else ap.part
    aag = build_aag(ap.geom)
    feats = structural_features(aag, ap.part)
    axes = [np.array([a.x, a.y, a.z]) for a in ap.part.obb.axes]
    sym = part_symmetry(ap.geom, axes)
    return infer_process(part, aag, feats, sym)


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("laser_plate", "laser_or_waterjet"),
        ("turned_shaft", "cnc_turned"),
        ("milled_pocket_block", "cnc_milled"),
        ("printed_like_part", "fdm_3d_printed"),
        ("sheet_metal_u", "sheet_metal"),
    ],
)
def test_process_ranked_first_with_and_without_names(analyses, fixture, expected) -> None:  # type: ignore[no-untyped-def]
    assert _process(analyses, fixture).ranked[0].label == expected
    anonymous = _process(analyses, fixture, rename="Part1")
    assert anonymous.ranked[0].label == expected, [
        (h.label, h.confidence) for h in anonymous.ranked
    ]
    top = anonymous.ranked[0]
    assert 0 < top.confidence < 1 and top.evidence
    assert all(e.code and e.text for e in top.evidence)


def test_sharp_vertical_corner_is_evidence_against_milling(analyses) -> None:  # type: ignore[no-untyped-def]
    guess = _process(analyses, "printed_like_part", rename="Part1")
    assert "cnc_milled" not in [h.label for h in guess.ranked[:1]]


def test_user_override_wins(analyses) -> None:  # type: ignore[no-untyped-def]
    ap = analyses("laser_plate").parts[0]
    aag = build_aag(ap.geom)
    g = infer_process(ap.part, aag, [], part_symmetry(ap.geom), "cnc_milled")
    assert (
        g.source == "user" and g.ranked[0].label == "cnc_milled" and g.ranked[0].confidence == 1.0
    )


def test_name_keywords_use_word_boundaries() -> None:
    r = rules()
    assert name_hits("Bottom_plate", r) == {}
    assert "fdm_3d_printed" in name_hits("pla_cover", r)
    assert "fdm_3d_printed" in name_hits("PrintedMount", r)


def test_every_new_knowledge_number_has_a_source() -> None:
    r = rules()
    for row in r["min_wall_mm"].values():
        assert row["source"].startswith("http")
