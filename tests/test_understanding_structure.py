"""U1: attributed adjacency graph and structural features."""

from __future__ import annotations

from stepscribe.understanding.aag import build_aag
from stepscribe.understanding.structural import base_body, structural_features


def _feats(analyses, name):  # type: ignore[no-untyped-def]
    ap = analyses(name).parts[0]
    aag = build_aag(ap.geom)
    return ap, aag, structural_features(aag, ap.part)


def test_edge_classification_on_cube_l_block_and_fillets(analyses) -> None:  # type: ignore[no-untyped-def]
    cube = build_aag(analyses("cube_10").parts[0].geom)
    assert cube.counts == {"convex": 12, "concave": 0, "smooth": 0}
    lb = build_aag(analyses("l_bracket").parts[0].geom)
    assert lb.counts["concave"] == 1 and lb.counts["convex"] >= 12
    rounded = build_aag(analyses("slot_and_fillets").parts[0].geom)
    assert rounded.counts["smooth"] == 12  # 4 corner fillets x 2 tangent edges + 2 slot ends x 2
    dihedrals = {round(e.dihedral_deg) for e in cube.edges}
    assert dihedrals == {90}


def test_connected_components_and_neighbors(analyses) -> None:  # type: ignore[no-untyped-def]
    aag = build_aag(analyses("cube_10").parts[0].geom)
    assert len(aag.connected_components()) == 1
    assert all(len(aag.neighbors(i)) == 4 for i in range(6))
    assert aag.connected_components(edge_ok=lambda e: e.kind == "smooth") != [[0]]


def test_ribbed_plate_has_three_ribs_and_no_steps(analyses) -> None:  # type: ignore[no-untyped-def]
    _ap, _aag, feats = _feats(analyses, "ribbed_plate")
    ribs = [f for f in feats if f.kind == "rib"]
    assert len(ribs) == 3 and not [f for f in feats if f.kind == "step"]
    for r in ribs:
        assert abs((r.thickness_mm or 0) - 2.0) < 0.01
        assert abs((r.height_mm or 0) - 12.0) < 0.01
        assert abs((r.length_mm or 0) - 50.0) < 0.01


def test_gusseted_bracket_has_one_gusset(analyses) -> None:  # type: ignore[no-untyped-def]
    _ap, _aag, feats = _feats(analyses, "gusseted_l_bracket")
    g = [f for f in feats if f.kind == "gusset"]
    assert len(g) == 1 and abs((g[0].angle_deg or 0) - 90.0) < 0.5
    assert abs((g[0].thickness_mm or 0) - 3.0) < 0.01
    assert not [f for f in feats if f.kind == "step"]


def test_stepped_block_has_two_steps(analyses) -> None:  # type: ignore[no-untyped-def]
    ap, aag, feats = _feats(analyses, "stepped_block")
    steps = [f for f in feats if f.kind == "step"]
    assert len(steps) == 2 and all(abs((s.height_mm or 0) - 6.0) < 0.01 for s in steps)
    assert base_body(ap.part, aag).startswith("block")


def test_sheet_metal_u_has_two_bends_and_three_flanges(analyses) -> None:  # type: ignore[no-untyped-def]
    _ap, _aag, feats = _feats(analyses, "sheet_metal_u")
    bends = [f for f in feats if f.kind == "bend"]
    assert len(bends) == 2
    for b in bends:
        assert abs((b.angle_deg or 0) - 90.0) < 0.5 and abs((b.radius_mm or 0) - 3.0) < 0.01
        assert abs((b.thickness_mm or 0) - 2.0) < 0.05
    assert len([f for f in feats if f.kind == "flange"]) == 3


def test_lightened_plate_counts_cutouts_not_screw_holes(analyses) -> None:  # type: ignore[no-untyped-def]
    _ap, _aag, feats = _feats(analyses, "lightened_plate")
    cut = [f for f in feats if f.kind == "lightening_cutout"]
    assert len(cut) == 4 and "280" in cut[0].description


def test_every_feature_traces_to_face_ids(analyses) -> None:  # type: ignore[no-untyped-def]
    for name in ("ribbed_plate", "sheet_metal_u", "lightened_plate", "stepped_block"):
        ap, _aag, feats = _feats(analyses, name)
        ids = {f"F{i + 1:04d}" for i in range(ap.part.topology.faces)}
        for f in feats:
            assert f.face_ids and set(f.face_ids) <= ids and 0 < f.confidence <= 1
