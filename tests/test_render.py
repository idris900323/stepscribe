"""Phase 7: renders (dimensions, content, label placement, determinism, size limits)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from assembly_fixtures import Node, write_assembly
from build123d import Box, Location
from PIL import Image

from stepscribe.analysis import AnalyzeOptions
from stepscribe.api import analyze_full, build_context_pack
from stepscribe.render import (
    build_scene,
    item_labels,
    render_labelled,
    render_part_images,
)
from stepscribe.render.overlay import Label, draw_labels
from stepscribe.render.raster import render_scene
from stepscribe.render.scene import apply_explode, scene_bounds, standard_camera


def test_assembly_images_exist_have_size_and_content(step_files, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    folder = build_context_pack(
        step_files["two_plates_assembly"], tmp_path, AnalyzeOptions(no_timestamp=True), images=True
    )
    for view in ("iso", "front", "top", "exploded"):
        path = folder / "images" / f"assembly_{view}.png"
        assert path.stat().st_size < 1_000_000
        arr = np.array(Image.open(path).convert("RGB"))
        assert arr.shape == (1200, 1600, 3)
        non_white = (arr.sum(axis=2) < 740).mean()
        assert 0.02 < non_white < 0.95, f"{view} looks blank or full"
    assert "images/assembly_iso.png" in (folder / "01_overview.md").read_text(encoding="utf-8")
    report = (folder / "data" / "report.json").read_text(encoding="utf-8")
    assert "images/parts/PRT001_iso.png" in report and "images/parts/PRT001_axis.png" in report


def test_known_hole_label_is_drawn_at_projected_position(analyses, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    analysis = analyses("plate_4xM3")
    (item,) = build_scene(analysis)
    part = analysis.report.parts[0]
    cam = standard_camera("top", [item])
    rendered = render_scene([item], cam)
    hole = part.holes[0]
    origin = np.array([hole.axis.origin.x, hole.axis.origin.y, hole.axis.origin.z])
    x, y, depth = rendered.project(item.to_world(origin))
    assert rendered.visible(x, y, depth)
    drawn = draw_labels(rendered.image, [Label(hole.id, x, y)])
    before = np.array(rendered.image.convert("L")).astype(int)
    after = np.array(drawn.convert("L")).astype(int)
    xi, yi = int(round(x)), int(round(y))
    window = (slice(yi - 5, yi + 6), slice(xi - 5, xi + 6))
    assert (np.abs(after[window] - before[window]) > 100).sum() >= 10, (
        "anchor marker missing within ±5 px"
    )


def test_hidden_part_gets_no_label(tmp_path: Path) -> None:
    """A cube fully enclosed by a bigger one is invisible, so it must not be labelled."""
    path = write_assembly(
        tmp_path / "enclosed.step",
        {"Big": Box(20, 20, 20), "Small": Box(4, 4, 4)},
        Node(
            "Enclosed",
            children=[Node("Outer", "Big", Location()), Node("Inner", "Small", Location())],
        ),
    )
    items = build_scene(analyze_full(path))
    camera = standard_camera("iso", items)
    labels = item_labels(items)
    inner = next(i.key for i in items if i.name == "Inner")
    outer = next(i.key for i in items if i.name == "Outer")
    plain = np.array(render_scene(items, camera).image)
    assert np.array_equal(
        np.array(render_labelled(items, camera, labels, only={inner}).image), plain
    )
    assert not np.array_equal(
        np.array(render_labelled(items, camera, labels, only={outer}).image), plain
    )


def test_render_is_deterministic(analyses) -> None:  # type: ignore[no-untyped-def]
    items = build_scene(analyses("l_bracket"))
    cam = standard_camera("iso", items)
    a = np.array(render_scene(items, cam).image)
    b = np.array(render_scene(items, cam).image)
    assert np.array_equal(a, b)


def test_explode_moves_items_apart_and_keeps_hierarchy(analyses) -> None:  # type: ignore[no-untyped-def]
    items = build_scene(analyses("nested_assembly"))
    lo0, hi0 = scene_bounds(items)
    apply_explode(items)
    lo1, hi1 = scene_bounds(items)
    assert np.linalg.norm(hi1 - lo1) > np.linalg.norm(hi0 - lo0)
    fingers = [i for i in items if i.group == "Arm"]
    base = next(i for i in items if i.name == "Base")
    # the two fingers of the Arm group move together (same group offset), Base moves separately
    assert np.linalg.norm(fingers[0].offset - fingers[1].offset) < np.linalg.norm(
        fingers[0].offset - base.offset
    )


def test_part_images_use_hole_ids(analyses, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    analysis = analyses("bolt_circle")
    render_part_images(analysis, tmp_path, "+Z", "-Y")
    assert (tmp_path / "images" / "parts" / "PRT001_iso.png").is_file()
    assert (tmp_path / "images" / "parts" / "PRT001_axis.png").is_file()
    assert analysis.report.parts[0].images == [
        "images/parts/PRT001_iso.png",
        "images/parts/PRT001_axis.png",
    ]


def test_cli_render_writes_png(step_files, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from typer.testing import CliRunner

    from stepscribe.cli import app

    out = tmp_path / "v.png"
    r = CliRunner().invoke(
        app, ["render", str(step_files["nested_assembly"]), "--view", "exploded", "-o", str(out)]
    )
    assert r.exit_code == 0, r.output
    assert Image.open(out).size == (1600, 1200)


def test_section_reports_known_wall_area(step_files, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from stepscribe.geometry.section_report import section_file

    text = section_file(step_files["tube"], "z=10", None, tmp_path / "s.png")
    expected = np.pi * (10**2 - 7**2)
    assert f"cut area {expected:.1f} mm²" in text
    assert (tmp_path / "s.png").is_file()
    assert "does not cut" in section_file(step_files["tube"], "z=500", None, tmp_path / "n.png")


def test_dimension_lines_and_caption_are_drawn_with_true_sizes(analyses) -> None:  # type: ignore[no-untyped-def]
    from stepscribe.render.dimensions import annotate, size_caption
    from stepscribe.render.scene import scene_bounds

    items = build_scene(analyses("plate_4xM3"))
    bounds = scene_bounds(items)
    assert (
        size_caption(bounds, "+Z", "-Y") == "Overall 60.0 (width) x 60.0 (depth) x 5.0 (height) mm"
    )
    cam = standard_camera("front", items)
    plain = render_scene(items, cam)
    marked = render_scene(items, cam)
    annotate(marked, cam, bounds, size_caption(bounds, "+Z", "-Y"), lines=True)
    diff = (
        np.abs(np.array(plain.image).astype(int) - np.array(marked.image).astype(int)).sum(axis=2)
        > 60
    )
    assert diff.sum() > 500, "dimension lines/caption missing"
    ys, xs = np.nonzero(diff)
    px = plain.project_many(
        np.array([[x, y, z] for x in (-30, 30) for y in (-30, 30) for z in (-2.5, 2.5)])
    )
    assert xs.min() <= px[:, 0].min() + 3 and xs.max() >= px[:, 0].max() - 3, (
        "line should span the model width"
    )
    assert ys.max() > px[:, 1].max(), "dimension line sits below the model"


def test_part_images_have_feature_panel_and_hole_sizes(analyses, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Part images get a feature table panel on the right; hole callouts carry diameters."""
    from stepscribe.render.features import feature_marks, panel_lines

    analysis = analyses("nema17_plate")
    ap = analysis.parts[0]
    texts = [m.label for m in feature_marks(ap.part, ap.geom)]
    assert any("dia 22.0" in t for t in texts) and any("dia 3.4" in t for t in texts)
    lines = panel_lines(ap.part)
    assert any(ln.startswith("Volume") for ln in lines) and any(
        "dia 3.4 thru" in ln for ln in lines
    )


def test_reconstruction_gives_outline_and_positions(analyses) -> None:  # type: ignore[no-untyped-def]
    from stepscribe.describe.reconstruct import reconstruction

    ap = analyses("slot_and_fillets").parts[0]
    text = "\n".join(reconstruction(ap.part, ap.geom))
    assert "Outer outline" in text and "R2.00" in text and "Cut-out outline" in text
    tube = analyses("tube").parts[0]
    assert "bore dia 14.00" in "\n".join(reconstruction(tube.part, tube.geom))
