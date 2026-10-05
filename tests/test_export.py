# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Portable export: layout, IDs, budgets, caps, determinism, caching."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any

import pytest

import stepscribe.exporter as exporter
from stepscribe.analysis import AnalyzeOptions
from stepscribe.describe.budget import estimate_tokens
from stepscribe.exporter import ExportResult, export
from stepscribe.exporter_text import retitle, slug, table_of_contents, uniquify_headings

OPTS = AnalyzeOptions(no_timestamp=True)


@pytest.fixture(scope="module")
def plates(step_files: dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> ExportResult:
    out = tmp_path_factory.mktemp("export_plates")
    return export(step_files["two_plates_assembly"], out, OPTS)


@pytest.fixture(scope="module")
def arm(step_files: dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> ExportResult:
    out = tmp_path_factory.mktemp("export_arm")
    return export(step_files["mini_mobile_manipulator"], out, OPTS, images=False)


def _report(result: ExportResult) -> dict[str, Any]:
    return json.loads((Path(result.folder) / "data" / "report.json").read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_layout(plates: ExportResult) -> None:
    folder = Path(plates.folder)
    name = plates.name
    for rel in (
        f"{name}_FULL.md",
        f"{name}_COMPACT.md",
        f"{name}_SMALL.md",
        "MANIFEST.md",
        "manifest.json",
        "images",
        "chat_bundle",
        "pack/00_READ_ME_FIRST.md",
        "pack/01_overview.md",
        "pack/03_parts",
        "data/report.json",
        "data/report.schema.json",
    ):
        assert (folder / rel).exists(), rel
    assert plates.zip and Path(plates.zip).is_file()
    assert not list(folder.rglob("*review*"))


def test_full_names_every_part_weak_spot_and_question(arm: ExportResult) -> None:
    text = Path(arm.full).read_text(encoding="utf-8")
    report = _report(arm)
    for part in report["parts"]:
        assert re.search(rf"^#+ {part['id']} ", text, re.M), part["id"]
    u = report["understanding"]
    for w in u["weak_spots"]:
        assert w["id"] in text, w["id"]
    for q in u["questions"]:
        assert q["id"] in text, q["id"]
    assert "Reconstruction" in text


def test_headings_are_unique(arm: ExportResult) -> None:
    text = Path(arm.full).read_text(encoding="utf-8")
    anchors: list[str] = []
    fenced = False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        m = None if fenced else re.match(r"^#{1,6} +(.*)$", line)
        if m:
            anchors.append(slug(m.group(1)))
    assert len(anchors) == len(set(anchors))


def test_image_links_resolve_and_are_described(plates: ExportResult) -> None:
    folder = Path(plates.folder)
    text = Path(plates.full).read_text(encoding="utf-8")
    links = re.findall(r"!\[[^\]]*\]\((images/[^)]+)\)", text)
    assert links
    for rel in links:
        assert (folder / rel).is_file(), rel
    assert "Isometric view of the whole assembly" in text
    assert "images/parts/PRT001_axis.png" in text


def test_zip_holds_exactly_the_folder(plates: ExportResult) -> None:
    folder = Path(plates.folder)
    with zipfile.ZipFile(plates.zip) as zf:
        names = sorted(zf.namelist())
    expected = sorted(
        f"{folder.name}/{p.relative_to(folder).as_posix()}"
        for p in folder.rglob("*")
        if p.is_file() and p.name != exporter.STAMP
    )
    assert names == expected


def test_small_stays_inside_its_budget(
    step_files: dict[str, Path], tmp_path: Path, arm: ExportResult
) -> None:
    assert estimate_tokens(Path(arm.small).read_text(encoding="utf-8")) <= 12000
    tight = export(
        step_files["mini_mobile_manipulator"],
        tmp_path,
        OPTS,
        small_budget_tokens=900,
        images=False,
    )
    tokens = estimate_tokens(Path(tight.small).read_text(encoding="utf-8"))
    assert 200 < tokens <= 900 * 1.05
    text = Path(tight.small).read_text(encoding="utf-8")
    assert "## Reconstruction" not in text and "Reconstruction (" not in text


def test_chat_bundle_respects_its_caps(step_files: dict[str, Path], tmp_path: Path) -> None:
    res = export(
        step_files["two_plates_assembly"],
        tmp_path,
        OPTS,
        chat_max_images=2,
        chat_max_mb=0.04,
        make_zip=False,
    )
    bundle = Path(res.folder) / "chat_bundle"
    total = sum(f.stat().st_size for f in bundle.iterdir())
    assert total <= 0.04 * 1_000_000
    assert len([f for f in bundle.iterdir() if f.suffix == ".png"]) <= 2
    assert res.dropped_images
    manifest = Path(res.manifest).read_text(encoding="utf-8")
    assert "Left out of the chat bundle" in manifest
    assert res.zip is None


def test_exported_markdown_is_clean(plates: ExportResult, arm: ExportResult) -> None:
    for result in (plates, arm):
        for md in Path(result.folder).glob("*.md"):
            raw = md.read_bytes()
            assert b"\r" not in raw, md.name
            text = raw.decode("utf-8")
            assert not re.search(
                r"</?[a-zA-Z][^>\n]*>", re.sub(r"```.*?```", "", text, flags=re.S)
            ), md.name


def test_full_is_deterministic_and_golden(plates: ExportResult, golden) -> None:  # type: ignore[no-untyped-def]
    golden("two_plates_assembly_FULL.md", Path(plates.full).read_text(encoding="utf-8"))


def test_second_export_does_no_geometry_work(
    step_files: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = export(step_files["two_plates_assembly"], tmp_path, OPTS, images=False)
    assert not first.cached

    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("geometry work on a cached export")

    monkeypatch.setattr(exporter, "analyze_full", boom)
    again = export(step_files["two_plates_assembly"], tmp_path, OPTS, images=False)
    assert again.cached and again.full == first.full
    assert exporter.cached_export(step_files["two_plates_assembly"], tmp_path, OPTS, images=False)
    # a different budget is a different export
    assert (
        exporter.cached_export(
            step_files["two_plates_assembly"],
            tmp_path,
            OPTS,
            images=False,
            small_budget_tokens=500,
        )
        is None
    )


def test_refuses_to_overwrite_a_foreign_folder(step_files: dict[str, Path], tmp_path: Path) -> None:
    clash = tmp_path / "two_plates_assembly_stepscribe"
    clash.mkdir()
    (clash / "mine.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(ValueError, match="not a stepscribe export"):
        export(step_files["two_plates_assembly"], tmp_path, OPTS, images=False)
    assert (clash / "mine.txt").exists()


def test_embedded_images_option(step_files: dict[str, Path], tmp_path: Path) -> None:
    res = export(step_files["two_plates_assembly"], tmp_path, OPTS, embed_images=True)
    embedded = Path(res.folder) / "two_plates_assembly_FULL_embedded.md"
    assert "data:image/png;base64," in embedded.read_text(encoding="utf-8")


def test_python_api_is_exposed() -> None:
    import stepscribe

    assert callable(stepscribe.export)


def test_markdown_helpers() -> None:
    md = "# A\n\n## Holes\n\n```\n# not a heading\n```\n\n## Holes\n"
    assert retitle(md, 3, prefix="PRT001").splitlines()[0] == "### A"
    assert "#### PRT001: Holes" in retitle(md, 3, prefix="PRT001")
    assert "# not a heading" in retitle(md, 3, prefix="PRT001")
    assert "## Holes (2)" in uniquify_headings(md)
    assert slug("Part roles & process (2)") == "part-roles--process-2"
    assert "- [Holes](#holes)" in table_of_contents(md)
