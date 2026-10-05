"""Shared fixtures: generated STEP files, cached analyses and golden-file helpers."""

from __future__ import annotations

import sys
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "fixtures"))
sys.path.insert(0, str(ROOT / "fixtures" / "understanding"))

from assembly_fixtures import generate_assemblies  # noqa: E402
from assembly_fixtures_u import generate_kinematic  # noqa: E402
from cutout_fixtures import generate_cutouts  # noqa: E402
from generate_fixtures import generate_all  # noqa: E402
from mechanism_fixtures import generate_mechanisms  # noqa: E402
from part_fixtures import generate_parts  # noqa: E402
from risk_fixtures import generate_risks  # noqa: E402

from stepscribe.analysis import AnalyzeOptions  # noqa: E402
from stepscribe.api import Analysis, analyze_full  # noqa: E402

GOLDEN_DIR = ROOT / "golden"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--update-golden", action="store_true", help="rewrite golden files")


@pytest.fixture(autouse=True)
def _not_headless():  # type: ignore[no-untyped-def]
    """Building the MCP server marks the process headless; keep that from leaking between tests."""
    import os

    os.environ.pop("STEPSCRIBE_HEADLESS", None)
    yield
    os.environ.pop("STEPSCRIBE_HEADLESS", None)


@pytest.fixture(scope="session")
def step_files() -> dict[str, Path]:
    """Generate every STEP fixture once per session."""
    paths = generate_all()
    paths.update(generate_assemblies())
    paths.update(generate_parts())
    paths.update(generate_cutouts())
    paths.update(generate_kinematic())
    paths.update(generate_mechanisms())
    paths.update(generate_risks())
    return paths


@pytest.fixture(scope="session")
def analyses(step_files: dict[str, Path]):  # type: ignore[no-untyped-def]
    """Cached ``analyze_full`` results keyed by fixture name."""

    @cache
    def get(name: str) -> Analysis:
        return analyze_full(step_files[name], AnalyzeOptions(no_timestamp=True))

    return get


@pytest.fixture
def golden(request: pytest.FixtureRequest):  # type: ignore[no-untyped-def]
    """Compare text against ``tests/golden/<name>``; ``--update-golden`` rewrites it."""
    update = request.config.getoption("--update-golden")

    def check(name: str, text: str) -> None:
        path = GOLDEN_DIR / name
        if update or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
            return
        assert text == path.read_text(encoding="utf-8"), (
            f"{name} differs from golden; rerun with --update-golden"
        )

    return check


RENDER_FILES = {
    "test_render.py",
    "test_no_clipping.py",
    "test_export.py",
    "test_understanding_pack.py",
}
RENDER_TESTS = {
    "test_image_tools_return_small_pngs_and_panel_text",
    "test_section_and_render_return_png",
    "test_page_and_upload_flow",
    "test_interview_over_http",
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip the image tests on machines where off-screen OpenGL does not work (bare CI VMs)."""
    from stepscribe.render.raster import rendering_available

    if rendering_available():
        return
    skip = pytest.mark.skip(reason="off-screen rendering is not available on this machine")
    for item in items:
        if Path(str(item.fspath)).name in RENDER_FILES or item.name in RENDER_TESTS:
            item.add_marker(skip)
