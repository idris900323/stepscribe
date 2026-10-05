# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""License hygiene: canonical AGPL text, SPDX headers, no personal-use wording, build metadata."""

from __future__ import annotations

import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return [ROOT / f for f in out if (ROOT / f).is_file()]


def test_license_is_canonical_agpl() -> None:
    text = (ROOT / "LICENSE").read_bytes().decode("utf-8")
    assert b"\r" not in (ROOT / "LICENSE").read_bytes()
    assert text.lstrip().startswith("GNU AFFERO GENERAL PUBLIC LICENSE")
    assert "Version 3, 19 November 2007" in text
    assert "13. Remote Network Interaction; Use with the GNU General Public License." in text
    assert "END OF TERMS AND CONDITIONS" in text
    assert 33000 < len(text) < 36000


def test_spdx_headers_everywhere() -> None:
    missing = [
        str(p.relative_to(ROOT))
        for p in (ROOT / "stepscribe").rglob("*.py")
        if "SPDX-License-Identifier: AGPL-3.0-only" not in p.read_text(encoding="utf-8")[:300]
    ]
    assert not missing, missing


def test_no_personal_use_wording() -> None:
    pattern = re.compile(
        r"personal use license|non-commercial use only|personal, non-commercial", re.I
    )
    hits = []
    for p in _tracked_files():
        if p.suffix in {".png", ".step", ".stp", ".zip"} or p.name == "test_license.py":
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if pattern.search(text):
            hits.append(str(p.relative_to(ROOT)))
    assert not hits, hits


def test_legal_docs_exist() -> None:
    for name in ("COMMERCIAL.md", "CLA.md", "TRADEMARKS.md", "CONTRIBUTING.md"):
        assert (ROOT / name).is_file(), name
    assert "AGPL" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Draft: to be reviewed by a lawyer" in (ROOT / "CLA.md").read_text(encoding="utf-8")


def test_gitattributes_forces_lf() -> None:
    text = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "* text=auto eol=lf" in text


def test_build_metadata_and_no_step_files(tmp_path: Path) -> None:
    pytest.importorskip("build")
    res = subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(tmp_path), str(ROOT)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    wheel = next(tmp_path.glob("*.whl"))
    sdist = next(tmp_path.glob("*.tar.gz"))
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        meta = next(n for n in names if n.endswith("METADATA"))
        metadata = z.read(meta).decode("utf-8")
        assert any(n.endswith("licenses/LICENSE") for n in names)
    assert re.search(r"^License-Expression: AGPL-3\.0-only$", metadata, re.M)
    with tarfile.open(sdist) as t:
        sd_names = t.getnames()
    assert any(n.endswith("/LICENSE") for n in sd_names)
    for n in names + sd_names:
        assert not n.lower().endswith((".step", ".stp")), n
        assert "robot_steps" not in n, n
