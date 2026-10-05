# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""README commands are real, and the release checks hold."""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stepscribe.cli import app

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def _readme_commands() -> list[list[str]]:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    out: list[list[str]] = []
    for block in re.findall(r"```bash\n(.*?)```", text, re.S):
        for line in block.splitlines():
            line = line.split("#", 1)[0].strip()
            if line.startswith("stepscribe "):
                out.append(shlex.split(line))
    return out


def test_readme_has_the_documented_sections() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for heading in (
        "## Use with AI tools",
        "### 1. Any AI, nothing to install on its side",
        "### 2. Claude Code plugin",
        "### 3. Agent Skill",
        "### 4. GitHub Copilot",
        "### 5. Other MCP clients",
        "## Standalone web page",
        "## License",
    ):
        assert heading in text, heading
    for link in ("COMMERCIAL.md", "CLA.md", "TRADEMARKS.md"):
        assert f"]({link})" in text


@pytest.mark.parametrize("argv", _readme_commands(), ids=lambda a: " ".join(a)[:60])
def test_readme_commands_and_options_exist(argv: list[str]) -> None:
    assert argv[0] == "stepscribe"
    command, options = argv[1], [a.split("=")[0] for a in argv[2:] if a.startswith("--")]
    res = runner.invoke(
        app, [command, "--help"], env={"NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "200"}
    )
    assert res.exit_code == 0, f"unknown command {command}"
    for opt in options:
        assert opt in re.sub(r"\[[0-9;]*m", "", res.output), f"{command} has no option {opt}"


def test_release_checks_pass() -> None:
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_release.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert res.returncode == 0, res.stdout
    bad = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_release.py"), "--tag", "v9.9.9"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert bad.returncode == 1 and "does not match" in bad.stdout


def test_release_dry_run_builds_wheel_sdist_and_skill_zip(tmp_path: Path) -> None:
    pytest.importorskip("build")
    out = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(out)],
        check=True,
        capture_output=True,
        cwd=ROOT,
        timeout=600,
    )
    subprocess.run([sys.executable, str(ROOT / "scripts" / "sync_skills.py")], check=True, cwd=ROOT)
    (out / "stepscribe-skill.zip").write_bytes(
        (ROOT / "dist" / "stepscribe-skill.zip").read_bytes()
    )
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_release.py"), "--dist", str(out)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert res.returncode == 0, res.stdout
