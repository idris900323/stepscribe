# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Skill, Claude Code plugin and marketplace: format, sync, headless."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml

import stepscribe

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "stepscribe"
SKILL_FILES = [
    *sorted((ROOT / "skills").glob("*/SKILL.md")),
    *sorted((PLUGIN / "skills").glob("*/SKILL.md")),
    *sorted((ROOT / ".github" / "skills").glob("*/SKILL.md")),
]
AGENT_FILES = sorted((PLUGIN / "agents").glob("*.md"))
RESERVED = ("claude", "anthropic")


def _front(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    assert m, f"{path} has no frontmatter"
    data = yaml.safe_load(m.group(1))
    assert isinstance(data, dict)
    return data, m.group(2)


# ------------------------------------------------------------------ skills and agents
@pytest.mark.parametrize("path", SKILL_FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_skill_frontmatter_is_valid(path: Path) -> None:
    fm, body = _front(path)
    name, desc = fm["name"], fm["description"]
    assert re.fullmatch(r"[a-z0-9-]{1,64}", name), name
    assert not any(word in name for word in RESERVED)
    assert desc.strip() and len(desc) <= 1024
    assert not re.search(r"<[a-zA-Z/]", name + desc)
    assert body.strip()
    assert fm.get("license") == "AGPL-3.0-only"


def test_the_canonical_skill_has_its_references_and_script() -> None:
    base = ROOT / "skills" / "stepscribe"
    for rel in (
        "SKILL.md",
        "references/pack_format.md",
        "references/review_checklist.md",
        "references/interview_guide.md",
        "scripts/check_env.py",
    ):
        assert (base / rel).is_file(), rel


def test_agent_file() -> None:
    assert AGENT_FILES
    for path in AGENT_FILES:
        fm, body = _front(path)
        assert fm["name"] == "mechanical-reviewer" and fm["description"].strip()
        denied = {t.strip() for t in str(fm["disallowedTools"]).split(",")}
        assert {"Write", "Edit"} <= denied  # no file editing
        for section in (
            "Summary",
            "Critical issues",
            "Important",
            "Minor",
            "Questions for the designer",
            "What is done well",
        ):
            assert section in body


# ------------------------------------------------------------------ manifests
def test_plugin_and_marketplace_manifests_and_version_sync() -> None:
    plugin = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", plugin["name"])
    assert not plugin["name"].startswith(("claude-", "anthropic-"))
    assert plugin["license"] == "AGPL-3.0-only" and plugin["author"]["name"]
    assert plugin["homepage"].startswith("https://")
    (entry,) = market["plugins"]
    assert entry["name"] == plugin["name"]
    assert entry["source"].startswith("./") and ".." not in entry["source"]
    assert (ROOT / entry["source"] / ".claude-plugin" / "plugin.json").is_file()
    assert market["owner"]["name"] and market["name"]
    # one version everywhere
    assert plugin["version"] == entry["version"] == stepscribe.__version__


def test_license_ships_with_the_plugin_and_the_skill_zip() -> None:
    text = (ROOT / "LICENSE").read_bytes().replace(b"\r\n", b"\n")
    assert (PLUGIN / "LICENSE").read_bytes().replace(b"\r\n", b"\n") == text
    subprocess.run([sys.executable, str(ROOT / "scripts" / "sync_skills.py")], check=True, cwd=ROOT)
    with zipfile.ZipFile(ROOT / "dist" / "stepscribe-skill.zip") as zf:
        names = zf.namelist()
        assert "stepscribe/SKILL.md" in names and "stepscribe/LICENSE" in names
        assert zf.read("stepscribe/LICENSE").replace(b"\r\n", b"\n") == text


def test_skill_copies_are_identical() -> None:
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sync_skills.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert res.returncode == 0, res.stdout


def test_claude_cli_validates_when_available() -> None:
    claude = shutil.which("claude")
    if not claude:
        pytest.skip("the claude CLI is not installed")
    for target in (PLUGIN, ROOT):
        res = subprocess.run(
            [claude, "plugin", "validate", str(target), "--strict"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert res.returncode == 0, res.stdout + res.stderr


# ------------------------------------------------------------------ headless guarantees
BANNED = re.compile(r"web ?ui|browser|localhost|127\.0\.0\.1|webbrowser|https?://", re.I)


def test_no_ai_facing_file_mentions_the_web_ui_or_a_browser() -> None:
    folders = [ROOT / "skills", PLUGIN / "skills", PLUGIN / "agents", ROOT / ".github" / "skills"]
    checked = 0
    for folder in folders:
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix in {".md", ".py", ".json"}:
                checked += 1
                hit = BANNED.search(path.read_text(encoding="utf-8"))
                assert not hit, f"{path.relative_to(ROOT)} mentions {hit.group(0)!r}"
    assert checked >= 10
    assert not list((PLUGIN / "skills").glob("*ui*"))  # no UI command


def test_there_is_no_hook_in_the_plugin() -> None:
    assert not (PLUGIN / "hooks").exists()
    plugin = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert "hooks" not in plugin


HEADLESS_SCRIPT = r"""
import http.server, json, os, sys, time


def main():
    os.environ["STEPSCRIBE_CACHE"] = sys.argv[2]

    def boom(*a, **k):
        raise AssertionError("an HTTP server was started")

    http.server.ThreadingHTTPServer.__init__ = boom
    http.server.HTTPServer.__init__ = boom

    from stepscribe.headless import HeadlessError, set_headless
    from stepscribe.mcp import tools

    set_headless()
    started = json.loads(tools.start_analysis(sys.argv[1]))
    status = json.loads(tools.get_job_status(started["job_id"]))
    end = time.time() + 240
    while not status["done"] and time.time() < end:
        time.sleep(1)
        status = json.loads(tools.get_job_status(started["job_id"]))
    assert status["done"] and not status["error"], status
    first = tools.export_pack(started["job_id"])
    text = tools.get_questions(sys.argv[1])
    qid = text.split("[", 1)[1].split("]", 1)[0]
    tools.submit_answer(sys.argv[1], qid, "skipped")
    second = tools.export_pack(sys.argv[1])
    assert "SMALL version" in first and "SMALL version" in second

    assert "stepscribe.ui" not in sys.modules, "the UI package was imported"
    assert "webbrowser" not in sys.modules, "webbrowser was imported"

    import stepscribe.ui.server as ui_server

    try:
        ui_server.serve(open_browser=False)
    except HeadlessError:
        print("HEADLESS-OK")


if __name__ == "__main__":
    main()
"""


def test_scripted_mcp_session_never_touches_the_ui_or_a_browser(
    step_files: dict[str, Path], tmp_path: Path
) -> None:
    script = tmp_path / "session.py"
    script.write_text(HEADLESS_SCRIPT, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "STEPSCRIBE_HEADLESS"}
    res = subprocess.run(
        [sys.executable, str(script), str(step_files["two_link_arm"]), str(tmp_path / "cache")],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert "HEADLESS-OK" in res.stdout, res.stdout + res.stderr


def test_headless_flag_blocks_server_and_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    from stepscribe import headless

    monkeypatch.delenv(headless.ENV, raising=False)
    assert not headless.is_headless()
    headless.require_interactive("anything")  # allowed when run on purpose
    monkeypatch.setenv(headless.ENV, "1")
    with pytest.raises(headless.HeadlessError):
        headless.require_interactive("open a browser")


def test_building_the_mcp_server_sets_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("mcp.server.mcpserver")
    from stepscribe import headless
    from stepscribe.mcp.server import build_server

    monkeypatch.delenv(headless.ENV, raising=False)
    build_server()
    assert headless.is_headless()
    monkeypatch.delenv(headless.ENV, raising=False)


# ------------------------------------------------------------------ the .mcp.json launches
def test_mcp_config_launches_the_server() -> None:
    pytest.importorskip("mcp.client")
    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters

    cfg = json.loads((PLUGIN / ".mcp.json").read_text(encoding="utf-8"))
    server = cfg["mcpServers"]["stepscribe"]
    scripts = str(Path(sys.executable).parent)
    env = {"PATH": scripts + os.pathsep + os.environ.get("PATH", ""), "STEPSCRIBE_HEADLESS": ""}
    # the config says "stepscribe"; resolve it in this environment's scripts folder
    command = shutil.which(server["command"], path=scripts)
    assert command, f"{server['command']} is not installed next to {sys.executable}"
    params = StdioServerParameters(command=command, args=server["args"], env=env)

    async def names() -> set[str]:
        async with Client(params) as client:
            listed = await client.list_tools()
            return {t.name for t in getattr(listed, "tools", listed)}

    got = asyncio.run(asyncio.wait_for(names(), 120))
    assert {"start_analysis", "get_job_status", "export_pack", "get_overview"} <= got


# ------------------------------------------------------------------ check_env
def _load_check_env() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_env", ROOT / "skills" / "stepscribe" / "scripts" / "check_env.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_env_with_stepscribe_present(capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_check_env()
    assert mod.main() == 0
    out = capsys.readouterr().out
    assert "stepscribe: installed" in out and "OpenCASCADE (OCP): available" in out


def test_check_env_with_stepscribe_absent_never_installs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load_check_env()
    monkeypatch.setattr(mod, "_has", lambda name: False)
    monkeypatch.setattr(mod, "_version", lambda dist: None)

    def no_subprocess(*a: object, **k: object) -> None:
        raise AssertionError("check_env must not run anything")

    monkeypatch.setattr(subprocess, "run", no_subprocess)
    monkeypatch.setattr(subprocess, "Popen", no_subprocess)
    assert mod.main() == 1
    out = capsys.readouterr().out
    assert "NOT installed" in out and "Ask the user before" in out
