"""Phase 0: scaffolding, schema contract, no-network rule."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from typer.testing import CliRunner

from stepscribe import __version__
from stepscribe.cli import app
from stepscribe.geometry.occ_utils import occt_version
from stepscribe.models.schema import json_schema

PKG = Path(__file__).parent.parent / "stepscribe"
BANNED = {
    "requests",
    "httpx",
    "urllib3",
    "aiohttp",
    "socket",
    "openai",
    "anthropic",
    "google",
    "ollama",
    "http",
}


def test_ocp_imports_and_reports_version() -> None:
    import OCP  # noqa: F401

    assert occt_version().startswith(("7.", "8."))


def test_cli_version_prints_occt() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output
    assert occt_version() in result.output


def test_no_network_or_llm_imports_in_core() -> None:
    """Rule 3: nothing outside stepscribe/llm may import an LLM SDK or a network library."""
    offenders = []
    for path in PKG.rglob("*.py"):
        if {"llm", "ui"} & set(path.relative_to(PKG).parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                if n.split(".")[0] in BANNED or n.startswith("urllib.request"):
                    offenders.append(f"{path.name}: {n}")
    assert not offenders, offenders


def test_json_schema_export_is_current() -> None:
    exported = json.loads(
        (PKG.parent / "schema" / "report.schema.json").read_text(encoding="utf-8")
    )
    assert exported == json_schema(), "run: stepscribe schema -o schema/report.schema.json"


def test_schema_forbids_extra_fields() -> None:
    defs = json_schema()["$defs"]
    assert all(
        d.get("additionalProperties") is False for d in defs.values() if d.get("type") == "object"
    )


def test_ui_only_listens_on_loopback() -> None:
    """The UI is exempt from the no-network scan because it is a local server: prove it is local."""
    source = (PKG / "ui" / "server.py").read_text(encoding="utf-8")
    assert '("127.0.0.1", port)' in source
    assert "0.0.0.0" not in source and '("", ' not in source
    assert "urlopen" not in source and "requests" not in source  # no outbound calls
