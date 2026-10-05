"""Print what is available for stepscribe: package version, OpenCASCADE, MCP. Never installs anything."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def check() -> dict[str, object]:
    """Facts about the environment, as a dict."""
    version = _version("stepscribe-cad") or _version("stepscribe")
    return {
        "python": ".".join(str(n) for n in sys.version_info[:3]),
        "python_ok": sys.version_info >= (3, 11),
        "stepscribe": version,
        "stepscribe_importable": _has("stepscribe"),
        "opencascade": _has("OCP"),
        "mcp_server": _has("mcp"),
    }


def main() -> int:
    info = check()
    print(
        f"Python {info['python']}"
        + ("" if info["python_ok"] else " (stepscribe needs 3.11 or newer)")
    )
    if info["stepscribe_importable"]:
        print(f"stepscribe: installed ({info['stepscribe'] or 'version unknown'})")
    else:
        print("stepscribe: NOT installed. Ask the user before running: pip install stepscribe")
    print("OpenCASCADE (OCP): " + ("available" if info["opencascade"] else "missing"))
    print(
        "MCP server dependencies: "
        + ("available" if info["mcp_server"] else 'missing (pip install "stepscribe[mcp]")')
    )
    return 0 if info["stepscribe_importable"] and info["opencascade"] else 1


if __name__ == "__main__":
    sys.exit(main())
