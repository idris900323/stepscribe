"""Release checks: one version everywhere, license metadata, and (with --dist) the built files.

    python scripts/check_release.py                      # versions and license metadata
    python scripts/check_release.py --tag v0.1.0         # the tag must match the version
    python scripts/check_release.py --dist dist          # wheel, sdist and skill zip must exist and be right

Exit code 1 lists every problem found.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LICENSE_ID = "AGPL-3.0-only"


def package_version() -> str:
    text = (ROOT / "stepscribe" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not m:
        raise SystemExit("no __version__ in stepscribe/__init__.py")
    return m.group(1)


def check_versions(tag: str | None) -> list[str]:
    """The package, plugin and marketplace entry carry the same version (and the tag, if given)."""
    version = package_version()
    plugin = json.loads(
        (ROOT / "plugins" / "stepscribe" / ".claude-plugin" / "plugin.json").read_text("utf-8")
    )
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text("utf-8"))
    found = {"package": version, "plugin.json": plugin["version"]}
    for entry in market["plugins"]:
        found[f"marketplace entry {entry['name']}"] = entry["version"]
    problems = [f"version mismatch: {found}"] if len(set(found.values())) > 1 else []
    if tag is not None and tag.lstrip("v") != version:
        problems.append(f"tag {tag} does not match version {version}")
    return problems


def check_license_metadata() -> list[str]:
    problems: list[str] = []
    project = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]
    if project.get("license") != LICENSE_ID:
        problems.append(f"pyproject license is {project.get('license')!r}, want {LICENSE_ID}")
    if "LICENSE" not in project.get("license-files", []):
        problems.append("pyproject license-files does not list LICENSE")
    plugin = json.loads(
        (ROOT / "plugins" / "stepscribe" / ".claude-plugin" / "plugin.json").read_text("utf-8")
    )
    if plugin.get("license") != LICENSE_ID:
        problems.append("plugin.json license is not " + LICENSE_ID)
    for path in (ROOT / "plugins" / "stepscribe" / "LICENSE", ROOT / "LICENSE"):
        if not path.is_file() or b"GNU AFFERO" not in path.read_bytes().upper():
            problems.append(f"{path.relative_to(ROOT)} is missing or not the AGPL text")
    for skill in (ROOT / "skills").glob("*/SKILL.md"):
        if f"license: {LICENSE_ID}" not in skill.read_text("utf-8"):
            problems.append(f"{skill.relative_to(ROOT)} has no license field")
    return problems


def check_dist(dist: Path) -> list[str]:
    """Wheel and sdist carry the license; the skill zip carries SKILL.md and LICENSE."""
    problems: list[str] = []
    wheels, sdists = sorted(dist.glob("*.whl")), sorted(dist.glob("*.tar.gz"))
    skill = dist / "stepscribe-skill.zip"
    if not wheels:
        problems.append("no wheel in " + str(dist))
    for w in wheels:
        with zipfile.ZipFile(w) as zf:
            names = zf.namelist()
            if not any(n.endswith("/licenses/LICENSE") or n.endswith("LICENSE") for n in names):
                problems.append(f"{w.name} has no LICENSE")
            meta = next((n for n in names if n.endswith("METADATA")), None)
            if meta is None or LICENSE_ID not in zf.read(meta).decode("utf-8"):
                problems.append(f"{w.name} metadata does not say {LICENSE_ID}")
    if not sdists:
        problems.append("no sdist in " + str(dist))
    for s in sdists:
        with tarfile.open(s) as tf:
            if not any(m.name.endswith("/LICENSE") for m in tf.getmembers()):
                problems.append(f"{s.name} has no LICENSE")
    if not skill.is_file():
        problems.append("no stepscribe-skill.zip in " + str(dist))
    else:
        with zipfile.ZipFile(skill) as zf:
            names = zf.namelist()
            for need in ("stepscribe/SKILL.md", "stepscribe/LICENSE"):
                if need not in names:
                    problems.append(f"{skill.name} lacks {need}")
    return problems


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--tag", help="release tag, e.g. v0.1.0")
    ap.add_argument("--dist", type=Path, help="folder with the built wheel, sdist and skill zip")
    args = ap.parse_args(argv)
    problems = check_versions(args.tag) + check_license_metadata()
    if args.dist is not None:
        problems += check_dist(args.dist)
    for p in problems:
        print("problem:", p)
    if not problems:
        print(f"release checks passed (version {package_version()})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
