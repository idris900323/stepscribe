"""Copy the canonical skill to every place it is shipped.

    python scripts/sync_skills.py          # write the copies and dist/stepscribe-skill.zip
    python scripts/sync_skills.py --check  # exit 1 if any copy differs (used by CI)

Copies: plugins/stepscribe/skills/stepscribe/ (Claude Code plugin) and .github/skills/stepscribe/
(GitHub Copilot). The zip (for upload where custom skills are supported) also holds the LICENSE.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "skills" / "stepscribe"
COPIES = [
    ROOT / "plugins" / "stepscribe" / "skills" / "stepscribe",
    ROOT / ".github" / "skills" / "stepscribe",
]
ZIP = ROOT / "dist" / "stepscribe-skill.zip"
SKIP = {"__pycache__"}


def files(folder: Path) -> dict[str, bytes]:
    """Relative POSIX path -> content for every file under *folder* (line endings normalised)."""
    out: dict[str, bytes] = {}
    for p in sorted(folder.rglob("*")):
        if p.is_file() and not (set(p.relative_to(folder).parts) & SKIP) and p.suffix != ".pyc":
            out[p.relative_to(folder).as_posix()] = p.read_bytes().replace(b"\r\n", b"\n")
    return out


def differences() -> list[str]:
    """Human-readable list of copies that do not match the canonical skill."""
    want = files(CANONICAL)
    problems: list[str] = []
    for copy in COPIES:
        have = files(copy) if copy.is_dir() else {}
        for name in sorted(want.keys() | have.keys()):
            if want.get(name) != have.get(name):
                problems.append(f"{copy.relative_to(ROOT).as_posix()}/{name}")
    return problems


def write_zip() -> Path:
    """``dist/stepscribe-skill.zip``: the skill folder plus LICENSE, fixed timestamps."""
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    ZIP.unlink(missing_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        entries = {f"stepscribe/{k}": v for k, v in files(CANONICAL).items()}
        entries["stepscribe/LICENSE"] = (ROOT / "LICENSE").read_bytes().replace(b"\r\n", b"\n")
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, entries[name])
    return ZIP


def sync() -> None:
    """Make every copy identical to the canonical skill and rebuild the zip."""
    for copy in COPIES:
        if copy.exists():
            shutil.rmtree(copy)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(CANONICAL, copy, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    write_zip()
    plugin_license = ROOT / "plugins" / "stepscribe" / "LICENSE"
    plugin_license.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "LICENSE", plugin_license)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only report differences")
    args = ap.parse_args(argv)
    if args.check:
        bad = differences()
        for b in bad:
            print(f"differs: {b}")
        return 1 if bad else 0
    sync()
    print(f"synced {len(COPIES)} copies and wrote {ZIP.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
