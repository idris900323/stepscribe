# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""File and folder discovery."""

from __future__ import annotations

from pathlib import Path

STEP_SUFFIXES = {".step", ".stp"}


def find_step_files(path: str | Path) -> list[Path]:
    """A STEP file, or all ``*.step|*.stp`` (any case) under a folder, recursively and sorted."""
    p = Path(path)
    if p.is_file():
        return [p]
    if not p.is_dir():
        return []
    return sorted(
        (f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in STEP_SUFFIXES),
        key=lambda f: str(f).lower(),
    )
