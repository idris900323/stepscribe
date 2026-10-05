# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Deterministic JSON serialisation: floats are rounded only here (Rule 6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stepscribe import config
from stepscribe.models.schema import Report


def _round(obj: Any, nd: int) -> Any:
    if isinstance(obj, float):
        r = round(obj, nd)
        return 0.0 if r == 0 else r
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, nd) for v in obj]
    return obj


def report_to_dict(report: Report, precision: int = config.FLOAT_PRECISION) -> dict[str, Any]:
    """Plain, rounded dict of a report."""
    data: dict[str, Any] = report.model_dump(mode="json")
    rounded: dict[str, Any] = _round(data, precision)
    return rounded


def report_to_json(report: Report, precision: int = config.FLOAT_PRECISION) -> str:
    """Byte-stable JSON text for a report (same input gives identical output)."""
    return json.dumps(report_to_dict(report, precision), indent=2, ensure_ascii=False) + "\n"


def write_report(report: Report, path: str | Path, precision: int = config.FLOAT_PRECISION) -> Path:
    """Write ``report.json`` (UTF-8, LF newlines)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_to_json(report, precision), encoding="utf-8", newline="\n")
    return out
