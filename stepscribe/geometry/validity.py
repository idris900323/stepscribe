# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Validity checks: BRepCheck and solid presence."""

from __future__ import annotations

from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopoDS import TopoDS_Shape

from stepscribe.geometry.occ_utils import unique_subshapes


def is_valid_solid(shape: TopoDS_Shape) -> bool:
    """True if OCCT's analyzer accepts the shape and it contains at least one solid."""
    if not unique_subshapes(shape, TopAbs_SOLID):
        return False
    return bool(BRepCheck_Analyzer(shape).IsValid())
