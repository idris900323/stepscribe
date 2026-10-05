# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Interference check (``--check-interference``): common volume of overlapping pairs (10.6)."""

from __future__ import annotations

from typing import Any

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

from stepscribe import config
from stepscribe.assembly.relations import labels
from stepscribe.assembly.tree import InstanceData
from stepscribe.describe.phrases import fmt

MIN_VOLUME = 0.001  # mm3


def find_interferences(instances: list[InstanceData]) -> list[dict[str, Any]]:
    """Pairs whose solids overlap by more than 0.001 mm3 (broad phase: inflated AABB overlap)."""
    name = labels(instances)
    lo = np.array([i.bounds[0] for i in instances]) - config.AABB_INFLATE
    hi = np.array([i.bounds[1] for i in instances]) + config.AABB_INFLATE
    found: list[dict[str, Any]] = []
    for i in range(len(instances)):
        for j in range(i + 1, len(instances)):
            if np.any(lo[i] > hi[j]) or np.any(lo[j] > hi[i]):
                continue
            common = BRepAlgoAPI_Common(instances[i].shape, instances[j].shape)
            if not common.IsDone():
                continue
            props = GProp_GProps()
            BRepGProp.VolumeProperties_s(common.Shape(), props)
            vol = float(props.Mass())
            if vol > MIN_VOLUME:
                a, b = instances[i], instances[j]
                found.append(
                    {
                        "instance_a": a.id,
                        "instance_b": b.id,
                        "volume_mm3": vol,
                        "sentence": f"{name[a.id]} and {name[b.id]} overlap by {fmt(vol, 2)} mm³.",
                    }
                )
    return found
