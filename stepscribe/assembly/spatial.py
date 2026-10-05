# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Spatial facts: above / below / left / right / front / behind with distances."""

from __future__ import annotations

import numpy as np

from stepscribe.assembly.tree import InstanceData
from stepscribe.describe.phrases import fmt
from stepscribe.geometry.occ_utils import Vec
from stepscribe.models.schema import BBox, Contact, SpatialFact

MAX_FACTS_PER_INSTANCE = 5
NEIGHBOURS_PER_INSTANCE = 3
CENTRE_TOL = 1.0  # mm
AXES = {"X": 0, "Y": 1, "Z": 2}


def axis_vector(label: str) -> Vec:
    """'+Z' -> unit vector."""
    v = np.zeros(3)
    v[AXES[label[1]]] = 1.0 if label[0] == "+" else -1.0
    return v


def frame(up: str, front: str) -> tuple[Vec, Vec, Vec]:
    """(up, right, front) unit vectors. Right is seen by a viewer facing the model's front."""
    u, f = axis_vector(up), axis_vector(front)
    right = np.cross(-f, u)
    return u, right, f


def _words(axis: int, sign: float) -> str:
    return {
        (0, 1): "above",
        (0, -1): "below",
        (1, 1): "right_of",
        (1, -1): "left_of",
        (2, 1): "in_front_of",
        (2, -1): "behind",
    }[(axis, 1 if sign > 0 else -1)]


_PHRASE = {
    "above": "above",
    "below": "below",
    "right_of": "to the right of",
    "left_of": "to the left of",
    "in_front_of": "in front of",
    "behind": "behind",
}


def _gap(a_lo: Vec, a_hi: Vec, b_lo: Vec, b_hi: Vec, direction: Vec) -> float:
    """Gap between boxes along a unit axis direction (0 if they overlap along it)."""
    idx = int(np.argmax(np.abs(direction)))
    if direction[idx] > 0:
        return max(0.0, float(b_lo[idx] - a_hi[idx]))
    return max(0.0, float(a_lo[idx] - b_hi[idx]))


def _fact(
    a: InstanceData,
    b_name: str,
    b_lo: Vec,
    b_hi: Vec,
    ref_centre: Vec,
    frame_: tuple[Vec, Vec, Vec],
    b_id: str,
) -> SpatialFact | None:
    """Fact about A relative to B along the dominant axis of the centre offset."""
    lo, hi = a.bounds
    centre = 0.5 * (lo + hi)
    if (
        np.all(lo >= b_lo - 1e-6)
        and np.all(hi <= b_hi + 1e-6)
        and not np.allclose(centre, ref_centre)
    ):
        return SpatialFact(
            subject=a.id,
            relation="inside",
            object=b_id,
            distance_mm=None,
            sentence=f"{a.name} is inside the bounding box of {b_name}.",
        )
    off = centre - ref_centre
    comps = [float(np.dot(off, v)) for v in frame_]
    k = int(np.argmax(np.abs(comps)))
    if abs(comps[k]) < CENTRE_TOL:
        return SpatialFact(
            subject=a.id,
            relation="centered_on",
            object=b_id,
            distance_mm=None,
            sentence=f"{a.name} is centred on {b_name}.",
        )
    sign = comps[k]
    word = _words(k, sign)
    direction = frame_[k] * (1 if sign > 0 else -1)
    gap = _gap(b_lo, b_hi, lo, hi, direction)
    return SpatialFact(
        subject=a.id,
        relation=word,
        object=b_id,
        distance_mm=gap,
        sentence=f"{a.name} is {_PHRASE[word]} {b_name}, {fmt(gap)} mm between their bounding boxes.",
    )


def spatial_facts(
    instances: list[InstanceData],
    contacts: list[Contact],
    global_bbox: BBox,
    up: str,
    front: str,
) -> list[SpatialFact]:
    """Per-instance facts vs the assembly centre and vs up to 3 contacted neighbours."""
    if len(instances) < 2:
        return []
    fr = frame(up, front)
    g_lo = np.array([global_bbox.min.x, global_bbox.min.y, global_bbox.min.z])
    g_hi = np.array([global_bbox.max.x, global_bbox.max.y, global_bbox.max.z])
    centre = 0.5 * (g_lo + g_hi)
    by_id = {i.id: i for i in instances}
    neighbours: dict[str, list[str]] = {i.id: [] for i in instances}
    for c in sorted(contacts, key=lambda c: (c.min_distance_mm, c.id)):
        neighbours[c.instance_a].append(c.instance_b)
        neighbours[c.instance_b].append(c.instance_a)
    facts: list[SpatialFact] = []
    for inst in instances:
        lo, hi = inst.bounds
        mine: list[SpatialFact] = []
        off = 0.5 * (lo + hi) - centre
        comps = [float(np.dot(off, v)) for v in fr]
        k = int(np.argmax(np.abs(comps)))
        if abs(comps[k]) >= CENTRE_TOL:
            word = _words(k, comps[k])
            mine.append(
                SpatialFact(
                    subject=inst.id,
                    relation=word,
                    object="assembly",
                    distance_mm=abs(comps[k]),
                    sentence=f"{inst.name} is {_PHRASE[word]} the assembly centre by {fmt(abs(comps[k]))} mm.",
                )
            )
        seen: set[str] = set()
        for nid in neighbours[inst.id]:
            if (
                nid in seen
                or len(mine) >= MAX_FACTS_PER_INSTANCE
                or len(seen) >= NEIGHBOURS_PER_INSTANCE
            ):
                continue
            seen.add(nid)
            other = by_id[nid]
            f = _fact(
                inst,
                other.name,
                other.bounds[0],
                other.bounds[1],
                0.5 * (other.bounds[0] + other.bounds[1]),
                fr,
                other.id,
            )
            if f is not None:
                mine.append(f)
        facts.extend(mine)
    return facts
