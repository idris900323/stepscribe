# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Relationship graph and plain-English sentences from contacts and joints."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable

import numpy as np

from stepscribe.assembly.spatial import axis_vector
from stepscribe.assembly.tree import InstanceData
from stepscribe.describe.phrases import fmt
from stepscribe.geometry.occ_utils import angle_between
from stepscribe.models.schema import Contact, FastenerJoint, Relation

REST_ANGLE_DEG = 30.0
CONTAINS_CONF = 0.7
CONTACT_CONF = 0.9
NEAR_MISS_CONF = 0.6
MOUNT_KINDS = ("motor_mount", "board_mount", "servo_mount")
MOUNTABLE = re.compile(r"motor|servo|board|raspberry|arduino|esp32", re.IGNORECASE)
ORDER = [
    "bolted_to",
    "mounted_on",
    "rests_on",
    "inserted_into",
    "contains",
    "adjacent_to",
    "aligned_with",
]


def labels(instances: list[InstanceData]) -> dict[str, str]:
    """Readable instance names: the leaf name, or the full path when the leaf is ambiguous."""
    counts: dict[str, int] = defaultdict(int)
    for i in instances:
        counts[i.name] += 1
    return {i.id: (i.name if counts[i.name] == 1 else i.path) for i in instances}


def _range(ids: list[str]) -> str:
    nums = sorted(int(i[1:]) for i in ids)
    runs: list[list[int]] = []
    for n in nums:
        if runs and n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    parts = [f"J{r[0]:03d}" if len(r) == 1 else f"J{r[0]:03d}–J{r[-1]:03d}" for r in runs]
    return ", ".join(parts)


def _size(joint: FastenerJoint) -> str:
    return joint.suggested_fastener.split(" ")[0] if joint.suggested_fastener else "fasteners"


def _is_mountable(inst: InstanceData) -> bool:
    p = inst.ap.part
    return p.likely_purchased_hardware and bool(MOUNTABLE.search(p.hardware_guess or p.name))


def _mount_tag(inst: InstanceData) -> tuple[str, float] | None:
    for t in inst.ap.part.semantic_tags:
        if t.kind in MOUNT_KINDS:
            return t.label, t.confidence
    return None


def build_relations(
    instances: list[InstanceData],
    contacts: list[Contact],
    joints: list[FastenerJoint],
    up: str,
) -> list[Relation]:
    """Derive relations; subject/object are instance IDs, ``sentence`` uses instance names."""
    by_id = {i.id: i for i in instances}
    name = labels(instances)
    up_vec = axis_vector(up)
    rels: list[tuple[tuple[object, ...], Relation]] = []
    bolted_pairs: set[frozenset[str]] = set()

    def centre(i: InstanceData) -> float:
        return float(np.dot(0.5 * (i.bounds[0] + i.bounds[1]), up_vec))

    def add(pred: str, subj: str, obj: str, via: list[str], sentence: str, conf: float) -> None:
        rels.append(
            (
                (ORDER.index(pred), subj, obj),
                Relation(
                    id="",
                    subject=subj,
                    predicate=pred,
                    object=obj,
                    via=via,
                    sentence=sentence,
                    confidence=round(conf, 2),
                ),
            )
        )

    # bolted_to / mounted_on from joints
    pair_joints: dict[tuple[str, str], list[FastenerJoint]] = defaultdict(list)
    for j in joints:
        for first, second in zip(j.instance_ids, j.instance_ids[1:], strict=False):
            lo_id, hi_id = sorted((first, second))
            pair_joints[(lo_id, hi_id)].append(j)
    for (id_a, id_b), js in sorted(pair_joints.items()):
        ia, ib = by_id[id_a], by_id[id_b]
        bolted_pairs.add(frozenset((id_a, id_b)))
        sizes = sorted({_size(j) for j in js})
        what = f"{len(js)} × {sizes[0]}" if len(sizes) == 1 else f"{len(js)} fasteners"
        if len(sizes) == 1 and sizes[0].startswith("M") and sizes[0][1:].replace(".", "").isdigit():
            what += " screws"
        ids = _range([j.id for j in js])
        conf = sum(j.confidence for j in js) / len(js)
        subj, obj = (
            (ia, ib)
            if centre(ia) > centre(ib) or (centre(ia) == centre(ib) and ia.id < ib.id)
            else (ib, ia)
        )
        mount: tuple[InstanceData, InstanceData, tuple[str, float] | None] | None = None
        for cand_s, cand_o in ((subj, obj), (obj, subj)):
            tag = _mount_tag(cand_o)
            if tag and (_is_mountable(cand_s) or _mount_tag(cand_s) is None):
                mount = (cand_s, cand_o, tag)
                break
            if _is_mountable(cand_s) and _mount_tag(cand_o) is None:
                mount = (cand_s, cand_o, None)
                break
        if mount is not None:
            s, o, tag = mount
            extra = f"; likely a {tag[0]} ({tag[1]:.2f})" if tag else ""
            add(
                "mounted_on",
                s.id,
                o.id,
                [j.id for j in js],
                f"{name[s.id]} is mounted on {name[o.id]} with {what} ({ids}){extra}.",
                min(conf, CONTACT_CONF),
            )
        else:
            add(
                "bolted_to",
                subj.id,
                obj.id,
                [j.id for j in js],
                f"{name[subj.id]} is bolted to {name[obj.id]} with {what} ({ids}).",
                conf,
            )

    related: set[frozenset[str]] = set(bolted_pairs)
    for c in contacts:
        ca, cb = by_id[c.instance_a], by_id[c.instance_b]
        key = frozenset((ca.id, cb.id))
        if c.kind == "planar_face" and c.normal is not None:
            n = np.array([c.normal.x, c.normal.y, c.normal.z])
            if angle_between(n, -up_vec) <= REST_ANGLE_DEG:
                add(
                    "rests_on",
                    ca.id,
                    cb.id,
                    [c.id],
                    f"{name[ca.id]} rests on {name[cb.id]} (planar contact {fmt(c.contact_area_mm2 or 0.0)} mm²).",
                    CONTACT_CONF,
                )
                related.add(key)
                continue
            if angle_between(n, up_vec) <= REST_ANGLE_DEG:
                add(
                    "rests_on",
                    cb.id,
                    ca.id,
                    [c.id],
                    f"{name[cb.id]} rests on {name[ca.id]} (planar contact {fmt(c.contact_area_mm2 or 0.0)} mm²).",
                    CONTACT_CONF,
                )
                related.add(key)
                continue
            add(
                "adjacent_to",
                ca.id,
                cb.id,
                [c.id],
                f"{name[ca.id]} touches {name[cb.id]} along a planar face ({fmt(c.contact_area_mm2 or 0.0)} mm²).",
                CONTACT_CONF,
            )
        elif c.kind == "cylindrical_fit":
            fit = {
                "clearance": f"diametral clearance {fmt(abs(c.fit_value_mm or 0.0), 2)} mm",
                "line_to_line": "line-to-line fit",
                "interference": f"diametral interference {fmt(abs(c.fit_value_mm or 0.0), 2)} mm",
            }[c.fit or "clearance"]
            add(
                "inserted_into",
                ca.id,
                cb.id,
                [c.id],
                f"{name[ca.id]} is inserted into {name[cb.id]} ({fit}).",
                CONTACT_CONF,
            )
        else:
            if key in related:
                continue
            word = {
                "near_miss": f"is almost touching ({fmt(c.min_distance_mm, 2)} mm from)",
                "line": "touches along a line",
                "point": "touches at a point",
            }
            conf = NEAR_MISS_CONF if c.kind == "near_miss" else CONTACT_CONF
            if c.kind == "near_miss":
                add(
                    "adjacent_to",
                    ca.id,
                    cb.id,
                    [c.id],
                    f"{name[ca.id]} {word['near_miss']} {name[cb.id]}.",
                    conf,
                )
            else:
                add(
                    "adjacent_to",
                    ca.id,
                    cb.id,
                    [c.id],
                    f"{name[ca.id]} {word[c.kind]} {name[cb.id]}.",
                    conf,
                )
    # contains
    for outer in instances:
        if outer.ap.part.shape_class.label != "housing":
            continue
        outer_b = outer.bounds
        for inner in instances:
            if inner is outer:
                continue
            inner_b = inner.bounds
            if np.all(inner_b[0] >= outer_b[0] - 1e-6) and np.all(inner_b[1] <= outer_b[1] + 1e-6):
                add(
                    "contains",
                    outer.id,
                    inner.id,
                    [],
                    f"{name[outer.id]} contains {name[inner.id]} (bounding box inside a housing).",
                    CONTAINS_CONF,
                )
    _merge_mutual_rests(rels, contacts, name, add)
    rels.sort(key=lambda kv: kv[0])
    out = [r for _k, r in rels]
    for idx, r in enumerate(out, 1):
        r.id = f"R{idx:03d}"
    return out


__all__ = ["build_relations", "labels", "math"]


def _merge_mutual_rests(
    rels: list[tuple[tuple[object, ...], Relation]],
    contacts: list[Contact],
    name: dict[str, str],
    add: Callable[[str, str, str, list[str], str, float], None],
) -> None:
    """Parts that touch from both sides (a plate sandwiched between two others, stacked
    alternately) would read "A rests on B" and "B rests on A"; say it once instead."""
    area = {c.id: c.contact_area_mm2 or 0.0 for c in contacts}
    rests = {(r.subject, r.object): r for _k, r in rels if r.predicate == "rests_on"}
    drop: set[int] = set()
    for (a, b), r1 in sorted(rests.items()):
        r2 = rests.get((b, a))
        if r2 is None or a > b:
            continue
        drop.update((id(r1), id(r2)))
        via = sorted({*r1.via, *r2.via})
        total = sum(area.get(v, 0.0) for v in via)
        add(
            "adjacent_to",
            a,
            b,
            via,
            f"{name[a]} and {name[b]} touch on opposite faces (planar contact {fmt(total)} mm² in total).",
            max(r1.confidence, r2.confidence),
        )
    if drop:
        rels[:] = [(k, r) for k, r in rels if id(r) not in drop]
