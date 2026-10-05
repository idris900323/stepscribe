# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Assembly analysis: tree, BOM, contacts, joints, relations, spatial facts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from stepscribe.assembly.bom import build_bom
from stepscribe.assembly.contacts import detect_contacts
from stepscribe.assembly.interference import find_interferences
from stepscribe.assembly.joints import detect_joints, shopping_list
from stepscribe.assembly.relations import build_relations
from stepscribe.assembly.spatial import spatial_facts
from stepscribe.assembly.tree import (
    InstanceData,
    build_subassemblies,
    to_schema_instance,
    union_bounds,
)
from stepscribe.geometry.properties import vec3
from stepscribe.models.schema import AssemblyInfo

if TYPE_CHECKING:
    from stepscribe.analysis import AnalyzeOptions
    from stepscribe.api import Analysis
    from stepscribe.progress import Tracker


def build_instances(analysis: Analysis) -> list[InstanceData]:
    """One :class:`InstanceData` per placed body, with stable IDs INS001..."""
    assert analysis.model is not None
    by_part = {ap.part.id: ap for ap in analysis.parts}
    rows: list[tuple[str, str, InstanceData]] = []
    for rec in analysis.model.instances:
        part_ids = analysis.proto_to_parts.get(rec.proto_key, [])
        for pid in part_ids:
            ap = by_part[pid]
            path = rec.path if len(part_ids) == 1 else f"{rec.path}/{ap.part.name}"
            rows.append(
                (
                    path,
                    pid,
                    InstanceData("", ap, rec.matrix, path, rec.parent_path, rec.subassembly),
                )
            )
    rows.sort(key=lambda r: (r[0], r[1]))
    out = []
    for n, (_p, _pid, inst) in enumerate(rows, 1):
        inst.id = f"INS{n:03d}"
        out.append(inst)
    return out


def build_assembly(
    analysis: Analysis, opts: AnalyzeOptions, tracker: Tracker | None = None
) -> AssemblyInfo | None:
    """Assembly info for files with more than one placed body; None for a single part."""
    if analysis.model is None or not analysis.parts:
        return None
    instances = build_instances(analysis)
    if len(instances) < 2:
        return None
    gbox = union_bounds(instances)
    contacts = detect_contacts(instances, tracker)
    joints = detect_joints(instances)
    relations = build_relations(instances, contacts, joints, opts.up)
    masses = [i.ap.part.mass.mass_g for i in instances]
    known = all(m is not None for m in masses)
    total = float(sum(m for m in masses if m is not None)) if known else None
    com = None
    if known and total:
        weights = np.array([m or 0.0 for m in masses])
        cents = np.array([i.to_global_point(_centroid(i)) for i in instances])
        com = vec3((weights[:, None] * cents).sum(axis=0) / total)
    return AssemblyInfo(
        root_name=analysis.model.root_name,
        up_axis=opts.up,
        front_axis=opts.front,
        instances=[to_schema_instance(i, None) for i in instances],
        subassemblies=build_subassemblies(instances),
        bom=build_bom(instances),
        global_bbox=gbox,
        total_mass_g=total,
        center_of_mass=com,
        contacts=contacts,
        fastener_joints=joints,
        relations=relations,
        spatial_facts=spatial_facts(instances, contacts, gbox, opts.up, opts.front),
        interferences=find_interferences(instances) if opts.check_interference else [],
        fastener_shopping_list=shopping_list(joints),
    )


def _centroid(inst: InstanceData) -> np.ndarray:
    c = inst.ap.part.mass.centroid
    return np.array([c.x, c.y, c.z])
