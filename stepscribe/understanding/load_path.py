# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Load paths: how heavy parts reach the ground.

The support graph has instances as nodes and connections (fastener joints, press fits, bearings,
planar rests) as edges, each with a strength proxy. A widest-path search from the ground link
finds, for every heavy part or motor, the route whose weakest connection is strongest.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from stepscribe.assembly.tree import InstanceData
from stepscribe.describe.phrases import fmt
from stepscribe.features.hole_standards import load_table
from stepscribe.models.schema import AssemblyInfo, KinematicModel, LoadPath
from stepscribe.understanding.kinematics import KinContext, hardware_kind


@dataclass
class Edge:
    """A connection between two instances."""

    a: str
    b: str
    strength: float
    ref: str  # fastener joint ID, contact ID or "INSa-INSb"
    kind: str  # fastened | joint | press | rest
    fasteners: int = 0
    diameter: float = 0.0
    joint_ids: list[str] = field(default_factory=list)

    def other(self, x: str) -> str:
        return self.b if x == self.a else self.a


@dataclass
class LoadInfo:
    """Load paths plus the data weak-spot rules need."""

    paths: list[LoadPath]
    edges: dict[str, Edge]  # ref -> edge, only edges on some path
    through: dict[str, list[str]]  # ref -> load sources whose path crosses that edge
    sources: list[str]
    total_mass: float | None
    ground: set[str] = field(default_factory=set)


def floor_instances(ctx: KinContext, model: KinematicModel, asm: AssemblyInfo) -> set[str]:
    """Instances that stand on the lowest plane (feet, wheels, a base plate): the load's destination.

    If nothing touches the lowest plane within tolerance (a wall-mounted design), the largest
    part of the ground link stands in for the floor.
    """
    from stepscribe.understanding.kinematics import axis_vector

    up = axis_vector(asm.up_axis)
    tol = float(load_table("weak_spot_rules.yaml")["stability"]["support_tolerance_mm"])

    def low(i: InstanceData) -> float:
        lo, hi = i.bounds
        corners = np.array(
            [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        )
        return float((corners @ up).min())

    floor = min(low(i) for i in ctx.insts)
    ids = {i.id for i in ctx.insts if low(i) <= floor + tol}
    if ids:
        return ids
    ground = next((g for g in model.links if g.is_ground), None)
    if ground is None:
        return set()
    biggest = max(
        (ctx.by_id[i] for i in ground.instance_ids), key=lambda m: m.ap.part.mass.volume_mm3
    )
    return {biggest.id}


def build_edges(asm: AssemblyInfo, ctx: KinContext) -> list[Edge]:
    cfg = load_table("weak_spot_rules.yaml")["load_path"]["strength"]
    by_pair: dict[tuple[str, str], Edge] = {}

    def add(a: str, b: str, e: Edge) -> None:
        key = tuple(sorted((a, b)))
        cur = by_pair.get(key)  # type: ignore[arg-type]
        if cur is None:
            by_pair[key] = e  # type: ignore[index]
        else:
            cur.strength += e.strength
            cur.fasteners += e.fasteners
            cur.joint_ids += e.joint_ids
            cur.kind = "fastened" if e.kind == "fastened" or cur.kind == "fastened" else cur.kind
            cur.diameter = max(cur.diameter, e.diameter)

    for j in asm.fastener_joints:
        ids = list(j.instance_ids) + (
            [j.existing_fastener_instance_id] if j.existing_fastener_instance_id else []
        )
        s = float(cfg["fastener_per_unit_d2"]) * (j.common_diameter_mm / 3.0) ** 2
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = sorted((ids[x], ids[y]))
                add(a, b, Edge(a, b, s, j.id, "fastened", 1, j.common_diameter_mm, [j.id]))
    for c in ctx.conns:
        a, b = sorted((c.a, c.b))
        ref = (ctx.contacts_of.get((a, b)) or [f"{a}-{b}"])[0]
        if c.kind == "soft":
            s = min(c.area * float(cfg["planar_rest_per_mm2"]), float(cfg["planar_rest_cap"]))
            add(a, b, Edge(a, b, max(s, 1.0), ref, "rest"))
        elif c.kind in ("revolute", "prismatic", "cylindrical"):
            add(a, b, Edge(a, b, float(cfg["bearing_or_joint"]), ref, "joint"))
        elif not any(e.code == "fastener_joint" for e in c.evidence):
            add(a, b, Edge(a, b, float(cfg["press_or_body"]), ref, "press"))
    return sorted(by_pair.values(), key=lambda e: (e.a, e.b))


def _widest_paths(
    edges: list[Edge], ground_ids: set[str]
) -> tuple[dict[str, float], dict[str, tuple[str, Edge]]]:
    adj: dict[str, list[Edge]] = {}
    for e in edges:
        adj.setdefault(e.a, []).append(e)
        adj.setdefault(e.b, []).append(e)
    best: dict[str, float] = {g: float("inf") for g in ground_ids}
    parent: dict[str, tuple[str, Edge]] = {}
    heap = [(-float("inf"), g) for g in sorted(ground_ids)]
    heapq.heapify(heap)
    while heap:
        neg, node = heapq.heappop(heap)
        cur = -neg
        if cur < best.get(node, -1.0):
            continue
        for e in adj.get(node, []):
            nxt = e.other(node)
            cand = min(cur, e.strength)
            if cand > best.get(nxt, 0.0) + 1e-9:
                best[nxt] = cand
                parent[nxt] = (node, e)
                heapq.heappush(heap, (-cand, nxt))
    return best, parent


def load_paths(asm: AssemblyInfo, model: KinematicModel, ctx: KinContext) -> LoadInfo:
    cfg = load_table("weak_spot_rules.yaml")["load_path"]
    ground_ids = floor_instances(ctx, model, asm)
    if not ground_ids:
        return LoadInfo([], {}, {}, [], None)
    masses = {i.id: i.ap.part.mass.mass_g for i in ctx.insts}
    known = all(m is not None for m in masses.values())
    total = float(sum(m for m in masses.values() if m is not None)) if known else None
    volumes = {i.id: i.ap.part.mass.volume_mm3 for i in ctx.insts}
    total_vol = sum(volumes.values()) or 1.0
    frac = float(cfg["load_mass_min_fraction"])
    sources: list[str] = []
    for i in ctx.insts:
        if i.id in ground_ids:
            continue
        heavy = (
            (masses[i.id] or 0.0) >= frac * total if total else volumes[i.id] >= frac * total_vol
        )
        if heavy or hardware_kind(i) == "motor":
            sources.append(i.id)
    edges = build_edges(asm, ctx)
    _best, parent = _widest_paths(edges, ground_ids)
    paths: list[LoadPath] = []
    used: dict[str, Edge] = {}
    through: dict[str, list[str]] = {}
    for n, src in enumerate(sorted(sources), 1):
        if src not in parent:
            continue
        chain: list[str] = [src]
        route: list[Edge] = []
        cur = src
        while cur in parent:
            prev, e = parent[cur]
            chain += [e.ref, prev]
            route.append(e)
            cur = prev
        for e in route:
            used[e.ref] = e
            through.setdefault(e.ref, []).append(src)
        weakest = min(route, key=lambda e: e.strength)
        fast = [e.fasteners for e in route if e.kind == "fastened"]
        names = {i.id: i.name for i in ctx.insts}
        mass_txt = f", {fmt(masses[src] or 0.0, 0)} g" if masses[src] is not None else ""
        hops = " -> ".join(
            f"{names[chain[k]]} --{chain[k + 1]}({_edge_word(used[chain[k + 1]])})--> {names[chain[k + 2]]}"
            for k in range(0, len(chain) - 2, 2)
        )
        paths.append(
            LoadPath(
                id=f"LP{n:03d}",
                load_source=src,
                path=chain,
                weakest_connection=weakest.ref,
                fastener_count_min=min(fast) if fast else None,
                description=(
                    f"{names[src]} ({src}{mass_txt}) reaches ground through {hops}; "
                    f"weakest connection {weakest.ref} ({_edge_word(weakest)})"
                ),
            )
        )
    return LoadInfo(paths, used, through, sources, total, ground_ids)


def _edge_word(e: Edge) -> str:
    if e.kind == "fastened":
        return f"{e.fasteners} fastener(s), dia {fmt(e.diameter, 1)}"
    return {
        "joint": "bearing or joint",
        "press": "press fit or mount",
        "rest": "rests, no fasteners",
    }[e.kind]
