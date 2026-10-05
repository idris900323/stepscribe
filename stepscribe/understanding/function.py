# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Function hypotheses: what each part is for, and what each group of parts is.

Function comes from shape + connections + motion together. Every role score is a sum of weighted
evidence items (weights in ``knowledge/roles.yaml``); each evidence item refers to concrete IDs.
A role supported only by the part's name is capped, and close top-two roles are flagged.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from stepscribe.assembly.tree import InstanceData
from stepscribe.describe.phrases import fmt
from stepscribe.features.hole_standards import load_table
from stepscribe.models.schema import (
    Evidence,
    Hypothesis,
    KinematicModel,
    Part,
    SubassemblyRole,
    WeakSpot,
)
from stepscribe.understanding.kinematics import (
    KinContext,
    axis_label,
    hardware_kind,
    wheel_joint_ids,
)
from stepscribe.understanding.ml_roles import RoleModel, load_role_models, role_scores

if TYPE_CHECKING:
    from stepscribe.api import Analysis


def cfg() -> dict:  # type: ignore[type-arg]
    out: dict = load_table("roles.yaml")["part_roles"]  # type: ignore[type-arg]
    return out


@dataclass
class Scorer:
    """Role scores with their evidence for one instance (or one single part)."""

    scores: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    evidence: dict[str, list[Evidence]] = field(default_factory=lambda: defaultdict(list))

    def add(self, roles: dict[str, float], code: str, refs: list[str], text: str) -> None:
        """Add evidence; the same code for the same role counts once (its refs are merged)."""
        for role, weight in roles.items():
            for e in self.evidence[role]:
                if e.code == code:
                    e.refs = sorted(set(e.refs) | set(refs))
                    break
            else:
                self.scores[role] += float(weight)
                self.evidence[role].append(
                    Evidence(code=code, refs=refs, weight=float(weight), text=text)
                )


def _has_word(text: str, word: str) -> bool:
    end = r"(?![a-z0-9])" if len(word) <= 3 else ""
    return re.search(r"(?<![a-z0-9])" + re.escape(word) + end, text) is not None


def name_hits(names: list[str], keywords: dict[str, list[str]]) -> dict[str, str]:
    text = " ".join(names)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower()
    text = re.sub(r"[_\-.]+", " ", text)
    # "M5standoff" and "608zz": also look at the text with digit-to-letter boundaries opened
    text_split = re.sub(r"(\d)([a-z])", r"\1 \2", text)
    hits: dict[str, str] = {}
    for role, words in keywords.items():
        for w in words:
            word = w.replace("_", " ").strip()
            if _has_word(text, word) or _has_word(text_split, word):
                hits[role] = w
                break
    return hits


def _confidence(role: str, scorer: Scorer, c: dict) -> float:  # type: ignore[type-arg]
    score = scorer.scores[role]
    conf = score / (score + float(c["confidence_k"]))
    if all(e.code.startswith("name:") for e in scorer.evidence[role]):
        conf = min(conf, float(c["name_only_cap"]))
    return round(conf, 3)


def _top(scorer: Scorer, c: dict, n: int = 3) -> list[Hypothesis]:  # type: ignore[type-arg]
    rows = [
        Hypothesis(label=r, confidence=_confidence(r, scorer, c), evidence=scorer.evidence[r])
        for r, s in scorer.scores.items()
        if s > 0
    ]
    rows.sort(key=lambda h: (-h.confidence, h.label))
    return rows[:n] or [Hypothesis(label="unknown", confidence=0.0, evidence=[])]


def part_score(part: Part, scorer: Scorer, c: dict) -> None:  # type: ignore[type-arg]
    """Evidence available from the part alone: shape class, semantic tags, hardware, structure."""
    sc = part.shape_class
    if sc.label in c["shape"]:
        scorer.add(
            c["shape"][sc.label],
            f"shape_class:{sc.label}",
            [part.id],
            f"shape classified as {sc.label.replace('_', ' ')} ({sc.confidence:.2f})",
        )
    for t in part.semantic_tags:
        if t.kind in c["tag"]:
            scorer.add(c["tag"][t.kind], f"tag:{t.kind}", [part.id, *t.feature_ids[:3]], t.evidence)
    if part.likely_purchased_hardware and part.hardware_guess:
        from stepscribe.understanding.kinematics import FASTENER_WORDS

        g = part.hardware_guess.lower()
        kind = (
            "linear_bearing"
            if "linear" in g
            else "bearing"
            if "bearing" in g
            else "motor"
            if "motor" in g or "servo" in g
            else "fastener"
            if any(w in g for w in FASTENER_WORDS)
            else None
        )
        if kind in c["hardware"]:
            scorer.add(
                c["hardware"][kind],
                f"hardware:{kind}",
                [part.id],
                f"recognised as {part.hardware_guess}",
            )
    u = part.understanding
    if u is not None:
        cut = [f for f in u.structural_features if f.kind == "lightening_cutout"]
        if len(cut) >= 2:
            scorer.add(
                c["structure"]["many_cutouts_and_fasteners"],
                "structure:cutouts",
                [part.id, *[f.id for f in cut[:3]]],
                f"{len(cut)} lightening cutouts in a plate-like part",
            )


def score_instances(
    analysis: Analysis, model: KinematicModel, ctx: KinContext
) -> dict[str, Scorer]:
    """One :class:`Scorer` per instance, using shape, tags, motion, neighbours, mechanisms, names."""
    c = cfg()
    asm = analysis.report.assembly
    assert asm is not None
    scorers: dict[str, Scorer] = {i.id: Scorer() for i in ctx.insts}
    for inst in ctx.insts:
        part_score(inst.ap.part, scorers[inst.id], c)
    link_of = ctx.link_of
    ground = next(g for g in model.links if g.is_ground)
    wheels = wheel_joint_ids(model, ctx)
    joints_in = defaultdict(list)  # link -> joints where it is the child
    joints_out = defaultdict(list)
    for j in model.joints:
        joints_in[j.child_link].append(j)
        joints_out[j.parent_link].append(j)
    wheel_links = {j.child_link for j in model.joints if j.id in wheels}
    k = c["kinematic"]

    for inst in ctx.insts:
        s = scorers[inst.id]
        part = inst.ap.part
        link = link_of[inst.id]
        hw = hardware_kind(inst)
        if link == ground.id and hw is None:
            s.add(
                k["in_ground_link"],
                "kinematic:ground_link",
                [inst.id, link],
                f"belongs to the ground link {link}",
            )
        if link == ground.id and hw is None and _largest_in_link(inst, model, ctx):
            s.add(
                k["largest_in_ground_link"],
                "kinematic:largest_in_ground",
                [inst.id, link],
                f"the largest part of the ground link {link}",
            )
        revs = [
            j
            for j in (*joints_in[link], *joints_out[link])
            if j.kind == "revolute" and j.id not in wheels
        ]
        main_part = _largest_in_link(inst, model, ctx)
        if len(revs) >= 2 and hw is None and main_part and link != ground.id:
            s.add(
                k["between_two_revolutes"],
                "kinematic:between_revolutes",
                [inst.id, revs[0].id, revs[1].id],
                f"link {link} is joined by revolute joints {revs[0].id} and {revs[1].id}",
            )
        elif len(revs) >= 2 and hw is None and not main_part and link != ground.id:
            s.add(
                k["link_mate_of_arm"],
                "kinematic:link_mate",
                [inst.id, link, revs[0].id],
                f"shares link {link} with the main arm part, between revolute joints {revs[0].id} and {revs[1].id}",
            )
        attached = [*joints_in[link], *joints_out[link]]
        if (
            len(attached) == 1
            and link != ground.id
            and link not in wheel_links
            and hw is None
            and main_part
            and model.topology != "closed_loop"
            and attached[0].parent_link != ground.id
        ):
            s.add(
                k["last_link_in_chain"],
                "kinematic:last_link",
                [inst.id, link],
                f"link {link} is the last link of its chain",
            )
        if link in wheel_links and part.shape_class.label in ("disc", "ring", "gear_like", "other"):
            a, b, c3 = part.obb.size_sorted
            if c3 <= 0.5 * a and b >= 0.9 * a:
                j = next(j for j in joints_in[link] if j.id in wheels)
                s.add(
                    k["wheel_on_wheel_joint"],
                    "kinematic:wheel",
                    [inst.id, j.id],
                    f"a disc on {j.id}, a horizontal revolute joint at the lowest plane",
                )
        for j in joints_in[link]:
            if j.kind == "prismatic" and hw is None and _largest_in_link(inst, model, ctx):
                s.add(
                    k["prismatic_carriage"],
                    "kinematic:carriage",
                    [inst.id, j.id],
                    f"the main part of link {link}, which slides on {j.id}",
                )
    # connections: bearings, motors, rods
    by_id = ctx.by_id
    for cn in ctx.conns:
        codes = {e.code: e for e in cn.evidence}
        if "bearing:outer" in codes:
            ref = codes["bearing:outer"].refs
            bearing = next(
                (r for r in ref[:2] if hardware_kind(by_id[r]) in ("bearing", "linear_bearing")),
                None,
            )
            holder = next((r for r in ref[:2] if r != bearing), None)
            if holder and bearing:
                scorers[holder].add(
                    k["carries_a_bearing"],
                    "kinematic:bearing_seat",
                    [holder, bearing],
                    f"{holder} holds bearing {bearing} in a bore",
                )
                n_bearings = sum(
                    1
                    for cc in ctx.conns
                    for e in cc.evidence
                    if e.code == "bearing:outer" and holder in e.refs[:2]
                )
                if n_bearings >= 3:
                    scorers[holder].add(
                        k["carries_many_bearings"],
                        "kinematic:many_bearings",
                        [holder],
                        f"{holder} holds {n_bearings} bearings: it frames the whole mechanism",
                    )
                shaft = _shaft_in_bearing(bearing, ctx)
                if shaft:
                    scorers[holder].add(
                        c["neighbour"]["holds_bearing_and_shaft"],
                        "neighbour:bearing_and_shaft",
                        [holder, bearing, shaft],
                        f"{holder} holds bearing {bearing} that carries shaft {shaft}",
                    )
        if "bearing:linear" in codes:
            ref = codes["bearing:linear"].refs
            rod = next(
                (
                    r
                    for r in ref[:2]
                    if hardware_kind(by_id[r]) not in ("bearing", "linear_bearing")
                ),
                None,
            )
            if rod:
                scorers[rod].add(
                    k["guide_rod"],
                    "kinematic:guide_rod",
                    [rod, *ref[:2]],
                    f"{rod} guides a linear bearing along its axis",
                )
        if "motor:shaft" in codes:
            motor, partner = codes["motor:shaft"].refs[:2]
            scorers[partner].add(
                k["on_motor_shaft"],
                "kinematic:motor_shaft",
                [partner, motor],
                f"{partner} sits on the shaft of motor {motor}",
            )
        if "motor:body" in codes:
            ref = codes["motor:body"].refs
            body_motor = next((r for r in ref[:2] if hardware_kind(by_id[r]) == "motor"), None)
            other = next((r for r in ref[:2] if r != body_motor), None)
            if body_motor and other:
                scorers[other].add(
                    c["neighbour"]["bolted_to_motor"],
                    "neighbour:motor",
                    [other, body_motor],
                    f"{other} carries motor {body_motor}",
                )
    for fj in asm.fastener_joints:
        mot = next((i for i in fj.instance_ids if hardware_kind(by_id[i]) == "motor"), None)
        if mot:
            for other in fj.instance_ids:
                if other != mot and hardware_kind(by_id[other]) is None:
                    scorers[other].add(
                        c["neighbour"]["bolted_to_motor"],
                        "neighbour:motor",
                        [other, mot, fj.id],
                        f"{other} is bolted to motor {mot} ({fj.id})",
                    )
    for spec in ctx.screws:
        scorers[spec.screw_id].add(
            k["screw_axis"],
            "kinematic:lead_screw",
            [spec.screw_id],
            "turns a nut that moves a slide",
        )
    for m in model.mechanisms:
        if m.kind in ("gear_pair", "belt_drive"):
            role = "gear" if m.kind == "gear_pair" else "pulley"
            for iid in m.instance_ids[:2]:
                scorers[iid].add(
                    c["mechanism"][role],
                    f"mechanism:{m.kind}",
                    [iid, m.id],
                    f"a toothed wheel of {m.id}: {m.description}",
                )
    _position_evidence(ctx, scorers, c, asm.up_axis)
    names = {i.id: [i.name, i.ap.part.name] for i in ctx.insts}
    for iid, nm in names.items():
        for role, word in name_hits(nm, c["name_keywords"]).items():
            scorers[iid].add(
                {role: c["name_weight"]},
                f"name:{word}",
                [iid],
                f"the name '{nm[0]}' contains '{word}'",
            )
    return scorers


def _largest_in_link(inst: InstanceData, model: KinematicModel, ctx: KinContext) -> bool:
    link = next(g for g in model.links if inst.id in g.instance_ids)
    members = [ctx.by_id[i] for i in link.instance_ids if hardware_kind(ctx.by_id[i]) is None]
    return bool(members) and max(members, key=lambda m: m.ap.part.mass.volume_mm3).id == inst.id


def _shaft_in_bearing(bearing: str, ctx: KinContext) -> str | None:
    for cn in ctx.conns:
        for e in cn.evidence:
            if e.code == "bearing:inner" and bearing in e.refs[:2]:
                return next(r for r in e.refs[:2] if r != bearing)
    return None


def _position_evidence(ctx: KinContext, scorers: dict[str, Scorer], c: dict, up_label: str) -> None:  # type: ignore[type-arg]
    from stepscribe.understanding.kinematics import axis_vector

    up = axis_vector(up_label)
    lows = {}
    highs = {}
    for i in ctx.insts:
        lo, hi = i.bounds
        corners = np.array(
            [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        )
        lows[i.id] = float((corners @ up).min())
        highs[i.id] = float((corners @ up).max())
    floor = min(lows.values())
    fab = [i for i in ctx.insts if hardware_kind(i) is None]
    if not fab:
        return
    body = [
        i for i in fab if i.ap.part.shape_class.label not in ("shaft", "fastener", "ring", "tube")
    ]
    foot = {i.id: i.ap.part.mass.volume_mm3 for i in body if lows[i.id] <= floor + 1.0}
    if foot:
        big = max(foot, key=lambda k: foot[k])
        scorers[big].add(
            c["position"]["lowest_and_largest"],
            "position:lowest_largest",
            [big],
            "the largest fabricated part standing on the lowest plane",
        )
    top = max(fab, key=lambda i: highs[i.id])
    height = max(highs.values()) - floor
    total = sum(i.ap.part.mass.volume_mm3 for i in ctx.insts) or 1.0
    if height > 0 and top.ap.part.mass.volume_mm3 < 0.05 * total:
        scorers[top.id].add(
            c["position"]["topmost_on_tall_part"],
            "position:topmost",
            [top.id],
            "the small part at the top of the design",
        )


def assign_roles(
    analysis: Analysis,
    model: KinematicModel,
    ctx: KinContext,
    confirmed: dict[str, str] | None = None,
    ml_models: dict[str, RoleModel] | None = None,
) -> None:
    """Fill ``part.understanding.roles`` (top 3) from the best-scoring instance of each part.

    Learned role models (optional, none installed by default) add one more evidence source.
    """
    c = cfg()
    scorers = score_instances(analysis, model, ctx)
    _neighbours_of_confirmed(scorers, ctx, confirmed or {})
    best: dict[str, dict[str, tuple[float, list[Evidence]]]] = defaultdict(dict)
    for inst in ctx.insts:
        s = scorers[inst.id]
        for role, score in s.scores.items():
            cur = best[inst.part_id].get(role)
            if cur is None or score > cur[0]:
                best[inst.part_id][role] = (score, s.evidence[role])
    models = ml_models if ml_models is not None else load_role_models()
    ml_weight = float(c.get("ml", {}).get("weight", 0.8))
    for ap in analysis.parts:
        if ap.part.understanding is None:
            continue
        merged = Scorer()
        for role, (score, ev) in best.get(ap.part.id, {}).items():
            merged.scores[role] = score
            merged.evidence[role] = ev
        for role, (name, w) in role_scores(ap.part, models, ml_weight).items():
            merged.add(
                {role: w}, f"ml:{name}", [ap.part.id], f"the learned model '{name}' suggests it"
            )
        ap.part.understanding.roles = _top(merged, c)


def _neighbours_of_confirmed(
    scorers: dict[str, Scorer], ctx: KinContext, confirmed: dict[str, str]
) -> None:
    """A part the designer confirmed as an arm link makes its link-mates look like joint brackets."""
    weight = float(cfg()["kinematic"].get("neighbour_of_confirmed_arm", 0.6))
    for inst in ctx.insts:
        if confirmed.get(inst.part_id) != "arm_link":
            continue
        link = ctx.link_of[inst.id]
        for other in ctx.insts:
            if (
                other.id != inst.id
                and ctx.link_of[other.id] == link
                and hardware_kind(other) is None
            ):
                scorers[other.id].add(
                    {"joint_bracket": weight},
                    "neighbour:confirmed_arm_link",
                    [other.id, inst.id],
                    f"{inst.id} in the same link was confirmed by the designer as an arm link",
                )


def assign_single_part_roles(analysis: Analysis) -> None:
    """Roles for a file with one part: shape, tags, structure and name only."""
    c = cfg()
    for ap in analysis.parts:
        if ap.part.understanding is None:
            continue
        s = Scorer()
        part_score(ap.part, s, c)
        for role, word in name_hits([ap.part.name], c["name_keywords"]).items():
            s.add(
                {role: c["name_weight"]},
                f"name:{word}",
                [ap.part.id],
                f"the name '{ap.part.name}' contains '{word}'",
            )
        ap.part.understanding.roles = _top(s, c)


# ---------------------------------------------------------------- subassembly roles and summary


def subassembly_roles(
    analysis: Analysis, model: KinematicModel, ctx: KinContext
) -> list[SubassemblyRole]:
    """Roles of structures found in the kinematic model, named by the links they cover."""
    rule = load_table("roles.yaml")["subassembly_roles"]
    out: list[SubassemblyRole] = []
    ground = next(g for g in model.links if g.is_ground)
    wheels = wheel_joint_ids(model, ctx)
    out.append(
        SubassemblyRole(
            subassembly=f"{ground.id} ground link",
            roles=[
                Hypothesis(
                    label="chassis",
                    confidence=0.7,
                    evidence=[
                        Evidence(
                            code="kinematic:ground",
                            refs=[ground.id],
                            weight=1.0,
                            text=f"{ground.id} is the ground link: {ground.description}",
                        )
                    ],
                )
            ],
        )
    )
    arm = _arm_links(model, wheels)
    if len(arm) >= int(rule["min_revolutes_for_arm"]):
        joints = [j for j in model.joints if j.child_link in arm]
        out.append(
            SubassemblyRole(
                subassembly="arm (" + ", ".join(sorted(arm)) + ")",
                roles=[
                    Hypothesis(
                        label="arm",
                        confidence=0.8,
                        evidence=[
                            Evidence(
                                code="kinematic:serial_revolutes",
                                refs=[j.id for j in joints],
                                weight=1.0,
                                text=f"{len(joints)} joints in a chain from the ground link: "
                                + ", ".join(j.id for j in joints),
                            )
                        ],
                    )
                ],
            )
        )
    if len(wheels) >= 2:
        motors = [
            m for m in model.mechanisms if m.kind == "direct_drive" and m.output_joint_id in wheels
        ]
        out.append(
            SubassemblyRole(
                subassembly="wheels (" + ", ".join(sorted(wheels)) + ")",
                roles=[
                    Hypothesis(
                        label="drivetrain",
                        confidence=0.75,
                        evidence=[
                            Evidence(
                                code="kinematic:wheels",
                                refs=sorted(wheels),
                                weight=1.0,
                                text=f"{len(wheels)} wheel joints at the lowest plane"
                                + (f", {len(motors)} driven directly by motors" if motors else ""),
                            )
                        ],
                    )
                ],
            )
        )
    for j in model.joints:
        if j.kind == "prismatic" and j.parent_link == ground.id:
            up_dot = abs(
                float(
                    np.dot(
                        [j.axis.direction.x, j.axis.direction.y, j.axis.direction.z], _up(analysis)
                    )
                )
            )
            label = "lift" if up_dot > 0.9 else "linear_axis"
            out.append(
                SubassemblyRole(
                    subassembly=f"{j.child_link} on {j.id}",
                    roles=[
                        Hypothesis(
                            label=label,
                            confidence=0.7,
                            evidence=[
                                Evidence(
                                    code="kinematic:prismatic",
                                    refs=[j.id],
                                    weight=1.0,
                                    text=f"a prismatic joint {j.id} along {axis_label(np.array([j.axis.direction.x, j.axis.direction.y, j.axis.direction.z]))}",
                                )
                            ],
                        )
                    ],
                )
            )
    return out


def _up(analysis: Analysis) -> np.ndarray:
    from stepscribe.understanding.kinematics import axis_vector

    assert analysis.report.assembly is not None
    return axis_vector(analysis.report.assembly.up_axis)


def _arm_links(model: KinematicModel, wheels: set[str]) -> set[str]:
    ground = next(g for g in model.links if g.is_ground)
    chain: set[str] = set()
    frontier = [ground.id]
    while frontier:
        cur = frontier.pop()
        for j in model.joints:
            if (
                j.parent_link == cur
                and j.id not in wheels
                and j.kind in ("revolute", "cylindrical")
            ):
                if not any(e.code == "motor:shaft" for e in j.evidence):
                    chain.add(j.child_link)
                    frontier.append(j.child_link)
    return chain


def design_summary(
    analysis: Analysis,
    model: KinematicModel,
    ctx: KinContext | None,
    spots: list[WeakSpot] | None = None,
) -> str:
    """Three to six deterministic sentences summarising the design like an engineer would."""
    report = analysis.report
    parts = report.parts
    asm = report.assembly
    if asm is None or ctx is None:
        p = parts[0]
        role = p.understanding.roles[0] if p.understanding and p.understanding.roles else None
        proc = p.understanding.process.ranked[0] if p.understanding else None
        bits = [
            f"Likely a single {p.shape_class.label.replace('_', ' ')} "
            f"({' x '.join(fmt(v) for v in sorted(p.obb.size_sorted, reverse=True))} mm), "
            f"{len(p.holes)} hole(s)"
        ]
        if role and role.label != "unknown":
            bits[0] += f", probably a {role.label.replace('_', ' ')}"
        if proc and proc.label != "unknown":
            bits.append(
                f"It looks {proc.label.replace('_', ' ')} (confidence {proc.confidence:.2f})"
            )
        return ". ".join(bits + _risk_sentence(spots)) + "."
    wheels = wheel_joint_ids(model, ctx)
    driven = {
        m.output_joint_id for m in model.mechanisms if m.output_joint_id and m.input_instance_id
    }
    feeders = {
        str(m.parameters["input_joint"])
        for m in model.mechanisms
        if m.parameters.get("input_joint")
    }
    arm = _arm_links(model, wheels)
    arm_joints = [j for j in model.joints if j.child_link in arm and j.id not in feeders]
    movers = [j for j in model.joints if j.id not in wheels and j.id not in feeders]
    topo = model.topology
    if topo == "mobile_manipulator":
        n_driven = sum(w in driven for w in wheels)
        drive = f"{n_driven} motor-driven" if n_driven else "none motor-driven"
        head = f"a mobile manipulator: a {len(wheels)}-wheel base ({drive}) carrying a {len(arm_joints)}-DOF serial arm"
    elif topo == "mobile_base":
        head = (
            f"a mobile base with {len(wheels)} wheels ({sum(w in driven for w in wheels)} driven)"
        )
    elif topo == "serial_chain" and len(arm_joints) >= 2:
        head = f"a {len(arm_joints)}-DOF serial arm"
    elif topo == "serial_chain" and movers and movers[0].kind == "prismatic":
        head = "a linear slide"
    elif topo == "serial_chain" and len(movers) == 1:
        head = f"a single {movers[0].kind} joint mechanism"
    elif topo == "closed_loop":
        head = f"a closed-loop linkage with an estimated {model.dof} degree(s) of freedom"
    elif topo == "static":
        head = "a static structure with no moving joints"
    else:
        head = f"a mechanism with {len(model.joints)} joints ({topo.replace('_', ' ')})"
    sentences = [f"Likely {head}"]
    listed = arm_joints or (movers if len(movers) <= 4 else [])
    if listed:
        parts_txt = []
        for j in listed:
            t = f"{j.id} {j.kind} about {axis_label(np.array([j.axis.direction.x, j.axis.direction.y, j.axis.direction.z]))}"
            mech = next((m for m in model.mechanisms if m.id == j.driven_by), None)
            if mech and mech.kind == "belt_drive":
                t += f" via a {fmt(mech.ratio or 0, 1)}:1 {mech.parameters.get('profile', 'belt')} belt"
            elif mech and mech.kind == "gear_pair":
                t += f" via a {fmt(mech.ratio or 0, 1)}:1 gear pair"
            elif mech and mech.kind == "direct_drive":
                t += " direct drive"
            parts_txt.append(t)
        sentences[0] += " (" + ", ".join(parts_txt) + ")"
    for m in model.mechanisms:
        if m.kind == "lead_screw":
            lead = m.parameters.get("lead_mm")
            sentences.append(
                "The slide is driven by a lead screw"
                + (f" ({fmt(float(lead), 0)} mm lead)" if lead else "")
            )
            break
    fab = [p for p in parts if not p.likely_purchased_hardware and p.understanding]
    procs: dict[str, int] = defaultdict(int)
    for p in fab:
        top = p.understanding.process.ranked[0] if p.understanding else None
        if top and top.label != "unknown":
            procs[top.label] += 1
    if procs:
        label, n = max(procs.items(), key=lambda kv: (kv[1], kv[0]))
        if n * 2 >= len(fab):
            sentences.append(f"Most fabricated parts look {label.replace('_', ' ')}")
        else:
            sentences.append(
                f"Fabricated parts are a mix, mostly {label.replace('_', ' ')} ({n} of {len(fab)})"
            )
    total = sum(int(r["qty"]) for r in asm.fastener_shopping_list)
    if total:
        sizes: dict[str, int] = defaultdict(int)
        for r in asm.fastener_shopping_list:
            hit = re.search(r"\bM\d+(?:\.\d+)?", str(r["item"]))
            if hit:
                sizes[hit.group(0)] += int(r["qty"])
        mostly = f", mostly {max(sizes, key=lambda k: sizes[k])}" if sizes else ""
        sentences.append(f"{total} fasteners{mostly}")
    sentences += _risk_sentence(spots)
    return ". ".join(sentences) + "."


RISK_WORDS = {
    "tipping": "tipping over ({value:.1f} deg)",
    "single_fastener": "a single-fastener connection",
    "cantilever": "an overhanging part held at one end",
    "short_screw": "a screw that is too short",
    "thin_wall": "a wall thinner than the process guideline",
    "edge_distance": "a hole too close to an edge",
    "cutout_web": "a thin web beside a cut-out",
    "floating_part": "a part not connected to the rest",
    "joint_play": "a joint on a single bearing",
    "interference": "overlapping parts",
    "unfilleted_bend": "sharp internal corners on a printed bracket",
    "sharp_internal_corner": "sharp internal corners",
    "unsupported_span": "a long unsupported plate span",
}


def _risk_phrase(w: WeakSpot) -> str:
    if w.category == "tipping" and (w.value or 0.0) <= 0.0:
        return f"a centre of mass outside the support polygon ({w.id})"
    return (
        RISK_WORDS.get(w.category, w.category.replace("_", " ")).format(value=w.value or 0.0)
        + f" ({w.id})"
    )


def _risk_sentence(spots: list[WeakSpot] | None) -> list[str]:
    """'Main risks: ...' from the two most severe findings (one per category)."""
    if not spots:
        return []
    seen: set[str] = set()
    picked: list[WeakSpot] = []
    for w in spots:  # already sorted by severity
        if w.severity in ("info",) or w.category in seen:
            continue
        seen.add(w.category)
        picked.append(w)
        if len(picked) == 2:
            break
    if not picked:
        return []
    return ["Main risks: " + ", ".join(_risk_phrase(w) for w in picked)]
