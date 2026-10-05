# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Understanding layer: structure, symmetry, process, kinematics, mechanisms, risks, roles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from stepscribe.geometry.part_geom import PartGeom
from stepscribe.models.schema import Part, PartUnderstanding
from stepscribe.understanding.aag import build_aag
from stepscribe.understanding.process import infer_process
from stepscribe.understanding.structural import base_body, structural_features
from stepscribe.understanding.symmetry import part_symmetry

if TYPE_CHECKING:
    from stepscribe.analysis import AnalyzeOptions
    from stepscribe.api import Analysis
    from stepscribe.models.schema import KinematicModel, Understanding
    from stepscribe.progress import Tracker
    from stepscribe.understanding.answers import Effects


def part_understanding(
    geom: PartGeom, part: Part, override_process: str | None = None
) -> PartUnderstanding:
    """AAG summary, structural features, symmetry and manufacturing process of one part."""
    feature_faces: dict[str, str] = {}
    for h in part.holes:
        for s in h.segments:
            for fid in s.face_ids:
                feature_faces[fid] = h.id
    for fl in part.fillets:
        for fid in fl.face_ids:
            feature_faces.setdefault(fid, fl.id)
    aag = build_aag(geom, feature_faces)
    features = structural_features(aag, part)
    axes = [np.array([a.x, a.y, a.z]) for a in part.obb.axes]
    sym = part_symmetry(geom, axes)
    process = infer_process(part, aag, features, sym, override_process)
    return PartUnderstanding(
        aag=aag.summary(base_body(part, aag)),
        structural_features=features,
        symmetry=sym,
        process=process,
    )


def single_part_kinematics(analysis: Analysis) -> KinematicModel:
    """A one-link, static model for a file that holds a single part."""
    from stepscribe.models.schema import KinematicModel, RigidGroup

    part = analysis.report.parts[0]
    link = RigidGroup(
        id="L0",
        instance_ids=[part.id],
        is_ground=True,
        mass_g=part.mass.mass_g,
        description=f"{part.name} (single part)",
    )
    return KinematicModel(
        links=[link],
        joints=[],
        dof=0,
        topology="static",
        description="a single rigid part: no joints",
        mermaid=f'flowchart LR\n  L0["L0 Ground: {part.name}"]',
    )


class UnderstandingPipeline:
    """The understanding stages with the expensive ones cached.

    Building it runs the heavy stages once (kinematics, mechanisms, optional motion check).
    :meth:`finish` then runs the cheap stages (load paths, stability, roles, weak spots, summary,
    questions) and can be called again whenever the designer's answers change; it restores the
    mutable state first, so the result depends only on the answers it is given.
    """

    def __init__(
        self, analysis: Analysis, opts: AnalyzeOptions, tracker: Tracker | None = None
    ) -> None:
        self.analysis = analysis
        self.opts = opts
        self.asm = analysis.report.assembly
        self.ctx = None
        if tracker is not None:
            tracker.stage("understanding", "kinematics")
        if self.asm is not None:
            from stepscribe.understanding.kinematics import build_kinematics_ex
            from stepscribe.understanding.mechanisms import detect_mechanisms

            self.kin, self.ctx = build_kinematics_ex(analysis)
            detect_mechanisms(analysis, self.kin, self.ctx)
            if opts.check_motion:
                from stepscribe.understanding.motion_check import check_motion

                if tracker is not None:
                    tracker.stage("understanding", "motion check")
                check_motion(analysis, self.kin, self.ctx, float(opts.motion_budget_s))
        else:
            self.kin = single_part_kinematics(analysis)
        if tracker is not None:
            tracker.stage("understanding", "load paths, weak spots and stability")
        self._snapshot()

    # ---------------------------------------------------------------- state
    def _snapshot(self) -> None:
        import copy

        self._parts = {
            ap.part.id: (ap.part.mass.model_copy(deep=True), copy.deepcopy(ap.part.understanding))
            for ap in self.analysis.parts
        }
        self._joints = {
            j.id: (j.driven_by, j.range_deg_or_mm, j.description) for j in self.kin.joints
        }
        a = self.asm
        self._asm = (
            None
            if a is None
            else (
                a.total_mass_g,
                copy.deepcopy(a.center_of_mass),
                copy.deepcopy(a.bom),
                copy.deepcopy(a.subassemblies),
            )
        )

    def _restore(self) -> None:
        import copy

        for ap in self.analysis.parts:
            mass, und = self._parts[ap.part.id]
            ap.part.mass = mass.model_copy(deep=True)
            ap.part.understanding = copy.deepcopy(und)
        for j in self.kin.joints:
            j.driven_by, j.range_deg_or_mm, j.description = self._joints[j.id]
        if self.asm is not None and self._asm is not None:
            self.asm.total_mass_g, com, bom, subs = self._asm
            self.asm.center_of_mass = copy.deepcopy(com)
            self.asm.bom = copy.deepcopy(bom)
            self.asm.subassemblies = copy.deepcopy(subs)

    # ---------------------------------------------------------------- stages
    def _payload_point(self) -> Any:
        """Where a payload sits: the far end of the arm, else the top of the design."""
        import numpy as np

        if self.ctx is None:
            return None
        from stepscribe.understanding.function import _arm_links
        from stepscribe.understanding.kinematics import axis_vector, wheel_joint_ids

        model, ctx = self.kin, self.ctx
        arm = _arm_links(model, wheel_joint_ids(model, ctx))
        members = [ctx.by_id[i] for g in model.links if g.id in arm for i in g.instance_ids]
        up = axis_vector(self.asm.up_axis) if self.asm else np.array([0.0, 0.0, 1.0])
        pool = members or list(ctx.insts)
        best = max(pool, key=lambda i: float(np.dot(0.5 * (i.bounds[0] + i.bounds[1]), up)))
        top = 0.5 * (best.bounds[0] + best.bounds[1])
        if members:
            joint = next((j for j in model.joints if j.child_link in arm), None)
            if joint is not None:
                anchor = np.array([joint.axis.origin.x, joint.axis.origin.y, joint.axis.origin.z])
                best = max(
                    members,
                    key=lambda i: float(np.linalg.norm(0.5 * (i.bounds[0] + i.bounds[1]) - anchor)),
                )
                top = 0.5 * (best.bounds[0] + best.bounds[1])
        return top

    def finish(
        self,
        answers: dict[str, dict] | None = None,  # type: ignore[type-arg]
        context: dict | None = None,  # type: ignore[type-arg]
    ) -> Understanding:
        """Run the cheap stages with the given answers applied; sets ``report.understanding``."""
        from stepscribe.understanding import answers as ans
        from stepscribe.understanding.kinematics import wheel_joint_ids
        from stepscribe.understanding.questions import generate_questions

        answers = answers or {}
        report = self.analysis.report
        wheels = wheel_joint_ids(self.kin, self.ctx) if self.ctx is not None else None
        effects = ans.Effects()
        qtext: dict[str, str] = {}
        if answers:
            # a first pass without answers defines the questions, so IDs can be resolved
            self._restore()
            draft = self._compute(ans.Effects(), context, wheels, generate=False)
            report.understanding = draft
            known = {q.id: q for q in generate_questions(report, None, context, wheels, limit=None)}
            effects = ans.collect_effects(answers, known)
            qtext = {k: q.text for k, q in known.items()}
        self._restore()
        und = self._compute(effects, context, wheels, generate=True, answers=answers, qtext=qtext)
        return und

    def _compute(
        self,
        effects: Effects,
        context: dict | None,  # type: ignore[type-arg]
        wheels: set[str] | None,
        generate: bool,
        answers: dict | None = None,  # type: ignore[type-arg]
        qtext: dict[str, str] | None = None,
    ) -> Understanding:
        from stepscribe.models.schema import Answer, Understanding
        from stepscribe.understanding import answers as ans
        from stepscribe.understanding.function import (
            assign_roles,
            assign_single_part_roles,
            design_summary,
            subassembly_roles,
        )
        from stepscribe.understanding.load_path import load_paths
        from stepscribe.understanding.questions import generate_questions
        from stepscribe.understanding.stability import stability
        from stepscribe.understanding.weak_spots import find_weak_spots, number

        analysis, kin, ctx, asm = self.analysis, self.kin, self.ctx, self.asm
        report = analysis.report
        conflicts: list[str] = []
        parts = [ap.part for ap in analysis.parts]
        if effects.material:
            ans.apply_material(analysis, effects.material)
        conflicts += ans.apply_process(parts, effects.process)
        sub_roles = []
        if asm is not None and ctx is not None:
            assign_roles(analysis, kin, ctx, confirmed=effects.roles)
            sub_roles = subassembly_roles(analysis, kin, ctx)
        else:
            assign_single_part_roles(analysis)
        conflicts += ans.apply_roles(parts, effects.roles)
        by_joint = {j.id: j for j in kin.joints}
        for jid, who in effects.drives.items():
            if jid in by_joint and not by_joint[jid].driven_by:
                by_joint[jid].driven_by = who
        for jid, rng in effects.ranges.items():
            if jid in by_joint:
                by_joint[jid].range_deg_or_mm = rng
                by_joint[
                    jid
                ].description += f"; range {rng[0]:g} to {rng[1]:g} given by the designer"
        stab = load = None
        payload = None
        if asm is not None and ctx is not None:
            if effects.payload_g:
                point = self._payload_point()
                if point is not None:
                    payload = (float(effects.payload_g), point)
            load = load_paths(asm, kin, ctx)
            stab = stability(ctx, kin, asm.up_axis, payload)
        spots = find_weak_spots(analysis, kin, ctx, load, stab)
        spots = [
            w
            for w in spots
            if not (w.category == "floating_part" and set(w.refs) & effects.suppress_floating)
        ]
        for a_id, b_id in effects.fasteners_missing:
            spots.append(
                ans.info_spot(
                    f"Designer says fasteners are missing from the CAD between {a_id} and {b_id}: "
                    "the connection is treated as bolted for the load path",
                    [a_id, b_id],
                )
            )
        spots = number(spots)
        summary = design_summary(analysis, kin, ctx, spots)
        extras = [
            f"Designer: {k} - {v}"
            for k, v in sorted(effects.text.items())
            if k in ("purpose", "load", "environment")
        ]
        if payload is not None:
            extras.append(f"Designer payload: {effects.payload_g:g} g at the end of the arm")
        if extras:
            summary = summary + " " + " ".join(e + "." for e in extras)
        und = Understanding(
            kinematics=kin,
            load_paths=load.paths if load else [],
            weak_spots=spots,
            stability=stab,
            subassembly_roles=sub_roles,
            conflicts=conflicts,
            answers=[
                Answer(
                    question_id=q,
                    status=a["status"],
                    value=a.get("value"),
                    note=a.get("note"),
                    answered_at=a.get("answered_at"),
                    question=(qtext or {}).get(q),
                )
                for q, a in sorted((answers or {}).items())
            ],
            summary=summary,
        )
        report.understanding = und
        if generate:
            und.questions = generate_questions(report, answers, context, wheels)
        return und


def build_understanding(
    analysis: Analysis, opts: AnalyzeOptions, tracker: Tracker | None = None
) -> Understanding | None:
    """The understanding of a file (single part or assembly), with any answers in the options."""
    if not analysis.report.parts:
        return None
    from stepscribe.understanding.answers import answers_from_context

    pipe = UnderstandingPipeline(analysis, opts, tracker)
    analysis.understanding_pipeline = pipe
    return pipe.finish(answers_from_context(opts.extra.get("answers")), opts.extra.get("context"))
