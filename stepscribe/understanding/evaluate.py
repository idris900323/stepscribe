# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Score the understanding layer against hand-written labels.

A label file (YAML) describes one STEP file::

    file: robot.step              # relative to the folder of STEP files
    joints:                       # what moves
      - {type: revolute, axis: [1, 0, 0], point: [0, 0, 30]}   # point is optional
    mechanisms:
      - {kind: belt_drive, ratio: 3.0}                          # ratio is optional
    roles:                        # part name -> accepted top roles
      Base: [base_plate, chassis_frame]
    weak_spots: [tipping, single_fastener]                      # categories that should appear

Matching is one-to-one and greedy; precision and recall are reported per section.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from stepscribe.models.schema import Report

AXIS_TOL_DEG = 10.0
POINT_TOL_MM = 10.0
RATIO_TOL = 0.05


@dataclass
class Score:
    """True positives, predictions and labels of one section, plus the mismatches."""

    tp: int = 0
    predicted: int = 0
    labelled: int = 0
    misses: list[str] = field(default_factory=list)
    extras: list[str] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / self.predicted if self.predicted else 1.0

    @property
    def recall(self) -> float:
        return self.tp / self.labelled if self.labelled else 1.0

    def add(self, other: Score) -> None:
        self.tp += other.tp
        self.predicted += other.predicted
        self.labelled += other.labelled
        self.misses += other.misses
        self.extras += other.extras


def load_labels(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "file" not in data:
        raise ValueError(f"{path}: a label file needs a 'file:' entry")
    return dict(data)


def _axis_close(a: np.ndarray, b: np.ndarray) -> bool:
    c = abs(float(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(float(min(1.0, c)))) <= AXIS_TOL_DEG


def _dist_to_axis(point: np.ndarray, origin: np.ndarray, d: np.ndarray) -> float:
    d = d / np.linalg.norm(d)
    v = point - origin
    return float(np.linalg.norm(v - d * float(np.dot(v, d))))


def score_joints(report: Report, labels: list[dict[str, Any]]) -> Score:
    u = report.understanding
    pred = list(u.kinematics.joints) if u else []
    s = Score(predicted=len(pred), labelled=len(labels))
    used: set[int] = set()
    for lab in labels:
        want = np.array(lab["axis"], dtype=float)
        hit = None
        for i, j in enumerate(pred):
            if i in used or j.kind != lab["type"]:
                continue
            d = np.array([j.axis.direction.x, j.axis.direction.y, j.axis.direction.z])
            if not _axis_close(d, want):
                continue
            if "point" in lab:
                o = np.array([j.axis.origin.x, j.axis.origin.y, j.axis.origin.z])
                if _dist_to_axis(np.array(lab["point"], dtype=float), o, d) > POINT_TOL_MM:
                    continue
            hit = i
            break
        if hit is None:
            s.misses.append(f"joint {lab['type']} about {lab['axis']} not found")
        else:
            used.add(hit)
            s.tp += 1
    s.extras = [f"{j.id} {j.kind}" for i, j in enumerate(pred) if i not in used]
    return s


def score_mechanisms(report: Report, labels: list[dict[str, Any]]) -> Score:
    u = report.understanding
    pred = list(u.kinematics.mechanisms) if u else []
    s = Score(predicted=len(pred), labelled=len(labels))
    used: set[int] = set()
    for lab in labels:
        hit = None
        for i, m in enumerate(pred):
            if i in used or m.kind != lab["kind"]:
                continue
            if "ratio" in lab and (
                m.ratio is None
                or abs(m.ratio - float(lab["ratio"])) > RATIO_TOL * float(lab["ratio"])
            ):
                continue
            hit = i
            break
        if hit is None:
            s.misses.append(f"mechanism {lab['kind']} not found")
        else:
            used.add(hit)
            s.tp += 1
    s.extras = [f"{m.id} {m.kind}" for i, m in enumerate(pred) if i not in used]
    return s


def score_roles(report: Report, labels: dict[str, list[str]]) -> Score:
    by_name = {p.name: p for p in report.parts}
    s = Score(predicted=len(labels), labelled=len(labels))
    for name, accepted in labels.items():
        p = by_name.get(name)
        got = (
            p.understanding.roles[0].label
            if p and p.understanding and p.understanding.roles
            else None
        )
        if got in accepted:
            s.tp += 1
        else:
            s.misses.append(f"{name}: got {got}, accepted {accepted}")
    return s


def score_weak_spots(report: Report, labels: list[str]) -> Score:
    u = report.understanding
    found = {w.category for w in (u.weak_spots if u else [])}
    s = Score(predicted=len(found), labelled=len(labels))
    s.tp = sum(1 for c in labels if c in found)
    s.misses = [f"weak spot '{c}' not raised" for c in labels if c not in found]
    s.extras = sorted(found - set(labels))
    return s


def score_report(report: Report, labels: dict[str, Any]) -> dict[str, Score]:
    return {
        "joints": score_joints(report, labels.get("joints", [])),
        "mechanisms": score_mechanisms(report, labels.get("mechanisms", [])),
        "roles": score_roles(report, labels.get("roles", {})),
        "weak_spots": score_weak_spots(report, labels.get("weak_spots", [])),
    }
