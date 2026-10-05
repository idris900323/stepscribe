# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Manufacturing-process inference.

Signals are computed from the attributed adjacency graph, the part's features and its
symmetry; weights and thresholds come from ``knowledge/process_rules.yaml``. The output is a
ranked list of hypotheses, each with traceable evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from stepscribe.features.hole_standards import load_table
from stepscribe.geometry.occ_utils import canonical_dir
from stepscribe.models.schema import (
    Evidence,
    Hypothesis,
    Part,
    ProcessGuess,
    StructuralFeature,
    SymmetryInfo,
)
from stepscribe.understanding.aag import AAG, FaceAttr, angle_deg, parallel

PROCESSES = [
    "sheet_metal",
    "laser_or_waterjet",
    "cnc_turned",
    "cnc_milled",
    "injection_molded",
    "fdm_3d_printed",
    "sla_or_sls_printed",
    "extruded_profile",
    "purchased",
]


def rules() -> dict[str, Any]:
    return load_table("process_rules.yaml")


@dataclass
class Signal:
    """One fired signal: a name, how strongly it fired (0-1), the IDs involved and a sentence."""

    name: str
    strength: float
    refs: list[str]
    text: str


def _unit(v: Any) -> np.ndarray:
    a = np.array([v.x, v.y, v.z], dtype=float)
    return np.asarray(a / np.linalg.norm(a))


def access_axes(part: Part, aag: AAG) -> list[np.ndarray]:
    """Tool-access directions: hole axes, pocket-floor normals, else the dominant plane normal."""
    dirs: list[np.ndarray] = [_unit(h.axis.direction) for h in part.holes]
    idx = {a.id: a for a in aag.attrs}
    for p in part.pockets:
        a = idx.get(p.floor_face_id)
        if a is not None and a.normal is not None:
            dirs.append(a.normal)
    if not dirs:
        planes = sorted(aag.planes(), key=lambda a: -a.area)
        if planes and planes[0].normal is not None:
            dirs.append(planes[0].normal)
    out: list[np.ndarray] = []
    for d in dirs:
        d = canonical_dir(d)
        if not any(parallel(d, o, 0.002) for o in out):
            out.append(d)
    return out


def _profile_extrusion(aag: AAG, thr: dict[str, Any]) -> Signal | None:
    planes = sorted(aag.planes(), key=lambda a: -a.area)
    if len(planes) < 2:
        return None
    n = planes[0].normal
    assert n is not None
    offsets: set[float] = set()
    for a in aag.attrs:
        if a.kind == "plane" and a.normal is not None:
            if parallel(a.normal, n, 0.002):
                offsets.add(
                    round(float((a.offset or 0.0) * float(np.sign(np.dot(a.normal, n)))), 2)
                )
            elif abs(float(np.dot(a.normal, n))) > 0.02:
                return None  # a slanted plane: not a straight extrusion
        elif a.kind == "cylinder":
            if a.axis_dir is None or not parallel(a.axis_dir, n, 0.002):
                return None
        else:
            return None
    if len(offsets) != int(thr["profile_cap_offsets"]):
        return None
    return Signal(
        "profile_extrusion",
        1.0,
        [planes[0].id],
        "every face is one of two parallel caps or a wall perpendicular to them (an extruded 2D profile)",
    )


def _coaxial_fraction(aag: AAG, axis: np.ndarray, point: np.ndarray) -> float:
    total = sum(a.area for a in aag.attrs) or 1.0
    ok = 0.0
    for a in aag.attrs:
        if a.kind == "cylinder" and a.axis_dir is not None and a.axis_point is not None:
            d = a.axis_point - point
            off = d - axis * float(np.dot(d, axis))
            if parallel(a.axis_dir, axis, 0.002) and np.linalg.norm(off) < 0.05:
                ok += a.area
        elif a.kind == "plane" and a.normal is not None and parallel(a.normal, axis, 0.002):
            ok += a.area
        elif a.kind == "cone":
            ok += a.area  # cones on a turned part are coaxial chamfers/tapers
    return ok / total


def compute_signals(
    part: Part,
    aag: AAG,
    structural: list[StructuralFeature],
    sym: SymmetryInfo,
    r: dict[str, Any],
) -> dict[str, Signal]:
    thr = r["thresholds"]
    out: dict[str, Signal] = {}
    total_area = sum(a.area for a in aag.attrs) or 1.0
    bends = [f for f in structural if f.kind == "bend"]
    if bends:
        out["has_bends"] = Signal(
            "has_bends",
            1.0,
            [f.id for f in bends],
            f"{len(bends)} bend(s) of constant sheet thickness",
        )
    flanges = [f for f in structural if f.kind == "flange"]
    if flanges:
        out["flanges"] = Signal(
            "flanges", 1.0, [f.id for f in flanges], "flat legs joined through bends"
        )
    if part.shape_class.label == "sheet_metal":
        out["shape_sheet_metal"] = Signal(
            "shape_sheet_metal", part.shape_class.confidence, [], "shape classified as sheet metal"
        )
    if part.shape_class.label == "plate":
        out["plate_shape"] = Signal("plate_shape", 1.0, [], "classified as a flat plate")
    prof = _profile_extrusion(aag, thr)
    if prof and not bends:
        out["profile_extrusion"] = prof
    # turned: high-fold rotational symmetry with coaxial faces
    for rot in sym.rotational:
        if int(rot["fold"]) >= int(thr["axisymmetric_min_fold"]):
            ax = rot["axis"]["direction"]
            org = rot["axis"]["origin"]
            axis = np.array([ax["x"], ax["y"], ax["z"]])
            frac = _coaxial_fraction(
                aag, axis / np.linalg.norm(axis), np.array([org["x"], org["y"], org["z"]])
            )
            if frac >= float(thr["axisymmetric_area_fraction"]):
                out["axisymmetric"] = Signal(
                    "axisymmetric",
                    1.0,
                    [],
                    f"{rot['fold']}-fold rotational symmetry with {100 * frac:.0f}% of the area on coaxial faces",
                )
            break
    axes = access_axes(part, aag)
    turned = "axisymmetric" in out
    if not turned and len(axes) <= int(thr["access_axes_max"]) and part.topology.faces > 6:
        out["tool_accessible"] = Signal(
            "tool_accessible", 1.0, [], f"all features reachable from {len(axes)} direction(s)"
        )
    elif not turned and len(axes) > int(thr["access_axes_max"]):
        out["not_tool_accessible"] = Signal(
            "not_tool_accessible", 1.0, [], f"features point in {len(axes)} different directions"
        )
    sharp, radiused = [], []
    for e in aag.edges:
        if e.kind != "concave" or e.length < float(thr["sharp_corner_min_length_mm"]):
            continue
        fa, fb = aag.attrs[e.a], aag.attrs[e.b]
        if fa.kind == "plane" and fb.kind == "plane":
            if any(parallel(e.tangent, ax, 0.02) for ax in axes):
                sharp.append(e)
    for a in aag.attrs:
        if (
            a.kind == "cylinder"
            and a.convex is False
            and a.radius is not None
            and a.radius >= float(thr["radiused_corner_min_radius_mm"])
            and a.axis_dir is not None
            and any(parallel(a.axis_dir, ax, 0.02) for ax in axes)
            and sum(1 for e in aag.edges_of_type(a.index, "smooth")) >= 2
        ):
            radiused.append(a)
    if sharp:
        ids = sorted({aag.attrs[e.a].id for e in sharp} | {aag.attrs[e.b].id for e in sharp})
        out["sharp_vertical_internal_corners"] = Signal(
            "sharp_vertical_internal_corners",
            1.0,
            ids,
            f"{len(sharp)} sharp internal corner(s) running along a tool direction (a cutter leaves a radius)",
        )
    if (
        any(
            e.kind == "concave" and e.length >= float(thr["sharp_corner_min_length_mm"])
            for e in aag.edges
            if aag.attrs[e.a].kind == "plane" and aag.attrs[e.b].kind == "plane"
        )
        and not turned
    ):
        ids = sorted({aag.attrs[e.a].id for e in aag.edges if e.kind == "concave"})
        out["sharp_internal_corners"] = Signal(
            "sharp_internal_corners", 1.0, ids[:6], "sharp concave edges between flat faces"
        )
    if radiused:
        out["radiused_vertical_corners"] = Signal(
            "radiused_vertical_corners",
            1.0,
            [a.id for a in radiused],
            f"{len(radiused)} internal corner(s) with radius >= {thr['radiused_corner_min_radius_mm']} mm along a tool direction",
        )
    if part.pockets or any(not h.is_through for h in part.holes):
        out["blind_features"] = Signal(
            "blind_features",
            1.0,
            [p.id for p in part.pockets] + [h.id for h in part.holes if not h.is_through],
            "pockets or blind holes",
        )
    draft = _draft_signal(aag, thr)
    if draft:
        out["draft"] = draft
    if part.bosses and (any(f.kind == "rib" for f in structural)):
        out["ribs_and_bosses"] = Signal(
            "ribs_and_bosses", 1.0, [b.id for b in part.bosses], "ribs and bosses together"
        )
    free = sum(
        a.area
        for a in aag.attrs
        if a.kind
        in ("bspline", "bezier", "other", "revolution", "extrusion", "offset", "sphere", "torus")
    )
    if free / total_area >= float(thr["freeform_area_fraction"]):
        out["freeform_faces"] = Signal(
            "freeform_faces",
            1.0,
            [],
            f"{100 * free / total_area:.0f}% of the area is freeform or doubly curved",
        )
    if part.min_wall_thickness_mm:
        m = part.min_wall_thickness_mm / float(thr["nozzle_mm"])
        if (
            abs(m - round(m)) * float(thr["nozzle_mm"]) <= float(thr["nozzle_multiple_tol_mm"])
            and round(m) >= 2
        ):
            out["wall_multiple_of_nozzle"] = Signal(
                "wall_multiple_of_nozzle",
                1.0,
                [],
                f"thinnest wall {part.min_wall_thickness_mm:.2f} mm is a multiple of a {thr['nozzle_mm']} mm nozzle",
            )
    if (
        any(t.kind == "extrusion_profile" for t in part.semantic_tags)
        or part.shape_class.label == "extrusion_profile"
    ):
        out["extrusion_tag"] = Signal(
            "extrusion_tag", 1.0, [], "constant-section extrusion profile"
        )
    if part.likely_purchased_hardware:
        out["hardware"] = Signal(
            "hardware", 1.0, [], f"recognised as hardware ({part.hardware_guess})"
        )
    return out


def _draft_signal(aag: AAG, thr: dict[str, Any]) -> Signal | None:
    planes = sorted(aag.planes(), key=lambda a: -a.area)
    if not planes:
        return None
    pull = planes[0].normal
    assert pull is not None
    walls, drafted = 0.0, 0.0
    ids: list[str] = []
    for a in aag.planes():
        if a.normal is None or parallel(a.normal, pull, 0.02):
            continue
        ang = abs(90.0 - angle_deg(a.normal, pull if float(np.dot(a.normal, pull)) >= 0 else -pull))
        if ang > 15.0:
            continue
        walls += a.area
        if float(thr["draft_min_deg"]) <= ang <= float(thr["draft_max_deg"]):
            drafted += a.area
            ids.append(a.id)
    if walls > 0 and drafted / walls >= float(thr["draft_area_fraction"]):
        return Signal(
            "draft",
            1.0,
            ids[:8],
            f"{100 * drafted / walls:.0f}% of the walls lean 0.5-3 deg from the pull direction",
        )
    return None


NAME_PROCESS_SIGNAL = "name_hint"


def _has_word(text: str, word: str) -> bool:
    """Keyword at a word start; short keywords (<= 3 letters) must also end a word."""
    import re

    end = r"(?![a-z0-9])" if len(word) <= 3 else ""
    return re.search(r"(?<![a-z0-9])" + re.escape(word) + end, text) is not None


def name_hits(name: str, r: dict[str, Any]) -> dict[str, str]:
    import re

    low = re.sub(r"([a-z])([A-Z])", r" ", name).lower()
    low = re.sub(r"[_\-.]+", " ", low)
    hits: dict[str, str] = {}
    for proc, words in r["name_keywords"].items():
        for w in words:
            if _has_word(low, w.replace("_", " ").strip()):
                hits[proc] = w
                break
    return hits


def user_process(ctx_text: str | None, r: dict[str, Any]) -> str | None:
    """Map a free-text ``manufacturing`` entry to a process label."""
    if not ctx_text:
        return None
    low = ctx_text.lower()
    for proc, words in r["context_keywords"].items():
        if any(w in low for w in words):
            return str(proc)
    return None


def infer_process(
    part: Part,
    aag: AAG,
    structural: list[StructuralFeature],
    sym: SymmetryInfo,
    override: str | None = None,
) -> ProcessGuess:
    """Rank manufacturing processes for *part*; *override* (a process label) wins at 1.0."""
    r = rules()
    if override in PROCESSES:
        ev = Evidence(code="user", refs=[part.id], weight=1.0, text="process given by the designer")
        return ProcessGuess(
            ranked=[Hypothesis(label=str(override), confidence=1.0, evidence=[ev])], source="user"
        )
    signals = compute_signals(part, aag, structural, sym, r)
    hits = name_hits(part.name, r)
    k = float(r["confidence_k"])
    ranked: list[Hypothesis] = []
    for proc in PROCESSES:
        w = r["weights"].get(proc, {})
        evidence: list[Evidence] = []
        score = 0.0
        for sname, weight in w.items():
            if sname == "name_hint":
                if proc in hits:
                    evidence.append(
                        Evidence(
                            code=f"name:{hits[proc]}",
                            refs=[part.id],
                            weight=float(weight),
                            text=f"part name contains '{hits[proc]}'",
                        )
                    )
                    score += float(weight)
                continue
            sig = signals.get(sname)
            if sig is None:
                continue
            contrib = float(weight) * sig.strength
            evidence.append(
                Evidence(code=f"signal:{sname}", refs=sig.refs, weight=contrib, text=sig.text)
            )
            score += contrib
        if (
            proc == "fdm_3d_printed"
            and score <= 0
            and part.topology.faces > 12
            and not part.likely_purchased_hardware
        ):
            score = float(r["fallback_floor"])
            evidence.append(
                Evidence(
                    code="fallback:complex_shape",
                    refs=[part.id],
                    weight=score,
                    text="complex shape, no stronger signal",
                )
            )
        score = max(score, 0.0)
        ranked.append(
            Hypothesis(label=proc, confidence=round(score / (score + k), 3), evidence=evidence)
        )
    ranked = [h for h in ranked if h.confidence > 0]
    ranked.sort(key=lambda h: (-h.confidence, PROCESSES.index(h.label)))
    if not ranked:
        ranked = [Hypothesis(label="unknown", confidence=0.0, evidence=[])]
    return ProcessGuess(ranked=ranked, source="inferred")


def min_wall_for(process: str | None) -> tuple[float, str] | None:
    """(guideline mm, source) for *process*, or None when no sourced value exists."""
    row = rules()["min_wall_mm"].get(process or "")
    if row:
        return float(row["value"]), str(row["source"])
    return None


__all__ = [
    "PROCESSES",
    "FaceAttr",
    "access_axes",
    "infer_process",
    "min_wall_for",
    "name_hits",
    "user_process",
]
