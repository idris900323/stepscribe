# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""JSON views of the understanding layer and of an interview session, for the web UI."""

from __future__ import annotations

from typing import Any


def understanding_view(report: dict[str, Any]) -> dict[str, Any] | None:
    """Everything the Understanding tab shows, from a ``report.json``-shaped dict."""
    u = report.get("understanding")
    if not u:
        return None
    k = u["kinematics"]
    names = {p["id"]: p["name"] for p in report["parts"]}
    qtext = {q["id"]: q["text"] for q in u.get("questions", [])}
    roles = []
    for p in report["parts"]:
        pu = p.get("understanding")
        if not pu:
            continue
        r = pu["roles"][0] if pu.get("roles") else None
        proc = pu["process"]["ranked"][0] if pu["process"].get("ranked") else None
        roles.append(
            {
                "id": p["id"],
                "name": names[p["id"]],
                "role": r["label"].replace("_", " ") if r else "",
                "confidence": r["confidence"] if r else None,
                "evidence": r["evidence"][0]["text"] if r and r["evidence"] else "",
                "process": proc["label"].replace("_", " ") if proc else "",
                "process_confidence": proc["confidence"] if proc else None,
                "process_by_designer": pu["process"].get("source") == "user",
            }
        )
    return {
        "summary": u["summary"],
        "topology": k["topology"].replace("_", " "),
        "kinematics": k["description"],
        "mermaid": k["mermaid"],
        "dof": k["dof"],
        "joints": [
            {
                "id": j["id"],
                "kind": j["kind"],
                "parent": j["parent_link"],
                "child": j["child_link"],
                "confidence": j["confidence"],
                "driven_by": j.get("driven_by") or "",
                "description": j["description"],
            }
            for j in k["joints"]
        ],
        "mechanisms": [
            {"id": m["id"], "kind": m["kind"].replace("_", " "), "description": m["description"]}
            for m in k["mechanisms"]
        ],
        "weak_spots": [
            {
                "id": w["id"],
                "severity": w["severity"],
                "category": w["category"].replace("_", " "),
                "message": w["message"],
                "suggestion": w.get("suggestion") or "",
                "assumption": w.get("depends_on_assumption") or "",
            }
            for w in u["weak_spots"]
        ],
        "load_paths": [lp["description"] for lp in u["load_paths"]],
        "stability": (u.get("stability") or {}).get("description", ""),
        "roles": roles,
        "conflicts": u.get("conflicts", []),
        "answers": [
            {
                "id": a["question_id"],
                "status": a["status"],
                "value": a.get("value"),
                "question": qtext.get(a["question_id"], a["question_id"]),
            }
            for a in u.get("answers", [])
        ],
        "questions": [question_view(q) for q in u.get("questions", [])],
    }


def question_view(q: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": q["id"],
        "kind": q["kind"],
        "priority": q["priority"],
        "text": q["text"],
        "why": q["why"],
        "answer_type": q["answer_type"],
        "options": q.get("options") or [],
        "unit": q.get("unit"),
        "default_guess": q.get("default_guess"),
        "refs": q.get("refs", []),
    }
