# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""User-supplied design intent (``design_context.yaml``)"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONTEXT_TEMPLATE = """\
# design_context.yaml: what the geometry cannot tell the reader.
# Fill in what you know and delete the rest. Pass it with: stepscribe pack robot.step --context design_context.yaml

purpose: ""            # what the design is for, e.g. "Base frame of a 4-wheel rover"
manufacturing: ""      # 3D print (FDM/SLA), CNC, laser cut, sheet metal, casting, ...
material: ""           # default material for every part, e.g. "aluminium 6061" (keywords: alu, steel, pla, petg, abs, nylon, cf, brass)
parts:                 # per-part material override, keyed by part name as shown in the pack
  # TopPlate:
  #   material: alu
loads: ""              # expected forces, torques, payload
environment: ""        # indoor, outdoor, temperature, dust, water
constraints:           # hard limits the design must meet
  # - "max mass 2 kg"
questions:             # specific questions for the reviewer
  # - "Is the base plate stiff enough?"
"""


def load_context(path: str | Path) -> dict[str, Any]:
    """Parse a design context YAML file; empty values are dropped."""
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError("design context must be a YAML mapping")
    return {k: v for k, v in raw.items() if v not in (None, "", [], {})}


def default_material(ctx: dict[str, Any]) -> str | None:
    """Default material, if given."""
    m = ctx.get("material")
    return str(m) if m else None


def part_materials(ctx: dict[str, Any]) -> dict[str, str]:
    """Per-part material overrides by part name."""
    parts = ctx.get("parts") or {}
    return {
        str(name): str(v["material"])
        for name, v in parts.items()
        if isinstance(v, dict) and v.get("material")
    }


def write_answer_stubs(path: str | Path, questions: list[Any]) -> int:
    """Append an ``answers:`` section (or missing entries) to a context file; returns the count.

    Entries already present are left alone, so earlier answers survive re-runs.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8") if p.is_file() else CONTEXT_TEMPLATE
    existing = yaml.safe_load(text) or {}
    have = set((existing.get("answers") or {}).keys()) if isinstance(existing, dict) else set()
    new = [q for q in questions if q.id not in have]
    if not new:
        return 0
    lines = (
        []
        if "answers:" in text
        else ["", "answers:  # questions the tool could not settle; fill status and value"]
    )
    for q in new:
        lines.append(f"  {q.id}:  # {q.text}")
        lines.append('    status: ""   # answered | skipped | not_sure')
        lines.append('    value: ""')
        lines.append('    note: ""')
    if "answers:" in text and not text.endswith("\n"):
        text += "\n"
    p.write_text(text.rstrip("\n") + "\n" + "\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return len(new)


def default_process(ctx: dict[str, Any]) -> str | None:
    """Default manufacturing process from the free-text ``manufacturing`` entry."""
    from stepscribe.understanding.process import rules, user_process

    return user_process(str(ctx.get("manufacturing") or ""), rules())


def part_processes(ctx: dict[str, Any]) -> dict[str, str]:
    """Per-part process overrides (``parts: Name: {process: ...}``)."""
    from stepscribe.understanding.process import rules, user_process

    parts = ctx.get("parts") or {}
    out: dict[str, str] = {}
    for name, v in parts.items():
        if isinstance(v, dict) and v.get("process"):
            proc = user_process(str(v["process"]), rules())
            if proc:
                out[str(name)] = proc
    return out


def apply_context(opts: Any, ctx: dict[str, Any]) -> None:
    """Copy material, process and answers from a design context into analysis options."""
    if opts.material is None and opts.density is None:
        opts.material = default_material(ctx)
    opts.extra["part_materials"] = part_materials(ctx)
    proc = default_process(ctx)
    if proc:
        opts.extra["default_process"] = proc
    opts.extra["part_processes"] = part_processes(ctx)
    opts.extra["context"] = ctx
    if ctx.get("answers"):
        opts.extra["answers"] = ctx["answers"]


def context_text(ctx: dict[str, Any]) -> str:
    """Compact bullet text of a context (used inside ``context_pack.md``)."""
    lines: list[str] = []
    for key, value in ctx.items():
        if key == "answers":
            continue  # printed as a readable Q&A list in 04_design_context.md
        label = key.replace("_", " ")
        if isinstance(value, dict):
            inner = "; ".join(f"{k}: {v}" for k, v in value.items())
            lines.append(f"- {label}: {inner}")
        elif isinstance(value, list):
            lines.append(f"- {label}: " + "; ".join(str(v) for v in value))
        else:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines)
