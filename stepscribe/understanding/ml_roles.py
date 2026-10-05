# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Optional learned role models: one more evidence source, never a replacement for the rules.

A role model is a Python package that registers a callable in the entry-point group
``stepscribe.role_models``. The callable takes a :class:`~stepscribe.models.schema.Part` and
returns ``{role: probability}`` with roles from ``roles.yaml``. Nothing is installed by default and
nothing runs unless a model is installed and ``understanding/ml`` is not switched off in
``roles.yaml``. See ``docs/future_ml.md`` for the bar a model has to clear before it is worth using.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points

from stepscribe.models.schema import Part

RoleModel = Callable[[Part], dict[str, float]]
GROUP = "stepscribe.role_models"


def load_role_models() -> dict[str, RoleModel]:
    """Role models installed in this environment, by entry-point name (empty by default)."""
    out: dict[str, RoleModel] = {}
    for ep in entry_points(group=GROUP):
        try:
            out[ep.name] = ep.load()
        except Exception:  # noqa: BLE001 - a broken optional plugin must not break the analysis
            continue
    return out


def role_scores(
    part: Part, models: dict[str, RoleModel], weight: float
) -> dict[str, tuple[str, float]]:
    """Role -> (model name, evidence weight) from every model; a probability times *weight*."""
    best: dict[str, tuple[str, float]] = {}
    for name, model in sorted(models.items()):
        try:
            probs = model(part)
        except Exception:  # noqa: BLE001
            continue
        for role, p in probs.items():
            w = float(p) * weight
            if w > 0 and (role not in best or w > best[role][1]):
                best[role] = (name, w)
    return best
