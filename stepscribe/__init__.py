# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""stepscribe: turn STEP files into an LLM Context Pack."""

__version__ = "0.1.2"

__all__ = ["__version__", "analyze", "build_context_pack", "export"]


def analyze(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Analyze a STEP file or folder; see :func:`stepscribe.api.analyze`."""
    from stepscribe.api import analyze as _analyze

    return _analyze(*args, **kwargs)


def build_context_pack(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Build a context pack; see :func:`stepscribe.api.build_context_pack`."""
    from stepscribe.api import build_context_pack as _b

    return _b(*args, **kwargs)


def export(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Write the portable export; see :func:`stepscribe.exporter.export`."""
    from stepscribe.exporter import export as _export

    return _export(*args, **kwargs)
