# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Headless flag: inside an AI client nothing may start a server or open a browser.

The MCP server sets the flag at startup. Code that would start an HTTP server or open a browser
calls :func:`require_interactive` first and gets an error when the flag is set. Only
``stepscribe ui``, which the user runs on purpose, ever gets past that check.
"""

from __future__ import annotations

import os

ENV = "STEPSCRIBE_HEADLESS"


class HeadlessError(RuntimeError):
    """Raised when something interactive is attempted in headless mode."""


def set_headless() -> None:
    """Mark this process (and its children) as headless."""
    os.environ[ENV] = "1"


def is_headless() -> bool:
    """True once :func:`set_headless` has been called in this process or a parent."""
    return os.environ.get(ENV) == "1"


def require_interactive(action: str) -> None:
    """Raise :class:`HeadlessError` if *action* (e.g. "start the local server") is not allowed."""
    if is_headless():
        raise HeadlessError(f"cannot {action}: stepscribe is running headless inside an AI client")
