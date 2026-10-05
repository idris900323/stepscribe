# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Process helpers shared by folder runs and the web UI."""

from __future__ import annotations

import subprocess
import sys
from typing import Any


def kill_tree(proc: Any) -> None:
    """Stop a worker and the part-pool processes it started (terminate() alone leaves those)."""
    if proc.pid is not None and sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False
        )
    else:
        proc.terminate()
    proc.join(timeout=10)
