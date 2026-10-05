# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""kill_tree stops a worker that is still running."""

from __future__ import annotations

import multiprocessing as mp
import time

from stepscribe.batch import _failure
from stepscribe.procutil import kill_tree


def _sleep() -> None:
    time.sleep(60)


def test_kill_tree_stops_worker() -> None:
    proc = mp.get_context("spawn").Process(target=_sleep)
    proc.start()
    assert proc.is_alive()
    kill_tree(proc)
    assert not proc.is_alive()


def test_failure_keeps_elapsed_time() -> None:
    assert _failure("a.step", "timed out", 12.345)["seconds"] == 12.35
