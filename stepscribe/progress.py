# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Progress and ETA reporting for long runs (analysis quality is never traded for speed).

The estimate is adaptive: it starts from a prior cost per face and replaces it with the measured
rate as parts finish. Later stages (assembly contacts, rendering) are added to the estimate as
soon as their size is known.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

PRIOR_SECONDS_PER_FACE = 0.06  # starting guess; measured values replace it after the first part
PRIOR_SECONDS_PER_PAIR = 12.0  # measured: BRepExtrema between complex solids is slow
PRIOR_SECONDS_PER_IMAGE = 4.0
ASSEMBLY_PRIOR_FRACTION = 5.0  # measured: assembly contacts cost several times the part analysis


@dataclass
class Progress:
    """A snapshot sent to the callback."""

    stage: str
    message: str
    fraction: float  # 0..1 overall, monotonic non-decreasing
    elapsed_s: float
    eta_s: float | None


ProgressFn = Callable[[Progress], None]


class Tracker:
    """Collects timings and emits :class:`Progress` through an optional callback."""

    def __init__(self, callback: ProgressFn | None = None) -> None:
        self.callback = callback
        self.start = time.time()
        self.total_faces = 0
        self.faces_done = 0
        self.part_seconds = 0.0
        self.n_parts = 0
        self.parts_done = 0
        self.expect_assembly = False
        self.pairs_total = 0
        self.pairs_done = 0
        self.pair_seconds = 0.0
        self.images_total = 0
        self.images_done = 0
        self.image_seconds = 0.0
        self.stage_name = "reading"
        self.images_wanted = False
        self._last_fraction = 0.0
        self._mark = time.time()

    # ---- planning
    def plan_parts(
        self, n_parts: int, total_faces: int, expect_assembly: bool, images: int = 0
    ) -> None:
        """Called once the STEP file is read and part sizes are known."""
        self.n_parts, self.total_faces = n_parts, total_faces
        self.expect_assembly = expect_assembly
        self.images_total = images or (5 + 2 * min(n_parts, 20) if self.images_wanted else 0)
        self.stage("parts", f"analysing {n_parts} part(s), {total_faces} faces")

    def plan_pairs(self, pairs: int) -> None:
        """Called after the contact broad phase."""
        self.pairs_total = pairs
        self.stage("assembly", f"checking {pairs} touching candidate pair(s)")

    # ---- events
    def stage(self, name: str, message: str) -> None:
        self.stage_name = name
        self._mark = time.time()
        self.emit(message)

    def part_done(self, name: str, faces: int, seconds: float) -> None:
        self.parts_done += 1
        self.faces_done += faces
        self.part_seconds += seconds
        self.emit(f"part {self.parts_done}/{self.n_parts}: {name} ({faces} faces, {seconds:.1f}s)")

    def pair_done(self, seconds: float) -> None:
        self.pairs_done += 1
        self.pair_seconds += seconds
        if self.pairs_done % 5 == 0 or self.pairs_done == self.pairs_total:
            self.emit(f"contact pair {self.pairs_done}/{self.pairs_total}")

    def image_done(self, name: str, seconds: float) -> None:
        self.images_done += 1
        self.image_seconds += seconds
        self.emit(f"rendered {name} ({self.images_done}/{self.images_total})")

    # ---- estimate
    def _part_rate(self) -> float:
        if self.faces_done:
            return self.part_seconds / self.faces_done
        return PRIOR_SECONDS_PER_FACE

    def estimate(self) -> tuple[float, float | None]:
        """(overall fraction, remaining seconds) from what is known so far."""
        rate = self._part_rate()
        done_s = time.time() - self.start
        if self.stage_name == "reading" or not self.total_faces:
            return 0.0, None
        parts_total_s = rate * self.total_faces
        parts_left_s = max(0.0, rate * (self.total_faces - self.faces_done))
        if self.pairs_total:
            per_pair = (
                self.pair_seconds / self.pairs_done if self.pairs_done else PRIOR_SECONDS_PER_PAIR
            )
            asm_total_s = per_pair * self.pairs_total
            asm_left_s = per_pair * (self.pairs_total - self.pairs_done)
        elif self.expect_assembly:
            asm_total_s = parts_total_s * ASSEMBLY_PRIOR_FRACTION
            asm_left_s = asm_total_s if self.stage_name in ("parts", "assembly") else 0.0
        else:
            asm_total_s = asm_left_s = 0.0
        per_img = (
            self.image_seconds / self.images_done if self.images_done else PRIOR_SECONDS_PER_IMAGE
        )
        img_total_s = per_img * self.images_total
        img_left_s = per_img * (self.images_total - self.images_done)
        total = parts_total_s + asm_total_s + img_total_s
        left = parts_left_s + asm_left_s + img_left_s
        fraction = 1.0 - left / (done_s + left) if (done_s + left) > 0 and total > 0 else 0.0
        return fraction, left

    def emit(self, message: str) -> None:
        if self.callback is None:
            return
        fraction, eta = self.estimate()
        self._last_fraction = max(self._last_fraction, min(fraction, 0.99))
        self.callback(
            Progress(self.stage_name, message, self._last_fraction, time.time() - self.start, eta)
        )

    def finish(self, message: str = "done") -> None:
        self.stage_name = "done"
        if self.callback:
            self.callback(Progress("done", message, 1.0, time.time() - self.start, 0.0))


def format_eta(seconds: float | None) -> str:
    """'about 2 min 05 s' / 'estimating...'."""
    if seconds is None:
        return "estimating..."
    s = int(round(seconds))
    if s < 60:
        return f"about {s} s"
    return f"about {s // 60} min {s % 60:02d} s"
