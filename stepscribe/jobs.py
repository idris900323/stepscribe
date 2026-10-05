# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Background analysis jobs for clients with short call timeouts (the MCP server).

A job runs one export in a separate process, so a start call returns at once and the client polls
for progress. Results are cached on disk by file hash and options, so asking again is instant.
Nothing here imports the web UI.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import tempfile
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from stepscribe.analysis import AnalyzeOptions
from stepscribe.progress import PRIOR_SECONDS_PER_FACE, Progress

FACES_PER_MB = 600  # measured on the robot corpus: about 600 faces per MB of STEP text
ASSEMBLY_FACTOR = 6.0  # parts plus contacts, from progress.ASSEMBLY_PRIOR_FRACTION + 1


def cache_dir() -> Path:
    """Where exports of MCP jobs live (override with ``STEPSCRIBE_CACHE``)."""
    root = os.environ.get("STEPSCRIBE_CACHE")
    return Path(root) if root else Path(tempfile.gettempdir()) / "stepscribe_cache"


def job_out_dir(path: Path, options: AnalyzeOptions) -> Path:
    """Cache folder for one file and one set of options."""
    h = hashlib.sha256(path.read_bytes())
    h.update(repr(sorted(asdict(options).items())).encode())
    return cache_dir() / h.hexdigest()[:16]


def initial_eta_s(path: Path) -> float:
    """A rough first estimate from the file size; the running job replaces it with a measured one."""
    mb = path.stat().st_size / 1e6
    return round(
        mb * FACES_PER_MB * PRIOR_SECONDS_PER_FACE * (ASSEMBLY_FACTOR if mb > 0.5 else 1), 0
    )


def _worker(spec: dict[str, Any], queue: Any) -> None:
    from stepscribe.batch import Job, process_file

    def push(p: Progress) -> None:
        queue.put(
            {
                "type": "progress",
                "stage": p.stage,
                "message": p.message,
                "fraction": p.fraction,
                "elapsed_s": p.elapsed_s,
                "eta_s": p.eta_s,
            }
        )

    try:
        summary = process_file(Job(**spec), push)
        queue.put({"type": "done", "summary": summary})
    except Exception as exc:  # noqa: BLE001
        queue.put(
            {
                "type": "error",
                "message": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=5),
            }
        )


@dataclass
class JobState:
    """One running or finished job."""

    id: str
    path: str
    out_dir: str
    started: float
    process: Any = None
    queue: Any = None
    last: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] | None = None
    error: str | None = None
    done: bool = False


class JobManager:
    """Starts export jobs in worker processes and reports their progress."""

    def __init__(self) -> None:
        self.jobs: dict[str, JobState] = {}

    def start(
        self,
        path: str | Path,
        options: AnalyzeOptions,
        export_params: dict[str, Any] | None = None,
        images: bool = True,
    ) -> JobState:
        """Begin an export of *path*; reuse a running job for the same file and options."""
        p = Path(path)
        out = job_out_dir(p, options)
        for state in self.jobs.values():
            if state.out_dir == str(out) and not state.done:
                return state
        job_id = uuid.uuid4().hex[:8]
        spec = {
            "path": str(p),
            "out_dir": str(out),
            "mode": "export",
            "options": asdict(options),
            "images": images,
            "export": export_params or {},
        }
        queue: Any = mp.get_context("spawn").Queue()
        proc = mp.get_context("spawn").Process(target=_worker, args=(spec, queue))
        state = JobState(job_id, str(p), str(out), time.time(), proc, queue)
        self.jobs[job_id] = state
        proc.start()
        return state

    def poll(self, job_id: str) -> JobState:
        """Drain progress messages and update the job's state."""
        state = self.jobs.get(job_id)
        if state is None:
            raise ValueError(f"unknown job '{job_id}'")
        self._drain(state)
        if not state.done and state.process is not None and not state.process.is_alive():
            self._drain(state)  # a message can arrive between the first drain and the exit
            if not state.done:
                state.error, state.done = "the analysis process stopped unexpectedly", True
        return state

    @staticmethod
    def _drain(state: JobState) -> None:
        while state.queue is not None and not state.queue.empty():
            msg = state.queue.get()
            if msg["type"] == "progress":
                state.last = msg
            elif msg["type"] == "done":
                state.summary, state.done = msg["summary"], True
                if msg["summary"]["status"] != "ok":
                    state.error = "; ".join(str(e) for e in msg["summary"]["errors"]) or "failed"
            else:
                state.error, state.done = msg["message"], True


MANAGER = JobManager()
