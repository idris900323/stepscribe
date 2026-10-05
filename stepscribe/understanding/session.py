# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""An interview session: answers applied on the cached analysis, undo, and a "what changed" line. The STEP file is never read again; each answer re-runs only
the cheap stages, well under a second on a mid-size assembly."""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from stepscribe.describe.phrases import fmt
from stepscribe.models.schema import Question, Understanding
from stepscribe.understanding import UnderstandingPipeline
from stepscribe.understanding.answers import answers_from_context

if TYPE_CHECKING:
    from stepscribe.analysis import AnalyzeOptions
    from stepscribe.api import Analysis

SESSION_LIMIT = 15  # questions per session (configurable)


@dataclass
class Change:
    """What one answer (or an undo) changed."""

    text: str
    seconds: float
    details: list[str] = field(default_factory=list)


def _metrics(u: Understanding, analysis: Analysis) -> dict[str, Any]:
    asm = analysis.report.assembly
    return {
        "mass": asm.total_mass_g if asm else analysis.report.parts[0].mass.mass_g,
        "spots": {(w.category, tuple(w.refs)): w for w in u.weak_spots},
        "roles": {
            p.id: p.understanding.roles[0].label
            for p in analysis.report.parts
            if p.understanding and p.understanding.roles
        },
        "process": {
            p.id: p.understanding.process.ranked[0].label
            for p in analysis.report.parts
            if p.understanding and p.understanding.process.ranked
        },
        "tipping": u.stability.tipping_angle_deg if u.stability else None,
        "summary": u.summary,
        "conflicts": len(u.conflicts),
    }


def describe_change(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    out: list[str] = []
    bm, am = before["mass"], after["mass"]
    if bm != am:
        old = "unknown" if bm is None else f"{fmt(bm, 0)} g"
        new = "unknown" if am is None else f"{fmt(am, 0)} g"
        out.append(f"Mass updated: {old} -> {new}")
    added = set(after["spots"]) - set(before["spots"])
    gone = set(before["spots"]) - set(after["spots"])
    if added or gone:
        bits = []
        if added:
            bits.append(f"{len(added)} added")
        if gone:
            bits.append(f"{len(gone)} resolved")
        out.append(f"{len(added) + len(gone)} weak spot(s) re-evaluated ({', '.join(bits)})")
    for pid, role in sorted(after["roles"].items()):
        if before["roles"].get(pid) != role:
            out.append(f"{pid} role: {before['roles'].get(pid, '-')} -> {role}")
    for pid, proc in sorted(after["process"].items()):
        if before["process"].get(pid) != proc:
            out.append(f"{pid} process: {before['process'].get(pid, '-')} -> {proc}")
    bt, at = before["tipping"], after["tipping"]
    if bt != at:
        old = "unknown" if bt is None else f"{fmt(bt, 1)} deg"
        new = "unknown" if at is None else f"{fmt(at, 1)} deg"
        out.append(f"Tipping angle: {old} -> {new}")
    if after["conflicts"] > before["conflicts"]:
        out.append("A conflict between the answer and the geometry was recorded")
    if not out and before["summary"] != after["summary"]:
        out.append("The design summary was updated")
    return out


class Session:
    """Holds the analysis, the designer's answers (in order, for undo) and the cheap re-analysis."""

    def __init__(
        self,
        analysis: Analysis,
        opts: AnalyzeOptions,
        context: dict[str, Any] | None = None,
        limit: int = SESSION_LIMIT,
    ) -> None:
        self.analysis = analysis
        self.opts = opts
        self.context = context if context is not None else dict(opts.extra.get("context") or {})
        self.limit = limit
        pipe = analysis.understanding_pipeline
        if pipe is None:
            pipe = UnderstandingPipeline(analysis, opts)
            analysis.understanding_pipeline = pipe
        self.pipe: UnderstandingPipeline = pipe
        self.answers: dict[str, dict[str, Any]] = answers_from_context(opts.extra.get("answers"))
        self.order: list[str] = []
        self.removed: set[str] = set()  # answers undone since the last save
        self.asked_this_session = 0
        self.understanding: Understanding = self.pipe.finish(self.answers, self.context)

    # ------------------------------------------------------------------ reading
    @property
    def questions(self) -> list[Question]:
        return self.understanding.questions

    def next_question(self) -> Question | None:
        if self.asked_this_session >= self.limit:
            return None
        return self.questions[0] if self.questions else None

    def all_questions(self) -> list[Question]:
        """Every question the design raises, answered or not (not capped)."""
        from stepscribe.understanding.kinematics import wheel_joint_ids
        from stepscribe.understanding.questions import generate_questions

        ctx = self.pipe.ctx
        wheels = wheel_joint_ids(self.pipe.kin, ctx) if ctx is not None else None
        return generate_questions(self.analysis.report, None, self.context, wheels, limit=None)

    def remaining(self) -> int:
        return min(len(self.questions), max(0, self.limit - self.asked_this_session))

    # ------------------------------------------------------------------ writing
    def answer(
        self, question_id: str, status: str, value: Any = None, note: str | None = None
    ) -> Change:
        """Record an answer ("answered", "skipped" or "not_sure"), re-run, describe the change."""
        if status not in ("answered", "skipped", "not_sure"):
            raise ValueError(f"unknown status {status!r}")
        before = _metrics(self.understanding, self.analysis)
        t0 = time.time()
        self.answers[question_id] = {
            "status": status,
            "value": value,
            "note": note,
            "answered_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        if question_id in self.order:
            self.order.remove(question_id)
        self.removed.discard(question_id)
        self.order.append(question_id)
        self.asked_this_session += 1
        self.understanding = self.pipe.finish(self.answers, self.context)
        return self._change(before, t0)

    def undo(self) -> Change | None:
        """Remove the most recent answer and rebuild."""
        if not self.order:
            return None
        before = _metrics(self.understanding, self.analysis)
        t0 = time.time()
        undone = self.order.pop()
        self.answers.pop(undone, None)
        self.removed.add(undone)
        self.asked_this_session = max(0, self.asked_this_session - 1)
        self.understanding = self.pipe.finish(self.answers, self.context)
        return self._change(before, t0, undone=True)

    def _change(self, before: dict[str, Any], t0: float, undone: bool = False) -> Change:
        after = _metrics(self.understanding, self.analysis)
        details = describe_change(before, after)
        head = "Undone. " if undone else ""
        text = head + ("; ".join(details) if details else "Noted; nothing else changed.")
        return Change(text, time.time() - t0, details)

    # ------------------------------------------------------------------ saving
    def save(self, path: str | Path) -> int:
        """Write this session's answers into design_context.yaml (an ``answers:`` section).

        Only the answers given (or undone) in this session are written, merged with the file's
        current content, so another writer's answers are kept.
        """
        removed = self.removed - set(self.order)
        texts = {q.id: q.text for q in self.questions}
        n = save_answers(path, self.answers, texts, only=set(self.order), remove=removed)
        self.removed -= removed
        return n


@contextmanager
def _locked(path: Path, timeout_s: float = 15.0) -> Iterator[None]:
    """Cross-process lock for one context file (a ``.lock`` file made with O_EXCL)."""
    lock = path.with_name(path.name + ".lock")
    deadline = time.time() + timeout_s
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            try:  # a writer that died leaves its lock behind
                if time.time() - lock.stat().st_mtime > 60:
                    lock.unlink(missing_ok=True)
                    continue
            except OSError:
                continue
            if time.time() > deadline:
                raise TimeoutError(f"{path} is locked by another writer") from None
            time.sleep(0.02)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def read_answers(path: str | Path) -> dict[str, dict[str, Any]]:
    """The answers stored in a context file (empty stubs ignored)."""
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        raw = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("answers")
    except yaml.YAMLError:
        return {}
    return answers_from_context(raw if isinstance(raw, dict) else None)


def save_answers(
    path: str | Path,
    answers: dict[str, dict[str, Any]],
    texts: dict[str, str] | None = None,
    only: Iterable[str] | None = None,
    remove: Iterable[str] = (),
) -> int:
    """Update the ``answers:`` section of a context file; everything before it is kept.

    Safe with several writers (UI, MCP, CLI): the file is locked, re-read and merged just before
    writing, and replaced atomically. ``only`` limits the write to the IDs this writer changed, so
    answers another writer saved meanwhile are never overwritten with stale values; ``remove``
    deletes undone answers.
    """
    from stepscribe.pack.design_context import CONTEXT_TEMPLATE

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mine = {k: v for k, v in answers.items() if only is None or k in set(only)}
    with _locked(p):
        text = p.read_text(encoding="utf-8") if p.is_file() else CONTEXT_TEMPLATE
        lines = text.split("\n")
        cut = next((i for i, ln in enumerate(lines) if ln.startswith("answers:")), len(lines))
        head = "\n".join(lines[:cut]).rstrip("\n")
        existing: dict[str, Any] = {}
        if cut < len(lines):
            try:
                existing = (yaml.safe_load("\n".join(lines[cut:])) or {}).get("answers") or {}
            except yaml.YAMLError:
                existing = {}
        merged = {**existing, **mine}
        for qid in remove:
            merged.pop(qid, None)
        out = [
            head,
            "",
            "answers:  # given by the designer; status is answered | skipped | not_sure",
        ]
        for qid in sorted(merged):
            row = merged[qid] or {}
            note = (texts or {}).get(qid)
            out.append(f"  {qid}:" + (f"  # {note}" if note else ""))
            for key in ("status", "value", "note", "answered_at"):
                if row.get(key) not in (None, ""):
                    dumped = yaml.safe_dump(
                        {key: row[key]}, default_flow_style=True, allow_unicode=True
                    )
                    out.append("    " + dumped.strip().strip("{}").strip())
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        tmp.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8", newline="\n")
        os.replace(tmp, p)
    return len(mine)
