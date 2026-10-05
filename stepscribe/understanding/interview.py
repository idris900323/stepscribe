# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Terminal interview: ask the designer the most valuable question, apply the answer, repeat.

Keys: a number picks an option, ``y``/``n`` answer yes/no, Enter accepts the tool's guess,
``s`` skips, ``?`` means "not sure", ``b`` goes back one answer, ``d`` is done.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console

from stepscribe.models.schema import Question
from stepscribe.understanding.session import Session

HELP = "number = choose | Enter = accept the guess | s = skip | ? = not sure | b = back | d = done"


def parse_answer(q: Question, raw: str) -> tuple[str, Any] | str | None:
    """('answered', value) for an answer; 'skip', 'unsure', 'back', 'done'; None if unusable."""
    text = raw.strip()
    low = text.lower()
    if low == "s":
        return "skip"
    if low == "?":
        return "unsure"
    if low == "b":
        return "back"
    if low == "d":
        return "done"
    if text == "":
        return ("answered", q.default_guess) if q.default_guess else None
    options = q.options or []
    if q.answer_type in ("single_choice", "yes_no") and options:
        if text.isdigit() and 1 <= int(text) <= len(options):
            return ("answered", options[int(text) - 1].value)
        match = next((o for o in options if o.value == text or o.label.lower() == low), None)
        return ("answered", match.value) if match else None
    if q.answer_type == "multi_choice" and options:
        picks = [t.strip() for t in text.split(",") if t.strip()]
        values = []
        for t in picks:
            if t.isdigit() and 1 <= int(t) <= len(options):
                values.append(options[int(t) - 1].value)
            else:
                return None
        return ("answered", values) if values else None
    if q.answer_type == "yes_no":
        if low in ("y", "yes"):
            return ("answered", "yes")
        if low in ("n", "no"):
            return ("answered", "no")
        return None
    if q.answer_type in ("number", "number_with_unit"):
        try:
            return ("answered", float(text.split()[0]))
        except (ValueError, IndexError):
            return None
    return ("answered", text)


def show_question(console: Console, q: Question, n: int, remaining: int) -> None:
    console.print(f"\nQuestion {n} - about {remaining} more useful one(s)", markup=False)
    console.print(q.text, markup=False)
    console.print(f"  why it matters: {q.why}", markup=False)
    for i, o in enumerate(q.options or [], 1):
        star = "  (my guess)" if q.default_guess == o.value else ""
        console.print(f"  {i}. {o.label}{star}", markup=False)
    if q.unit:
        console.print(f"  (give a number in {q.unit})", markup=False)
    if q.default_guess and not q.options:
        console.print(f"  my guess: {q.default_guess}", markup=False)
    console.print(f"  [{HELP}]", markup=False)


def run_interview(
    session: Session,
    console: Console,
    ask: Callable[[str], str] = input,
    context_file: Path | None = None,
) -> int:
    """Drive *session* until the queue is empty, the limit is reached or the user is done.

    Returns the number of answers recorded (answered, skipped or not sure).
    """
    count = 0
    while True:
        q = session.next_question()
        if q is None:
            console.print("\nNo more questions worth asking.")
            break
        show_question(console, q, session.asked_this_session + 1, session.remaining())
        try:
            raw = ask("> ")
        except EOFError:
            break
        parsed = parse_answer(q, raw)
        if parsed is None:
            console.print("  I did not understand that; try again.", markup=False)
            continue
        if parsed == "done":
            break
        if parsed == "back":
            change = session.undo()
            console.print(
                "  " + (change.text if change else "Nothing to go back to."), markup=False
            )
            count = max(0, count - 1) if change else count
            continue
        if parsed == "skip":
            change = session.answer(q.id, "skipped")
        elif parsed == "unsure":
            change = session.answer(q.id, "not_sure")
        else:
            assert isinstance(parsed, tuple)
            change = session.answer(q.id, "answered", parsed[1])
        count += 1
        console.print(f"  {change.text}", markup=False)
    if context_file is not None and session.answers:
        session.save(context_file)
        console.print(f"\nAnswers saved to {context_file}.")
    return count
