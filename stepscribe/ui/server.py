# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Local web UI: upload a STEP file, watch progress with an ETA, inspect the context pack.

Standard library only. Binds to 127.0.0.1 and runs each analysis in a separate process so a
heavy file can never freeze the page.
"""

from __future__ import annotations

import io
import json
import multiprocessing as mp
import shutil
import tempfile
import threading
import time
import uuid
import webbrowser
import zipfile
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from stepscribe import __version__
from stepscribe.headless import require_interactive
from stepscribe.io.discovery import find_step_files
from stepscribe.procutil import kill_tree
from stepscribe.ui.worker import run_job, summarise_pack

INDEX = Path(__file__).with_name("index.html")
FAVICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="6" fill="#1f6feb"/>'
    '<path d="M9 22V10h8a4 4 0 010 8h-4" fill="none" stroke="#fff" stroke-width="3" '
    'stroke-linecap="round"/></svg>'
)
MIME = {
    ".png": "image/png",
    ".md": "text/plain; charset=utf-8",
    ".json": "application/json",
    ".html": "text/html; charset=utf-8",
}


@dataclass
class JobState:
    """What the page can poll for."""

    id: str
    name: str
    started: float = field(default_factory=time.time)
    stage: str = "queued"
    message: str = "starting"
    fraction: float = 0.0
    eta_s: float | None = None
    elapsed_s: float = 0.0
    done: bool = False
    error: str | None = None
    pack: str | None = None
    result: dict[str, Any] | None = None
    process: Any = None
    cmd_q: Any = None  # commands to the interview session in the worker
    reply_q: Any = None
    call_lock: Any = field(default_factory=threading.Lock)


class App:
    """Shared server state: work directory and jobs."""

    def __init__(self, workdir: Path, sample_dir: Path | None) -> None:
        self.workdir = workdir
        self.sample_dir = sample_dir
        self.jobs: dict[str, JobState] = {}
        self.lock = threading.Lock()

    def start(self, path: Path, display_name: str, params: dict[str, Any]) -> JobState:
        job_id = uuid.uuid4().hex[:10]
        out = self.workdir / job_id / "out"
        out.mkdir(parents=True, exist_ok=True)
        state = JobState(job_id, display_name)
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        state.cmd_q, state.reply_q = ctx.Queue(), ctx.Queue()
        job = {**params, "path": str(path), "out": str(out)}
        # not a daemon: the analysis starts its own worker processes for the parts
        proc = ctx.Process(
            target=run_job, args=(job, queue, state.cmd_q, state.reply_q), daemon=False
        )
        state.process = proc
        with self.lock:
            self.jobs[job_id] = state
        proc.start()
        threading.Thread(target=self._drain, args=(state, queue, proc), daemon=True).start()
        return state

    def _drain(self, state: JobState, queue: Any, proc: Any) -> None:
        while not state.done:
            try:
                msg = queue.get(timeout=0.5)
            except Exception:  # noqa: BLE001 - queue.Empty
                if not proc.is_alive() and queue.empty():
                    state.error = state.error or "the analysis process stopped unexpectedly"
                    state.done = True
                state.elapsed_s = time.time() - state.started
                continue
            kind = msg["type"]
            if kind == "progress":
                state.stage, state.message = msg["stage"], msg["message"]
                state.fraction, state.eta_s = msg["fraction"], msg["eta_s"]
                state.elapsed_s = msg["elapsed_s"]
            elif kind == "done":
                state.pack = msg["pack"]
                state.result = summarise_pack(Path(msg["pack"]))
                state.fraction, state.eta_s, state.stage = 1.0, 0.0, "done"
                state.message = f"finished in {msg['seconds']} s"
                state.done = True
            elif kind == "error":
                state.error = msg["message"]
                state.done = True
        proc.join(timeout=2)

    def call(self, job_id: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
        """Send one command to a job's interview session and wait for its reply."""
        s = self.jobs.get(job_id)
        if s is None or not s.done or s.error or s.cmd_q is None:
            return {"error": "the analysis is not finished"}
        if s.process is None or not s.process.is_alive():
            return {"error": "the interview session has ended; run the analysis again"}
        with s.call_lock:
            s.cmd_q.put(payload)
            try:
                return dict(s.reply_q.get(timeout=timeout))
            except Exception:  # noqa: BLE001 - queue.Empty
                return {"error": "the session did not answer in time"}

    def shutdown(self) -> None:
        for s in self.jobs.values():
            if s.cmd_q is not None:
                s.cmd_q.put({"cmd": "quit"})
            if s.process is not None and s.process.is_alive():
                s.process.join(timeout=3)
                if s.process.is_alive():
                    kill_tree(s.process)

    def cancel(self, job_id: str) -> None:
        state = self.jobs.get(job_id)
        if state and state.process is not None and state.process.is_alive():
            kill_tree(state.process)
            state.error, state.done = "cancelled", True


def _json(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    """Request handler class bound to *app*."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        # ------------------------------------------------------------ GET
        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            q = parse_qs(url.query)
            if url.path == "/":
                body = (
                    INDEX.read_text(encoding="utf-8")
                    .replace("{{VERSION}}", __version__)
                    .encode("utf-8")
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path in ("/favicon.svg", "/favicon.ico"):
                body = FAVICON.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "image/svg+xml")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/samples":
                files = find_step_files(app.sample_dir) if app.sample_dir else []
                _json(
                    self,
                    [
                        {"path": str(f), "name": f.name, "mb": round(f.stat().st_size / 1e6, 1)}
                        for f in files
                    ],
                )
            elif url.path == "/api/status":
                self._status(q.get("id", [""])[0])
            elif url.path == "/api/zip":
                self._zip(q.get("id", [""])[0])
            elif url.path == "/api/questions/next":
                _json(self, app.call(q.get("id", [""])[0], {"cmd": "state"}))
            elif url.path == "/api/pack/preview":
                _json(self, app.call(q.get("id", [""])[0], {"cmd": "preview"}))
            elif url.path == "/api/pack/download":
                jid = q.get("id", [""])[0]
                res = app.call(jid, {"cmd": "save"})
                if "error" in res:
                    _json(self, res, 400)
                else:
                    self._zip(jid)
            elif url.path == "/api/render":
                self._render(q)
            elif url.path == "/api/report":
                s = app.jobs.get(q.get("id", [""])[0])
                _json(self, (s.result if s and s.result else {"error": "unknown job"}))
            elif url.path.startswith("/files/"):
                self._file(unquote(url.path[len("/files/") :]))
            else:
                self.send_error(404)

        def _status(self, job_id: str) -> None:
            s = app.jobs.get(job_id)
            if s is None:
                _json(self, {"error": "unknown job"}, 404)
                return
            _json(
                self,
                {
                    "id": s.id,
                    "name": s.name,
                    "stage": s.stage,
                    "message": s.message,
                    "fraction": s.fraction,
                    "eta_s": s.eta_s,
                    "elapsed_s": s.elapsed_s or time.time() - s.started,
                    "done": s.done,
                    "error": s.error,
                    "result": s.result,
                    "job": s.id,
                },
            )

        def _render(self, q: dict[str, list[str]]) -> None:
            jid = q.get("id", [""])[0]
            refs = [r for r in q.get("highlight", [""])[0].split(",") if r]
            res = app.call(jid, {"cmd": "render", "refs": refs, "view": q.get("view", ["iso"])[0]})
            if "path" not in res:
                _json(self, res, 400)
                return
            data = Path(res["path"]).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _zip(self, job_id: str) -> None:
            s = app.jobs.get(job_id)
            if s is None or not s.pack:
                self.send_error(404)
                return
            buf = io.BytesIO()
            root = Path(s.pack)
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in sorted(root.rglob("*")):
                    if f.is_file():
                        zf.write(f, f"{root.name}/{f.relative_to(root).as_posix()}")
            data = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{root.name}.zip"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _file(self, rel: str) -> None:
            job_id, _, sub = rel.partition("/")
            s = app.jobs.get(job_id)
            if s is None or not s.pack:
                self.send_error(404)
                return
            root = Path(s.pack).resolve()
            target = (root / sub).resolve()
            if root not in target.parents or not target.is_file():
                self.send_error(404)
                return
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", MIME.get(target.suffix, "application/octet-stream"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        # ------------------------------------------------------------ POST
        def do_POST(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            length = int(self.headers.get("Content-Length", "0"))
            if url.path == "/api/upload":
                name = Path(unquote(q.get("name", "upload.step"))).name or "upload.step"
                dest_dir = app.workdir / "uploads" / uuid.uuid4().hex[:8]
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / name
                remaining = length
                with dest.open("wb") as fh:
                    while remaining > 0:
                        chunk = self.rfile.read(min(1 << 20, remaining))
                        if not chunk:
                            break
                        fh.write(chunk)
                        remaining -= len(chunk)
                state = app.start(dest, name, _params(q))
                _json(self, {"id": state.id})
            elif url.path == "/api/start_path":
                body = json.loads(self.rfile.read(length) or b"{}")
                p = Path(body.get("path", ""))
                if not p.is_file():
                    _json(self, {"error": f"file not found: {p}"}, 400)
                    return
                state = app.start(p, p.name, _params(body))
                _json(self, {"id": state.id})
            elif url.path == "/api/cancel":
                app.cancel(q.get("id", ""))
                _json(self, {"ok": True})
            elif url.path == "/api/answers":
                body = json.loads(self.rfile.read(length) or b"{}")
                res = app.call(
                    q.get("id", ""),
                    {
                        "cmd": "answer",
                        "id": body.get("question_id", ""),
                        "status": body.get("status", "answered"),
                        "value": body.get("value"),
                        "note": body.get("note"),
                    },
                )
                _json(self, res, 400 if "error" in res else 200)
            elif url.path == "/api/answers/undo":
                res = app.call(q.get("id", ""), {"cmd": "undo"})
                _json(self, res, 400 if "error" in res else 200)
            else:
                self.send_error(404)

    return Handler


def _params(src: dict[str, Any]) -> dict[str, Any]:
    def flag(k: str, default: bool) -> bool:
        v = src.get(k)
        return default if v is None else str(v).lower() in ("1", "true", "yes", "on")

    density = src.get("density")
    return {
        "material": (src.get("material") or "").strip() or None,
        "density": float(density) if density not in (None, "") else None,
        "up": src.get("up") or "+Z",
        "front": src.get("front") or "-Y",
        "images": flag("images", True),
        "interference": flag("interference", True),
    }


def serve(
    port: int = 8765,
    open_browser: bool = True,
    sample_dir: Path | None = None,
    preload: Path | None = None,
) -> None:
    """Run the UI until interrupted."""
    require_interactive("start the local server")
    workdir = Path(tempfile.mkdtemp(prefix="stepscribe_ui_"))
    app = App(workdir, sample_dir)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    if preload is not None:
        state = app.start(preload, preload.name, _params({}))
        url += f"?job={state.id}"
    print(f"stepscribe UI on {url}  (Ctrl+C to stop)")
    if open_browser:
        require_interactive("open a browser")
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.shutdown()
        shutil.rmtree(workdir, ignore_errors=True)
