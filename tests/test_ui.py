"""The local web UI: upload, progress with ETA, results, files and zip."""

from __future__ import annotations

import io
import json
import threading
import time
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from stepscribe.ui.server import App, make_handler


@pytest.fixture
def server(tmp_path: Path):  # type: ignore[no-untyped-def]
    app = App(tmp_path, tmp_path)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    app.shutdown()


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def wait_done(base: str, job: str, timeout: float = 240.0) -> dict:  # type: ignore[type-arg]
    seen: list[dict] = []  # type: ignore[type-arg]
    end = time.time() + timeout
    while time.time() < end:
        s = json.loads(get(f"{base}/api/status?id={job}"))
        seen.append(s)
        if s["done"]:
            s["history"] = seen
            return s
        time.sleep(0.3)
    raise AssertionError("job did not finish")


def test_page_and_upload_flow(server: str, step_files) -> None:  # type: ignore[no-untyped-def]
    assert b"stepscribe" in get(server + "/") and b"{{VERSION}}" not in get(server + "/")
    data = step_files["two_plates_assembly"].read_bytes()
    req = urllib.request.Request(
        server + "/api/upload?name=two_plates.step&material=alu&images=1", data=data, method="POST"
    )
    job = json.loads(urllib.request.urlopen(req, timeout=30).read())["id"]
    s = wait_done(server, job)
    assert s["error"] is None, s["error"]
    fractions = [h["fraction"] for h in s["history"]]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0, (
        "progress must never go backwards"
    )
    assert any(h["eta_s"] is not None for h in s["history"]) or len(s["history"]) < 3
    r = s["result"]
    assert r["instances"] == 2 and len(r["relations"]) == 2 and r["total_mass_g"] > 0
    assert "TopPlate is bolted to BasePlate" in " ".join(r["relations"])
    img = r["images"][0]
    assert get(f"{server}/files/{job}/{img}")[:4] == b"\x89PNG"
    z = zipfile.ZipFile(io.BytesIO(get(f"{server}/api/zip?id={job}")))
    assert any(n.endswith("context_pack.md") for n in z.namelist())


def test_bad_path_and_traversal_are_rejected(server: str) -> None:
    req = urllib.request.Request(
        server + "/api/start_path", data=json.dumps({"path": "nope.step"}).encode(), method="POST"
    )
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e2:
        get(server + "/files/unknown/../../etc/passwd")
    assert e2.value.code == 404


def post(url: str, body: dict | None = None) -> dict:  # type: ignore[type-arg]
    req = urllib.request.Request(url, data=json.dumps(body or {}).encode(), method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())  # type: ignore[no-any-return]


def test_interview_over_http(server: str, step_files) -> None:  # type: ignore[no-untyped-def]
    """Ask me questions: next question, answer, change line, highlight image, undo, download."""
    data = step_files["two_link_arm"].read_bytes()
    req = urllib.request.Request(
        server + "/api/upload?name=arm.step&images=0", data=data, method="POST"
    )
    job = json.loads(urllib.request.urlopen(req, timeout=30).read())["id"]
    s = wait_done(server, job)
    assert s["error"] is None, s["error"]
    u = s["result"]["understanding"]
    assert u and u["topology"] == "serial chain" and len(u["joints"]) == 2 and u["questions"]
    page = get(server + "/").decode("utf-8")
    assert "Ask me questions" in page and "http://" not in page.replace("http://127", "")
    state = json.loads(get(f"{server}/api/questions/next?id={job}"))
    q = state["question"]
    assert q["kind"] in ("material", "process", "purpose") and state["remaining"] >= 1
    assert state["understanding"]["summary"].startswith("Likely a 2-DOF serial arm")
    mat = next(x for x in state["understanding"]["questions"] if x["kind"] == "material")
    done = post(
        f"{server}/api/answers?id={job}",
        {"question_id": mat["id"], "status": "answered", "value": "pla"},
    )
    assert "Mass updated" in done["change"]["text"] and done["change"]["seconds"] < 1.0
    assert all(x["id"] != mat["id"] for x in done["understanding"]["questions"])
    assert any(a["status"] == "answered" for a in done["understanding"]["answers"])
    img = get(f"{server}/api/render?id={job}&highlight=KJ1")
    assert img[:4] == b"\x89PNG"
    back = post(f"{server}/api/answers/undo?id={job}")
    assert back["change"]["text"].startswith("Undone") and any(
        x["id"] == mat["id"] for x in back["understanding"]["questions"]
    )
    post(
        f"{server}/api/answers?id={job}",
        {"question_id": mat["id"], "status": "answered", "value": "alu"},
    )
    preview = json.loads(get(f"{server}/api/pack/preview?id={job}"))
    assert "Confirmed by designer" in preview["context_pack"]
    z = zipfile.ZipFile(io.BytesIO(get(f"{server}/api/pack/download?id={job}")))
    names = z.namelist()
    assert any(n.endswith("05_understanding.md") for n in names)


def test_favicon_is_served_locally(server: str) -> None:
    assert b"<svg" in get(server + "/favicon.svg")
