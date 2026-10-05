"""Phase 11: LLM adapter against a mocked HTTP server (no real model, no network)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stepscribe.cli import app
from stepscribe.llm.adapter import LLMConfig, LLMConfigError, chat
from stepscribe.llm.evaluate import Question, accuracy, build_questions, score

SEEN: list[dict] = []  # type: ignore[type-arg]


class Mock(BaseHTTPRequestHandler):
    def log_message(self, *a) -> None:  # type: ignore[no-untyped-def]
        return

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        last = body["messages"][-1]
        text = last["content"] if isinstance(last["content"], str) else last["content"][0]["text"]
        answer = "1" if "How many unique parts" in text else "mock answer"
        if self.path.endswith("/api/chat"):
            payload = {"message": {"role": "assistant", "content": answer}}
        else:
            payload = {"choices": [{"message": {"role": "assistant", "content": answer}}]}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def mock_url():  # type: ignore[no-untyped-def]
    SEEN.clear()
    httpd = HTTPServer(("127.0.0.1", 0), Mock)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_config_needs_explicit_endpoint_and_has_no_default() -> None:
    with pytest.raises(LLMConfigError, match="no default endpoint"):
        LLMConfig.from_env({})
    cfg = LLMConfig.from_env(
        {"STEPSCRIBE_LLM_BASE_URL": "http://localhost:11434", "STEPSCRIBE_LLM_MODEL": "m"}
    )
    assert cfg.provider == "ollama" and cfg.api_key is None
    other = LLMConfig.from_env(
        {"STEPSCRIBE_LLM_BASE_URL": "https://x/v1", "STEPSCRIBE_LLM_MODEL": "m"}
    )
    assert other.provider == "openai"
    with pytest.raises(LLMConfigError):
        LLMConfig.from_env(
            {
                "STEPSCRIBE_LLM_BASE_URL": "http://x",
                "STEPSCRIBE_LLM_MODEL": "m",
                "STEPSCRIBE_LLM_PROVIDER": "nope",
            }
        )


def test_ollama_request_shape(mock_url: str, tmp_path: Path) -> None:
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNGdata")
    cfg = LLMConfig(mock_url, "llama", None, "ollama", True)
    assert chat(cfg, "hello", [img]) == "mock answer"
    req = SEEN[-1]
    assert req["path"] == "/api/chat" and req["body"]["stream"] is False
    assert req["body"]["messages"][-1]["images"] and req["body"]["options"]["temperature"] == 0


def test_openai_request_shape_auth_and_vision(mock_url: str, tmp_path: Path) -> None:
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNGdata")
    cfg = LLMConfig(mock_url + "/v1", "gpt", "secret", "openai", True)
    assert chat(cfg, "hello", [img]) == "mock answer"
    req = SEEN[-1]
    assert req["path"] == "/v1/chat/completions" and req["auth"] == "Bearer secret"
    parts = req["body"]["messages"][-1]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    chat(LLMConfig(mock_url + "/v1", "gpt", None, "openai", False), "hello", [img])
    assert isinstance(SEEN[-1]["body"]["messages"][-1]["content"], str), "no images unless vision"


def test_scoring() -> None:
    num = Question("q", "3.4", True, 0.15)
    assert score("3.4", num) and score("about 3.45 mm", num)
    assert not score("3.9", num) and not score("none", num)
    word = Question("q", "plate", False)
    assert score("It is a Plate.", word) and not score("tube", word)
    assert accuracy([]) == 0.0


def test_questions_are_derived_from_report(analyses) -> None:  # type: ignore[no-untyped-def]
    qs = build_questions(analyses("two_plates_assembly").report)
    by_text = {q.text.split("?")[0]: q.expected for q in qs}
    assert by_text["How many unique parts does the design have"] == "1"
    assert by_text["How many part instances are in the assembly"] == "2"
    assert by_text["How many fastener joints were found"] == "4"


def test_cli_ask_and_eval_through_mock(mock_url: str, step_files, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("STEPSCRIBE_LLM_BASE_URL", mock_url)
    monkeypatch.setenv("STEPSCRIBE_LLM_MODEL", "mock")
    monkeypatch.delenv("STEPSCRIBE_LLM_VISION", raising=False)
    r = CliRunner().invoke(app, ["ask", str(step_files["plate_4xM3"]), "How many holes?"])
    assert r.exit_code == 0 and "mock answer" in r.output
    sent = SEEN[-1]["body"]["messages"][-1]["content"]
    assert "Likely:" in sent and "Question: How many holes?" in sent
    r = CliRunner().invoke(app, ["eval", str(step_files["plate_4xM3"]), "--limit", "3"])
    assert r.exit_code == 0 and "Accuracy:" in r.output


def test_cli_without_config_explains_and_exits(step_files, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    for k in ("STEPSCRIBE_LLM_BASE_URL", "STEPSCRIBE_LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    r = CliRunner().invoke(app, ["ask", str(step_files["cube_10"]), "x"])
    assert r.exit_code == 2 and "no default endpoint" in " ".join(r.output.split())
