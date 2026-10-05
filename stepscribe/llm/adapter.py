# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Optional, opt-in LLM adapter (``pip install stepscribe[llm]``).

Speaks Ollama's native API and any OpenAI-compatible chat endpoint over plain HTTP (httpx only,
no vendor SDKs). There is no default endpoint and no key is ever written to disk: everything
comes from the environment.

    STEPSCRIBE_LLM_BASE_URL   e.g. http://localhost:11434  or  https://api.example.com/v1
    STEPSCRIBE_LLM_MODEL      e.g. llama3.1:8b
    STEPSCRIBE_LLM_API_KEY    optional bearer token
    STEPSCRIBE_LLM_PROVIDER   optional: ollama | openai (default: guessed from the URL)
    STEPSCRIBE_LLM_VISION     optional: 1 to also send images
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an experienced mechanical engineer. You are given a context pack describing a CAD "
    "design: measured facts are plain, inferences start with 'Likely:' and carry a confidence. "
    "Refer to parts, holes and joints by their IDs. If the pack does not contain what you need, "
    "say what is missing instead of inventing numbers."
)


class LLMConfigError(Exception):
    """Missing or invalid configuration."""


@dataclass(frozen=True)
class LLMConfig:
    """Endpoint settings read from the environment."""

    base_url: str
    model: str
    api_key: str | None
    provider: str  # ollama | openai
    vision: bool
    timeout_s: float = 600.0

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> LLMConfig:
        """Build a config; raises :class:`LLMConfigError` with instructions if incomplete."""
        e = dict(os.environ) if env is None else env
        base = e.get("STEPSCRIBE_LLM_BASE_URL", "").strip().rstrip("/")
        model = e.get("STEPSCRIBE_LLM_MODEL", "").strip()
        if not base or not model:
            raise LLMConfigError(
                "Set STEPSCRIBE_LLM_BASE_URL and STEPSCRIBE_LLM_MODEL (and STEPSCRIBE_LLM_API_KEY "
                "if the endpoint needs one). There is no default endpoint."
            )
        provider = e.get("STEPSCRIBE_LLM_PROVIDER", "").strip().lower()
        if not provider:
            provider = "ollama" if ":11434" in base or "ollama" in base.lower() else "openai"
        if provider not in ("ollama", "openai"):
            raise LLMConfigError("STEPSCRIBE_LLM_PROVIDER must be 'ollama' or 'openai'")
        return cls(
            base_url=base,
            model=model,
            api_key=e.get("STEPSCRIBE_LLM_API_KEY") or None,
            provider=provider,
            vision=e.get("STEPSCRIBE_LLM_VISION", "").strip().lower() in ("1", "true", "yes"),
        )


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def chat(cfg: LLMConfig, user_text: str, images: list[Path] | None = None) -> str:
    """Send one system + user turn and return the model's text answer."""
    import httpx

    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    pics = [p for p in (images or []) if p.is_file()] if cfg.vision else []
    if cfg.provider == "ollama":
        user: dict[str, object] = {"role": "user", "content": user_text}
        if pics:
            user["images"] = [_b64(p) for p in pics]
        url = f"{cfg.base_url}/api/chat"
        body: dict[str, object] = {
            "model": cfg.model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, user],
        }
    else:
        content: object = user_text
        if pics:
            content = [{"type": "text", "text": user_text}] + [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(p)}"}}
                for p in pics
            ]
        url = cfg.base_url + (
            "/chat/completions" if cfg.base_url.endswith("/v1") else "/v1/chat/completions"
        )
        body = {
            "model": cfg.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        }
    resp = httpx.post(url, json=body, headers=headers, timeout=cfg.timeout_s)
    resp.raise_for_status()
    data = resp.json()
    if cfg.provider == "ollama":
        return str(data["message"]["content"]).strip()
    return str(data["choices"][0]["message"]["content"]).strip()


def build_prompt(pack_text: str, task: str) -> str:
    """The user turn: the context pack followed by the task."""
    return f"{pack_text}\n\n---\n\n{task}"
