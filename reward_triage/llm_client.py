"""Provider-agnostic LLM client for reward generation (keys never enter workers)."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from reward_triage.local_rewards import sample_code


@dataclass
class LLMResponse:
    text: str
    model: str
    meta: dict[str, Any]


class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, *, temperature: float = 0.7, n: int = 1) -> list[LLMResponse]:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Deterministic mock for dry runs and tests (no network)."""

    def __init__(self, model: str = "mock-llm") -> None:
        self.model = model

    def complete(self, prompt: str, *, temperature: float = 0.7, n: int = 1) -> list[LLMResponse]:
        out: list[LLMResponse] = []
        for i in range(n):
            digest = hashlib.sha256(f"{prompt}|{temperature}|{i}".encode()).hexdigest()[:8]
            # Sparse-but-valid shaped reward; Tier0 must not hard-reject.
            w_prog = 0.5 + (int(digest[:2], 16) / 255.0)
            w_goal = 1.0 + (int(digest[2:4], 16) / 255.0)
            w_col = 1.0 + (int(digest[4:6], 16) / 255.0)
            code = f'''
def compute_reward(transition):
    progress = float(transition.get("progress", 0.0))
    success = 1.0 if transition.get("success") else 0.0
    collision = 1.0 if (transition.get("collision") or transition.get("out_of_bounds")) else 0.0
    total = {w_prog:.4f} * progress + {w_goal:.4f} * success - {w_col:.4f} * collision - 0.01
    return total, {{"progress": progress, "goal": success, "collision": collision, "time": 1.0}}
'''.strip()
            out.append(
                LLMResponse(
                    text=code,
                    model=self.model,
                    meta={"temperature": temperature, "mock": True, "digest": digest},
                )
            )
        return out


class LocalCorpusLLMClient(LLMClient):
    """Hand-written reward families — stand-in for Grok when no API key."""

    def __init__(self, model: str = "local-corpus") -> None:
        self.model = model

    def complete(self, prompt: str, *, temperature: float = 0.7, n: int = 1) -> list[LLMResponse]:
        out: list[LLMResponse] = []
        for i in range(n):
            code, meta = sample_code(f"{prompt}|T={temperature}", index=i)
            out.append(
                LLMResponse(
                    text=code,
                    model=self.model,
                    meta={"temperature": temperature, **meta},
                )
            )
        return out


class XAIClient(LLMClient):
    """xAI Chat Completions client. API key from XAI_API_KEY only."""

    def __init__(
        self,
        model: str = "grok-4.3",
        *,
        api_key: str | None = None,
        base_url: str = "https://api.x.ai/v1/chat/completions",
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("XAI_API_KEY", "")
        self.base_url = base_url
        self.timeout_s = timeout_s
        if not self.api_key:
            raise RuntimeError("XAI_API_KEY is not set (generator process only)")

    def complete(self, prompt: str, *, temperature: float = 0.7, n: int = 1) -> list[LLMResponse]:
        # One completion request; if n>1 ask the model for n variants via prompt.
        body = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You write Python reward functions. Reply with ONLY a "
                        "compute_reward(transition) function that returns "
                        "(float, dict). No markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        out: list[LLMResponse] = []
        for i in range(n):
            req = urllib.request.Request(
                self.base_url,
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"xAI HTTP {exc.code}: {detail}") from exc
            text = payload["choices"][0]["message"]["content"]
            out.append(
                LLMResponse(
                    text=_strip_code_fence(text),
                    model=str(payload.get("model", self.model)),
                    meta={"temperature": temperature, "index": i, "raw_id": payload.get("id")},
                )
            )
        return out


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def make_llm_client(provider: str, model: str) -> LLMClient:
    key = provider.strip().lower()
    if key in ("mock", "none", "dry"):
        return MockLLMClient(model=model or "mock-llm")
    if key in ("local", "corpus", "handwritten", "agent"):
        return LocalCorpusLLMClient(model=model or "local-corpus")
    if key in ("xai", "grok"):
        return XAIClient(model=model)
    raise ValueError(f"Unknown LLM provider: {provider}")
