"""LLM client for any OpenAI-compatible chat endpoint (Ollama, OpenAI, Groq, ...), via httpx.

Failures raise ``LLMError`` with an actionable message. There is deliberately no fallback answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import httpx


class LLMError(RuntimeError):
    """The LLM endpoint was unreachable, returned an error, or returned something unusable."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None


class LLMClient(Protocol):
    def complete(self, messages: Sequence[dict[str, str]]) -> LLMResponse: ...


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int = 512,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._base_url = base_url
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout, transport=transport)

    def complete(self, messages: Sequence[dict[str, str]]) -> LLMResponse:
        payload = {
            "model": self._model,
            "messages": list(messages),
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        try:
            response = self._client.post(self._url, json=payload, headers=self._headers)
        except httpx.ConnectError as exc:
            raise LLMError(
                f"cannot reach the LLM endpoint at {self._base_url}. Is the server running "
                "(for Ollama: start it and run `ollama pull <model>`)?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"LLM request timed out after {self._timeout:.0f}s") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        if response.status_code >= 400:
            raise LLMError(f"LLM endpoint returned HTTP {response.status_code}: {response.text[:300]}")
        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage") or {}
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError("LLM endpoint returned a response in an unexpected format") from exc
        if not isinstance(text, str) or not text.strip():
            raise LLMError("LLM returned an empty completion")
        return LLMResponse(
            text=text.strip(),
            model=str(data.get("model", self._model)),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
