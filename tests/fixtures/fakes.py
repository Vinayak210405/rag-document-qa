"""Test doubles."""

from __future__ import annotations

from typing import Sequence

from app.generation.llm import LLMError, LLMResponse


class FakeLLM:
    """Returns a canned answer (or raises) and records every prompt it was given."""

    def __init__(self, text: str = "Answer [1].", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: Sequence[dict[str, str]]) -> LLMResponse:
        self.calls.append(list(messages))
        if self.error:
            raise self.error
        return LLMResponse(text=self.text, model="fake-llm", prompt_tokens=11, completion_tokens=3)


class FakeOcr:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pages: list[int] = []

    def ocr_page(self, pdf_bytes: bytes, page_number: int) -> str:
        self.pages.append(page_number)
        return self.text


__all__ = ["FakeLLM", "FakeOcr", "LLMError"]
