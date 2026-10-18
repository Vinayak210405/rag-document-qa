"""Settings: defaults, env overrides, validation, and secret handling."""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from app.config import Settings


@pytest.fixture(autouse=True)
def clean_rag_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove RAG_* variables so tests never depend on the developer's shell."""
    for key in [k for k in os.environ if k.startswith("RAG_")]:
        monkeypatch.delenv(key)


def make() -> Settings:
    # _env_file=None: never read a developer's real .env during tests.
    return Settings(_env_file=None)


def test_defaults_target_local_ollama_with_no_key() -> None:
    s = make()
    assert s.llm_base_url == "http://localhost:11434/v1"
    assert s.llm_api_key is None
    assert s.log_level == "INFO"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_MODEL", "my-model")
    monkeypatch.setenv("RAG_LOG_LEVEL", "DEBUG")
    s = make()
    assert s.llm_model == "my-model"
    assert s.log_level == "DEBUG"


def test_invalid_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LOG_LEVEL", "LOUD")
    with pytest.raises(ValidationError):
        make()


def test_api_key_is_not_leaked_in_repr_or_str(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_API_KEY", "sk-super-secret")
    s = make()
    assert "sk-super-secret" not in repr(s)
    assert "sk-super-secret" not in str(s)
    assert s.llm_api_key is not None
    assert s.llm_api_key.get_secret_value() == "sk-super-secret"
