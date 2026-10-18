"""Application settings, loaded from environment variables and an optional .env file.

Design notes
------------
* Every variable uses the ``RAG_`` prefix (e.g. ``RAG_LLM_MODEL``) so we never
  collide with unrelated variables in the user's shell.
* Secrets are ``SecretStr`` so they do not leak into logs or ``repr()``.
* Settings are added *per milestone*, when the code that needs them exists.
  Chunk size / top-k are deliberately absent until Milestone 1 decides their units.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated configuration for the whole application."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Filesystem layout. `storage_dir` holds generated state (vector index,
    # manifests, benchmark results) and is git-ignored.
    data_dir: Path = Path("data")
    storage_dir: Path = Path("storage")

    # Embedding model identifier (resolved by the embedding provider in M1).
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Any OpenAI-compatible chat endpoint. Default targets local Ollama,
    # which needs no API key.
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "llama3.2:3b"
    llm_api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached ``Settings`` instance."""
    return Settings()
