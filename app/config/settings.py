"""Application settings, loaded from environment variables and an optional .env file.

Design notes
------------
* Every variable uses the ``RAG_`` prefix (e.g. ``RAG_LLM_MODEL``) so we never
  collide with unrelated variables in the user's shell.
* Secrets are ``SecretStr`` so they do not leak into logs or ``repr()``.
* Settings are added *per milestone*, when the code that needs them exists.
* Chunk sizes are in **characters** (model-independent; ~4 chars ~ 1 English token).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
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
    # benchmark results) and is git-ignored.
    data_dir: Path = Path("data")
    storage_dir: Path = Path("storage")

    # --- Embeddings -------------------------------------------------------
    # "hashing" is a dependency-free lexical embedder: it runs offline with no
    # download (used by tests and for trying the app without PyTorch), but it
    # is NOT semantic. Use "sentence_transformers" for real retrieval quality.
    embedding_backend: Literal["sentence_transformers", "hashing"] = "sentence_transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # bge models are trained to embed short queries with this instruction prefix.
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "

    # --- Chunking / retrieval --------------------------------------------
    chunk_max_chars: int = Field(500, ge=100, le=4000)
    chunk_overlap_chars: int = Field(80, ge=0)
    default_top_k: int = Field(4, ge=1, le=20)
    retriever: Literal["dense", "hybrid"] = "dense"  # hybrid = dense + BM25 fused with RRF
    hybrid_candidates: int = Field(20, ge=1, le=200)
    rrf_k: int = Field(60, ge=0)
    chroma_collection: str = Field("documents", min_length=3)  # Chroma requires >= 3 chars

    # --- Uploads ----------------------------------------------------------
    max_upload_bytes: int = Field(20 * 1024 * 1024, ge=1)

    # --- LLM: any OpenAI-compatible chat endpoint (default: local Ollama) ---
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "llama3.2:3b"
    llm_api_key: SecretStr | None = None
    llm_timeout_seconds: float = Field(120.0, gt=0)
    llm_temperature: float = Field(0.0, ge=0, le=2)
    llm_max_tokens: int = Field(512, ge=16)

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> Self:
        if self.chunk_overlap_chars >= self.chunk_max_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_max_chars")
        return self

    @property
    def chroma_dir(self) -> Path:
        return self.storage_dir / "chroma"


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached ``Settings`` instance."""
    return Settings()
