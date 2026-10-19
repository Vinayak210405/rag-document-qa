"""Shared fixtures. Nothing here touches the network or downloads a model."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from app.chunking import Chunker, SentenceChunker
from app.embeddings import HashingEmbedder
from app.indexing import ChromaVectorStore
from app.retrieval.dense import DenseRetriever
from app.service import RagService
from tests.fixtures.fakes import FakeLLM

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "sample_corpus"


@pytest.fixture(scope="session")
def corpus_files() -> dict[str, bytes]:
    """The 7 sample documents as {file name: bytes}."""
    return {p.name: p.read_bytes() for p in sorted(CORPUS_DIR.glob("*.txt"))}


@pytest.fixture
def make_service(tmp_path: Path) -> Callable[..., RagService]:
    """Factory: a real Chroma store in tmp_path + the offline hashing embedder + a fake LLM."""

    def _make(llm: FakeLLM | None = None, chunker: Chunker | None = None, subdir: str = "chroma") -> RagService:
        embedder = HashingEmbedder()
        store = ChromaVectorStore(tmp_path / subdir, "documents", embedder.model_name, embedder.dimension)
        return RagService(
            embedder=embedder,
            store=store,
            retriever=DenseRetriever(store, embedder),
            chunker=chunker or SentenceChunker(500, 80),
            llm=llm or FakeLLM(),
            default_top_k=4,
        )

    return _make
