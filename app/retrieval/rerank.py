"""Second-stage reranking: re-score the first-stage candidates with a (query, passage) scorer.

First-stage retrievers (dense / hybrid) encode query and passage separately, which is fast but
coarse. A cross-encoder reads them together, which is slower but usually more precise, so we only
apply it to a small candidate pool. ``RerankedRetriever`` is orchestration only: it asks any base
retriever for ``candidates`` results, scores them, and sorts (ties keep first-stage order).

Rerankers:
- ``CrossEncoderReranker``: a pretrained sentence-transformers CrossEncoder (needs a model download).
- ``LexicalOverlapReranker``: a deterministic, offline STAND-IN (query-term coverage). It exists so the
  orchestration can be tested and benchmarked without a download. It is NOT a cross-encoder and its
  benchmark numbers must never be reported as cross-encoder results.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Collection, Protocol, Sequence

from app.retrieval.bm25 import tokenize
from app.retrieval.types import RetrievedChunk


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """One relevance score per passage (higher = more relevant)."""
        ...


class RerankerError(RuntimeError):
    """The reranker is unavailable or failed (never silently replaced by the first-stage order)."""


class LexicalOverlapReranker:
    """Fraction of distinct query tokens present in the passage. Deterministic stand-in, see module doc."""

    name = "lexical-overlap(stand-in)"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        q = set(tokenize(query))
        if not q:
            return [0.0] * len(passages)
        return [len(q & set(tokenize(p))) / len(q) for p in passages]


class CrossEncoderReranker:
    """Pretrained cross-encoder (e.g. ``cross-encoder/ms-marco-MiniLM-L-6-v2``); loaded lazily."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size: int = 16) -> None:
        self.name = model_name
        self._batch = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RerankerError("install the 'embeddings' extra: uv sync --extra embeddings") from exc
            try:
                self._model = CrossEncoder(self.name)
            except Exception as exc:  # network / missing model: surface it, do not fall back silently
                raise RerankerError(f"could not load cross-encoder {self.name!r}: {exc}") from exc
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        scores = self._load().predict([(query, p) for p in passages], batch_size=self._batch)
        return [float(s) for s in scores]


class RerankedRetriever:
    def __init__(self, base, reranker: Reranker, candidates: int = 20) -> None:
        if candidates < 1:
            raise ValueError("candidates must be >= 1")
        self._base, self._reranker, self.candidates = base, reranker, candidates

    def retrieve(self, query: str, top_k_: int, groups: Collection[str] | None = None) -> list[RetrievedChunk]:
        pool = self._base.retrieve(query, max(self.candidates, top_k_), groups)
        if not pool:
            return []
        scores = self._reranker.score(query, [c.text for c in pool])
        if len(scores) != len(pool):
            raise RerankerError(f"reranker returned {len(scores)} scores for {len(pool)} passages")
        order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))[:top_k_]
        return [replace(pool[i], rank=r, score=scores[i]) for r, i in enumerate(order, start=1)]
