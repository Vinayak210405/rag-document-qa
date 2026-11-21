"""Hybrid retrieval: dense cosine + BM25, fused with RRF. Also a BM25-only retriever for ablations.

The BM25 index is rebuilt from the same store snapshot the dense retriever uses, keyed on the
store version, so the lexical and dense views can never disagree about which chunks exist.
"""

from __future__ import annotations

from app.embeddings.base import Embedder
from app.indexing.chroma_store import ChromaVectorStore, IndexSnapshot
from app.retrieval.bm25 import BM25Index
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.similarity import cosine_similarity, top_k
from app.retrieval.types import RetrievedChunk, build_retrieved


class _SnapshotRetriever:
    def __init__(self, store: ChromaVectorStore) -> None:
        self._store = store
        self._version = -1
        self._snapshot: IndexSnapshot | None = None
        self._bm25: BM25Index | None = None

    def _state(self) -> tuple[IndexSnapshot, BM25Index]:
        if self._snapshot is None or self._bm25 is None or self._version != self._store.version:
            self._snapshot = self._store.load_snapshot()
            self._bm25 = BM25Index(self._snapshot.texts)
            self._version = self._store.version
        return self._snapshot, self._bm25

    _build = staticmethod(build_retrieved)


class BM25Retriever(_SnapshotRetriever):
    """Keyword-only retrieval. Chunks with score 0 (no query term) are never returned."""

    def retrieve(self, query: str, top_k_: int) -> list[RetrievedChunk]:
        snapshot, bm25 = self._state()
        if len(snapshot) == 0:
            return []
        scores = bm25.scores(query)
        hits = [(i, s) for i, s in top_k(scores, top_k_) if s > 0]
        return [self._build(snapshot, r, i, s) for r, (i, s) in enumerate(hits, start=1)]


class HybridRetriever(_SnapshotRetriever):
    """Dense + BM25 candidate lists fused with RRF; ``score`` of a result is its RRF score."""

    def __init__(
        self, store: ChromaVectorStore, embedder: Embedder, *,
        candidates: int = 20, rrf_k: int = 60, dense_weight: float = 1.0, bm25_weight: float = 1.0,
    ) -> None:
        super().__init__(store)
        if candidates < 1:
            raise ValueError("candidates must be >= 1")
        self._embedder = embedder
        self.candidates, self.rrf_k = candidates, rrf_k
        self._weights = (dense_weight, bm25_weight)

    def retrieve(self, query: str, top_k_: int) -> list[RetrievedChunk]:
        snapshot, bm25 = self._state()
        if len(snapshot) == 0:
            return []
        n = max(self.candidates, top_k_)
        dense = [i for i, _ in top_k(cosine_similarity(self._embedder.embed_query(query), snapshot.embeddings), n)]
        lexical = [i for i, s in top_k(bm25.scores(query), n) if s > 0]
        fused = reciprocal_rank_fusion([dense, lexical], k=self.rrf_k, weights=self._weights)[:top_k_]
        return [self._build(snapshot, r, i, s) for r, (i, s) in enumerate(fused, start=1)]
