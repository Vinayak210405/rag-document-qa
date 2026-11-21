"""Dense retrieval: embed the query, score every stored chunk with OUR cosine, take OUR top-k.

Brute-force exact search: O(n*d) per query. Fine to ~1e5 chunks; beyond that, use an
approximate index (Chroma's HNSW) and compare it against this baseline.
"""

from __future__ import annotations

from app.embeddings.base import Embedder
from app.indexing.chroma_store import ChromaVectorStore, IndexSnapshot
from app.retrieval.similarity import cosine_similarity, top_k
from app.retrieval.types import RetrievedChunk, build_retrieved


class DenseRetriever:
    def __init__(self, store: ChromaVectorStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder
        self._snapshot: IndexSnapshot | None = None
        self._snapshot_version = -1

    def _current_snapshot(self) -> IndexSnapshot:
        """Cache the vector matrix in memory; reload whenever the store has been written to."""
        if self._snapshot is None or self._snapshot_version != self._store.version:
            self._snapshot = self._store.load_snapshot()
            self._snapshot_version = self._store.version
        return self._snapshot

    def retrieve(self, query: str, top_k_: int) -> list[RetrievedChunk]:
        snapshot = self._current_snapshot()
        if len(snapshot) == 0:
            return []
        scores = cosine_similarity(self._embedder.embed_query(query), snapshot.embeddings)
        return [
            build_retrieved(snapshot, rank, index, score)
            for rank, (index, score) in enumerate(top_k(scores, top_k_), start=1)
        ]
