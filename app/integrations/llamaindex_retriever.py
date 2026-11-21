"""LlamaIndex retrieval over OUR Chroma collection, for comparison against our own dense retriever (Milestone 10).

Purpose: show that the framework-independent core is not a worse reimplementation. LlamaIndex reads the same vectors we
wrote (through ``BaseEmbedding`` adapter around our embedder, so the query vector is identical) and ranks with Chroma's
HNSW index, whereas ``DenseRetriever`` does exact brute-force cosine in NumPy.

Differences to keep in mind:
- Scores are on a different scale (LlamaIndex reports ``exp(-distance)``, ours is cosine similarity); ranks are comparable.
- Ties and approximate search can order results differently from our deterministic tie-break.
- Access control (``groups``) is NOT supported here: this retriever exists for benchmarking, not serving.
Requires ``uv sync --extra llamaindex``.
"""

from __future__ import annotations

from typing import Collection

from app.embeddings.base import Embedder
from app.indexing.chroma_store import ChromaVectorStore
from app.retrieval.types import _CORE_KEYS, RetrievedChunk


class LlamaIndexUnavailableError(RuntimeError):
    pass


class LlamaIndexRetriever:
    def __init__(self, store: ChromaVectorStore, embedder: Embedder) -> None:
        try:
            from llama_index.core import VectorStoreIndex
            from llama_index.core.embeddings import BaseEmbedding
            from llama_index.vector_stores.chroma import ChromaVectorStore as LlamaChroma
        except ImportError as exc:
            raise LlamaIndexUnavailableError("LlamaIndex is not installed. Run: uv sync --extra llamaindex") from exc

        class _Adapter(BaseEmbedding):  # routes LlamaIndex's embedding calls to OUR embedder
            def _get_query_embedding(self, query: str) -> list[float]:
                return embedder.embed_query(query).tolist()

            def _get_text_embedding(self, text: str) -> list[float]:
                return embedder.embed_documents([text])[0].tolist()

            async def _aget_query_embedding(self, query: str) -> list[float]:
                return self._get_query_embedding(query)

            async def _aget_text_embedding(self, text: str) -> list[float]:
                return self._get_text_embedding(text)

        self._index = VectorStoreIndex.from_vector_store(
            LlamaChroma(chroma_collection=store.collection), embed_model=_Adapter(model_name=embedder.model_name)
        )

    def retrieve(self, query: str, top_k_: int, groups: Collection[str] | None = None) -> list[RetrievedChunk]:
        if groups is not None:
            raise NotImplementedError("the LlamaIndex comparison retriever does not enforce access control")
        nodes = self._index.as_retriever(similarity_top_k=top_k_).retrieve(query)
        out: list[RetrievedChunk] = []
        for rank, n in enumerate(nodes, start=1):
            meta = dict(n.node.metadata)
            out.append(RetrievedChunk(
                rank=rank, score=float(n.score or 0.0), chunk_id=n.node.node_id, doc_id=str(meta["doc_id"]),
                doc_name=str(meta["doc_name"]), ordinal=int(meta["ordinal"]), page=int(meta["page"]) if "page" in meta else None,
                heading=str(meta["heading"]) if "heading" in meta else None, text=n.node.get_content(),
                metadata={k: v for k, v in meta.items() if k not in _CORE_KEYS},
            ))
        return out
