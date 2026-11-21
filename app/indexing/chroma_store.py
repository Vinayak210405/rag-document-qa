"""Persistent vector storage on ChromaDB.

Chroma is used as a *durable store* for (id, text, embedding, metadata). Similarity ranking is
done by our own code in ``app.retrieval``, so the baseline stays transparent and testable.

Verified against chromadb 1.5.9: reopening an existing collection with different metadata
silently keeps the OLD metadata, so the embedding-model guard below is our own explicit check.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import ChromaError

from app.chunking.base import Chunk
from app.utils.logging import log_event

logger = logging.getLogger(__name__)

_PAGE = 5000  # rows fetched per Chroma call
_BATCH = 500  # rows written per Chroma call


class StorageError(RuntimeError):
    """The vector database could not be opened or written (message names the folder and fixes)."""


class IndexMismatchError(RuntimeError):
    """The stored index was built with a different embedding model than the one configured."""


@dataclass(frozen=True)
class IndexSnapshot:
    """The whole index in memory, sorted by (doc_id, ordinal) for deterministic tie-breaking."""

    ids: tuple[str, ...]
    texts: tuple[str, ...]
    metadatas: tuple[dict[str, Any], ...]
    embeddings: np.ndarray  # (n, d) float32

    def __len__(self) -> int:
        return len(self.ids)


@dataclass(frozen=True)
class DocumentInfo:
    doc_id: str
    name: str
    chunks: int


class ChromaVectorStore:
    def __init__(
        self, persist_dir: Path | None, collection_name: str, embedding_model: str, embedding_dim: int
    ) -> None:
        """Open (or create) the store. ``persist_dir=None`` gives a throw-away in-memory store.

        In-memory clients share state *within a process*, so callers must use a unique
        ``collection_name`` for each isolated in-memory store (the benchmark runner does).
        """
        chroma_settings = ChromaSettings(anonymized_telemetry=False)
        try:
            if persist_dir is None:
                self._client = chromadb.EphemeralClient(settings=chroma_settings)
            else:
                persist_dir.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=str(persist_dir), settings=chroma_settings)
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=None,  # we always supply vectors; never trigger Chroma's default model
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_model": embedding_model,
                    "embedding_dim": embedding_dim,
                },
            )
        except (ChromaError, OSError, sqlite3.Error) as exc:
            raise StorageError(
                f"could not open the vector database at {persist_dir!s}: {exc}. Likely causes: another "
                "server process is still using this folder (stop it with Ctrl+C), the folder is read-only, "
                "or antivirus is blocking the database file. Fix: close other servers, or point "
                "RAG_STORAGE_DIR at a different writable folder (e.g. C:\\rag-storage). The index can be "
                "rebuilt by re-uploading your documents."
            ) from exc
        stored = self._collection.metadata or {}
        if stored.get("embedding_model") != embedding_model or stored.get("embedding_dim") != embedding_dim:
            raise IndexMismatchError(
                f"index at {persist_dir!s} was built with embedding model "
                f"{stored.get('embedding_model')!r} (dim {stored.get('embedding_dim')}), but the configured "
                f"model is {embedding_model!r} (dim {embedding_dim}). Vectors from different models are not "
                "comparable. Re-ingest into a fresh storage directory (or delete this one)."
            )
        self.version = 0  # bumped on every write; lets retrievers invalidate cached matrices

    def drop(self) -> None:
        """Delete the collection (used to clean up throw-away in-memory benchmark stores)."""
        self._client.delete_collection(self._collection.name)

    def count(self) -> int:
        return self._collection.count()

    def replace_document(self, doc_id: str, chunks: Sequence[Chunk], embeddings: np.ndarray) -> bool:
        """Store a document's chunks, first removing any older chunks of the same document.

        Returns True if an older version existed. Not atomic (delete then add); the full document
        lifecycle (versioning, atomicity, hash-based skipping) arrives in Milestone 6.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        existing = self._collection.get(where={"doc_id": doc_id}, include=[])
        replaced = len(existing["ids"]) > 0
        if replaced:
            self._collection.delete(where={"doc_id": doc_id})
        for start in range(0, len(chunks), _BATCH):
            batch = chunks[start : start + _BATCH]
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings[start : start + _BATCH].tolist(),
                documents=[c.text for c in batch],
                metadatas=[self._metadata(c) for c in batch],
            )
        self.version += 1
        log_event(logger, "index_write", doc_id=doc_id, chunks=len(chunks), replaced=replaced)
        return replaced

    @staticmethod
    def _metadata(chunk: Chunk) -> dict[str, Any]:
        meta: dict[str, Any] = {"doc_id": chunk.doc_id, "doc_name": chunk.doc_name, "ordinal": chunk.ordinal}
        if chunk.page is not None:  # Chroma rejects None values, so optional keys are omitted
            meta["page"] = chunk.page
        if chunk.heading:
            meta["heading"] = chunk.heading
        for key, value in chunk.metadata.items():  # document-level fields; core keys can never be overridden
            meta.setdefault(key, value)
        return meta

    def _fetch_all(self, include: list[str]) -> dict[str, list[Any]]:
        out: dict[str, list[Any]] = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
        offset = 0
        while True:
            page = self._collection.get(limit=_PAGE, offset=offset, include=include)
            ids = page["ids"]
            if not ids:
                break
            out["ids"].extend(ids)
            for key in ("documents", "metadatas"):
                if key in include:
                    out[key].extend(page[key])
            if "embeddings" in include:
                out["embeddings"].extend(list(page["embeddings"]))
            offset += len(ids)
        return out

    def load_snapshot(self) -> IndexSnapshot:
        data = self._fetch_all(["documents", "metadatas", "embeddings"])
        if not data["ids"]:
            return IndexSnapshot((), (), (), np.zeros((0, 0), dtype=np.float32))
        order = sorted(range(len(data["ids"])), key=lambda i: (
            data["metadatas"][i]["doc_id"], data["metadatas"][i]["ordinal"]))
        return IndexSnapshot(
            ids=tuple(data["ids"][i] for i in order),
            texts=tuple(data["documents"][i] for i in order),
            metadatas=tuple(dict(data["metadatas"][i]) for i in order),
            embeddings=np.asarray([data["embeddings"][i] for i in order], dtype=np.float32),
        )

    def list_documents(self) -> list[DocumentInfo]:
        data = self._fetch_all(["metadatas"])
        counts: dict[str, tuple[str, int]] = {}
        for meta in data["metadatas"]:
            name, n = counts.get(meta["doc_id"], (meta["doc_name"], 0))
            counts[meta["doc_id"]] = (name, n + 1)
        return [DocumentInfo(doc_id=k, name=v[0], chunks=v[1]) for k, v in sorted(counts.items())]
