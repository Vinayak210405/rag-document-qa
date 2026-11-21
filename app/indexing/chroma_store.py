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

from app.chunking.base import Chunk, make_chunk_id
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
    content_hash: str | None = None
    ingest_version: int | None = None  # 1 on first ingest, +1 every time the content changes


@dataclass(frozen=True)
class DocumentState:
    """What the index currently holds for one document (used to skip unchanged re-uploads)."""

    doc_id: str
    content_hash: str | None
    index_signature: str | None  # chunker + embedding model the stored chunks were built with
    ingest_version: int
    chunk_ids: tuple[str, ...]
    groups: str = ""  # access-control groups the chunks were stored with (comma-separated; "" = unrestricted)


@dataclass(frozen=True)
class AuditReport:
    """Consistency findings. An empty report means: no stale, mixed or duplicate content was found."""

    documents: int
    chunks: int
    stale_chunks: tuple[str, ...]  # chunk_id does not match sha256(doc_id, ordinal, text): corrupt or hand-edited
    mixed_versions: tuple[str, ...]  # docs whose chunks carry more than one content hash (interrupted update)
    duplicate_ordinals: tuple[str, ...]  # docs with two chunks at the same position
    duplicate_texts: tuple[tuple[str, ...], ...]  # groups of chunk_ids (different docs) with identical text

    @property
    def clean(self) -> bool:
        return not (self.stale_chunks or self.mixed_versions or self.duplicate_ordinals)


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
        """Store a document's chunks, replacing any older version. Returns True if an older version existed.

        Order matters for crash safety: new chunks are written FIRST (upsert, content-derived IDs so unchanged
        chunks are simply overwritten in place), and only then are the old chunks that are no longer part of the
        document deleted. An interruption can therefore leave stale extra chunks (found by ``audit``) but never
        a document with missing content. Chroma has no multi-operation transaction, so this is ordering, not
        atomicity.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        existing = self._collection.get(where={"doc_id": doc_id}, include=[])
        old_ids = set(existing["ids"])
        for start in range(0, len(chunks), _BATCH):
            batch = chunks[start : start + _BATCH]
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings[start : start + _BATCH].tolist(),
                documents=[c.text for c in batch],
                metadatas=[self._metadata(c) for c in batch],
            )
        stale = sorted(old_ids - {c.chunk_id for c in chunks})
        for start in range(0, len(stale), _BATCH):
            self._collection.delete(ids=stale[start : start + _BATCH])
        self.version += 1
        log_event(logger, "index_write", doc_id=doc_id, chunks=len(chunks), replaced=bool(old_ids), removed=len(stale))
        return bool(old_ids)

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk of a document; returns how many were deleted (0 if it was not indexed)."""
        existing = self._collection.get(where={"doc_id": doc_id}, include=[])
        ids = list(existing["ids"])
        for start in range(0, len(ids), _BATCH):
            self._collection.delete(ids=ids[start : start + _BATCH])
        if ids:
            self.version += 1
        log_event(logger, "index_delete", doc_id=doc_id, chunks=len(ids))
        return len(ids)

    def document_state(self, doc_id: str) -> DocumentState | None:
        data = self._collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        if not data["ids"]:
            return None
        metas = data["metadatas"]
        hashes = {m.get("content_hash") for m in metas}
        sigs = {m.get("index_signature") for m in metas}
        return DocumentState(
            doc_id=doc_id,
            content_hash=hashes.pop() if len(hashes) == 1 else None,  # None = mixed: treat as changed
            index_signature=sigs.pop() if len(sigs) == 1 else None,
            ingest_version=max(int(m.get("ingest_version", 1)) for m in metas),
            chunk_ids=tuple(sorted(data["ids"])),
            groups=str(metas[0].get("groups", "")),
        )

    def audit(self) -> AuditReport:
        data = self._fetch_all(["documents", "metadatas"])
        stale: list[str] = []
        positions: dict[tuple[str, int], int] = {}
        hashes: dict[str, set[str | None]] = {}
        by_text: dict[str, list[tuple[str, str]]] = {}
        for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            doc_id, ordinal = str(meta["doc_id"]), int(meta["ordinal"])
            if make_chunk_id(doc_id, ordinal, text) != cid:
                stale.append(cid)
            positions[(doc_id, ordinal)] = positions.get((doc_id, ordinal), 0) + 1
            hashes.setdefault(doc_id, set()).add(meta.get("content_hash"))
            by_text.setdefault(text, []).append((doc_id, cid))
        return AuditReport(
            documents=len(hashes), chunks=len(data["ids"]), stale_chunks=tuple(sorted(stale)),
            mixed_versions=tuple(sorted(d for d, h in hashes.items() if len(h) > 1)),
            duplicate_ordinals=tuple(sorted({d for (d, _), n in positions.items() if n > 1})),
            duplicate_texts=tuple(
                tuple(sorted(c for _, c in group)) for group in by_text.values()
                if len({d for d, _ in group}) > 1
            ),
        )

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
        counts: dict[str, list] = {}
        for meta in data["metadatas"]:
            entry = counts.setdefault(meta["doc_id"], [meta["doc_name"], 0, meta.get("content_hash"), 1])
            entry[1] += 1
            entry[3] = max(entry[3], int(meta.get("ingest_version", 1)))
        return [
            DocumentInfo(doc_id=k, name=v[0], chunks=v[1], content_hash=v[2], ingest_version=v[3])
            for k, v in sorted(counts.items())
        ]
