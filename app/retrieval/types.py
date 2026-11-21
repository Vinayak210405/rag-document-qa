"""Types returned by retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved passage with its score and provenance."""

    rank: int  # 1-based position in the result list
    score: float  # cosine similarity in [-1, 1]
    chunk_id: str
    doc_id: str
    doc_name: str
    ordinal: int
    page: int | None
    heading: str | None
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)  # doc_family, doc_version, status_class, ...


_CORE_KEYS = {"doc_id", "doc_name", "ordinal", "page", "heading"}


def build_retrieved(snapshot, rank: int, index: int, score: float) -> RetrievedChunk:
    """Build a RetrievedChunk from row ``index`` of an index snapshot (shared by all retrievers)."""
    meta = snapshot.metadatas[index]
    return RetrievedChunk(
        rank=rank, score=score, chunk_id=snapshot.ids[index], doc_id=str(meta["doc_id"]),
        doc_name=str(meta["doc_name"]), ordinal=int(meta["ordinal"]),
        page=int(meta["page"]) if "page" in meta else None,
        heading=str(meta["heading"]) if "heading" in meta else None, text=snapshot.texts[index],
        metadata={k: v for k, v in meta.items() if k not in _CORE_KEYS},
    )
