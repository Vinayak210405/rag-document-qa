"""Types returned by retrieval."""

from __future__ import annotations

from dataclasses import dataclass


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
