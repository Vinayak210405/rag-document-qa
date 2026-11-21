"""Chunk data type, stable chunk IDs, and the interface every chunking strategy implements."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

from app.ingestion.models import ParsedDocument


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage with full provenance."""

    chunk_id: str
    doc_id: str
    doc_name: str
    ordinal: int  # 0-based position within the document
    text: str
    page: int | None
    heading: str | None  # section heading in force at the start of the chunk
    metadata: dict[str, str | int | bool] = field(default_factory=dict)  # document-level fields, not part of the ID

    @property
    def char_count(self) -> int:
        return len(self.text)


def make_chunk_id(doc_id: str, ordinal: int, text: str) -> str:
    """Stable 16-hex-char ID: sha256(doc_id, position, text).

    Same document + same position + same text -> same ID on every run and machine.
    Changing the text (or moving the chunk) changes the ID, so stale chunks can be detected.
    """
    payload = f"{doc_id}\x00{ordinal}\x00{text}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


class Chunker(Protocol):
    """Interface shared by the sentence-aware baseline (M1) and semantic chunking (M7)."""

    @property
    def strategy_name(self) -> str: ...

    def chunk(self, doc: ParsedDocument) -> list[Chunk]: ...
