"""Data types and errors shared by the ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol


class IngestionError(Exception):
    """Base class for all document-ingestion failures (never swallowed silently)."""


class UnsupportedFileTypeError(IngestionError):
    """The file extension is not one of .pdf / .docx / .txt."""


class ExtractionError(IngestionError):
    """The file could not be opened or parsed."""


class EmptyDocumentError(ExtractionError):
    """Parsing succeeded but yielded no usable text (e.g. a scanned PDF)."""


@dataclass(frozen=True)
class Block:
    """A contiguous piece of text with its role and provenance."""

    text: str
    kind: Literal["heading", "body"]
    page: int | None  # 1-based page for PDFs; None for formats without pages


@dataclass(frozen=True)
class ParsedDocument:
    """Normalized, structured text of one source file plus provenance."""

    doc_id: str  # stable identifier (the sanitized file name)
    name: str
    content_hash: str  # sha256 of the raw bytes
    file_type: Literal["txt", "pdf", "docx"]
    blocks: tuple[Block, ...]
    page_count: int | None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, str] = field(default_factory=dict)  # header fields: doc_family, doc_version, ...

    @property
    def char_count(self) -> int:
        return sum(len(b.text) for b in self.blocks)


class OcrEngine(Protocol):
    """Extension point for scanned PDFs (not implemented in the baseline).

    Implementations render ``page_number`` (1-based) of the PDF and return its text.
    Pass one to ``parse_document(..., ocr=engine)``; it is only called for pages
    with no extractable text.
    """

    def ocr_page(self, pdf_bytes: bytes, page_number: int) -> str: ...
