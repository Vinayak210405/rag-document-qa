"""Entry point of ingestion: validate a file name, dispatch to a parser, return a ParsedDocument."""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.ingestion.models import (
    EmptyDocumentError,
    OcrEngine,
    ParsedDocument,
    UnsupportedFileTypeError,
)
from app.ingestion.normalize import safe_filename
from app.ingestion.parsers import parse_docx, parse_pdf, parse_txt

SUPPORTED_TYPES = {".txt": "txt", ".pdf": "pdf", ".docx": "docx"}


def parse_document(
    filename: str, content: bytes, *, ocr: OcrEngine | None = None
) -> ParsedDocument:
    """Parse raw file bytes into a normalized, structured document.

    Raises ``UnsupportedFileTypeError``, ``ExtractionError`` or ``EmptyDocumentError``
    with an actionable message; it never returns an empty document.
    """
    name = safe_filename(filename)
    suffix = Path(name).suffix.lower()
    file_type = SUPPORTED_TYPES.get(suffix)
    if file_type is None:
        raise UnsupportedFileTypeError(
            f"unsupported file type {suffix or '(none)'!r}; supported: "
            + ", ".join(sorted(SUPPORTED_TYPES))
        )
    if not content:
        raise EmptyDocumentError(f"{name} is empty (0 bytes)")

    page_count: int | None = None
    if file_type == "txt":
        blocks, warnings = parse_txt(content)
    elif file_type == "docx":
        blocks, warnings = parse_docx(content)
    else:
        blocks, page_count, warnings = parse_pdf(content, ocr)

    if not blocks:
        detail = f" ({'; '.join(warnings)})" if warnings else ""
        hint = (
            ": no extractable text. Scanned PDFs need OCR, which is not configured "
            "(see OcrEngine in app/ingestion/models.py)"
            if file_type == "pdf"
            else ": no extractable text"
        )
        raise EmptyDocumentError(f"{name}{hint}{detail}")

    return ParsedDocument(
        doc_id=name,
        name=name,
        content_hash=hashlib.sha256(content).hexdigest(),
        file_type=file_type,  # type: ignore[arg-type]
        blocks=tuple(blocks),
        page_count=page_count,
        warnings=tuple(warnings),
    )
