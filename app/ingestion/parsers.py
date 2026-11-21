"""Format-specific parsers. Each returns blocks (+ page count and warnings); none hides failures."""

from __future__ import annotations

import io
import re

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from app.ingestion.models import Block, ExtractionError, OcrEngine
from app.ingestion.normalize import normalize_text
from app.ingestion.structure import text_to_blocks


def parse_txt(content: bytes) -> tuple[list[Block], list[str]]:
    """Decode as UTF-8 (BOM tolerated), falling back to Windows-1252 with a warning."""
    warnings: list[str] = []
    try:
        raw = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            raw = content.decode("cp1252")
        except UnicodeDecodeError as exc:
            raise ExtractionError("text file is neither valid UTF-8 nor Windows-1252") from exc
        warnings.append("file is not UTF-8; decoded as Windows-1252")
    text = normalize_text(raw)
    return text_to_blocks(text, page=None, is_document_start=True), warnings


def parse_docx(content: bytes) -> tuple[list[Block], list[str]]:
    """Read paragraphs (headings detected from Word styles) and tables (one block per row)."""
    try:
        document = docx.Document(io.BytesIO(content))
    except Exception as exc:  # python-docx raises several unrelated types for bad input
        raise ExtractionError(f"could not open DOCX file: {exc}") from exc

    blocks: list[Block] = []
    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            text = re.sub(r"\s*\n\s*", " ", normalize_text(item.text))
            if not text:
                continue
            style = item.style.name if item.style is not None and item.style.name else ""
            kind = "heading" if style.startswith("Heading") or style == "Title" else "body"
            blocks.append(Block(text, kind, None))
        elif isinstance(item, Table):
            for row in item.rows:
                cells: list[str] = []
                for cell in row.cells:
                    cell_text = re.sub(r"\s+", " ", normalize_text(cell.text))
                    if cell_text and (not cells or cells[-1] != cell_text):  # merged cells repeat
                        cells.append(cell_text)
                if cells:
                    blocks.append(Block(" | ".join(cells), "body", None))
    return blocks, []


def _dehyphenate(text: str) -> str:
    """Rejoin words split by a line-end hyphen (``exam-\\nple``); lowercase-only to stay safe."""
    return re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)


def parse_pdf(
    content: bytes, ocr: OcrEngine | None = None, max_pages: int | None = None
) -> tuple[list[Block], int, list[str]]:
    """Extract text page by page, keeping page numbers; report pages that yield nothing."""
    try:
        reader = PdfReader(io.BytesIO(content))
        page_count = len(reader.pages)
    except Exception as exc:  # pypdf raises several unrelated types for bad input
        raise ExtractionError(f"could not open PDF file: {exc}") from exc

    if max_pages is not None and page_count > max_pages:
        raise ExtractionError(f"PDF has {page_count} pages, above the configured limit of {max_pages}")
    blocks: list[Block] = []
    warnings: list[str] = []
    empty_pages: list[int] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # reported as a warning, never hidden
            warnings.append(f"page {number}: text extraction failed ({exc})")
            raw = ""
        text = normalize_text(_dehyphenate(raw))
        if not text and ocr is not None:
            text = normalize_text(ocr.ocr_page(content, number))
            if text:
                warnings.append(f"page {number}: text recovered with OCR")
        if not text:
            empty_pages.append(number)
            continue
        blocks.extend(text_to_blocks(text, page=number, is_document_start=not blocks))

    if empty_pages and blocks:
        warnings.append(
            f"{len(empty_pages)} of {page_count} pages had no extractable text "
            f"(pages {empty_pages[:10]}); they may be scanned images and are not indexed"
        )
    return blocks, page_count, warnings
