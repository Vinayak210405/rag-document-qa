"""Document parsing (PDF/DOCX/TXT), normalization, and structure detection."""

from app.ingestion.loader import SUPPORTED_TYPES, parse_document
from app.ingestion.validation import InvalidFileError
from app.ingestion.models import (
    Block,
    EmptyDocumentError,
    ExtractionError,
    IngestionError,
    OcrEngine,
    ParsedDocument,
    UnsupportedFileTypeError,
)

__all__ = [
    "Block", "EmptyDocumentError", "ExtractionError", "IngestionError", "InvalidFileError", "OcrEngine",
    "ParsedDocument", "SUPPORTED_TYPES", "UnsupportedFileTypeError", "parse_document",
]
