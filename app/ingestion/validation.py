"""Content-level validation of untrusted uploads, run BEFORE any parser touches the bytes.

The file extension is user-controlled, so we check that the bytes really look like the claimed type, and bound
the work a hostile file can cause: DOCX is a ZIP archive (zip bombs), PDFs can have enormous page counts.
This reduces risk; it does not make parsing untrusted files safe (run ingestion in a sandbox for that).
"""

from __future__ import annotations

import io
import zipfile

from app.ingestion.models import ExtractionError

MAX_ZIP_ENTRIES = 2000
MAX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200


class InvalidFileError(ExtractionError):
    """The bytes do not match the claimed file type, or exceed safety limits."""


def validate_signature(file_type: str, content: bytes) -> None:
    if file_type == "pdf":
        if b"%PDF-" not in content[:1024]:
            raise InvalidFileError("file is not a PDF (missing %PDF- header); the extension does not match the content")
    elif file_type == "docx":
        _validate_docx(content)
    elif file_type == "txt":
        if b"\x00" in content[:8192]:
            raise InvalidFileError("file looks binary (NUL bytes), not plain text")


def _validate_docx(content: bytes) -> None:
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise InvalidFileError("file is not a DOCX (not a valid ZIP archive)") from exc
    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_ZIP_ENTRIES:
            raise InvalidFileError(f"DOCX has too many archive entries ({len(infos)} > {MAX_ZIP_ENTRIES})")
        total = sum(i.file_size for i in infos)
        if total > MAX_UNCOMPRESSED_BYTES:
            raise InvalidFileError(f"DOCX expands to {total} bytes (limit {MAX_UNCOMPRESSED_BYTES}); possible zip bomb")
        for i in infos:
            if i.compress_size and i.file_size / i.compress_size > MAX_COMPRESSION_RATIO and i.file_size > 1_000_000:
                raise InvalidFileError(f"DOCX entry {i.filename!r} has a suspicious compression ratio")
        if "word/document.xml" not in {i.filename for i in infos}:
            raise InvalidFileError("file is a ZIP archive but not a Word document (no word/document.xml)")
