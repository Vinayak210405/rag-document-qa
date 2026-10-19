"""Text and file-name normalization."""

from __future__ import annotations

import re
import unicodedata

from app.ingestion.models import IngestionError

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MAX_NAME = 200


def normalize_text(text: str) -> str:
    """Return cleaned text with predictable whitespace.

    * Unicode NFKC: folds ligatures (``fi`` -> ``fi``), full-width forms, NBSP.
    * Removes soft hyphens and control characters (keeps ``\\n`` and tabs-as-space).
    * Normalizes line endings; collapses runs of spaces/tabs; limits blank lines to one.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_filename(name: str) -> str:
    """Reduce an uploaded file name to a safe base name (no directories, no odd characters).

    Used as the document ID, so it must be stable and must not allow path traversal.
    """
    base = re.split(r"[\\/]", name or "")[-1].strip()
    base = _CONTROL.sub("", base)
    base = re.sub(r"[^\w.\- ()]", "_", base)
    if base in {"", ".", ".."} or set(base) <= {"."}:
        raise IngestionError(f"invalid file name: {name!r}")
    return base[:_MAX_NAME]
