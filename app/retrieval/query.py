"""Query validation and normalization."""

from __future__ import annotations

import re

from app.ingestion.normalize import normalize_text

MAX_QUERY_CHARS = 1000


class InvalidQueryError(ValueError):
    """The user's question is empty or too long."""


def normalize_query(query: str) -> str:
    """Clean a user question: Unicode-normalise, collapse whitespace, reject empty/oversized input."""
    cleaned = re.sub(r"\s+", " ", normalize_text(query or ""))
    if not cleaned:
        raise InvalidQueryError("question is empty")
    if len(cleaned) > MAX_QUERY_CHARS:
        raise InvalidQueryError(f"question is too long ({len(cleaned)} > {MAX_QUERY_CHARS} characters)")
    return cleaned
