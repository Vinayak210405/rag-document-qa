"""Rule-based sentence splitting (no NLP model needed)."""

from __future__ import annotations

import re

# A boundary is sentence punctuation + whitespace + something that can start a sentence.
# Decimals ("2.0") never match because they have no whitespace after the dot.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_ABBREVIATIONS = frozenset(
    {"e.g.", "i.e.", "vs.", "mr.", "mrs.", "ms.", "dr.", "prof.", "fig.", "approx.", "inc.", "ltd.", "co."}
)
_INITIAL = re.compile(r"[A-Za-z]\.")


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, skipping known abbreviations and single-letter initials."""
    parts: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        candidate = text[start : match.start()]
        tokens = candidate.split()
        last = tokens[-1] if tokens else ""
        if last.lower() in _ABBREVIATIONS or _INITIAL.fullmatch(last):
            continue
        parts.append(candidate)
        start = match.end()
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]
