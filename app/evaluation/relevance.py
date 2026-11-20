"""Map gold evidence quotes onto chunks, independent of the chunking strategy.

A chunk "contains" a quote if the quote appears in it as a whole (whitespace-insensitive) AND the chunk
belongs to the quote's document. A quote cut in half by a chunk boundary is contained by neither chunk;
``reachable`` diagnostics report such losses so chunking damage is visible, not silent.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from app.evaluation.dataset import Evidence, Passage


def _squash(text: str) -> str:
    return " ".join(text.split())


def contains_quote(text: str, quote: str) -> bool:
    """True if ``quote`` appears in ``text`` (whitespace differences ignored)."""
    return _squash(quote) in _squash(text)


def chunk_grade(doc_id: str, text: str, evidence: Sequence[Evidence]) -> int:
    """Highest evidence grade fully contained in this chunk (0 if none, or if from another document)."""
    squashed = _squash(text)
    return max((e.grade for e in evidence if e.doc == doc_id and _squash(e.quote) in squashed), default=0)


def covered_evidence(chunks: Iterable[tuple[str, str]], evidence: Sequence[Evidence]) -> set[int]:
    """Indices of ``evidence`` items fully contained in at least one ``(doc_id, text)`` chunk."""
    squashed = [(doc, _squash(text)) for doc, text in chunks]
    return {
        i
        for i, e in enumerate(evidence)
        if any(doc == e.doc and _squash(e.quote) in text for doc, text in squashed)
    }


def contains_passage(doc_id: str, text: str, passages: Sequence[Passage]) -> bool:
    """True if the chunk fully contains any of the given passages (used for distractors)."""
    squashed = _squash(text)
    return any(p.doc == doc_id and _squash(p.quote) in squashed for p in passages)
