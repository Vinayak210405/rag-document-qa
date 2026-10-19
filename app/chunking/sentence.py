"""Sentence-aware chunking with overlap: the baseline strategy.

Algorithm
---------
1. Turn blocks into *units*: each heading is one unit, each body block is split into sentences.
2. Group units by page (chunks never cross a page, so page citations stay exact).
3. Greedily pack whole units while the joined text stays <= ``max_chars``.
4. When a chunk closes, carry its trailing units (<= ``overlap_chars``) into the next chunk, so
   a fact that straddles a boundary appears intact in at least one chunk.
5. A heading is never left stranded at the end of a chunk: it moves to the next one.
6. A single unit longer than ``max_chars`` is hard-split on whitespace (with overlap).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import groupby
from typing import Sequence

from app.chunking.base import Chunk, make_chunk_id
from app.chunking.sentences import split_sentences
from app.ingestion.models import ParsedDocument


@dataclass(frozen=True)
class _Unit:
    text: str
    is_heading: bool
    page: int | None
    heading: str | None  # heading in force for this unit


def _join(units: Sequence[_Unit]) -> str:
    """Join units: a newline around headings, a space between sentences."""
    parts: list[str] = []
    for i, unit in enumerate(units):
        if i:
            parts.append("\n" if units[i - 1].is_heading or unit.is_heading else " ")
        parts.append(unit.text)
    return "".join(parts)


def _hard_split(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split one over-long unit into <= max_chars pieces, preferring whitespace cuts."""
    pieces: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            cut = text.rfind(" ", start + max_chars // 2, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)  # always make progress
    return pieces


class SentenceChunker:
    """Baseline chunker. Sizes are in characters (see settings docs)."""

    def __init__(self, max_chars: int = 500, overlap_chars: int = 80) -> None:
        if max_chars < 50:
            raise ValueError("max_chars must be >= 50")
        if not 0 <= overlap_chars < max_chars:
            raise ValueError("overlap_chars must satisfy 0 <= overlap_chars < max_chars")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    @property
    def strategy_name(self) -> str:
        return f"sentence(max={self.max_chars},overlap={self.overlap_chars})"

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page, group in groupby(self._units(doc), key=lambda u: u.page):
            for packed, lead in self._pack(list(group)):
                text = _join(packed)
                ordinal = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(doc.doc_id, ordinal, text),
                        doc_id=doc.doc_id,
                        doc_name=doc.name,
                        ordinal=ordinal,
                        text=text,
                        page=page,
                        heading=packed[min(lead, len(packed) - 1)].heading,  # first NEW unit
                    )
                )
        return chunks

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _units(doc: ParsedDocument) -> list[_Unit]:
        units: list[_Unit] = []
        heading: str | None = None
        for block in doc.blocks:
            if block.kind == "heading":
                heading = block.text
                units.append(_Unit(block.text, True, block.page, heading))
            else:
                units.extend(
                    _Unit(sentence, False, block.page, heading)
                    for sentence in split_sentences(block.text)
                )
        return units

    def _overlap_tail(self, closed: Sequence[_Unit]) -> list[_Unit]:
        tail: list[_Unit] = []
        size = 0
        for unit in reversed(closed):
            size += len(unit.text) + 1
            if size > self.overlap_chars:
                break
            tail.insert(0, unit)
        if tail and all(u.is_heading for u in tail):  # never overlap on a lone heading
            return []
        return tail

    def _pack(self, units: list[_Unit]) -> list[tuple[list[_Unit], int]]:
        """Return ``(units, lead)`` pairs; ``lead`` = how many leading units are overlap."""
        packed: list[tuple[list[_Unit], int]] = []
        current: list[_Unit] = []
        lead = 0  # leading units of `current` carried over as overlap
        fresh = 0  # units in `current` that are new (not carried over as overlap)
        for unit in units:
            if len(unit.text) > self.max_chars:
                if fresh:
                    packed.append((current, lead))
                current, lead, fresh = [], 0, 0
                packed.extend(
                    ([replace(unit, text=piece)], 0)
                    for piece in _hard_split(unit.text, self.max_chars, self.overlap_chars)
                )
                continue
            if fresh and len(_join([*current, unit])) > self.max_chars:
                closed, carry = current, []
                if (
                    len(closed) > 1
                    and closed[-1].is_heading
                    and len(_join([closed[-1], unit])) <= self.max_chars
                ):
                    carry, closed = [closed[-1]], closed[:-1]
                packed.append((closed, lead))
                tail = self._overlap_tail(closed)
                while tail and len(_join([*tail, *carry, unit])) > self.max_chars:
                    tail = tail[1:]  # shrink overlap so the size limit always holds
                current, lead, fresh = [*tail, *carry], len(tail), len(carry)
            current.append(unit)
            fresh += 1
        if fresh:
            packed.append((current, lead))
        return packed
