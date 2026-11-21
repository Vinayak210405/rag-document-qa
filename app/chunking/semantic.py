"""Semantic chunking: split where the meaning shifts, using sentence embeddings.

Algorithm (per page, so page citations stay exact)
-------------------------------------------------
1. Reuse the baseline's units: each heading is a unit, each body block is split into sentences.
2. Embed every unit. For neighbours i, i+1 compute the cosine DISTANCE ``1 - cos(e_i, e_{i+1})``.
3. A breakpoint goes between i and i+1 when the distance is above the ``percentile``-th percentile of this page's
   distances (relative threshold: "the biggest topic jumps on this page"), or when unit i+1 is a heading.
4. Groups smaller than ``min_chars`` are merged forward (never across a heading) so we do not emit tiny fragments.
5. Groups longer than ``max_chars`` are packed with the baseline packer (no overlap: boundaries are semantic).

Costs vs the baseline: one embedding per sentence at INGEST time (counted in ``units_embedded``), more latency,
and a dependence on the embedder's quality. The benefit is NOT assumed: it is measured in the ablation
(docs/decisions.md D25). With the offline hashing embedder, "semantic" distances are lexical-overlap distances, so any
benchmark with it is a stand-in, not evidence about semantic chunking.
"""

from __future__ import annotations

from itertools import groupby

import numpy as np

from app.chunking.base import Chunk, make_chunk_id
from app.chunking.sentence import SentenceChunker, _join, _Unit
from app.embeddings.base import Embedder
from app.ingestion.models import ParsedDocument
from app.retrieval.similarity import cosine_similarity


class SemanticChunker:
    def __init__(
        self, embedder: Embedder, max_chars: int = 500, min_chars: int = 100, percentile: float = 90.0,
    ) -> None:
        if max_chars < 50 or not 0 <= min_chars < max_chars:
            raise ValueError("require max_chars >= 50 and 0 <= min_chars < max_chars")
        if not 0 < percentile < 100:
            raise ValueError("percentile must be in (0, 100)")
        self._embedder = embedder
        self.max_chars, self.min_chars, self.percentile = max_chars, min_chars, percentile
        self._packer = SentenceChunker(max_chars, 0)
        self.units_embedded = 0  # cumulative embedding calls' units, for the cost comparison

    @property
    def strategy_name(self) -> str:
        return f"semantic(max={self.max_chars},min={self.min_chars},p={self.percentile:g})"

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page, group in groupby(SentenceChunker._units(doc), key=lambda u: u.page):
            units = list(group)
            for segment in self._segments(units):
                for packed, lead in self._packer._pack(segment):
                    text = _join(packed)
                    ordinal = len(chunks)
                    chunks.append(Chunk(
                        chunk_id=make_chunk_id(doc.doc_id, ordinal, text), doc_id=doc.doc_id, doc_name=doc.name,
                        ordinal=ordinal, text=text, page=page, heading=packed[0].heading,
                        metadata=dict(doc.metadata),
                    ))
        return chunks

    def _segments(self, units: list[_Unit]) -> list[list[_Unit]]:
        if len(units) < 2:
            return [units] if units else []
        vectors = np.asarray(self._embedder.embed_documents([u.text for u in units]), dtype=np.float64)
        self.units_embedded += len(units)
        distances = np.array([
            1.0 - float(cosine_similarity(vectors[i], vectors[i + 1 : i + 2])[0]) for i in range(len(units) - 1)
        ])
        threshold = float(np.percentile(distances, self.percentile)) if len(distances) >= 3 else float("inf")
        segments: list[list[_Unit]] = [[units[0]]]
        for i in range(1, len(units)):
            if units[i].is_heading or distances[i - 1] > threshold:
                segments.append([units[i]])
            else:
                segments[-1].append(units[i])
        return self._merge_small(segments)

    def _merge_small(self, segments: list[list[_Unit]]) -> list[list[_Unit]]:
        merged: list[list[_Unit]] = []
        carry: list[_Unit] = []
        for seg in segments:
            seg = carry + seg
            carry = []
            size = len(_join(seg))
            if size < self.min_chars and not seg[-1].is_heading:
                carry = seg  # too small: join it with the next segment
            elif size < self.min_chars and seg[-1].is_heading:
                carry = seg
            else:
                merged.append(seg)
        if carry:
            if merged:
                merged[-1].extend(carry)
            else:
                merged.append(carry)
        return merged
