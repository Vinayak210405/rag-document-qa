"""BM25 (Okapi) keyword scoring, implemented from scratch.

For a query term t and document d:

    score(q, d) = sum_t  idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
    idf(t)      = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))        (Lucene variant: never negative)

``tf`` is the term count in d, ``n_t`` the number of documents containing t, ``N`` the corpus
size, ``|d|`` the document length in tokens. ``k1`` controls term-frequency saturation and ``b``
length normalisation. Query terms are de-duplicated (a repeated query word is not double-counted).

Tokenisation keeps identifiers intact: "INV-2041" -> ["inv-2041", "inv", "2041"], so both the exact
identifier and its parts can match. No stemming or stop-word removal: those are language-specific
choices we would have to evaluate, not assume.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")
_PART = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase tokens; compound identifiers (a-b, v1.2, a_b) also yield their parts."""
    tokens: list[str] = []
    for match in _TOKEN.findall(text.lower()):
        tokens.append(match)
        parts = _PART.findall(match)
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


class BM25Index:
    """Immutable BM25 index over a list of texts; row i of every result refers to ``texts[i]``."""

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("require k1 >= 0 and 0 <= b <= 1")
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(t)) for t in texts]
        self._len = np.array([sum(c.values()) for c in self._tf], dtype=np.float64)
        self._avgdl = float(self._len.mean()) if len(texts) else 0.0
        df: Counter[str] = Counter()
        for counts in self._tf:
            df.update(counts.keys())
        n = len(texts)
        self._idf = {t: math.log(1.0 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def __len__(self) -> int:
        return len(self._tf)

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for ``query`` (0.0 where no query term matches)."""
        out = np.zeros(len(self._tf), dtype=np.float64)
        if not len(self._tf) or self._avgdl == 0:
            return out
        norm = self.k1 * (1.0 - self.b + self.b * self._len / self._avgdl)
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            tf = np.array([c.get(term, 0) for c in self._tf], dtype=np.float64)
            out += idf * tf * (self.k1 + 1.0) / (tf + norm)
        return out
